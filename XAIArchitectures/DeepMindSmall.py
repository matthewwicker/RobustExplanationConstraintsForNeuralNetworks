import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import GradCertModule
import pytorch_lightning as pl


class DeepMindSmall(pl.LightningModule):
    def __init__(self, dataset="MNIST", mode="Grad", nclass=10):
        self.num_cls = nclass
        self.mode = mode
        super().__init__()
        self.save_hyperparameters()
        self.activations = [torch.relu] # This needs to reflect what is used in the forward pass :)
        if(dataset=="MNIST"):
            self.num_cls = 10
            self.in_chans = 1
            self.lat_dim = 3200
            self.std = torch.tensor([0.3081])[None, :, None, None]
        elif(dataset=="CIFAR10"):
            self.num_cls = 10
            self.in_chans = 3
            self.lat_dim = 4608
            elf.std = torch.tensor([0.2023, 0.1994, 0.2010])[None, :, None, None]
        elif(dataset=="MEDMNIST"):
            self.num_cls = 10
            self.in_chans = 3
            self.lat_dim = 3200
        elif(dataset == "PNEUMONIAMNIST"):
            self.num_cls = 2
            self.in_chans = 1
            self.lat_dim = 15488
        elif(dataset == "TISSUEMNIST"):
            self.num_cls = 8
            self.in_chans = 1
            self.lat_dim = 3200
        elif(dataset == "DERMAMNIST"):
            self.num_cls = 7
            self.in_chans = 3
            self.lat_dim = 3200
            
        self.ALPHA = 0.0             # Regularization Parameter (Weights the Reg. Term)
        self.EPSILON = 0.0           # Input Peturbation Budget at Training Time
        self.GAMMA = 0.0             # Model Peturbation Budget at Training Time 
                                     #(Changed to proportional budget rather than absolute)

        self.LEARN_RATE = 0.001      # Learning Rate Hyperparameter
        self.MAX_EPOCHS = 10         # Maximum Epochs to Train the Model for

        self.EPSILON_LINEAR = True   # Put Epsilon on a Linear Schedule?
        self.GAMMA_LINEAR = True     # Put Gamma on a Linear Schedule?
        
        self.layers = []
        
        self.conv1 = nn.Conv2d(self.in_chans, 16, 4, 2)
        self.layers.append("Conv2")
        self.conv2 = nn.Conv2d(16, 32, 4, 1)
        self.layers.append("Conv1")
        self.layers.append("Flatten")
        self.fc1 = nn.Linear(self.lat_dim, 100)
        self.layers.append("Linear")
        self.fc2 = nn.Linear(100, self.num_cls)
        self.layers.append("Linear")
        
        if(self.EPSILON_LINEAR):
            self.eps = 0.0
        else:
            self.eps = self.EPSILON
        if(self.GAMMA_LINEAR):
            self.gam = 0.0
        else:
            self.gam = self.GAMMA
        self.mode = mode.upper()
        self.inputfooling = False
        
    def set_params(self, **kwargs):
        self.ALPHA =  kwargs.get('alpha', 0.00)
        self.GAMMA =  kwargs.get('gamma', 0.00)
        self.EPSILON =  kwargs.get('epsilon', 0.00)
        self.LEARN_RATE =  kwargs.get('learn_rate', 0.001)
        self.MAX_EPOCHS =  int(kwargs.get('max_epochs', 15))
        self.EPSILON_LINEAR = bool(kwargs.get('epsilon_linear', True))
        self.GAMMA_LINEAR = bool(kwargs.get('gamma_linear', True))
        
    def inputfooling_ON(self):
        self.inputfooling = True
        
    def inputfooling_OFF(self):
        self.inputfooling = False
        
    def classify(self, x):
        outputs = self.forward(x)
        return F.softmax(outputs, dim=1), torch.max(outputs, 1)[1]
    
    def forward(self, x):
        if(self.inputfooling):
            beta = 5
            try:
                x = F.softplus(self.conv1(x), beta=beta)
                x = F.softplus(self.conv2(x), beta=beta)
                x = torch.flatten(x, 1) 
                x = F.softplus(self.fc1(x), beta=beta)
                x = self.fc2(x)
            except:
                x = torch.flatten(x,0) 
                x = F.softplus(self.fc1(x), beta=beta)
                x = self.fc2(x)   
        else:
            try:
                x = F.relu(self.conv1(x))
                #print("First: ", x)
                x = F.relu(self.conv2(x))
                #print("Second: ", x)
                x = torch.flatten(x, 1)
                #print("Third: ", x)
                x = F.relu(self.fc1(x))
                #print("Fourth: ", x)
                x = self.fc2(x)
            except:
                x = torch.flatten(x,0) 
                x = F.relu(self.fc1(x))
                x = self.fc2(x)              
        return x

    def training_step(self, batch, batch_idx):
        #weights = [t for t in model.parameters()]
        x, y = batch
        y_hat = self(x)
        regval = 0.0
        if(self.mode == 'GRAD'):
            regval = GradCertModule.GradCertRegularizer(self, x, y, self.eps, self.gam, nclasses=self.num_cls)
        elif(self.mode == 'ADV'):
            regval = GradCertModule.RobustnessRegularizer(self, x, y, self.eps, self.gam, nclasses=self.num_cls)
        elif(self.mode == 'PGD'):
            regval = GradCertModule.PGDRegularizer(self, x, y, self.eps)
        elif(self.mode == 'BOTH'):
            regval += GradCertModule.GradCertRegularizer(self, x, y, self.eps, self.gam, nclasses=self.num_cls)
            regval += GradCertModule.RobustnessRegularizer(self, x, y, self.eps, self.gam, nclasses=self.num_cls)
        elif self.mode == "HESSIAN":
            regval += GradCertModule.HessianRegularizer(x, y, y_hat)
        elif self.mode == "L2":
            regval += GradCertModule.L2Regularizer(self, x, y, y_hat)
        #print(y_hat.shape, y.shape)
        #print(y_hat, y)
        loss = F.cross_entropy(y_hat, y)  + (self.ALPHA*regval)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)
        if(self.EPSILON_LINEAR):
            self.eps += self.EPSILON/self.MAX_EPOCHS
        if(self.GAMMA_LINEAR):
            self.gam += self.GAMMA/self.MAX_EPOCHS
        
    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LEARN_RATE)
    
    def save(self, trainer):
        directory = "Models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        SCHEDULED = self.EPSILON_LINEAR or self.GAMMA_LINEAR
        MODEL_ID = "DeepMindSmall_e=%s_g=%s_a=%s_s=%s"%(self.EPSILON, self.GAMMA, self.ALPHA, SCHEDULED)
        trainer.save_checkpoint("Models/%s.ckpt"%(MODEL_ID))
        torch.save(model.state_dict(), "Models/%s.pt"%(MODEL_ID))
    
