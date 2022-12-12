import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import GradCertModule
import pytorch_lightning as pl

from typing import Union
import numpy as np 

class FullyConnected(pl.LightningModule):
    
    def __init__(self, hidden_dim: int = 128, hidden_lay: int = 1, learning_rate: float = 0.001,  
                 dataset="MNIST", mode="GRAD",
                 epsilon: float = 0.00, gamma: float = 0.00, alpha: float = 0.00):
        super().__init__()
        self.save_hyperparameters()
        HIDDEN_LAY = hidden_lay
        self.activations = [torch.relu] # This needs to reflect what is used in the forward pass :)
        self.lays=nn.ModuleList()
        self.layers = []
        self.dataset = dataset
        if(dataset=="MNIST"):
            self.class_weights = None
            self.in_dim = 28*28
            self.num_cls = 10
        elif(dataset=="ADULT"):
            #self.class_weights = torch.Tensor([1,2*3.017])
            self.class_weights = torch.Tensor([1,1])
            self.in_dim = 63 #110
            self.num_cls = 2
        elif(dataset=="CREDIT"):
            #self.class_weights = torch.Tensor([1,3.508])
            self.class_weights = torch.Tensor([1,1])
            self.in_dim = 146
            self.num_cls = 2
        elif(dataset=="GERMAN"):
            self.class_weights = torch.Tensor([2.319, 1])
            #self.class_weights = torch.Tensor([1,1])
            self.in_dim = 62
            self.num_cls = 2
        elif(dataset=="CRIME"):
            #self.class_weights = torch.Tensor([1,2*42.081])
            self.class_weights = torch.Tensor([1,1])
            self.in_dim = 54
            self.num_cls = 2
        elif(dataset=="CIFAR10"):
            self.class_weights = None
            self.in_dim = 32*32*3
            self.num_cls = 10
        elif(dataset=="HALFMOONS"):
            self.class_weights = None
            self.in_dim = 2
            self.num_cls = 2
        elif(dataset=="PNEUMONIAMNIST" or dataset =="pneumoniamnist"):
            self.class_weights = torch.Tensor([3,1])
            self.in_dim = 28*28
            self.num_cls = 2
        elif(dataset=="tissuemnist"):
            self.class_weights = None
            self.in_dim = 28*28
            self.num_cls = 8
        elif(dataset=="breastmnist"):
            self.class_weights = None
            self.in_dim = 28*28
            self.num_cls = 2  
        elif("organ" in dataset):
            self.class_weights = None
            self.in_dim = 28*28
            self.num_cls = 11 
            
        self.l1 = torch.nn.Linear(self.in_dim, self.hparams.hidden_dim)
        self.lays.append(self.l1); self.activations.append(torch.relu)
        self.layers.append("Linear")
        
        for i in range(HIDDEN_LAY - 1):
            self.lays.append(torch.nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim))
            self.activations.append(torch.relu)
            self.layers.append("Linear")
            
        self.lf = torch.nn.Linear(self.hparams.hidden_dim, self.num_cls)
        self.lays.append(self.lf)
        self.layers.append("Linear")
        
        self.ALPHA = alpha             # Regularization Parameter (Weights the Reg. Term)
        self.EPSILON = epsilon           # Input Peturbation Budget at Training Time
        self.GAMMA = gamma             # Model Peturbation Budget at Training Time 
                                     #(Changed to proportional budget rather than absolute)

        self.LEARN_RATE = 0.001      # Learning Rate Hyperparameter
        self.MAX_EPOCHS = 10         # Maximum Epochs to Train the Model for

        self.EPSILON_LINEAR = True   # Put Epsilon on a Linear Schedule?
        self.GAMMA_LINEAR = True     # Put Gamma on a Linear Schedule?
        
        if(self.EPSILON_LINEAR):
            self.eps = 0.0
        else:
            self.eps = self.EPSILON
        if(self.GAMMA_LINEAR):
            self.gam = 0.0
        else:
            self.gam = self.GAMMA
        self.mode = mode.upper()
        print("SET MODE TO: ", self.mode)
        self.inputfooling = False
       
    def set_params(self, **kwargs):
        self.ALPHA =  kwargs.get('alpha', 0.00)
        self.GAMMA =  kwargs.get('gamma', 0.00)
        self.EPSILON =  kwargs.get('epsilon', 0.00)
        self.LEARN_RATE =  kwargs.get('learn_rate', 0.001)
        self.MAX_EPOCHS =  int(kwargs.get('max_epochs', 15))
        self.EPSILON_LINEAR = bool(kwargs.get('epsilon_linear', True))
        self.GAMMA_LINEAR = bool(kwargs.get('gamma_linear', True))
        #self.mode = kwargs.get('mode', "GRAD")
        if(self.EPSILON_LINEAR):
            self.eps = 0.0
        else:
            self.eps = self.EPSILON
        if(self.GAMMA_LINEAR):
            self.gam = 0.0
        else:
            self.gam = self.GAMMA
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i in range(len(self.lays)-1):
            if(self.inputfooling):
                beta = 5
                x = F.softplus(self.lays[i](x), beta=beta)
            else:
                x = torch.relu(self.lays[i](x))
        x = self.lf(x)
        return x
    
    def inputfooling_ON(self):
        self.inputfooling = True
        
    def inputfooling_OFF(self):
        self.inputfooling = False
        
    def classify(self, x):
        outputs = self.forward(x)
        return F.softmax(outputs, dim=1), torch.max(outputs, 1)[1]
    
    def loss(self, x, y, weight=-1):
        if(type(weight) == int):
            return F.cross_entropy(x, y)
        else:
            return F.cross_entropy(x, y, weight=weight)
        
    def training_step(self, batch, batch_idx):
        #print(self.mode)
        #weights = [t for t in model.parameters()]
        x, y = batch
        y_hat = self(x)
        regval = 0.0
        if(self.mode == 'GRAD'):
            x = x.view(x.size(0), -1)
            regval = GradCertModule.GradCertRegularizer(self, x, y, self.eps, self.gam, nclasses=self.num_cls)
            #print(regval)
            loss = self.loss(y_hat, y, weight=self.class_weights) + (self.ALPHA*regval)
        elif(self.mode == 'ADV'):
            x = x.view(x.size(0), -1)
            regval = GradCertModule.RobustnessRegularizer(self, x, y, self.eps, self.gam, nclasses=self.num_cls)
            loss = self.loss(y_hat, y, weight=self.class_weights) + (self.ALPHA*regval)
        elif(self.mode == 'BOTH'):
            x = x.view(x.size(0), -1)
            regval += GradCertModule.GradCertRegularizer(self, x, y, self.eps, self.gam, nclasses=self.num_cls)
            regval += GradCertModule.RobustnessRegularizer(self, x, y, self.eps, self.gam, nclasses=self.num_cls)
            loss = self.loss(y_hat, y, weight=self.class_weights) + (self.ALPHA*regval)
        elif(self.mode == 'PGD'):
            regval = GradCertModule.PGDRegularizer(self, x, y, self.eps)
            loss = self.loss(y_hat, y, weight=self.class_weights)
        elif(self.mode == 'L2ADV'):
            regval = GradCertModule.L2AdvRegularizer(self, x, y, self.eps)
            loss = self.loss(y_hat, y, weight=self.class_weights)
        elif(self.mode == 'IGSUMNORM'):
            regval = GradCertModule.L2AdvRegularizer(self, x, y, self.eps, iters=5)
            loss = GradCertModule.PGDRegularizer(self, x, y, self.eps, iters=5)
        elif(self.mode == 'FAIRWASH'):
            ig = GradCertModule.InputGrad(self, x, y, nclasses=self.num_cls)
            regval = torch.sum(torch.abs(ig)[self.sens_idx])
            loss = self.loss(y_hat, y, weight=self.class_weights) + (self.ALPHA*regval)
        elif(self.mode == 'FAIRWASH-GRAD'):
            regval_grad = GradCertModule.GradCertRegularizer(self, x, y, self.eps, self.gam, nclasses=self.num_cls)
            ig = GradCertModule.InputGrad(self, x, y, nclasses=self.num_cls)
            regval = torch.sum(torch.abs(ig)[self.sens_idx])
            loss = self.loss(y_hat, y, weight=self.class_weights) + (self.ALPHA*regval) + (self.ALPHA*regval_grad)
        elif self.mode == "HESSIAN":
            regval += GradCertModule.HessianRegularizer(self, x, y)
            loss = self.loss(y_hat, y, weight=self.class_weights) + (self.ALPHA*regval)
        elif self.mode == "L2":
            regval += GradCertModule.L2Regularizer(self, x, y, y_hat)
            loss = self.loss(y_hat, y, weight=self.class_weights) + (self.ALPHA*regval)
        else:
             loss = self.loss(y_hat, y, weight=self.class_weights)
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
    
    def predict_proba(self, data: Union[torch.FloatTensor, np.array]) -> np.array:
        """
        Computes probabilistic output for c classes
        :param data: torch tabular input
        :return: np.array
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = data.float()

        return self.forward(input).detach().numpy()
    
    def predict(self, data):
        """
        :param data: torch or list
        :return: np.array with prediction
        """
        
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = torch.squeeze(data).float()
        
        outputs = self.forward(input)
        s = F.softmax(outputs, dim=1)
        return s.detach().numpy()
    
    
    def save(self, trainer):
        directory = "Models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        SCHEDULED = self.EPSILON_LINEAR or self.GAMMA_LINEAR
        MODEL_ID = "FCN_e=%s_g=%s_a=%s_s=%s"%(self.EPSILON, self.GAMMA, self.ALPHA, SCHEDULED)
        trainer.save_checkpoint("Models/%s.ckpt"%(MODEL_ID))
        torch.save(model.state_dict(), "Models/%s.pt"%(MODEL_ID))
    