import torch
#torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.nn import functional as F

import torchvision
import pytorch_lightning as pl
import sys
sys.path.append('..')
import GradCertModule
import XAIArchitectures

import pytorch_lightning as pl
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule

import argparse

import random
import numpy as np
SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--alp")
parser.add_argument("--eps")
parser.add_argument("--gam")

args = parser.parse_args()

ALPHA = float(args.alp)              # Regularization Parameter (Weights the Reg. Term)
EPSILON = float(args.eps)            # Input Peturbation Budget at Training Time
GAMMA = float(args.gam)              # Model Peturbation Budget at Training Time 
                         #(Changed to proportional budget rather than absolute)
    
LEARN_RATE = 0.001      # Learning Rate Hyperparameter
HIDDEN_DIM = 100         # Hidden Neurons Hyperparameter
HIDDEN_LAY = 1           # Hidden Layers Hyperparameter
MAX_EPOCHS = 30

EPSILON_LINEAR = True    # Put Epsilon on a Linear Schedule?
GAMMA_LINEAR = True      # Put Gamma on a Linear Schedule?
if(EPSILON != 0 and GAMMA != 0):
    MODE = 'GRAD'
else:
    MODE = 'NONE'

model = XAIArchitectures.DeepMindSmall(dataset="MNIST", mode=MODE)
model.set_params(alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA, 
                 hidden_dim=HIDDEN_DIM, hidden_lay=HIDDEN_LAY,
                learn_rate=LEARN_RATE, max_epochs=MAX_EPOCHS,
                epsilon_linear=EPSILON_LINEAR, gamma_linear=GAMMA_LINEAR)


#l = [0.0004, 0.012, 0.025, 0.05].index(EPSILON) +1
AVAIL_GPUS = min(1, torch.cuda.device_count())

BATCH_SIZE = 256
PATH_DATASETS = 'Datasets'
NUM_WORKERS = 5
dm = MNISTDataModule(batch_size=100, num_workers=90)

model = XAIArchitectures.FullyConnected()
model.set_params(alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA, 
                 hidden_dim=HIDDEN_DIM, hidden_lay=HIDDEN_LAY,
                learn_rate=LEARN_RATE, max_epochs=MAX_EPOCHS,
                epsilon_linear=EPSILON_LINEAR, gamma_linear=GAMMA_LINEAR)


trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", devices=1)
trainer.fit(model, datamodule=dm)
result = trainer.test(model, datamodule=dm)

ACC = round(result[0]['test_acc'],3)
print(ACC)

# Finally, we save the model IDed by its relevant parameters
import os
directory = "SEED_Models"
if not os.path.exists(directory):
    os.makedirs(directory)
SCHEDULED = model.EPSILON_LINEAR or model.GAMMA_LINEAR
if(MODE == 'ADV'):
    MODEL_ID = "FCNA_e=%s_g=%s_a=%s_s=%s"%(model.EPSILON, model.GAMMA, model.ALPHA, SCHEDULED)
elif(MODE == 'BOTH'):
    MODEL_ID = "FCNB_e=%s_g=%s_a=%s_s=%s"%(model.EPSILON, model.GAMMA, model.ALPHA, SCHEDULED)
else:
    MODEL_ID = "FCN_e=%s_g=%s_a=%s_s=%s"%(model.EPSILON, model.GAMMA, model.ALPHA, SCHEDULED)
trainer.save_checkpoint("SEED_Models/%s.ckpt"%(MODEL_ID))
torch.save(model.state_dict(), "SEED_Models/%s.pt"%(MODEL_ID))

