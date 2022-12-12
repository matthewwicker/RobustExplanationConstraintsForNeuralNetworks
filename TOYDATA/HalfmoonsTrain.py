#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Custom Imports
import sys
sys.path.append("../..")
sys.path.append("..")
import GradCertModule
import XAIArchitectures
# Deep Learning Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
import pytorch_lightning as pl
# Standard Lib Imports
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import sklearn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_moons(n_samples=1000, noise=0.15, random_state=42)
X_train = X[0:800]
X_test = X[800:]

y_train = y[0:800]
y_test = y[800:]


# Parse command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", default=0)
parser.add_argument("--eps", default=0)

args = parser.parse_args()
alpha = float(args.alpha)
epsilon = float(args.eps)

ALPHA = alpha
EPSILON = epsilon



# In[2]:


class custDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X).float()
        self.y = y
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

CustTrain = custDataset(X_train, y_train)    
CustTest = custDataset(X_test, y_test)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train, val, test, batch_size=32):
        super().__init__()
        self.train_data = train
        self.val_data = val
        self.test_data = test
        self.batch_size = batch_size
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
    
dm = CustomDataModule(CustTrain, CustTest, CustTest)



# In[3]:


ALPHA = alpha           # Regularization Parameter (Weights the Reg. Term)
EPSILON = epsilon          # Input Peturbation Budget at Training Time
GAMMA = 0.00        # Model Peturbation Budget at Training Time 
                       #(Changed to proportional budget rather than absolute)
 
LEARN_RATE = 0.001   # Learning Rate Hyperparameter
HIDDEN_DIM = 256       # Hidden Neurons Hyperparameter
HIDDEN_LAY = 2         # Hidden Layers Hyperparameter
MAX_EPOCHS = 35

EPSILON_LINEAR = True   # Put Epsilon on a Linear Schedule?
GAMMA_LINEAR = True     # Put Gamma on a Linear Schedule?


# In[4]:


mode = 'GRAD'

    
model = XAIArchitectures.FullyConnected(hidden_dim=HIDDEN_DIM, hidden_lay=HIDDEN_LAY, dataset="HALFMOONS", mode=mode)
model.set_params(alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA, 
                learn_rate=LEARN_RATE, max_epochs=MAX_EPOCHS,
                epsilon_linear=EPSILON_LINEAR, gamma_linear=GAMMA_LINEAR)



# In[5]:


trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="cpu", devices=1)
trainer.fit(model, datamodule=dm)
result = trainer.test(model, datamodule=dm)


# In[6]:

import os
directory = "Models"
if not os.path.exists(directory):
    os.makedirs(directory)
    
MODEL_ID = "Halfmoons_a=%s_e=%s"%(alpha, epsilon)
trainer.save_checkpoint("Models/%s.ckpt"%(MODEL_ID))
torch.save(model.state_dict(), "Models/%s.pt"%(MODEL_ID))


"""

X_preds = []
y_preds = []
for x in np.linspace(-1.5,2.5,150):
    for y in np.linspace(-0.75,1.25,150):
        y_hat = model(torch.Tensor([[x,y]]))
        X_preds.append([x,y])
        y_preds.append(torch.argmax(y_hat).detach().numpy())

X_preds = np.asarray(X_preds)
y_preds = np.asarray(y_preds)


# In[8]:


import seaborn as sns
sns.set_context('poster')
plt.figure(figsize=(8,8), dpi=100)
plt.title(r"GradCert Training ($\alpha$ = 0.2)")
plt.scatter(x=X_preds[:,0], y=X_preds[:,1], c=y_preds, cmap='coolwarm')

plt.show()
"""
