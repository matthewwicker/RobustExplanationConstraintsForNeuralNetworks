#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Custom Imports
import sys
sys.path.append("..")
import data_utils
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = "ADULT"

if(dataset == "GERMAN"):
    negative_cls = 0
    sensitive_features = [] #['status_sex']
    drop_columns = []
    train_ds, test_ds = data_utils.get_german_data(sensitive_features, drop_columns=drop_columns)

elif(dataset == "CREDIT"):
    negative_cls = 1
    sensitive_features = [] #['x2']
    drop_columns = []
    train_ds, test_ds = data_utils.get_credit_data(sensitive_features, drop_columns=drop_columns)
    
elif(dataset == "ADULT"):
    sensitive_features = [] #['sex', 'race']
    drop_columns = ['native-country'] #, 'education']
    train_ds, test_ds = data_utils.get_adult_data(sensitive_features, drop_columns=drop_columns)
    
elif(dataset == "CRIME"):
    negative_cls = 1
    CRIME_DROP_COLUMNS = [
    'HispPerCap', 'LandArea', 'LemasPctOfficDrugUn', 'MalePctNevMarr',
    'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'MedRent',
    'MedYrHousBuilt', 'OwnOccHiQuart', 'OwnOccLowQuart',
    'OwnOccMedVal', 'PctBornSameState', 'PctEmplManu',
    'PctEmplProfServ', 'PctEmploy', 'PctForeignBorn', 'PctImmigRec5',
    'PctImmigRec8', 'PctImmigRecent', 'PctRecImmig10', 'PctRecImmig5',
    'PctRecImmig8', 'PctRecentImmig', 'PctSameCity85',
    'PctSameState85', 'PctSpeakEnglOnly', 'PctUsePubTrans',
    'PctVacMore6Mos', 'PctWorkMom', 'PctWorkMomYoungKids',
    'PersPerFam', 'PersPerOccupHous', 'PersPerOwnOccHous',
    'PersPerRentOccHous', 'RentHighQ', 'RentLowQ', 'Unnamed: 0',
    'agePct12t21', 'agePct65up', 'householdsize', 'indianPerCap',
    'pctUrban', 'pctWFarmSelf', 'pctWRetire', 'pctWSocSec', 'pctWWage',
    'whitePerCap'
    ]
    sensitive_features = []# ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']
    train_ds, test_ds = data_utils.get_crime_data(sensitive_features, drop_columns=CRIME_DROP_COLUMNS)


# In[2]:


X_train = train_ds.X_df.to_numpy()
y_train = torch.squeeze(torch.Tensor(train_ds.y_df.to_numpy()).to(torch.int64))

X_test = test_ds.X_df.to_numpy()
y_test = torch.squeeze(torch.Tensor(test_ds.y_df.to_numpy()).to(torch.int64))


# In[3]:


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


# In[4]:


z = 0
o = 0

for i in range(len(y_train)):
    if(y_train[i] == 0):
        z += 1
    elif(y_train[i] == 1):
        o += 1 
        
zero_weight = z/o#len(y_train)
one_weight = o/z #len(y_train)
print(z, o)
print(z/len(y_train), o/len(y_train))
print(zero_weight, one_weight)


# In[5]:

import sys
ALPHA = 0.5            # Regularization Parameter (Weights the Reg. Term)
EPSILON = 0.0          # Input Peturbation Budget at Training Time
GAMMA = float(sys.argv[1])            # Model Peturbation Budget at Training Time 
                       #(Changed to proportional budget rather than absolute)
    
LEARN_RATE = 0.0005     # Learning Rate Hyperparameter
    # Was 0.001 for previous runs
    # 0.0005 for 0.5/0.5
HIDDEN_DIM = 256       # Hidden Neurons Hyperparameter
HIDDEN_LAY = 2         # Hidden Layers Hyperparameter
MAX_EPOCHS = 25

EPSILON_LINEAR = True   # Put Epsilon on a Linear Schedule?
GAMMA_LINEAR = True     # Put Gamma on a Linear Schedule?


# In[6]:


model = XAIArchitectures.FullyConnected(hidden_dim=HIDDEN_DIM, hidden_lay=HIDDEN_LAY, dataset=dataset)
model.set_params(alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA, 
                learn_rate=LEARN_RATE, max_epochs=MAX_EPOCHS,
                epsilon_linear=EPSILON_LINEAR, gamma_linear=GAMMA_LINEAR)


# In[ ]:


trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="cpu", devices=90)
trainer.fit(model, datamodule=dm)
result = trainer.test(model, datamodule=dm)


# In[ ]:


import os
directory = "Models"
if not os.path.exists(directory):
    os.makedirs(directory)
SCHEDULED = EPSILON_LINEAR or GAMMA_LINEAR
MODEL_ID = "%s_FCN_e=%s_g=%s_a=%s_l=%s_h=%s_s=%s"%(dataset, EPSILON, GAMMA, ALPHA, HIDDEN_LAY, HIDDEN_DIM, SCHEDULED)
trainer.save_checkpoint("Models/%s.ckpt"%(MODEL_ID))
torch.save(model.state_dict(), "Models/%s.pt"%(MODEL_ID))

