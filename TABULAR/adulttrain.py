#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[2]:


full_data = pd.read_csv(
    "./Datasets/adult.csv",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python', skiprows=1,
        na_values="?", dtype={0:int, 1:str, 2:int, 3:str, 4:int, 5: str, 6:str , 7:str ,8:str ,9: str, 10:int, 11:int, 12:int, 13:str,14: str})

print('Dataset size: ', full_data.shape[0])


# In[3]:


str_list=[]
for data in [full_data]:
    for colname, colvalue in data.iteritems(): 
        if type(colvalue[1]) == str:
            str_list.append(colname) 
num_list = data.columns.difference(str_list)

full_size = full_data.shape[0]
print('Dataset size Before pruning: ', full_size)
for data in [full_data]:
    for i in full_data:
        data[i].replace('nan', np.nan, inplace=True)
    data.dropna(inplace=True)
real_size = full_data.shape[0]
print('Dataset size after pruning: ', real_size)
print('We eliminated ', (full_size-real_size), ' datapoints')

# Take
full_labels = full_data['Target'].copy()
full_data = full_data.drop(['Target'], axis=1)

# Label Encode Labels
label_encoder = LabelEncoder()
full_labels = label_encoder.fit_transform(full_labels)

# Segment categorical and non categorical data (will manipulate cat_data, and append them back later)
cat_data = full_data.select_dtypes(include=['object']).copy()
other_data = full_data.select_dtypes(include=['int']).copy()

newcat_data = pd.get_dummies(cat_data, columns=[
    "Workclass", "Education", "Country" ,"Relationship", "Martial Status", "Occupation", "Relationship",
    "Race", "Sex"
])


full_data = pd.concat([other_data, newcat_data], axis=1)

train_size = 30000
valid_size = 10000


# In[4]:


train_x = full_data.iloc[:train_size, :].to_numpy()
train_y = full_labels[:train_size]
print(train_x.shape)
print(train_y.shape)
print()

valid_x = full_data.iloc[train_size:(train_size+valid_size), :].to_numpy()
valid_y = full_labels[train_size:(train_size+valid_size)]
print(valid_x.shape)
print(valid_y.shape)
print()

test_x = full_data.iloc[(train_size+valid_size):, :].to_numpy()
test_y = full_labels[(train_size+valid_size):]
print(test_x.shape)
print(test_y.shape)
num_features = test_x.shape[1]


# In[5]:


print(test_x[0].shape)


# In[6]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms

import sys
sys.path.append("..")
import GradCertModule
import XAIArchitectures
import pytorch_lightning as pl


class custDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

AdultTrain = custDataset(train_x, train_y)    
AdultVal = custDataset(valid_x, valid_y) 
AdultTest = custDataset(test_x, test_y)

class AdultDataModule(pl.LightningDataModule):
    def __init__(self, train, val, test, batch_size=1000):
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

    #def predict_dataloader(self):
    #    return DataLoader(self.mnist_predict, batch_size=self.batch_size)
    
dm = AdultDataModule(AdultTrain, AdultVal, AdultTest)


# In[7]:


ALPHA = 0.0            # Regularization Parameter (Weights the Reg. Term)
EPSILON = 0.0          # Input Peturbation Budget at Training Time
GAMMA = 0.0            # Model Peturbation Budget at Training Time 
                       #(Changed to proportional budget rather than absolute)
    
LEARN_RATE = 0.001     # Learning Rate Hyperparameter
HIDDEN_DIM = 256       # Hidden Neurons Hyperparameter
HIDDEN_LAY = 1         # Hidden Layers Hyperparameter
MAX_EPOCHS = 10

EPSILON_LINEAR = True   # Put Epsilon on a Linear Schedule?
GAMMA_LINEAR = True     # Put Gamma on a Linear Schedule?


# In[8]:

HIDDEN_DIM = 100
HIDDEN_LAY = 1

model = XAIArchitectures.FullyConnected(hidden_dim=HIDDEN_DIM, hidden_lay=HIDDEN_LAY, dataset="ADULT")
model.set_params(alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA, 
                learn_rate=LEARN_RATE, max_epochs=MAX_EPOCHS,
                epsilon_linear=EPSILON_LINEAR, gamma_linear=GAMMA_LINEAR)


# In[9]:


trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="cpu", devices=1)
trainer.fit(model, datamodule=dm)
result = trainer.test(model, datamodule=dm)


