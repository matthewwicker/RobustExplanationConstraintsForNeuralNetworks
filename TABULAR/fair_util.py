import sys
sys.path.append("..")
sys.path.append("../..")
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

def fairness_computation(model, X_test, y_test, sens_inds):
    X_maj = []
    y_maj = []
    X_min = []
    y_min = []

    for i in range(len(X_test)):
        if(X_test[i][sens_inds[0]] == 0.0):
            X_maj.append(X_test[i])
            y_maj.append(y_test[i])
        elif(X_test[i][sens_inds[0]] == 1.0):
            X_min.append(X_test[i])
            y_min.append(y_test[i])

    X_maj = np.asarray(X_maj)
    y_maj = np.asarray(y_maj)
    X_min = np.asarray(X_min)
    y_min = np.asarray(y_min)


    y_pred_min = []
    y_pred_maj = []

    maj_tp = 0
    maj_tn = 0
    maj_fp = 0
    maj_fn = 0

    min_tp = 0
    min_tn = 0
    min_fp = 0
    min_fn = 0

    acc = 0
    majority = 0
    minority = 0
    min_pos, min_neg = 0, 0
    maj_pos, maj_neg = 0, 0

    for i in range(len(X_test)):
        y_pred = model(torch.Tensor([X_test[i]]))
        corr = int(torch.argmax(y_pred) == y_test[i])
        acc += corr

        #cls = (int(torch.argmax(y_pred).detach().numpy()))
        # Majority Group Data
        if(X_test[i][sens_inds[0]] == 0.0):
            majority += 1
            if(corr == 1 and torch.argmax(y_pred) == 1):
                maj_tp += 1
            elif(corr == 1 and torch.argmax(y_pred) == 0):
                maj_tn += 1
            elif(corr == 0 and torch.argmax(y_pred) == 1):
                maj_fp += 1
            elif(corr == 0 and torch.argmax(y_pred) == 0):
                maj_fn += 1
            # Sanity check:
            if(y_test[i] == 1):
                maj_pos += 1
            else:
                maj_neg += 1
        # Minority Group Data
        elif(X_test[i][sens_inds[0]] == 1.0):
            minority += 1
            if(corr == 1 and torch.argmax(y_pred) == 1):
                min_tp += 1
            elif(corr == 1 and torch.argmax(y_pred) == 0):
                min_tn += 1
            elif(corr == 0 and torch.argmax(y_pred) == 1):
                min_fp += 1
            elif(corr == 0 and torch.argmax(y_pred) == 0):
                min_fn += 1
            # Sanity check:
            if(y_test[i] == 1):
                min_pos += 1
            else:
                min_neg += 1

    #print("Neural Net Acc: \t", acc/len(X_test))

    p_rate_maj = (maj_tp + maj_fp)/(maj_pos+maj_neg)
    p_rate_min = (min_tp + min_fp)/(min_pos+min_neg)
    demographic_parity = abs(p_rate_maj - p_rate_min)
    #print("Demographic Parity: \t", demographic_parity)

    maj_tp_r = maj_tp/maj_pos
    min_tp_r = min_tp/min_pos
    equalized_opp = abs(maj_tp_r - min_tp_r)
    #print("Equalized Opp Disc: \t", equalized_opp)

    maj_tp_r = (maj_tp+maj_tn)/(maj_pos+maj_neg)
    min_tp_r = (min_tp+min_tn)/(min_pos+min_neg)
    equalized_acc = abs(maj_tp_r - min_tp_r)
    #print("Equalized Acc Disc: \t", equalized_acc)
    
    """
    maj_tp /= (maj_pos) # Total positives maj
    maj_tn /= (maj_neg) # Total negatives maj
    maj_fp /= (maj_neg)
    maj_fn /= (maj_pos)

    min_tp /= (min_pos) # Total positives min
    min_tn /= (min_neg) # Total negatives min
    min_fp /= (min_neg)
    min_fn /= (min_pos)


    print(" ")
    print("Majority Stats/Minority Stats:")
    print("TP:", round(maj_tp,3), "/", round(min_tp,3))
    print("TN:", round(maj_tn,3), "/", round(min_tn,3))
    print("FP:", round(maj_fp,3), "/", round(min_fp,3))
    print("FN:", round(maj_fn,3), "/", round(min_fn,3))
    """
    
    return demographic_parity, equalized_opp, equalized_acc, acc/len(X_test)

from tqdm import trange
def detect_bias(model, inputs, labels, sens_inds):
    sens_exp = 0.0
    # Numerical issue handling is why there are small values
    for i in trange(len(inputs)):
        exp = GradCertModule.InputGrad(model, torch.Tensor([inputs[i]]), labels[i], nclasses=2)
        exp = np.abs(exp.detach().numpy())
        sens_exp += (sum(exp[sens_inds]) / (sum(exp) + 0.00000000001))
    sens_exp += 0.00000000001 
    sens_exp/=len(inputs)
    return sens_exp

import copy
def indiv_fairness(model, X_test, y_test, sens_inds):
    y_pred_1 = model.predict(X_test)
    other_test = []
    for i in X_test:
        indiv = copy.deepcopy(i)
        indiv[sens_inds] = 1 - indiv[sens_inds]
        other_test.append(indiv)
    y_pred_2 = model.predict(other_test)
    return np.mean(np.abs(y_pred_1 - y_pred_2))



