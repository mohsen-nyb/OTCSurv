import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import date
import datetime
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import math
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_ipcw, brier_score
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
import numpy as np
from lifelines import KaplanMeierFitter






# some util functions

_EPSILON = 1e-08

def div(x, y):
    return torch.div(x, (y + _EPSILON))


def log(x):
    return torch.log(x + _EPSILON)



def f_get_fc_mask(time, label, num_Event, num_Category):
 
    N = np.shape(time)[0]
    mask = np.ones([N, num_Event, num_Category]) 
    for i in range(N):
        if label[i] != 0:  
            mask[i,int(label[i]-1), int(time[i])-1:] = 0
        else: 
            mask[i,:,int(time[i]):] =  0 
            
    return mask


def f_get_fc_mask2(time, label, num_Event, num_Category):
 
    N = np.shape(time)[0]
    mask = np.ones([N, num_Event, num_Category]) 
    for i in range(N):
        if label[i] == 1:  
            mask[i,0, int(time[i])-1:] = 0
            
        elif label[i] == 2:
            mask[i,:, int(time[i])-1:] = 0
            
        else: 
            mask[i,:,int(time[i]):] =  0 
            
    return mask




def get_rx_vectors(onto_rx, v, max_length, pad_idx):
    
    lst_l1=[]
    lst_l2=[]
    lst_l3=[]
    for code_l4 in v.nonzero()+1:
        code_l1, code_l2, code_l3 = onto_rx[onto_rx['atc4']==itoc_rx4[code_l4.item()]][['atc1', 'atc2', 'atc3']].values.squeeze()
        lst_l1.append(ctoi_rx1[code_l1])
        lst_l2.append(ctoi_rx2[code_l2])
        lst_l3.append(ctoi_rx3[code_l3])
        
    padded_l1 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l1[0:len(lst_l1)]= torch.tensor(lst_l1)
    
    padded_l2 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l2[0:len(lst_l2)]= torch.tensor(lst_l2)

    padded_l3 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l3[0:len(lst_l3)]= torch.tensor(lst_l3)    
    
    padded_l4 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l4[0:len(v.nonzero()[:,0])]= (v.nonzero()+1)[:,0]
        
    return padded_l1, padded_l2, padded_l3, padded_l4





def get_rx_levels(onto_rx,x_rx, max_length, pad_idx):
    
    l1=[]
    l2=[]
    l3=[]
    l4=[]

    for row in x_rx:
        
        v1, v2, v3, v4 = get_rx_vectors(onto_rx, row, max_length, pad_idx)
        l1.append(v1.unsqueeze(0))
        l2.append(v2.unsqueeze(0))
        l3.append(v3.unsqueeze(0))
        l4.append(v4.unsqueeze(0))


    x_rx_l1 = torch.concat(l1, axis=0)
    x_rx_l2 = torch.concat(l2, axis=0)
    x_rx_l3 = torch.concat(l3, axis=0)
    x_rx_l4 = torch.concat(l4, axis=0) 
    
    
    return x_rx_l1, x_rx_l2, x_rx_l3, x_rx_l4




def get_dx_vectors(onto_dx, v, max_length, pad_idx):
    
    lst_l1=[]
    lst_l2=[]
    
    for code_l3 in v.nonzero()+1:
        code_l1, code_l2 = onto_dx[onto_dx['l3']==itoc_dx3[code_l3.item()]][['l1', 'l2']].values.squeeze()
        lst_l1.append(ctoi_dx1[code_l1])
        lst_l2.append(ctoi_dx2[code_l2])

    padded_l1 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l1[0:len(lst_l1)]= torch.tensor(lst_l1)
    
    
    padded_l2 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l2[0:len(lst_l2)]= torch.tensor(lst_l2)
    
    padded_l3 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l3[0:len(v.nonzero()[:,0])]= (v.nonzero()+1)[:,0]
    
    return padded_l1, padded_l2, padded_l3





def get_dx_levels(onto_dx, x_dx, max_length, pad_idx):
    
 
    l1=[]
    l2=[]
    l3=[]

    for row in tqdm(x_dx):
        v1, v2, v3 = get_dx_vectors(onto_dx, row, max_length, pad_idx)
        l1.append(v1.unsqueeze(0))
        l2.append(v2.unsqueeze(0))
        l3.append(v3.unsqueeze(0))

    x_dx_l1 = torch.concat(l1, axis=0)
    x_dx_l2 = torch.concat(l2, axis=0)  
    x_dx_l3 = torch.concat(l3, axis=0)
 
    
    return x_dx_l1, x_dx_l2, x_dx_l3






### C(t)-INDEX CALCULATION
def c_index(Prediction, Time_survival, Death, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] < Prediction)] = 1
  
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result



##### WEIGHTED C-INDEX & BRIER-SCORE
def CensoringProb(Y, T):

    T = T.reshape([-1]) # (N,) - np array
    Y = Y.reshape([-1]) # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y==0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)
    
    return G


### C(t)-INDEX CALCULATION: this account for the weighted average for unbaised estimation
def weighted_c_index(T_train, Y_train, Prediction, T_test, Y_test, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    G = CensoringProb(Y_train, T_train)

    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        tmp_idx = np.where(G[0,:] >= T_test[i])[0]

        if len(tmp_idx) == 0:
            W = (1./G[1, -1])**2
        else:
            W = (1./G[1, tmp_idx[0]])**2

        A[i, np.where(T_test[i] < T_test)] = 1. * W
        Q[i, np.where(Prediction[i] < Prediction)] = 1. # give weights

        if (T_test[i]<=Time and Y_test[i]==1):
            N_t[i,:] = 1.

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result



