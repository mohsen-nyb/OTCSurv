import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import date
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import math
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_ipcw, brier_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
from utils import *




def OrdinalRegLoss(surv_probs, target_label, mask):

    
    num_event = surv_probs.shape[1]
    
    neg_likelihood_loss = 0
    
    for i in range(num_event):
        predicted_surv_ = surv_probs[:,i,:]
        
        I_2 = (target_label == i + 1).float().squeeze()
 

        # for obsorved
        temp = torch.sum(log(mask[:,i,:] * predicted_surv_), dim=1) + torch.sum(log(1 - (1.0 - mask[:,i,:]) * predicted_surv_), dim=1)
        L_obsorved = I_2 * temp

        # for censored
        tmp2 = torch.sum(log(mask[:,i,:] * predicted_surv_), dim=1)
        L_censored = (1. - I_2) * tmp2

        neg_likelihood_loss += - (L_obsorved + L_censored)
    
    return torch.sum(neg_likelihood_loss) 





def randomized_ranking_loss(pred_duration, trg_label, trg_ett): 

    ranking_loss=0
    
    I_ob = (trg_label == 1).float().squeeze()
    
    trg_duration = (trg_ett.squeeze()-1)*I_ob + trg_ett.squeeze()*(1-I_ob)
    
    trg_duration_ob = trg_duration[I_ob==1].squeeze()
    pred_duration_ob = pred_duration[I_ob==1].squeeze()

    
    for idx, Ti in enumerate(trg_duration_ob):
        Ti_pred = pred_duration_ob[idx]
        
        mask = (trg_duration > Ti).float()
        
        selected_true_set = trg_duration[mask==1].squeeze()
        selected_pred_set = pred_duration[mask==1].squeeze()
        
        if selected_true_set.shape==torch.Size([]) or len(selected_true_set)==0:
            continue
        
        selection = torch.randint(0, len(selected_true_set), (1,))
        
        Tj = selected_true_set[selection].squeeze()
        Tj_pred = selected_pred_set[selection].squeeze()     
        
        loss = F.relu((Tj-Ti) - (Tj_pred-Ti_pred))
        ranking_loss += loss
        
    return ranking_loss
  

def randomized_weighted_ranking_loss(pred_duration, trg_label, trg_ett): 

    ranking_loss=0
    
    I_ob = (trg_label == 1).float().squeeze()
    
    trg_duration = (trg_ett.squeeze()-1)*I_ob + trg_ett.squeeze()*(1-I_ob)
    
    trg_duration_ob = trg_duration[I_ob==1].squeeze()
    pred_duration_ob = pred_duration[I_ob==1].squeeze()

    
    for idx, Ti in enumerate(trg_duration_ob):
        Ti_pred = pred_duration_ob[idx]
        
        mask = (trg_duration > Ti).float()
        
        selected_true_set = trg_duration[mask==1].squeeze()
        selected_pred_set = pred_duration[mask==1].squeeze()
        
        if selected_true_set.shape==torch.Size([]) or len(selected_true_set)==0:
            continue
        
        selection = torch.randint(0, len(selected_true_set), (1,))
        
        Tj = selected_true_set[selection].squeeze()
        Tj_pred = selected_pred_set[selection].squeeze()     
        
        loss = F.relu(-(Tj_pred-Ti_pred))
        ranking_loss += loss
        
    return ranking_loss    
    

    


def MSE_loss(pred_duration, trg_label, trg_ett, num_event=1): 

    ob_mse_loss=0
    for i in range(num_event):
        I = (trg_label == i+1).float().squeeze()
        true_duration_ob = trg_ett[I==1].squeeze()
        pred_duration_ob = pred_duration[I==1] 
        pred_duration_ob_i = pred_duration_ob[:,i] + 1
        loss = nn.MSELoss(pred_duration_ob_i, true_duration_ob)
        ob_mse_loss += loss
        
    return ob_mse_loss


def MAE_loss(pred_duration, trg_label, trg_ett, num_event=1): 

    ob_mae_loss=0
    for i in range(num_event):
        I = (trg_label == i+1).float().squeeze()
        true_duration_ob = trg_ett[I==1].squeeze()
        pred_duration_ob = pred_duration[I==1] 
        pred_duration_ob_i = pred_duration_ob[:,i] + 1
        loss = F.l1_loss(pred_duration_ob_i, true_duration_ob)
        ob_mae_loss += loss
        
    return ob_mae_loss


def contrast_loss(encoded_inp, target_label, target_ett, num_event=0, temp=None):
    
    contrast_loss_risk = 0.0

    for j in range(1,10):
        encoded = encoded_inp.reshape(encoded_inp.shape[0], -1)
        h_e_mask = (target_ett >= j) & (target_ett < j+2)
        h_e = encoded[torch.where((target_label==1) & h_e_mask)[0]]
        h_0_c_mask = (target_label==0)&(target_ett >= j+2)
        h_0_Ob_mask = (target_label==1)&(~ h_e_mask)
        h_0 = encoded[torch.where(h_0_c_mask | h_0_Ob_mask)[0]]
        contrast_loss_risk_numerator = torch.triu(torch.mm(h_e, torch.transpose(h_e, 0, 1)), diagonal=1)
        contrast_loss_risk_denominator = torch.exp(torch.mm(h_e, torch.transpose(h_0, 0, 1)).double())
        val = -torch.sum(contrast_loss_risk_numerator -  torch.log(torch.sum(torch.exp(contrast_loss_risk_numerator)) + torch.sum(contrast_loss_risk_denominator)))/len(h_e)
        if not val.isnan():
            contrast_loss_risk += val
    return contrast_loss_risk


def weighted_contrast_loss(encoded_inp, target_label, target_ett, num_event=0, window_size=2, temp=0.5):
 
    contrast_loss_risk = 0.0
    

    for j in range(1,10):
        encoded = encoded_inp.reshape(encoded_inp.shape[0], -1)
        h_e_mask = (target_ett >= j) & (target_ett < j+window_size)
        h_e = encoded[torch.where((target_label==1) & h_e_mask)[0]]
        h_0_c_mask = (target_label==0)&(target_ett >= j+window_size)
        h_0_Ob_mask = (target_label==1)&(~ h_e_mask)
        h_0 = encoded[torch.where(h_0_c_mask | h_0_Ob_mask)[0]]
        
        normalized_h_e = F.normalize(h_e, p=2, dim=1)
        normalized_h_0 = F.normalize(h_0, p=2, dim=1)
        
        T_e = target_ett[torch.where((target_label==1) & h_e_mask)[0]]
        T_o = target_ett[torch.where(h_0_c_mask | h_0_Ob_mask)[0]]
        
        if len(T_e)==0:
            continue
            
        weights_list=[torch.abs(T_o-t).squeeze().unsqueeze(0) for t in T_e]
        weights = torch.cat(weights_list, dim=0) 
        
        
        contrast_loss_risk_numerator = torch.triu(torch.mm(normalized_h_e, torch.transpose(normalized_h_e, 0, 1))/temp, diagonal=1) +torch.tril(torch.mm(normalized_h_e, torch.transpose(normalized_h_e, 0, 1))/temp, diagonal=-1)
            
        contrast_loss_risk_neg_weighted = torch.sum(torch.exp(torch.mm(normalized_h_e, torch.transpose(normalized_h_0, 0, 1)).double())* weights, dim=1)
        contrast_loss_risk_denominator = torch.log(torch.exp(contrast_loss_risk_numerator) + contrast_loss_risk_neg_weighted)
        contrast_loss_risk_denominator = torch.triu(contrast_loss_risk_denominator,diagonal=1) + torch.tril(contrast_loss_risk_denominator,diagonal=-1)
        val = -torch.sum(contrast_loss_risk_numerator -  contrast_loss_risk_denominator)/len(h_e)
        if not val.isnan():
            contrast_loss_risk += val
        
    return contrast_loss_risk




