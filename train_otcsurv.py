import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
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
from losses import *
from utils import *
from Model_architecture import * 
from utils import *



##############################     experiment     #############################
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

#hyperparameters
encoder_num_layers = 1
code_dim=128
num_features = code_dim+5
embed_dim=512
rep_dim = 252
hidden_dim = 128
num_epochs = 60
learning_rate = 0.00005
num_event = 1
embedding_demo_size = 2
encoder_num_layers = 1



######################################### Dataset Vocabulary ######################################


# Set random seed for PyTorch
torch.manual_seed(123)

# Set random seed for numpy
np.random.seed(1234)


class AKIDataset(Dataset):
    
    def __init__(self, df_dataset, x_dx, x_rx, num_sequence, num_category, num_event, added_time):
   
        self.df_dataset = df_dataset
        self.num_sequence = num_sequence
        self.added_time = added_time
        self.x_dx_l1,self.x_dx_l2,self.x_dx_l3 = x_dx
        self.x_rx_l1,self.x_rx_l2,self.x_rx_l3,self.x_rx_l4 = x_rx

        
        # formating data for model
        self.data_copy = self.df_dataset.values
        self.data_copy = self.data_copy.reshape(-1,self.num_sequence,self.df_dataset.shape[1])
        
        self.data_dx1 = self.x_dx_l1.values.reshape(-1,self.num_sequence,self.x_dx_l1.shape[1])
        self.data_dx2 = self.x_dx_l2.values.reshape(-1,self.num_sequence,self.x_dx_l2.shape[1])
        self.data_dx3 = self.x_dx_l3.values.reshape(-1,self.num_sequence,self.x_dx_l3.shape[1])
        
        self.data_rx1 = self.x_rx_l1.values.reshape(-1,self.num_sequence,self.x_rx_l1.shape[1])
        self.data_rx2 = self.x_rx_l2.values.reshape(-1,self.num_sequence,self.x_rx_l2.shape[1])
        self.data_rx3 = self.x_rx_l3.values.reshape(-1,self.num_sequence,self.x_rx_l3.shape[1])
        self.data_rx4 = self.x_rx_l4.values.reshape(-1,self.num_sequence,self.x_rx_l4.shape[1])
        
        self.data_final_copy = self.data_copy[:,-1,:]
        self.time = self.data_final_copy[:, 4] #tte
        
        self.features_demo = self.data_copy[:, :, 5:7].astype('float32')

        self.features_dx1 = self.data_dx1[:, :, 3:].astype('int16')
        self.features_dx2 = self.data_dx2[:, :, 3:].astype('int16')
        self.features_dx3 = self.data_dx3[:, :, 3:].astype('int16')
        
        self.features_rx1 = self.data_rx1[:, :, 3:].astype('int16')
        self.features_rx2 = self.data_rx2[:, :, 3:].astype('int16')
        self.features_rx3 = self.data_rx3[:, :, 3:].astype('int16')
        self.features_rx4 = self.data_rx4[:, :, 3:].astype('int16')
        
        self.day = self.data_copy[:, :, 2].astype('int16')
        self.tte = self.data_copy[:, :, 4].astype('float32') 
        self.event = self.data_copy[:, :, 3].astype('int8')
        
        self.last_meas = self.data_final_copy[:, 2] # last measurement time
        self.last_meas = self.last_meas - self.added_time
        self.label = self.data_final_copy[:, 3] # event type
        
        self.num_category = num_category
        self.num_event = num_event
                
        self.mask3 = f_get_fc_mask(self.time, self.label, self.num_event, self.num_category)
        
        
    def __len__(self):
        return len(self.data_final_copy)
    
    def __getitem__(self, index):
        
        x_demo = self.features_demo[index]
        x_dx1 = self.features_dx1[index]
        x_dx2 = self.features_dx2[index]
        x_dx3 = self.features_dx3[index]
        
        x_rx1 = self.features_rx1[index]
        x_rx2 = self.features_rx2[index]
        x_rx3 = self.features_rx3[index]
        x_rx4 = self.features_rx4[index]
        
        t = self.tte.reshape(-1, self.num_sequence,1)[index]
        y = self.event.reshape(-1, self.num_sequence,1)[index]
        day = self.day[index]
        m = self.mask3[index]
        
        return (x_dx1, x_dx2, x_dx3), (x_rx1, x_rx2, x_rx3, x_rx4), x_demo, t, y, day, m






class Trainer:

    def __init__(self,
            demo_num_features,
            code_dim,
            demo_dim,
            transformer_encoder_num_heads,
            d_transformer,
            rep_dim,
            hidden_dim,
            num_category,
            lambda_list,
            cl_window_size,
            tempreture,
            device,
            batch_size,
            learning_rate,
            max_num_sequence,
            use_checkpoint=False,
            model_checkpoint_path='checkpoint.pth.tar'
           ):
        
        
        self.max_c_index_test = 0
        self.num_features = code_dim + demo_dim
        self.demo_num_features = demo_num_features
        self.code_dim = code_dim
        self.demo_dim = demo_dim
        self.transformer_encoder_num_heads = transformer_encoder_num_heads
        self.d_transformer = d_transformer
        self.rep_dim = rep_dim
        self.hidden_dim = hidden_dim
        self.num_category = num_category
        self.lambda_list = lambda_list
        self.cl_window_size = cl_window_size
        self.tempreture = tempreture
        self.device = device
        self.use_checkpoint = use_checkpoint
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_sequence = max_num_sequence
        self.num_event = 1
        self.model_checkpoint_path = model_checkpoint_path
        
        self.onto_dx_params = {'l1': 20, 'l2': 177, 'l3': 1125} #ICD9 unique number of categories in each level
        self.onto_rx_params = {'l1': 15, 'l2': 88, 'l3': 207, 'l4': 517} #ATC unique number of categories in each level
        
        
        self.model = OTCSurv(num_features=self.num_features,
                demo_num_features=self.demo_num_features,
                onto_dx_params=self.onto_dx_params,
                onto_rx_params=self.onto_rx_params,
                code_dim=self.code_dim,
                demo_dim = self.demo_dim,
                transformer_encoder_num_heads=self.transformer_encoder_num_heads,
                d_transformer=self.d_transformer,
                rep_dim=self.rep_dim,
                hidden_dim=self.hidden_dim,
                num_event=self.num_event,
                num_category=self.num_category,
                device = self.device).to(self.device)

        
    def create_data_batch(self, train_set_df, test_set_df, x_dx_train,
                          x_rx_train, x_dx_test, x_rx_test,
                          added_time_df_tr_, added_time_df_test):

        self.akidataset_train = AKIDataset(df_dataset=train_set_df,
                                  x_dx=(x_dx_train[0], x_dx_train[1], x_dx_train[2]),
                                  x_rx=(x_rx_train[0], x_rx_train[1], x_rx_train[2], x_rx_train[3]),
                                  num_sequence=self.num_sequence,
                                  num_category=self.num_category,
                                  num_event=self.num_event,
                                  added_time=added_time_df_tr_['added_time'].values)



        self.akidataset_test = AKIDataset(df_dataset=test_set_df,
                                  x_dx=(x_dx_test[0], x_dx_test[1], x_dx_test[2]),
                                  x_rx=(x_rx_test[0], x_rx_test[1], x_rx_test[2], x_rx_test[3]),
                                 num_sequence=self.num_sequence,
                                 num_category=self.num_category,
                                 num_event=self.num_event,
                                 added_time=added_time_df_test['added_time'].values)


        train_loader = DataLoader(dataset=self.akidataset_train,batch_size=self.batch_size,shuffle=True)
        test_loader = DataLoader(dataset=self.akidataset_test,batch_size=self.batch_size,shuffle=True)

        tr_time = np.floor(self.akidataset_train.tte.reshape(-1, 5,1)[:,-1,:])
        tr_label = self.akidataset_train.label
        eval_time = [int(np.percentile(tr_time, 25)), int(np.percentile(tr_time, 50)), int(np.percentile(tr_time, 75))]


        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tr_time = tr_time
        self.tr_label = tr_label        
        
    
    def _save_checkpoint(self, state, filename="checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)
        
    def train(self):
        
        if self.device==None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



        if self.use_checkpoint:
            self.model.load_state_dict(torch.load(self.model_checkpoint_path))

        #optmizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay= 2 * 1e-5)
        #optimizer = optim.RMSprop(MyModel.parameters(), lr=learning_rate, weight_decay= 2 * 1e-5)

        # loss lists
        loss_list_epoch = []
        loss_list_test_epoch = []

        # c-index
        c_index_train_all = []
        c_index_test_all = []

        #mae
        mae_train_all = []
        mae_test_all = []

        #cl
        cl_loss_list_epoch = []


        #ranking
        ranking_loss_list_epoch = []


        #train
        for epoch in range(num_epochs):

            loss_list = []
            loss_list_test = []

            ranking_loss_list = [] 

            mae_train_list = []
            mae_test_list = []

            cl_loss_list=[]



            self.model.train()
            loop = tqdm(self.train_loader, total=len(self.train_loader), leave=False)


            pred_duration_train_list=[]
            target_label_train_list = []
            target_ett_train_list = []
            for x_dx_train,x_rx_train, x_demo_train, t_train, y_train, day_train, mask_train in loop:


                ########################### converting to pytorch tensors ###################################


                x_dx_train = [i.to(self.device) for i in x_dx_train]
                x_rx_train = [i.to(self.device) for i in x_rx_train]
                x_demo_train = x_demo_train.to(self.device)
                t_train = t_train.to(self.device)
                y_train = y_train.to(self.device)
                day_train = day_train.to(self.device)
                mask_train = mask_train.to(self.device)

                target_label = y_train[:,-1].reshape((-1, 1)).float()
                target_ett = t_train[:,-1].reshape((-1, 1)).float()
                target_label_train_list.append(target_label)
                target_ett_train_list.append(target_ett) 

                ################################## model - train ##############################################
                optimizer.zero_grad()

                sigmoid_probs, encoded_inp, projected_head = self.model(x_dx_train,x_rx_train, x_demo_train)
                surv_probs = torch.cumprod(sigmoid_probs, dim=-1)
                pred_duration_train = torch.sum(surv_probs, dim=-1)
                pred_duration_train_list.append(pred_duration_train)

                #calculating losses
                loss = OrdinalRegLoss(surv_probs, target_label, mask_train)
                ob_mae_loss = MAE_loss(pred_duration_train, target_label, target_ett)
                cl_loss = weighted_contrast_loss(projected_head,
                                                 target_label,
                                                 torch.floor(target_ett),
                                                 num_event = self.num_event,
                                                 window_size=self.cl_window_size,
                                                 temp=self.tempreture)
                ranking_loss = randomized_ranking_loss(pred_duration_train, target_label, target_ett)

                totall_loss =  self.lambda_list[0] * loss + \
                self.lambda_list[1]*ranking_loss + \
                self.lambda_list[2]*ob_mae_loss + \
                self.lambda_list[3] * cl_loss
                
                totall_loss.backward()

                loss_list.append(loss.item())
                ranking_loss_list.append(ranking_loss.item())
                cl_loss_list.append(cl_loss.item())


                optimizer.step()


                #######################train evaluation ############################
                mae_train_list.append(ob_mae_loss.item())

            pred_duration_train_all = torch.cat(pred_duration_train_list, dim=0)
            target_label_train_all = torch.cat(target_label_train_list, dim=0)
            target_ett_train_all = torch.cat(target_ett_train_list, dim=0)        

            c_index_train_list = []
            for i in range(num_event):     
                pred_duration_train_i = pred_duration_train_all[:,i]


                is_obsorved_train_i = (target_label_train_all == i + 1).float()
                is_obsorved_or_cencored_train_i = ((target_label_train_all == 0) | (target_label_train_all == i + 1)).float()
                n_target_ett2 = ((1-is_obsorved_or_cencored_train_i) * 10 + (is_obsorved_or_cencored_train_i * target_ett_train_all))  
                n_target_ett2_1 = (n_target_ett2.squeeze()-1) * is_obsorved_train_i.squeeze() + n_target_ett2.squeeze() * (1-is_obsorved_train_i.squeeze())



                c_index_train = concordance_index(torch.floor(n_target_ett2_1).cpu().detach().numpy(),
                                                  pred_duration_train_i.cpu().detach().numpy(), 
                                                  is_obsorved_train_i.cpu().detach().numpy())
                c_index_train_list.append(c_index_train)   



            ########################################### test  #################################################  
            self.model.eval()
            with torch.no_grad():

                num_positive=0
                num_totall=0

                pred_duration_test_list=[]
                target_label_test_list = []
                target_ett_test_list = []
                for x_dx_test, x_rx_test, x_demo_test, t_test, y_test, day_test, mask_test in self.test_loader:

                    x_dx_test = [i.to(self.device) for i in x_dx_test]
                    x_rx_test = [i.to(self.device) for i in x_rx_test]
                    x_demo_test = x_demo_test.to(self.device)
                    y_test = y_test.to(self.device)
                    t_test = t_test.to(self.device)
                    day_test = day_test.to(self.device)
                    mask_test = mask_test.to(self.device)


                    target_label_test = y_test[:,-1].reshape((-1, 1)).float()
                    target_ett_test = t_test[:,-1].reshape((-1, 1)).float()
                    target_day_test = day_test[:,-1].reshape((-1, 1)).float()
                    target_label_test_list.append(target_label_test)
                    target_ett_test_list.append(target_ett_test)

                    sigmoid_probs_test, encoded_inp_test, _ = self.model(x_dx_test, x_rx_test, x_demo_test)
                    surv_probs_test = torch.cumprod(sigmoid_probs_test, dim=-1)
                    pred_duration_test = torch.sum(surv_probs_test, dim=-1)

                    # test loss
                    loss_test = OrdinalRegLoss(surv_probs_test, target_label_test, mask_test)
                    ob_mae_loss_test = MAE_loss(pred_duration_test, target_label_test, target_ett_test)

                    total_loss_test =  loss_test + ob_mae_loss_test  
                    loss_list_test.append(loss_test.item())


                    pred_duration_test_list.append(pred_duration_test)

                    mae_test_list.append(ob_mae_loss_test.item())

                pred_duration_test_all = torch.cat(pred_duration_test_list, dim=0)
                target_label_test_all = torch.cat(target_label_test_list, dim=0)
                target_ett_test_all = torch.cat(target_ett_test_list, dim=0)

                c_index_test_list=[]
                for i in range(self.num_event):
                    pred_duration_test_i = pred_duration_test_all[:,i]


                    ## in calculation of c-index seprately for each event
                    ## if a sample is not censored and from other event than (i+1), the ett should be 212 (mean-life time)
                    is_obsorved_test_i = (target_label_test_all == i + 1).float()
                    is_obsorved_or_cencored_test_i = ((target_label_test_all == 0) | (target_label_test_all == i + 1)).float()
                    n_last_time_test = ((1-is_obsorved_or_cencored_test_i) * 10 + (is_obsorved_or_cencored_test_i * target_ett_test_all))  
                    n_last_time_test1 = (n_last_time_test.squeeze()-1) * is_obsorved_test_i.squeeze() + n_last_time_test.squeeze() * (1-is_obsorved_test_i.squeeze())


                    c_index_test = concordance_index(torch.floor(n_last_time_test1).cpu().detach().numpy(),
                                                    pred_duration_test_i.cpu().detach().numpy(), 
                                                    is_obsorved_test_i.cpu().detach().numpy())
                    c_index_test_list.append(c_index_test)           






            ####################################   print  ###############################################  
            print(f"[Epoch {epoch} / {num_epochs}]")   
            print(f'--train_loss = {np.mean(loss_list)} --test_loss = {np.mean(loss_list_test)}')
            print(f'--train_ranking_loss = {np.mean(ranking_loss_list)}')
            print(f'--train_cl_loss = {np.mean(cl_loss_list)}')
            print(f'--train_mae_loss = {np.mean(mae_train_list)} --test_mae_loss = {np.mean(mae_test_list)}')
            print(f'train_c-index: {c_index_train_list}\ntest_c-index: {c_index_test_list}')

            #loss 
            loss_list_epoch.append(np.mean(loss_list))
            loss_list_test_epoch.append(np.mean(loss_list_test))
            cl_loss_list_epoch.append(np.mean(cl_loss_list))
            ranking_loss_list_epoch.append(np.mean(ranking_loss_list))    

            #Metrics 
            c_index_train_all.append(c_index_train_list)
            c_index_test_all.append(c_index_test_list)
            mae_train_all.append(np.mean(mae_train_list))
            mae_test_all.append(np.mean(mae_test_list))

            ######################save check point###############
            if c_index_test_list[0] > self.max_c_index_test:
                
                self.max_c_index_test_list = c_index_test_list
                checkpoint = {
                        "state_dict": self.model.state_dict(),
                        "optimizer": optimizer.state_dict()
                            }
                self._save_checkpoint(checkpoint, filename="checkpoint.pth.tar")
                


def main():
    

    # Define command-line arguments and their descriptions
    parser = argparse.ArgumentParser(description='Your Model Training and Inference')

    # Model-related arguments
    
    parser.add_argument('--demo_num_features', type=int, default=2,
                        help='Number of demographic features')    
    
    parser.add_argument('--transformer_encoder_num_heads', type=int, default=2,
                        help='Number of transformerencoder heads')
    
    parser.add_argument('--code_dim', type=int, default=128,
                        help='Code dimension embedding')

    parser.add_argument('--demo_dim', type=int, default=5,
                        help='demo dimension embedding')
        
        
    parser.add_argument('--model_path', type=str, default='./checkpoint/checkpoint.pth.tar',
                        help='Path to the pre-trained model checkpoint')

    
    parser.add_argument('--d_transformer', type=int, default=512,
                        help='Embedding dimension for transformer encoder')
    
    parser.add_argument('--rep_dim', type=int, default=252,
                        help='Ultimate Representation dimension')
    
    
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for survival component')
    
    parser.add_argument('--num_category', type=int, default=9,
                        help='Number of time intervals')    
    
    parser.add_argument('--num_epochs', type=int, default=60,
                        help='Number of training epochs')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    

    
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='Weight for loglikelihood loss')
    parser.add_argument('--lambda2', type=float, default=1000,
                        help='Weight for ranking loss')
    parser.add_argument('--lambda3', type=float, default=0.0,
                        help='Weight for MSE loss')
    parser.add_argument('--lambda4', type=float, default=0.0,
                        help='Weight for SupWCon loss')
    
    parser.add_argument('--cl_window_size', type=int, default=5,
                        help='Context window size')
    parser.add_argument('--tempreture', type=float, default=1.0,
                        help='Temperature value')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for training (e.g., cuda or cpu)')

    parser.add_argument('--batch_size', type=int, default=2000,
                        help='batch_size')   
    
    parser.add_argument('--max_num_sequence', type=int, default=5,
                        help='max_num_sequence')        

    args = parser.parse_args()
    
    
    train_set_df=pd.read_csv('final_data/train_set_df_.csv')
    train_set_df.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_dx_l1_d_train=pd.read_csv('final_data/x_dx_l1_d_train_.csv')
    x_dx_l1_d_train.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_dx_l2_d_train=pd.read_csv('final_data/x_dx_l2_d_train_.csv')
    x_dx_l2_d_train.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_dx_l3_d_train=pd.read_csv('final_data/x_dx_l3_d_train_.csv')
    x_dx_l3_d_train.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_rx_l1_d_train=pd.read_csv('final_data/x_rx_l1_d_train_.csv')
    x_rx_l1_d_train.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_rx_l2_d_train=pd.read_csv('final_data/x_rx_l2_d_train_.csv')
    x_rx_l2_d_train.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_rx_l3_d_train=pd.read_csv('final_data/x_rx_l3_d_train_.csv')
    x_rx_l3_d_train.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_rx_l4_d_train=pd.read_csv('final_data/x_rx_l4_d_train_.csv')
    x_rx_l4_d_train.drop(['Unnamed: 0'], axis=1, inplace=True)

    added_time_df_train=pd.read_csv('final_data/added_time_df_tr_.csv')
    added_time_df_train.drop(['Unnamed: 0'], axis=1, inplace=True)


    test_set_df=pd.read_csv('final_data/test_set_df.csv')
    test_set_df.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_dx_l1_d_test=pd.read_csv('final_data/x_dx_l1_d_test.csv') 
    x_dx_l1_d_test.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_dx_l2_d_test=pd.read_csv('final_data/x_dx_l2_d_test.csv')
    x_dx_l2_d_test.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_dx_l3_d_test=pd.read_csv('final_data/x_dx_l3_d_test.csv')
    x_dx_l3_d_test.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_rx_l1_d_test=pd.read_csv('final_data/x_rx_l1_d_test.csv')
    x_rx_l1_d_test.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_rx_l2_d_test=pd.read_csv('final_data/x_rx_l2_d_test.csv')
    x_rx_l2_d_test.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_rx_l3_d_test=pd.read_csv('final_data/x_rx_l3_d_test.csv')
    x_rx_l3_d_test.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_rx_l4_d_test=pd.read_csv('final_data/x_rx_l4_d_test.csv')
    x_rx_l4_d_test.drop(['Unnamed: 0'], axis=1, inplace=True)

    added_time_df_test=pd.read_csv('final_data/added_time_df_test.csv')
    added_time_df_test.drop(['Unnamed: 0'], axis=1, inplace=True)


    x_dx_train=(x_dx_l1_d_train, x_dx_l2_d_train, x_dx_l3_d_train)
    x_rx_train=(x_rx_l1_d_train, x_rx_l2_d_train, x_rx_l3_d_train, x_rx_l4_d_train)

    x_dx_test=(x_dx_l1_d_test, x_dx_l2_d_test, x_dx_l3_d_test)
    x_rx_test=(x_rx_l1_d_test, x_rx_l2_d_test, x_rx_l3_d_test, x_rx_l4_d_test)

    
    # Initialize and load your model using args.model_path
    trainer = Trainer(num_features= args.code_dim+args.demo_dim,
            demo_num_features = arg.demo_num_features,
            code_dim = args.code_dim,
            demo_dim = args.demo_dim,
            transformer_encoder_num_heads = arg.transformer_encoder_num_heads,
            d_transformer = args.d_transformer,
            rep_dim = args.rep_dim,
            hidden_dim = args.hidden_dim,
            num_category= args.num_category,
            lambda_list=[args.lambda1, args.lambda2, args.lambda3, args.lambda4],
            cl_window_size=args.cl_window_size,
            tempreture=args.tempreture,
            device=args.device,
            batch_size = args.batch_size,
            learning_rate= args.learning_rate,
            max_num_sequence = args.max_num_sequence,
            use_checkpoint=True
                     )    
    
    
    trainer.create_data_batch(train_set_df, test_set_df,
                              x_dx_train, x_rx_train, x_dx_test, x_rx_test,
                              added_time_df_train, added_time_df_test)
    trainer.train()



if __name__ == '__main__':
    main()

