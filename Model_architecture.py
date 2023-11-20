import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import math


def positional_encoding2(seq_len, d_model, device):
  pe = np.zeros((seq_len, d_model))
  for pos in range(seq_len):
    for i in range(0, d_model, 2):
      pe[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
      pe[pos, i+1] = math.cos(pos / (10000 ** (i/d_model)))
  return torch.from_numpy(pe[np.newaxis, :]).to(device)


class SharedEncoder(nn.Module):
    def __init__(self, num_encoder_heads, X_dim, d_transformer, rep_dim, device, dropout=0.6):
        super().__init__()
        self.linear1 = nn.Linear(X_dim, d_transformer)
        self.encoder = nn.TransformerEncoderLayer(d_model=d_transformer, nhead=num_encoder_heads,dropout=0.6, batch_first=True)
        self.encoder2 = nn.TransformerEncoderLayer(d_model=d_transformer, nhead=num_encoder_heads,dropout=0.6, batch_first=True)
        self.encoder3 = nn.TransformerEncoderLayer(d_model=d_transformer, nhead=num_encoder_heads,dropout=0.6, batch_first=True)
        self.encoder4 = nn.TransformerEncoderLayer(d_model=d_transformer, nhead=num_encoder_heads,dropout=0.6, batch_first=True)
        self.linear2 = nn.Linear(d_transformer, rep_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.device = device
        
        self.src_position_embedding = positional_encoding2(5, d_transformer, device)
        
    def forward(self, inp_x):
        encoded_x = self.dropout1(self.linear1(inp_x))
        encoded_x = (encoded_x + self.src_position_embedding).type(torch.FloatTensor).to(self.device)
        encoded_x = self.encoder(encoded_x)
        encoded_x = self.encoder2(encoded_x)
        encoded_x = self.encoder3(encoded_x)
        encoded_x = self.encoder4(encoded_x)
        encoded_x = self.dropout2(self.linear2(encoded_x))
        return encoded_x
    
  

         
    
class FC(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_category, dropout=0.6):
        
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, int(0.5*hidden_dim))
        self.linear3 = nn.Linear(int(0.5*hidden_dim), num_category)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, encoded_inp):
        x = self.dropout1(F.relu(self.linear1(encoded_inp)))
        x = self.dropout2(F.relu(self.linear2(x)))
        x = torch.sigmoid(self.linear3(x))
        return x

    
class OTCSurv(nn.Module):
    def __init__(self, 
                 num_features,
                 demo_num_features,
                 onto_dx_params,
                 onto_rx_params,
                 code_dim,
                 demo_dim,
                 transformer_encoder_num_heads,
                 d_transformer,
                 rep_dim,
                 hidden_dim,
                 num_event,
                 num_category,
                 device):
        
        super().__init__()
        
        self.code_dim = code_dim
        self.pad_idx = 0
        self.device = device
        
        # EMBEDDING GENERATION
        self.embedding_demo = nn.Linear(demo_num_features, demo_dim)
        
        self.embedding_dx_l1 = nn.Embedding(onto_dx_params['l1'], code_dim)
        self.embedding_dx_l2 = nn.Embedding(onto_dx_params['l2'], code_dim)
        self.embedding_dx_l3 = nn.Embedding(onto_dx_params['l3'], code_dim)

        self.embedding_rx_l1 = nn.Embedding(onto_rx_params['l1'], code_dim)
        self.embedding_rx_l2 = nn.Embedding(onto_rx_params['l2'], code_dim)
        self.embedding_rx_l3 = nn.Embedding(onto_rx_params['l3'], code_dim)
        self.embedding_rx_l4 = nn.Embedding(onto_rx_params['l4'], code_dim)
        
        
        self.energy_dx = nn.Linear(code_dim * 2, 1)
        self.energy_rx = nn.Linear(code_dim * 2, 1)
        
        self.energy_visit_dx = nn.Linear(code_dim, 1)
        self.energy_visit_rx = nn.Linear(code_dim, 1)
        
        self.compresed_visit_energy = nn.Linear(rep_dim,1)
        
        self.softmax = nn.Softmax(dim=-2)
        
        
        self.encoder = SharedEncoder(num_encoder_heads=transformer_encoder_num_heads, 
                                     X_dim=num_features,
                                     d_transformer=d_transformer, 
                                     rep_dim=rep_dim,
                                     device = self.device)
        
        #contrastive component mlps
        self.mlp = nn.Linear(rep_dim, int(rep_dim/2))
        self.mlp2 = nn.Linear(int(rep_dim/2), int(rep_dim/2))
        self.dropout1 = nn.Dropout(0.4)


        
        
        self.fc_net = nn.ModuleList([FC(embed_dim=rep_dim, hidden_dim=hidden_dim, num_category=num_category) 
                                     for i in range(num_event)])

        
        
        self.hidden_dim = hidden_dim
        self.num_category = num_category
        self.num_event = num_event
        
        
    def make_padding_mask(self, x_dx, x_rx):   
        x_dx_mask = (x_dx != self.pad_idx)
        x_rx_mask = (x_rx != self.pad_idx)
        # (N, 5, max_len)
        
        return x_dx_mask.to(self.device), x_rx_mask.to(self.device)
    
        
    def get_context_vector(self, x_dx, x_rx, mask=True): 
        '''based on GRAM paper'''
        
        x_dx_l1, x_dx_l2, x_dx_l3 = x_dx
        x_rx_l1, x_rx_l2, x_rx_l3, x_rx_l4 = x_rx
        
        embedding_dx_l1 = self.embedding_dx_l1(x_dx_l1.int())  # (N, num_visits, num_dx_l3, num_codes)
        embedding_dx_l2 = self.embedding_dx_l2(x_dx_l2.int())
        embedding_dx_l3 = self.embedding_dx_l3(x_dx_l3.int())
        embedding_dx = torch.cat([embedding_dx_l1.unsqueeze(-2),embedding_dx_l2.unsqueeze(-2),embedding_dx_l3.unsqueeze(-2)], dim=-2)
        embedding_dx_l3_extend = embedding_dx_l3.unsqueeze(-2).repeat(1,1,1,3,1)
        # (N, num_visits, max_len, 3, num_codes)
        embedding_dx_cat = torch.cat([embedding_dx_l3_extend, embedding_dx], dim=-1)
        # (N, num_visits, max_len, 3, num_codes*2)
        
        embedding_rx_l1 = self.embedding_rx_l1(x_rx_l1.int())
        embedding_rx_l2 = self.embedding_rx_l2(x_rx_l2.int())
        embedding_rx_l3 = self.embedding_rx_l3(x_rx_l3.int())
        embedding_rx_l4 = self.embedding_rx_l4(x_rx_l4.int())
        embedding_rx = torch.cat([embedding_rx_l1.unsqueeze(-2),embedding_rx_l2.unsqueeze(-2),embedding_rx_l3.unsqueeze(-2), embedding_rx_l4.unsqueeze(-2)], dim=-2)
        embedding_rx_l4_extend = embedding_rx_l4.unsqueeze(-2).repeat(1,1,1,4,1)
        # (N, num_visits, max_len, 4, num_codes)     
        embedding_rx_cat = torch.cat([embedding_rx_l4_extend, embedding_rx], dim=-1)
        # (N, num_visits, max_len, 4, 2*num_codes) 
        
        
        energy_dx = F.relu(self.energy_dx(embedding_dx_cat))
        # (N, num_visits, max_len, 3, 1)
        energy_rx = F.relu(self.energy_rx(embedding_rx_cat))
        # (N, num_visits, max_len, 4, 1)
        
        if mask:
            dx_mask, rx_mask = self.make_padding_mask(x_dx_l1, x_rx_l1) # (N, num_visits, max_len)
            dx_mask = dx_mask.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3,1)
            rx_mask = rx_mask.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,4,1)
            energy_dx = energy_dx.masked_fill(dx_mask == 0, float("-1e20"))
            energy_rx = energy_rx.masked_fill(rx_mask == 0, float("-1e20"))
            
        attention_dx = self.softmax(energy_dx)
        attention_rx = self.softmax(energy_rx)
        
        


        # attention: (N, num_visits, max_len, 3, 1), nvdrw
        # embedding_dx: N, num_visits, max_len, 3, code_dim), nvdrs
        # we want context_vector: (N, num_visits, max_len, 1, code_dim), nvdws
        context_vector_dx = torch.einsum("nvdrw,nvdrs->nvdws", attention_dx, embedding_dx).squeeze()        
        #(N, num_visits, max_len, code_dim)

        # attention: (N, num_visits, max_len, 4, 1), nvdrw
        # embedding_dx: N, num_visits, max_len, 4, code_dim), nvdrs
        # we want context_vector: (N, num_visits, max_len, 1, code_dim), nvdws
        context_vector_rx = torch.einsum("nvdrw,nvdrs->nvdws", attention_rx, embedding_rx).squeeze()  
        #(N, num_visits, max_len, code_dim)
        
        #context_vector = torch.cat([context_vector_dx, context_vector_rx], dim=-2)
        # (N, num_visits, max_len*2, code_dim)
        
        return context_vector_dx, context_vector_rx, (attention_dx,attention_rx)
    # (N, num_visits, max_len, code_dim)
    
    def get_compressed_visit_vector(self, x_dx, x_rx, context_vector_dx, context_vector_rx, mask=True):
        '''attention pooling'''
        
        x_dx_l1, _, _ = x_dx
        x_rx_l1, _, _, _ = x_rx     
        context_vector = torch.cat([context_vector_dx, context_vector_rx], dim=-2)
        # (N, num_visits, 2*max_len, code_dim)
        
        energies_visit_dx = self.energy_visit_dx(context_vector_dx)
        #(N, num_visits, max_len, 1)
        energies_visit_rx = self.energy_visit_rx(context_vector_rx)
        #(N, num_visits, max_len, 1)
        
        if mask:
            dx_mask, rx_mask = self.make_padding_mask(x_dx_l1, x_rx_l1) # (N, num_visits, max_len)
            dx_mask = dx_mask.unsqueeze(-1)
            rx_mask = rx_mask.unsqueeze(-1)
            energies_visit_dx = energies_visit_dx.masked_fill(dx_mask == 0, float("-1e20"))
            energies_visit_rx = energies_visit_rx.masked_fill(rx_mask == 0, float("-1e20"))
            
        attentions_visit_dx = self.softmax(energies_visit_dx) 
        attentions_visit_rx = self.softmax(energies_visit_rx)
        attentions_visit = torch.cat([attentions_visit_dx, attentions_visit_rx], dim=-2)
        #attentions_visit : (N, num_visits, 2*max_len, 1) nvms
        
        
        
        #attentions_visit : (N, num_visits, 2*max_len, 1) nvms
        #context_vector : (N, num_visits, 2*max_len, code_dim) nvmc
        # compressed_context_vector : (N, num_visits, code_dim, 1) nvcs
        compressed_context_vector = torch.einsum("nvms,nvmc->nvcs" ,attentions_visit, context_vector)

        
        return compressed_context_vector.squeeze(), attentions_visit, (energies_visit_dx, energies_visit_rx)
    
    
    def get_compressed_incoded_visit(self, encoded_inp):
        compresed_visit_energies = self.compresed_visit_energy(encoded_inp) # (N,5,1)
        attentions_compresed_visit = self.softmax(compresed_visit_energies)    
        
        # attentions_compresed_visit.shape : (N,5,1)  nvm
        # encoded_inp.shape : (N,5,128)  nvs
        # compressed_incoded_visit.shape : (N,1,128) nms
        compressed_incoded_visit = torch.einsum("nvm,nvs->nms" ,attentions_compresed_visit, encoded_inp)
        
        return compressed_incoded_visit.squeeze(), attentions_compresed_visit, compresed_visit_energies#(N,embed_dim)

        
    def forward(self, x_dx, x_rx, x_demo):
        
        context_vector_dx, context_vector_rx, gram_att = self.get_context_vector(x_dx, x_rx) #(N, num_visits, max_len*2, code_dim)
        #context_vector = context_vector.reshape(context_vector.shape[0],context_vector.shape[1], -1)
        compressed_context_vector, att_visit, e_visit = self.get_compressed_visit_vector(x_dx, x_rx, context_vector_dx, context_vector_rx)
        
        demo_embedding = F.relu(self.embedding_demo(x_demo))

        encoder_input = torch.cat([demo_embedding, compressed_context_vector], dim=-1)
    

        encoded_inp = self.encoder(encoder_input)  # (N,5,rep_dim)
        compressed_incoded_visit, att_compresed_visit,e_compresed_visit  = self.get_compressed_incoded_visit(encoded_inp) 
        # (N,rep_dim)
        projected_head = self.dropout1(F.relu(self.mlp(compressed_incoded_visit)))
        projected_head = self.mlp2(projected_head)
        
        predicted_output = []
        for i in range(self.num_event):
            predicted_output_ = self.fc_net[i](compressed_incoded_visit.unsqueeze(1))
            predicted_output.append(predicted_output_)
        
        out_concat = torch.concat(predicted_output, axis=1)  # stack referenced on subject
 
        #(N, num_event, num_category)     

        
        return out_concat, compressed_incoded_visit, projected_head
        