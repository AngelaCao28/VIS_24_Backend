import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import json
import numpy as np
import pandas as pd

# Dataset
df = pd.read_csv("./data-frames/temp4.csv")
df = df.reset_index()
idx_all = np.repeat(True,len(df))

# Variables
device = torch.device('cpu')
num_chars = 7
num_players = 2454
num_teams = 98
input_vars = ['act', 'T', 'x', 'y', 'sg', 'thetag', 'scrad', 'deltaT', 'deltax', 'deltay', 's']
target_vars = ['act', 'x', 'y']
input_index = ['TID', 'PLID']

# Hyperparameters
num_action_cats = num_chars
act_padding_idx = 1
scale_grad_by_freq = True
num_contvars_in  = len(input_vars) - 1
num_contvars_out = len(target_vars) - 1
transformer_finaldenselayer_dim = 64
transformer_finaldenselayer_dim_1 = 64

cat_embedding_dim = 3
cont_embedding_dim = 5
num_layers = 1
dim_feedforward = 8
model_type = "Transformer"
transformer_nhead = 1

# Newly added hyperparameters
num_player_cats = num_players
player_embedding_dim = 16
num_team_cats = num_teams
team_embedding_dim = 16
emb_finaldenselayer_dim = 16
embdenselayer_dim = 8

d_model = cat_embedding_dim + cont_embedding_dim

# Positional encoding for Transformer encoder component
def ian_generate_positional_encoding(src):

  pos_encoding = torch.zeros_like(src[0])
  seq_len = pos_encoding.shape[0]
  d_model = pos_encoding.shape[1]

  for i in range(d_model):
    for pos in range(seq_len):
      if i % 2 == 0:
        pos_encoding[pos,i] = np.sin(pos/100**(2*i/d_model))
      else:
        pos_encoding[pos,i] = np.cos(pos/100**(2*i/d_model))
  return pos_encoding.float()

# Model Definition
class Soccer_Model_3ze(nn.Module):
    def __init__(self):  #pick up all specification vars from the global environment
        super(Soccer_Model_3ze, self).__init__()

        # Add player embedding and team embedding
        self.player_emb = nn.Embedding(num_player_cats,player_embedding_dim)
        self.team_emb = nn.Embedding(num_team_cats,team_embedding_dim)
        self.emb_lin1 = nn.Linear(player_embedding_dim+team_embedding_dim,emb_finaldenselayer_dim,bias=True)
        self.emb_lin2 = nn.Linear(emb_finaldenselayer_dim,embdenselayer_dim,bias=True)
        
        self.emb = nn.Embedding(num_action_cats,cat_embedding_dim,padding_idx=act_padding_idx,scale_grad_by_freq=scale_grad_by_freq)
        self.lin0 = nn.Linear(num_contvars_in,cont_embedding_dim,bias=True)
        if model_type == "Transformer":
          #self.msk = torch.nn.Transformer.generate_square_subsequent_mask(self,maxlen-1).to(device)
          self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=transformer_nhead,dim_feedforward=dim_feedforward).to(device)
          self.seqnet = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        if model_type == "LSTM":
          self.seqnet =  nn.LSTM(input_size=d_model,hidden_size=dim_feedforward,num_layers=num_layers,dropout=0.2,bias=True)

        if model_type == "Transformer":
          self.lin1 = nn.Linear(d_model+embdenselayer_dim,transformer_finaldenselayer_dim)
        else:
          self.lin1 = nn.Linear(dim_feedforward+embdenselayer_dim,transformer_finaldenselayer_dim)

        self.lin2 = nn.Linear(transformer_finaldenselayer_dim, transformer_finaldenselayer_dim_1,bias=True)
        self.lin3 = nn.Linear(transformer_finaldenselayer_dim_1, num_action_cats + num_contvars_out,bias=True)
        print(self)

    def forward(self, X, Index):
        global testX, X_cat_seqnet
        testX = X
        #X = testX
        X_acts = X[:,:,0].long()
        X_actsemb = self.emb(X_acts)
        X_cont = X[:,:,1:]
        X_cont = self.lin0(X_cont.float())
              
        X_cat = torch.cat([X_actsemb,X_cont],dim=2)
        X_cat = X_cat.float()

        src = X_cat + ian_generate_positional_encoding(X_cat).to(device)
        if model_type == "Transformer":
          X_cat_seqnet = self.seqnet(src)
        else:
          X_cat_seqnet,_ = self.seqnet(src)

        Index_player = Index[:,1].long()
        Index_team = Index[:,0].long()
        
        player_emb_vect = self.player_emb(Index_player)
        team_emb_vect = self.team_emb(Index_team)
        Index_cat = torch.cat([player_emb_vect,team_emb_vect],dim=1)
        Index_cat = Index_cat.float()
        Index_cat = self.emb_lin1(Index_cat)
        Index_cat = F.relu(Index_cat)
        Index_cat = self.emb_lin2(Index_cat)

        # Concatenation between seq and emb
        out = torch.cat([X_cat_seqnet[:,-1,:],Index_cat],dim=1)
        out = self.lin1(out)
        
        out = F.relu(out)
        out = self.lin2(out)
        out = F.relu(out)
        out = self.lin3(out)
        return out
    
# Loss function
def Soccer_Model_3_lossfn_3ze(X, Yhat, Y, cat_weight=5.,cont_weight=1.):
        global testX, testYhat, testY
        testX, testYhat, testY = X, Yhat,Y
                
        Y_acts = Y[:,0].long()
        Yminus1_acts = X[:,-1,0].long()
        Y_cont = Y[:,1:].float()

        Yhat_acts = Yhat[:,:num_chars]
        Yhat_cont = Yhat[:,num_chars:].float()

        CEL = nn.CrossEntropyLoss(weight=weights.to(device),reduction="none")

        Yhat_CEL  = torch.mean(CEL(Yhat_acts,Y_acts)) * cat_weight
        Yhat_MSEL= (Yhat_cont-Y_cont.float())**2
        
        idx_ignorecurrent = torch.logical_or(Y_acts == 0, Y_acts == 1)
        idx_ignoreminus1  = torch.logical_or(Yminus1_acts == 0, Yminus1_acts == 1)
        idx_ignore = torch.logical_or(idx_ignorecurrent,idx_ignoreminus1)
        if torch.sum(idx_ignore) == 0:
          Yhat_MSEL = torch.mean(Yhat_MSEL)**(1/2) * cont_weight
        elif torch.sum(~idx_ignore) == 0:
          Yhat_MSEL = torch.Tensor([0.]).to(device)
        else:
          Yhat_MSEL = Yhat_MSEL[~idx_ignore,:]
          Yhat_MSEL = torch.mean(Yhat_MSEL)**(1/2) * cont_weight
         
        Yhat_totL = Yhat_CEL + Yhat_MSEL
        return Yhat_totL, Yhat_CEL, Yhat_MSEL
    
# Dataloader definition
maxlen = 10
step = 1    

class SoccerDataset(Dataset):
    # cut the text in semi-redundant sequences of maxlen characters
    def __init__(self,idx=idx_all):
        self.idx = idx
        self.valid_slice_idxn = np.where(np.logical_and(self.idx,df["valid_slice_flag"]))[0]  #in both the idx and has a valid slice flag
    def __len__(self):
        return int(np.sum(df.loc[self.idx,"valid_slice_flag"]))
    def __getitem__(self, i):
        j = self.valid_slice_idxn[i]
        index = df.iloc[j-1].loc[input_index]
        y = df.iloc[j].loc[target_vars]
        x = df.iloc[(j-maxlen):j].loc[:,input_vars]
        return x.to_numpy(), y.to_numpy().astype(float), index.to_numpy().astype(int)

all_dataset = SoccerDataset()

batch_size, num_workers = 128, 2
all_loader = DataLoader(all_dataset,shuffle=False,batch_size=batch_size,num_workers=num_workers,drop_last=True)

# Data preprocessing
# replace own goal "h" with goal "g"
idx = df["act"] == "h"
df.loc[idx,"act"] = "g"

# set up idx2char and char2idx
chars = sorted(list(set(df['act'])))
num_chars_test = len(chars)
char2idx_test = dict((c, i) for i, c in enumerate(chars))
idx2char_test = dict((i, c) for i, c in enumerate(chars))

idx2char = {0: '@', 1: '_', 2: 'd', 3: 'g', 4: 'p', 5: 's', 6: 'x'}
char2idx = {'@': 0, '_': 1, 'd': 2, 'g': 3, 'p': 4, 's': 5, 'x': 6}

# replace characters with numbers
df['act'].replace(char2idx,inplace=True)

weights = char2idx.copy()
for x in weights: weights[x]=1
weights['@'] = 0
weights['_'] = 0
weights['d'] = 0.120
weights['g'] = 0
weights['p'] = 0.013
weights['s'] = 0.580
weights['x'] = 0.29

weights = torch.tensor(list(weights.values()))
cont_weights = torch.tensor([1.,1.])

def do_epoch(dataloader, model, loss_fn):
    torch.cuda.empty_cache(); import gc; gc.collect()

    with torch.no_grad():
      for batch, (X, Y, Index) in enumerate(dataloader):
          X, Y, Index = X.to(device), Y.to(device), Index.to(device)
            
          pred = model(X, Index)
    
          loss, lossCEL, lossMSE = loss_fn(X, pred, Y)
        
          if batch % 500 == 0:
            loss = loss.item()
            print(loss)

def go():
    torch.cuda.empty_cache(); import gc; gc.collect()
    model = Soccer_Model_3ze().to(device)
    model.load_state_dict(torch.load('./model-files/MDLstate_2024_03_08-085614', map_location=device))
    model.eval()

    loss_fn    = Soccer_Model_3_lossfn_3ze
    do_epoch(all_loader, model, loss_fn)

if __name__ == "__main__":
    go()