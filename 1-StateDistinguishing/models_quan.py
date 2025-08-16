import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

import math

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True, dropout=0):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V) #
        self.fc_k = nn.Linear(dim_K, dim_V) #
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.d = nn.Dropout(p=dropout)
        self.dropout = dropout
        self.eps = 1e-7
        
    def forward(self, Q, K):
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A_ = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)

        if self.dropout==0: A = A_
        else:
            dA_ = self.d(A_)
            A = dA_ / (torch.broadcast_to(dA_.sum(axis=2).reshape(-1,Q.size(1),1), (-1,Q.size(1), K.size(1)))+self.eps)
        if self.mask is None: A = A
        else:
            dA_ = A * self.mask
            A = dA_ / torch.broadcast_to(dA_.sum(axis=2).reshape(-1,Q.size(1),1), (-1,Q.size(1), K.size(1)))
        self.A = A
        self.Awosm = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=True, dropout=0, mask=None, minus=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln = ln, dropout=dropout, mask=mask, minus=minus)
        
    def forward(self, X):
        return self.mab(X, X)

class PAB(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln = True, dropout=0):
        super(PAB, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln = ln, dropout=dropout)
        
    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

########################################################################
########################################################################
class QuAN_Tbt(nn.Module): #Tbt
    def __init__(self, setsize, channel, dim_output, kersize, stride, \
                dim_hidden, num_heads, Nr, Nc, p_outputs, sab_L=2, sab_M=2, ln=True, dropout=0):
        super(QuAN_Tbt,self).__init__()
        self.Ch = channel
        self.Nr = Nr
        self.Nc = Nc
        self.setsize = setsize
        self.h = dim_hidden
        self.sig = nn.Sigmoid()
        self.emb = nn.Embedding(4, self.Ch)
        if sab_L==1:
            self.transformer = SAB(self.Ch*self.Nc*2, dim_hidden, num_heads, ln=ln, dropout=dropout)
        elif sab_L==2:
            self.transformer = nn.Sequential(
                    SAB(self.Ch*self.Nc*2, dim_hidden, num_heads, ln=ln, dropout=dropout), 
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout))
        elif sab_L==3:
            self.transformer = nn.Sequential(
                    SAB(self.Ch*self.Nc*2, dim_hidden, num_heads, ln=ln, dropout=dropout),
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout), 
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout))
        self.transformer_dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
                nn.LayerNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_output))
        
        if sab_M==1:
            self.enc = SAB(self.Nr//2, self.Nr//2, num_heads, ln=ln)
        elif sab_M==2:
            self.enc = nn.Sequential(
                SAB(self.Nr//2, self.Nr//2, num_heads, ln=ln),
                SAB(self.Nr//2, self.Nr//2, num_heads, ln=ln))
        elif sab_M==3:
            self.enc = nn.Sequential(
                SAB(self.Nr//2, self.Nr//2, num_heads, ln=ln),
                SAB(self.Nr//2, self.Nr//2, num_heads, ln=ln),
                SAB(self.Nr//2, self.Nr//2, num_heads, ln=ln))
        
        self.dec = nn.Sequential(
                PAB(self.Nr//2, num_heads, p_outputs, ln=ln),
                nn.Linear(self.Nr//2, dim_output),
                nn.Sigmoid())
        
        self.apply(weights_init)

    def forward(self, X):
        X = X.int()
        # embedding
        X = self.emb(X.view(-1, self.Nr*self.Nc, 1))
        X = X.view(-1, self.Nr//2, self.Ch*self.Nc*2)
        # transformer
        X = self.transformer_dec(self.sig(self.transformer(X)))
        self.aftertemp = X
        X = self.sig(self.enc(X.view(-1, self.setsize, self.Nr//2)))
        return self.dec(X)[:,0]
    

class SMLP_Tbm(nn.Module): # SMLP_Tbm
    def __init__(self, setsize, channel, dim_output, kersize, stride, \
                dim_hidden, num_heads, Nr, Nc, p_outputs, sab_L=1, sab_M=2, model_loaded=None, ln=True, dropout=0):
        super(SMLP_Tbm,self).__init__()
        self.Ch = channel
        self.Nr = Nr
        self.Nc = Nc
        self.setsize = setsize
        self.h = dim_hidden
        self.sig = nn.Sigmoid()
        self.emb = nn.Embedding(4, self.Ch)
        if sab_L==1:
            self.transformer = nn.Sequential(
                    nn.Linear(self.Ch*self.Nc*2, 2*dim_hidden), nn.Sigmoid(),
                    nn.Linear(2*dim_hidden, dim_hidden))
        elif sab_L==2:

            self.transformer = nn.Sequential(
                    nn.Linear(self.Ch*self.Nc*2, 4*dim_hidden), nn.Sigmoid(),
                    nn.Linear(4*dim_hidden, dim_hidden))
        self.transformer_dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
                nn.LayerNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_output))
        if sab_M==1:
            self.enc = nn.Sequential(
                nn.Linear(self.Nr//2, 4*self.Nr//2),
                nn.Sigmoid(),
                nn.Linear(4*self.Nr//2, self.Nr//2),
                nn.Sigmoid())
        elif sab_M==2:
            self.enc = nn.Sequential(
                nn.Linear(self.Nr//2, 3*self.Nr//2),
                nn.Sigmoid(),
                nn.Linear(3*self.Nr//2, self.Nr//2),
                nn.Sigmoid(),
                nn.Linear(self.Nr//2, 3*self.Nr//2),
                nn.Sigmoid(),
                nn.Linear(3*self.Nr//2, self.Nr//2),
                nn.Sigmoid())
        self.dec = nn.Sequential(
                nn.Linear(self.Nr//2, 3*self.Nr//2),
                nn.Sigmoid(),
                nn.Linear(3*self.Nr//2, dim_output))
        
        self.apply(weights_init)

    def forward(self, X):
        X = X.int()
        # embedding
        X = self.emb(X.view(-1, self.Nr*self.Nc, 1))
        X = X.view(-1, self.Nr//2, self.Ch*self.Nc*2)
        # transformer
        X = self.transformer_dec(self.sig(self.transformer(X)))
        X = self.enc(X.view(-1, self.setsize, self.Nr//2))
        X = self.dec(X)
        return self.sig(X.mean(axis=1))

class SMLP_Tbt(nn.Module): # SMLP_Tbt
    def __init__(self, setsize, channel, dim_output, kersize, stride, \
                dim_hidden, num_heads, Nr, Nc, p_outputs, sab_L=1, sab_M=2, model_loaded=None, ln=True, dropout=0):
        super(SMLP_Tbt,self).__init__()
        self.Ch = channel
        self.Nr = Nr
        self.Nc = Nc
        self.rNr = int((Nr - self.ks)/self.st + 1)
        self.rNc = int((Nc - self.ks)/self.st + 1)
        self.setsize = setsize
        self.h = dim_hidden
        self.sig = nn.Sigmoid()
        self.emb = nn.Embedding(4, self.Ch)
        if sab_L==1:
            self.transformer = SAB(self.Ch*self.Nc*2, dim_hidden, num_heads, ln=ln, dropout=dropout)
        elif sab_L==2:
            self.transformer = nn.Sequential(
                    SAB(self.Ch*self.Nc*2, dim_hidden, num_heads, ln=ln, dropout=dropout), 
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout))
        elif sab_L==3:
            self.transformer = nn.Sequential(
                    SAB(self.Ch*self.Nc*2, dim_hidden, num_heads, ln=ln, dropout=dropout),
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout), 
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout))
        self.transformer_dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
                nn.LayerNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_output))
            
        if sab_M==1:
            self.enc = nn.Sequential(
                nn.Linear(self.Nr//2, 4*self.Nr//2),
                nn.Sigmoid(),
                nn.Linear(4*self.Nr//2, self.Nr//2),
                nn.Sigmoid())
        elif sab_M==2:
            self.enc = nn.Sequential(
                nn.Linear(self.Nr//2, 3*self.Nr//2),
                nn.Sigmoid(),
                nn.Linear(3*self.Nr//2, self.Nr//2),
                nn.Sigmoid(),
                nn.Linear(self.Nr//2, 3*self.Nr//2),
                nn.Sigmoid(),
                nn.Linear(3*self.Nr//2, self.Nr//2),
                nn.Sigmoid())
        self.dec = nn.Sequential(
                nn.Linear(self.Nr//2, 3*self.Nr//2),
                nn.Sigmoid(),
                nn.Linear(3*self.Nr//2, dim_output))
        
        self.apply(weights_init)

    def forward(self, X):
        X = X.int()
        # embedding
        X = self.emb(X.view(-1, self.Nr*self.Nc, 1))
        X = X.view(-1, self.Nr//2, self.Ch*self.Nc*2)
        # transformer
        X = self.transformer_dec(self.sig(self.transformer(X)))
        X = self.enc(X.view(-1, self.setsize, self.Nr//2))
        X = self.dec(X)
        return self.sig(X.mean(axis=1))