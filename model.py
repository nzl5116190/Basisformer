import torch.nn as nn
import torch.nn.utils.weight_norm as wn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time
import math
import numpy as np
from utils import Coefnet,MLP_bottle

class Basisformer(nn.Module):
    def __init__(self,seq_len,pred_len,d_model,heads,basis_nums,block_nums,bottle,map_bottleneck,device,tau,is_MS=False,input_channel=0):
        super().__init__()
        self.d_model = d_model
        self.k = heads
        self.N = basis_nums
        self.coefnet = Coefnet(blocks=block_nums,d_model=d_model,heads=heads)
            
        self.pred_len = pred_len
        self.seq_len = seq_len
        
        self.MLP_x = MLP_bottle(seq_len,heads * int(seq_len/heads),int(seq_len/bottle))
        self.MLP_y = MLP_bottle(pred_len,heads * int(pred_len/heads),int(pred_len/bottle))
        self.MLP_sx = MLP_bottle(heads * int(seq_len/heads),seq_len,int(seq_len/bottle))
        self.MLP_sy = MLP_bottle(heads * int(pred_len/heads),pred_len,int(pred_len/bottle))
        
        self.project1 = wn(nn.Linear(seq_len,d_model))
        self.project2 = wn(nn.Linear(seq_len,d_model))
        self.project3 = wn(nn.Linear(pred_len,d_model))
        self.project4 = wn(nn.Linear(pred_len,d_model))
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.L1Loss(reduction='none')
        
        self.device = device
                        
        # smooth array
        arr = torch.zeros((seq_len+pred_len-2,seq_len+pred_len))
        for i in range(seq_len+pred_len-2):
            arr[i,i]=-1
            arr[i,i+1] = 2
            arr[i,i+2] = -1
        self.smooth_arr = arr.to(device)
        self.map_MLP = MLP_bottle(1,self.N*(self.seq_len+self.pred_len),map_bottleneck,bias=True)
        self.tau = tau
        self.epsilon = 1E-5
        self.is_MS = is_MS
        if is_MS:
            self.MLP_MS = wn(nn.Linear(input_channel,1))
            self.mean_MS = wn(nn.Linear(input_channel,1))
            self.std_MS = wn(nn.Linear(input_channel,1))
        
    def forward(self,x,mark,y=None,train=True,y_mark=None):
        mean_x = x.mean(dim=1,keepdim=True)
        std_x = x.std(dim=1,keepdim=True)
        feature = (x - mean_x) / (std_x + self.epsilon)
        B,L,C = feature.shape
        feature = feature.permute(0,2,1)
        feature = self.project1(feature)   #(B,C,d)
        
        m = self.map_MLP(mark[:,0].unsqueeze(1)).reshape(B,self.seq_len + self.pred_len,self.N)
        m = m / torch.sqrt(torch.sum(m**2,dim=1,keepdim=True)+self.epsilon)
        
        raw_m1 = m[:,:self.seq_len].permute(0,2,1)  #(B,L,N)
        raw_m2 = m[:,self.seq_len:].permute(0,2,1)   #(B,L',N)
        m1 = self.project2(raw_m1)    #(B,N,d)
        
        score,attn_x1,attn_x2 = self.coefnet(m1,feature)    #(B,k,C,N)
        if self.is_MS:
            score = self.MLP_MS(score.permute(0,1,3,2)).permute(0,1,3,2) # (B,k,1,N)

        base = self.MLP_y(raw_m2).reshape(B,self.N,self.k,-1).permute(0,2,1,3)   #(B,k,N,L/k)
        out = torch.matmul(score,base).permute(0,2,1,3).reshape(B,score.shape[2],-1)  #(B,C,k * (L/k))
        out = self.MLP_sy(out).reshape(B,score.shape[2],-1).permute(0,2,1)   #（BC,L）
        
        if self.is_MS:
            std_x = self.std_MS(std_x)
            mean_x = self.mean_MS(mean_x)
        output = out * (std_x + self.epsilon) + mean_x

        #loss
        if train:
            l_smooth = torch.einsum('xl,bln->xbn',self.smooth_arr,m)
            l_smooth = abs(l_smooth).mean()
            # l_smooth = self.criterion1(l_smooth,torch.zeros_like(l_smooth))
            
            # #back
            mean_y = y.mean(dim=1,keepdim=True)
            std_y = y.std(dim=1,keepdim=True)
            feature_y_raw = (y - mean_y) / (std_y + self.epsilon)
            
            feature_y = feature_y_raw.permute(0,2,1)
            feature_y = self.project3(feature_y)   #(BC,d)
            m2 = self.project4(raw_m2)    #(N,d)
            
            score_y,attn_y1,attn_y2 = self.coefnet(m2,feature_y)    #(B,k,C,N)
            logit_q = score.permute(0,2,3,1) #(B,C,N,k)
            logit_k = score_y.permute(0,2,3,1) #(B,C,N,k)

            # l_pos = torch.bmm(logit_q.view(-1,1,self.k), logit_k.view(-1,self.k,1)).reshape(-1,1)  #(B*C*N,1,1)
            l_neg = torch.bmm(logit_q.reshape(-1,self.N,self.k), logit_k.reshape(-1,self.N,self.k).permute(0,2,1)).reshape(-1,self.N) # (B,C*N,N)

            labels = torch.arange(0,self.N,1,dtype=torch.long).unsqueeze(0).repeat(B*score.shape[2],1).reshape(-1)

            labels = labels.to(self.device)

            cross_entropy_loss = nn.CrossEntropyLoss()
            l_entropy = cross_entropy_loss(l_neg/self.tau, labels)           
            
            return output,l_entropy,l_smooth,attn_x1,attn_x2,attn_y1,attn_y2
        else:
            # #back
            mean_y = y.mean(dim=1,keepdim=True)
            std_y = y.std(dim=1,keepdim=True)
            feature_y_raw = (y - mean_y) / (std_y + self.epsilon)
            
            feature_y = feature_y_raw.permute(0,2,1)
            feature_y = self.project3(feature_y)   #(B,C,d)
            m2 = self.project4(raw_m2)    #(B,N,d)
            
            score_y,attn_y1,attn_y2 = self.coefnet(m2,feature_y)    #(B,k,C,N)
            return output,m,attn_x1,attn_x2,attn_y1,attn_y2      
        
        
