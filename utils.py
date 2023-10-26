import torch.nn as nn
import torch.nn.utils.weight_norm as wn
import torch
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self,input_len,output_len):
        super().__init__()
        self.linear1 = nn.Sequential(
            wn(nn.Linear(input_len, output_len)),
            nn.ReLU(),
            wn(nn.Linear(output_len,output_len))
        )

        self.linear2 = nn.Sequential(
            wn(nn.Linear(output_len, output_len)),
            nn.ReLU(),
            wn(nn.Linear(output_len, output_len))
        )

        self.skip = wn(nn.Linear(input_len, output_len))
        self.act = nn.ReLU()
        
    def forward(self,x):
        x = self.act(self.linear1(x)+self.skip(x))
        x = self.linear2(x)
        
        return x
    
class MLP_bottle(nn.Module):
    def __init__(self,input_len,output_len,bottleneck,bias=True):
        super().__init__()
        self.linear1 = nn.Sequential(
            wn(nn.Linear(input_len, bottleneck,bias=bias)),
            nn.ReLU(),
            wn(nn.Linear(bottleneck,bottleneck,bias=bias))
        )

        self.linear2 = nn.Sequential(
            wn(nn.Linear(bottleneck, bottleneck)),
            nn.ReLU(),
            wn(nn.Linear(bottleneck, output_len))
        )

        self.skip = wn(nn.Linear(input_len, bottleneck,bias=bias))
        self.act = nn.ReLU()
        
    def forward(self,x):
        x = self.act(self.linear1(x)+self.skip(x))
        x = self.linear2(x)
        
        return x

class Coefnet(nn.Module):
    def __init__(self, blocks,d_model,heads,norm_layer=None, projection=None):
        super().__init__()
        layers = [BCAB(d_model,heads) for i in range(blocks)]
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        # heads = heads if blocks > 0 else 1
        self.last_layer = last_layer(d_model,heads)

    def forward(self, basis, series):
        attns1 = []
        attns2 = []
        for layer in self.layers:
            basis,series,basis_attn,series_attn = layer(basis,series)   #basis(B,N,d)  series(B,C,d)
            attns1.append(basis_attn)
            attns2.append(series_attn)
        
        coef = self.last_layer(series,basis)  #(B,k,C,N)
        
        return coef,attns1,attns2

class BCAB(nn.Module):
    def __init__(self, d_model,heads=8,index=0,d_ff=None,
                     dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention_basis = channel_AutoCorrelationLayer(d_model,heads,dropout=dropout)
        self.conv1_basis = wn(nn.Linear(d_model,d_ff))
        self.conv2_basis = wn(nn.Linear(d_ff,d_model))

        self.dropout_basis = nn.Dropout(dropout)
        self.activation_basis = F.relu if activation == "relu" else F.gelu
        
        self.cross_attention_ts = channel_AutoCorrelationLayer(d_model,heads,dropout=dropout)
        self.conv1_ts = wn(nn.Linear(d_model,d_ff))
        self.conv2_ts = wn(nn.Linear(d_ff,d_model))

        self.dropout_ts = nn.Dropout(dropout)
        self.activation_ts = F.relu if activation == "relu" else F.gelu
        self.layer_norm11 = nn.LayerNorm(d_model)
        self.layer_norm12 = nn.LayerNorm(d_model)
        self.layer_norm21 = nn.LayerNorm(d_model)
        self.layer_norm22 = nn.LayerNorm(d_model)

    def forward(self, basis,series):
        basis_raw = basis
        series_raw = series
        basis_add, basis_attn = self.cross_attention_basis(
            basis_raw, series_raw, series_raw,
        )
        basis_out = basis_raw + self.dropout_basis(basis_add)
        basis_out = self.layer_norm11(basis_out)

        y_basis = basis_out
        y_basis = self.dropout_basis(self.activation_basis(self.conv1_basis(y_basis)))
        y_basis = self.dropout_basis(self.conv2_basis(y_basis))
        basis_out = basis_out + y_basis
        
        basis_out = self.layer_norm12(basis_out)
        
        series_add,series_attn = self.cross_attention_ts(
            series_raw, basis_raw, basis_raw
        )
        series_out = series_raw + self.dropout_ts(series_add)
        
        series_out = self.layer_norm21(series_out)

        y_ts = series_out
        y_ts = self.dropout_ts(self.activation_ts(self.conv1_ts(y_ts)))
        y_ts = self.dropout_ts(self.conv2_ts(y_ts))
        series_out = series_out + y_ts
        series_out = series_raw
        
        series_out = self.layer_norm22(series_out)

        return basis_out, series_out, basis_attn, series_attn
    
class channel_AutoCorrelationLayer(nn.Module):
    def __init__(self,d_model,n_heads, mask=False,d_keys=None,
                 d_values=None,dropout=0):
        super().__init__()
        
        self.mask = mask

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = wn(nn.Linear(d_model,d_keys * n_heads))
        self.key_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.value_projection = wn(nn.Linear(d_model, d_values * n_heads))
        self.out_projection = wn(nn.Linear(d_values * n_heads, d_model))
        self.n_heads = n_heads
        self.scale = d_keys ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        num = len(queries.shape)
        if num == 2:
            L, _ = queries.shape
            S, _ = keys.shape
            H = self.n_heads

            queries = self.query_projection(queries).view(L, H, -1).permute(1,0,2)
            keys = self.key_projection(keys).view(S, H, -1).permute(1,0,2)
            values = self.value_projection(values).view(S, H, -1).permute(1,0,2)
            # queries = queries.view(L, H, -1).permute(1,0,2)
            # keys = keys.view(S, H, -1).permute(1,0,2)
            # values = values.view(S, H, -1).permute(1,0,2)

            dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, values)    #(H,L,D)

            out = out.permute(1,0,2).reshape(L,-1)
        else:
            B,L, _ = queries.shape
            B,S, _ = keys.shape
            H = self.n_heads

            queries = self.query_projection(queries).view(B,L, H, -1).permute(0,2,1,3)
            keys = self.key_projection(keys).view(B,S, H, -1).permute(0,2,1,3)
            values = self.value_projection(values).view(B,S, H, -1).permute(0,2,1,3)

            dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            
            attn = self.dropout(attn)

            out = torch.matmul(attn, values)    #(H,L,D)

            out = out.permute(0,2,1,3).reshape(B,L,-1)
            
        return self.out_projection(out),attn
    
class last_layer(nn.Module):
    def __init__(self,d_model,n_heads, mask=False,d_keys=None,
                 d_values=None,dropout=0):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = wn(nn.Linear(d_model,d_keys * n_heads))
        self.key_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.n_heads = n_heads
        self.scale = d_keys ** -0.5

    def forward(self, queries, keys):
        B,L, _ = queries.shape
        B,S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B,L, H, -1).permute(0,2,1,3)
        keys = self.key_projection(keys).view(B,S, H, -1).permute(0,2,1,3)

        dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale   #(B,H,L,S)

        return dots
    
