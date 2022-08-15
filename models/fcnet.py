import torch.nn as nn
from torch import clip, sigmoid
from .utils import View, Reflect, ToSymmetric
from numpy import prod

# reproducibility
import torch
#torch.manual_seed(0)


class FCNet(nn.Module):
    def __init__(self, in_dim, name, drop_prob):
        super().__init__()
        self.name = name
        out_shape = (128,16)
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim, prod(out_shape)//32),
            nn.BatchNorm1d(prod(out_shape)//32),
            nn.SiLU(inplace = True),
            nn.Dropout(drop_prob),
            
            nn.Linear(prod(out_shape)//32, prod(out_shape)//16),
            nn.BatchNorm1d(prod(out_shape)//16),
            nn.SiLU(inplace = True),
            nn.Dropout(drop_prob),
            
            nn.Linear(prod(out_shape)//16, prod(out_shape)//4), 
            nn.BatchNorm1d(prod(out_shape)//4),
            nn.SiLU(inplace = True),
            nn.Dropout(drop_prob),
            
            nn.Linear(prod(out_shape)//4, prod(out_shape)),
            View((-1,1,out_shape[0], out_shape[1])),
            #Reflect(),
        ) 

    def forward(self, x):
        return self.fc(x)
    

class _FCNet(nn.Module):
    def __init__(self, out_shape, in_dim, name):
        super().__init__()
        self.name = name
        self.fc = nn.Sequential(
            nn.Linear(in_dim, prod(out_shape)//32),
            nn.ReLU(inplace = True),
            nn.Linear(prod(out_shape)//32, prod(out_shape)//16),
            nn.ReLU(inplace = True),
            nn.Linear(prod(out_shape)//16, prod(out_shape)//4), 
            nn.ReLU(inplace = True),
            nn.Linear(prod(out_shape)//4, prod(out_shape)//2),
            View((-1,1,out_shape[0], out_shape[1]//2)),
            Reflect(),
        ) 
               
    def forward(self, x):
        return self.fc(x)
