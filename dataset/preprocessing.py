import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, Identity
from math import log10

def salt_and_pepper(x_in, prob):
    x_out = x_in.clone()
    noise_tensor=torch.rand_like(x_out)
    salt=torch.max(x_out)
    pepper=torch.min(x_out)
    x_out[noise_tensor < prob/2]=salt
    x_out[noise_tensor > 1-prob/2]=pepper
    return x_out

class Clip(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        eps = torch.sort(x.reshape(-1))[0][int(0.1*x.numel())].item() #90 percentile
        return torch.clip(x, eps)    

class Log(Module):
    def __init__(self, eps = 1e-3):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        return torch.clip(x, self.eps).log10()

class Equalize(Module):
    def __init__(self):
        super().__init__()
     
    def forward(self, x):
        shape = x.shape
        bs = x.shape[0]
        numel = x.shape[1]*x.shape[2]
        sorted_sequence = torch.sort(x.reshape(bs,-1))[0]
        x = torch.searchsorted(sorted_sequence, x.reshape(bs,-1))/numel
        return x.reshape(shape)

class MinMax(Module):
    def __init__(self):
        super().__init__()
     
    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min()) 

class Transform(Module):
    def __init__(self, to_log, to_minmax, to_equalize, in_shape = None, out_shape = None):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.transform = [Clip()]
        if to_log:
            self.transform.append(Log())
        if to_minmax:
            self.transform.append(MinMax())
        if to_equalize:
            self.transform.append(Equalize())
        if not len(self.transform):
            self.transform.append(Identity())
        self.transform = Sequential(*self.transform)

    def __call__(self, x):
        x = x.view(-1, *self.in_shape)
        x = self.transform(x)
        x = F.interpolate(x.unsqueeze(1), self.out_shape)
        return x.squeeze()
