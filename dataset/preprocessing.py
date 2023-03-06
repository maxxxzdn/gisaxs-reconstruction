import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, Identity
from math import log10

def salt_and_pepper(x_in, prob):
    """
    Add salt and pepper noise to an image
    Inputs:
        x_in: input image
        prob: probability of noise
    Outputs:
        x_out: noisy image
    """
    x_out = x_in.clone()
    noise_tensor=torch.rand_like(x_out)
    salt=torch.max(x_out)
    pepper=torch.min(x_out)
    x_out[noise_tensor < prob/2]=salt
    x_out[noise_tensor > 1-prob/2]=pepper
    return x_out

class Clip(Module):
    """
    Clip the input to the 10th percentile
    Used to remove outliers
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        eps = torch.sort(x.reshape(-1))[0][int(0.1*x.numel())].item() #90 percentile
        return torch.clip(x, eps)   

class Log(Module):
    """
    Logarithm of the input
    """
    def __init__(self, eps = 1e-3):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        return torch.clip(x, self.eps).log10()

class MinMax(Module):
    """
    MinMax normalization of the input
    """
    def __init__(self):
        super().__init__()
     
    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min()) 

class Transform(Module):
    """
    Preprocessing of the input
    Inputs:
        to_log (bool): apply log to the input
        to_minmax (bool): apply minmax normalization to the input
        to_equalize (bool): apply histogram equalization to the input
        in_shape (bool): shape of the input
        out_shape (bool): shape of the output
    """
    def __init__(self, to_log: bool, to_minmax: bool, to_equalize: bool, in_shape: list = None, out_shape: list = None):
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
        x = x.view(-1, *self.in_shape)[:,:,128:300]
        x = self.transform(x)
        return x
