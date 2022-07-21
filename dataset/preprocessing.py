from torch import sort, searchsorted, clip, sort
from torch.nn.functional import interpolate
from torch.nn import Module, Sequential, Identity
from torch.nn.functional import normalize
from math import log10


class Clip(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        eps = 60. #sort(x.reshape(-1))[0][int(0.1*x.numel())].item() #90 percentile
        return clip(x, eps)    

class Log(Module):
    def __init__(self, eps = 1e-3):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        return clip(x, self.eps).log10()

class Equalize(Module):
    def __init__(self):
        super().__init__()
     
    def forward(self, x):
        shape = x.shape
        bs = x.shape[0]
        numel = x.shape[1]*x.shape[2]
        sorted_sequence = sort(x.reshape(bs,-1))[0]
        x = searchsorted(sorted_sequence, x.reshape(bs,-1))/numel
        return x.reshape(shape)

class MinMax(Module):
    def __init__(self):
        super().__init__()
     
    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min()) 

class Transform(Module):
    def __init__(self, out_shape, to_log, to_minmax, to_equalize):
        super().__init__()
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
        x = x.view(-1, 1200, 120)
        x = self.transform(x)
        return interpolate(x.unsqueeze(1), self.out_shape).squeeze()
