from torch import sort, searchsorted, clip
from torch.nn.functional import interpolate
from torch.nn import Module, Sequential, Identity
from math import log10

MEAN = 125 #73.6592
STD = 130 #653.2066

class Log(Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
        
    def forward(self, x):
        x = clip(x, self.a)
        return x.log10()

class Equalize(Module):
    def __init__(self):
        super().__init__()
     
    def forward(self, x):
        shape = x.shape
        bs = x.shape[0]
        numel = x.shape[1]*x.shape[2]
        sorted_sequence = sort(x.reshape(bs,-1))[0]
        x = searchsorted(sorted_sequence, x.reshape(bs,-1))/numel
        return 2*x.reshape(shape) - 1

class MinMax(Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
     
    def forward(self, x):
        x = clip(x, self.a, self.b)
        x = (x - self.a) / (self.b - self.a)
        return 2*x - 1 

class Transform(Module):
    def __init__(self, to_log, to_minmax, to_equalize, a, b):
        super().__init__()
        self.transform = []
        if to_log:
            self.transform.append(Log(a))
            a, b = log10(a), log10(b)
        if to_minmax:
            self.transform.append(MinMax(a, b))
        if to_equalize:
            self.transform.append(Equalize())
        if not len(self.transform):
            self.transform.append(Identity())
        self.transform = Sequential(*self.transform)

    def __call__(self, x):
        x = x.view(-1, 1200, 120)
        x = self.transform(x)
        x = x.unsqueeze(1)
        return interpolate(x, (128,16)).squeeze()
