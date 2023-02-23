from torch.nn import Module
from torch import cat

class View(Module):
    """
    Reshapes a tensor to a given shape.
    """
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class ToSymmetric(Module):
    """
    Converts a lower triangular matrix to a symmetric matrix.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """
        Forward pass of the ToSymmetric transform.
        Input:
            x (torch.Tensor): input tensor
        """
        middle = x.shape[-1]//2
        x[:,:,:,middle:] = x.flip(-1)[:,:,:,middle:]
        return x
    
class Reflect(Module):
    """
    Reflects a tensor along the last dimension.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return cat([x, x.flip(-1)], -1)
    
class Squeeze(Module):
    """
    Squeezes a tensor.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.squeeze()