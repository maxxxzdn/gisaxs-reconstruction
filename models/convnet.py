import torch.nn as nn
from torch import clip
from .utils import View, ToSymmetric

# reproducibility
import torch
torch.manual_seed(0)


class ConvNet(nn.Module):
    def __init__(self, in_shape, latent_dim, n_channels, kernel_size, mode, name):
        super().__init__()
        self.name = name
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, n_channels//2 * (in_shape[0]//16) * (in_shape[1]//16)),
            nn.ReLU(inplace = True),
            View((-1,n_channels//2,(in_shape[0]//16),(in_shape[1]//16))),
        ) 
        convs = []
        convs.append(ConvBlock(n_channels//2, n_channels, kernel_size, mode))
        convs.append(ConvBlock(n_channels, n_channels, kernel_size, mode))
        convs.append(ConvBlock(n_channels, n_channels, kernel_size, mode))
        convs.append(ConvBlock(n_channels, n_channels//2, kernel_size, mode))
        convs.append(ConvBlock(n_channels//2, n_channels//4, kernel_size, None))
        convs.append(ConvBlock(n_channels//4, 1, kernel_size, None))
        self.net = nn.Sequential(*convs)
               
    def forward(self, z):
        return clip(self.net(self.fc(z)), 0., 1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mode, act_fn = nn.SiLU): #nn.SiLU, nn.GELU
        super().__init__()
        if mode == 'upsample':
            assert kernel_size % 2, 'odd kernel is required'
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size//2),
                act_fn(),
            )
        elif mode == 'transpose':
            if kernel_size % 2:
                kernel_size += 1
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 2, kernel_size//2-1),
                act_fn(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size//2),
                act_fn(),
            )
            
    def forward(self, x):
        return self.conv(x)

class _ConvNet(nn.Module):
    def __init__(self, latent_dim, n_channels):
        super().__init__()     
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, n_channels//2*8*4),
            View((-1,n_channels//2,8,4)),
            nn.BatchNorm2d(n_channels//2),
            nn.ReLU(),                      
        ) 
        self.conv1 = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            #nn.ConvTranspose2d(16, 32, (3,25), 1, (1,12)),
            nn.ConvTranspose2d(n_channels//2, n_channels, (24,4), 2, (11,1)),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),                      
        )  
        self.conv2 = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            #nn.ConvTranspose2d(32, 32, (3,25), 1, (1,12)), 
            nn.ConvTranspose2d(n_channels, n_channels, (24,4), 2, (11,1)),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),                                   
        ) 
        self.conv3 = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            #nn.ConvTranspose2d(32, 32, (3,25), 1, (1,12)),
            nn.ConvTranspose2d(n_channels, n_channels, (24,4), 2, (11,1)),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            #nn.ConvTranspose2d(32, 16, (3,25), 1, (1,12)),
            nn.ConvTranspose2d(n_channels, n_channels//2, (24,4), 2, (11,1)),
            nn.BatchNorm2d(n_channels//2),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            #nn.ConvTranspose2d(16, 1, (3,25), 1, (1,12)) 
            nn.ConvTranspose2d(n_channels//2, 1, (24,4), 2, (11,1)),  
        )
        
    def forward(self, z):
        z = self.fc(z)
        z = self.conv1(z)
        z = self.conv2(z)
        z = self.conv3(z)
        z = self.conv4(z)
        z = self.conv5(z)
        z[:,:,:,64:] = z.flip(-1)[:,:,:,64:] # enforce mirror symmetry
        return z
