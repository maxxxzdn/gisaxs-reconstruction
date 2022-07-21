import torch.nn as nn
from torch import Tensor, sigmoid
from torchvision.models.resnet import BasicBlock, conv3x3
from typing import Optional, Callable
from .utils import View, ToSymmetric

# reproducibility
import torch
torch.manual_seed(0)


class ResNet(nn.Module):
    def __init__(self, in_dim, max_channels, name):
        super().__init__()
        self.name = name
        c = [max_channels, max_channels//2, max_channels//4, max_channels//8]
        self.fc = nn.Sequential(
            nn.Linear(in_dim, c[0]*1*1), 
            nn.ReLU(inplace=True),
            View((-1,c[0],1,1)))
        self.avgpool = nn.Sequential(
            nn.Upsample(scale_factor=(8,1)),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            BasicBlock(c[0],c[0]), 
            BasicBlockTranspose(c[0],c[1],1, 
                downsample = nn.Sequential(
                    nn.ConvTranspose2d(c[0],c[1],4,2,1),
                    nn.BatchNorm2d(c[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))))
        self.layer3 = nn.Sequential(
            BasicBlock(c[1],c[1]), 
            BasicBlockTranspose(c[1],c[2],1, 
                downsample = nn.Sequential(
                    nn.ConvTranspose2d(c[1],c[2],4,2,1),
                    nn.BatchNorm2d(c[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))))
        self.layer2 = nn.Sequential(
            BasicBlock(c[2],c[2]), 
            BasicBlockTranspose(c[2],c[3],1, 
            downsample= nn.Sequential(
                nn.ConvTranspose2d(c[2],c[3],4,2,1), 
                nn.BatchNorm2d(c[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))))
        self.layer1 = nn.Sequential(
            BasicBlock(c[3],c[3]), 
            BasicBlock(c[3],c[3],1))
        self.maxpool = nn.Sequential(
            nn.ConvTranspose2d(c[3],c[3],4,2,1),
            nn.BatchNorm2d(c[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(c[3], c[3], 7, 1, 3),
            nn.BatchNorm2d(c[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(c[3], 1, 7, 1, 3),
            ToSymmetric())
        
    def forward(self, x):
        x = self.fc(x)
        x = self.avgpool(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.conv1(x)
        return sigmoid(x)
    
class BasicBlockTranspose(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv4x4transpose(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
def conv4x4transpose(in_planes: int, out_planes: int, stride: int = 2, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """4x4 transposed convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=4,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )