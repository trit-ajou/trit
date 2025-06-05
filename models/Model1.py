import torch
import torchvision
from .BaseModel import BaseModel  # Corrected import path based on file structure
from typing import List, Dict, Optional  # For type hinting
from .model1_util.vgg16_bn import vgg16_bn, init_weights
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# CRAFT ver.
class Model1(BaseModel):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = _DoubleConv(1024, 512, 256)
        self.upconv2 = _DoubleConv(512, 256, 128)
        self.upconv3 = _DoubleConv(256, 128, 64)
        self.upconv4 = _DoubleConv(128, 64, 32)

        num_class = 2 # Region Score, Affinity Score
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        last_conv = self.conv_cls[-1]          # nn.Conv2d(16 → 2)
        if last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, -4.0)
        self.to(self.device)

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)
        sources = [s.detach() for s in sources]  # 그래프 연결 완전 끊기

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1), feature
