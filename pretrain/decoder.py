# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from utils.misc import is_pow2n
from torch.nn import functional as F

class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d, scale):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.scale = scale
        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.up_sample41 = nn.ConvTranspose2d(cin, cin, kernel_size=(4, 1), stride=(4, 1), padding=(0, 0), bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cout),
        )
    
    def forward(self, x):
        if self.scale == (2,2):
            x = self.up_sample(x)
        elif self.scale == (4,1):
            x = self.up_sample41(x)
        else:
            raise ValueError(f"Unsupported scale: {self.scale}")        
        return self.conv(x)


class LightDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768, sbn=True):   # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        
        self.width = width
        up_sample_ratio_h, up_sample_ratio_w = up_sample_ratio
        assert is_pow2n(up_sample_ratio_w)
        assert is_pow2n(up_sample_ratio_h)
        n_w = round(math.log2(up_sample_ratio_w))
        n_h = round(math.log2(up_sample_ratio_h))
        #channels = [self.width // 2 ** i for i in range(n_w + 1)] # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        #self.factors = [(2,2), (2,2), (2,2), (2,2), (2,2)]
        self.factors = [(4,1), (2,2), (2,2), (2,2)]
        channels = [self.width]
        for (fh, fw) in self.factors:
            channels.append(max(1, int(channels[-1] // ((fh * fw) ** 0.5))))
        bn2d = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn2d, scale) for (cin, cout, scale) in zip(channels[:-1], channels[1:], self.factors)])
        self.proj = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1, bias=True)
        
        self.initialize()
    
    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
            #for name, param in self.dec[i].named_parameters():
            #    print(name)
        #x = self.refine(x) # 20260125
        return self.proj(x)
    
    def extra_repr(self) -> str:
        return f'width={self.width}'
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
