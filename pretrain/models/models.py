#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is 
#  to recognize text and layout from full-page images with end-to-end deep neural networks.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import torch
from torch.nn import Module, ModuleList
from torch.nn import Conv2d
from torch.nn import InstanceNorm2d
from torch.nn import Dropout, Dropout2d
from torch.nn import ReLU
from torch.nn.functional import pad
from typing import List
from timm.models.registry import register_model
import random


class DepthSepConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1, 1), dilation=(1, 1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        return x


class MixDropout(Module):
    def __init__(self, dropout_proba=0.4, dropout2d_proba=0.2):
        super(MixDropout, self).__init__()

        self.dropout = Dropout(dropout_proba)
        self.dropout2d = Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)


'''
Resnet like encoder with 6 downsampling stages, each stage is a ConvBlock or DSCBlock
'''
class FCN_Encoder2(Module):
    def __init__(self, params):
        super(FCN_Encoder2, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 256, stride=(2, 2), dropout=self.dropout),
            ConvBlock(256, 512, stride=(2, 2), dropout=self.dropout),
        ])
        self.blocks = ModuleList([
            DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
        ])
        self.mixer = Conv2d(128, 128, kernel_size=(1, 5), padding=(0, 2))

    def get_feature_map_channels(self) -> List[int]:
        return [32, 64, 128, 256, 512]

    def get_downsample_ratio(self):
        return (32, 32)


    def forward(self, inp_bchw, hierarchical: bool = False):
        x = inp_bchw
        feats = []

        # --- stem / downsampling stages ---
        for i, b in enumerate(self.init_blocks):
            x = b(x)
            if i in (1, 2, 3, 4):   # taps after channel increases to 32, 64, 128
                feats.append(x)

        #x = self.mixer(x)

        # --- depthwise-separable blocks + residual adds ---
        for j, b in enumerate(self.blocks):
            xt = b(x)
            x = x + xt if x.shape == xt.shape else xt
            if j == len(self.blocks) - 1:  # final stage: 256 ch
                feats.append(x)

        if hierarchical:
            # sanity: ensure the channel dims match what get_feature_map_channels() promises
            assert len(feats) == len(self.get_feature_map_channels()), \
                f"Collected {len(feats)} feature maps, expected {len(self.get_feature_map_channels())}"
            # (optional) lightweight runtime check on channels
            # for f, c in zip(feats, self.get_feature_map_channels()):
            #     assert f.shape[1] == c, f"Expected {c} channels, got {f.shape[1]}"
            return feats

        # non-hierarchical: return logits (global pooled)
        if hasattr(self, "head"):
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.head(pooled)
        else:
            # if no classifier is defined, return the final feature map
            return x


'''
DAN encoder with 6 downsampling stages, each stage is a ConvBlock or DSCBlock
'''
class FCN_Encoder(Module):
    def __init__(self, params):
        super(FCN_Encoder, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
        ])
        self.blocks = ModuleList([
            DSCBlock(128, 128, stride=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, stride=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, stride=(1, 1), dropout=self.dropout),
            DSCBlock(128, 256, stride=(1, 1), dropout=self.dropout),
        ])

    def get_feature_map_channels(self) -> List[int]:
        return [32, 64, 128, 256]

    def get_downsample_ratio(self):
        #return (32, 8)
        return (32, 8)


    def forward(self, inp_bchw, hierarchical: bool = False):
        x = inp_bchw
        feats = []

        # --- stem / downsampling stages ---
        for i, b in enumerate(self.init_blocks):
            x = b(x)
            if i in (1, 2, 3):   # taps after channel increases to 32, 64, 128
                feats.append(x)

        # --- depthwise-separable blocks + residual adds ---
        for j, b in enumerate(self.blocks):
            xt = b(x)
            x = x + xt if x.shape == xt.shape else xt
            if j == len(self.blocks) - 1:  # final stage: 256 ch
                feats.append(x)

        if hierarchical:
            # sanity: ensure the channel dims match what get_feature_map_channels() promises
            assert len(feats) == len(self.get_feature_map_channels()), \
                f"Collected {len(feats)} feature maps, expected {len(self.get_feature_map_channels())}"
            # (optional) lightweight runtime check on channels
            # for f, c in zip(feats, self.get_feature_map_channels()):
            #     assert f.shape[1] == c, f"Expected {c} channels, got {f.shape[1]}"
            return feats

        # non-hierarchical: return logits (global pooled)
        if hasattr(self, "head"):
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.head(pooled)
        else:
            # if no classifier is defined, return the final feature map
            return x
        

class ConvBlock(Module):

    def __init__(self, in_, out_, stride=(1, 1), k=3, activation=ReLU, dropout=0.4):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = Conv2d(in_channels=in_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv2 = Conv2d(in_channels=out_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv3 = Conv2d(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        #print('1')
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        #print('2')
        if pos == 1:
            #print('2.1')
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        #print('3')
        if pos == 2:
            #print('3.1')
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        x = self.activation(x)
        #print('4')

        if pos == 3:
            #print('4.1')
            x = self.dropout(x)
        return x


class DSCBlock(Module):

    def __init__(self, in_, out_, stride=(2, 1), activation=ReLU, dropout=0.4):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_, out_, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_, out_, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout/2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)
        return x




@register_model
def DAN_encoder(pretrained=False, **kwargs):
    params = {
        "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
        "dropout": 0.0, # 0.5,
    }
    return FCN_Encoder(params)

@register_model
def DAN_encoder2(pretrained=False, **kwargs):
    params = {
        "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
        "dropout": 0.0, # 0.5,
    }
    return FCN_Encoder2(params)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('your_convnet_small')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    #downsample_ratio = cnn.get_downsample_ratio()
    downsample_ratio_h, downsample_ratio_w = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    # 224/7 is 32, 224/28 is 8
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    print(f"shape={feats[-1].shape},  {feats[-1].shape[-2]} == {H} {feats[-1].shape[-1]} == {W}")
    assert feats[-1].shape[-2] == H // downsample_ratio_h
    assert feats[-1].shape[-1] == W // downsample_ratio_w
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
