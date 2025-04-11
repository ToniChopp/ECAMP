# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import math
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import InterpolationMode

import timm.models.vision_transformer
import ml_collections
import ipdb

from torch import Tensor
from typing import Optional, Callable


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    

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
        expansion: int = 4
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.expansion = expansion
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.patch_size = 16
        self.det_head = nn.Linear(self.embed_dim, self.embed_dim)
        del self.norm
        del self.head

    def forward_features(self, x):
        B = x.shape[0]
        h = x.shape[2]
        w = x.shape[3]
        x = self.patch_embed(x)

        height = h // self.patch_size
        width = w // self.patch_size

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = x[:, 1:, :]
        x = self.det_head(x)
        x = einops.rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                             b=B, h=height, w=width, p1=1, p2=1, c=self.embed_dim)    # type: ignore
        return x
    
    def forward(self, x):
        ipdb.set_trace()
        x = self.forward_features(x)
        return x


class DetectionLayers(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            channels=[512, 1024, 2048],
            expansion=4
        ):
        super().__init__()
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(embed_dim, channels[0], 3, padding=1),
        #     nn.BatchNorm2d(channels[0]),
        #     nn.ReLU(),
        #     nn.Conv2d(channels[0], channels[0], 3, padding=1),
        # )
        self.traspose = nn.Conv2d(embed_dim, channels[1], 1)
        self.layer1 = Bottleneck(channels[1], channels[1]//expansion, expansion=expansion, norm_layer=nn.BatchNorm2d)
        
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(channels[0], channels[1], 3, padding=1),
        #     nn.BatchNorm2d(channels[1]),
        #     nn.ReLU(),
        #     nn.Conv2d(channels[1], channels[1], 3, padding=1),
        # )
        self.transpose1 = nn.Conv2d(embed_dim, channels[0], 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer2 = Bottleneck(channels[0], channels[0]//expansion, expansion=expansion, norm_layer=nn.BatchNorm2d)
        # self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        # self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample = nn.Conv2d(embed_dim, channels[2], kernel_size=(1, 1), stride=(2, 2), bias=False)
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(channels[0], channels[2], 3, padding=1),
        #     nn.BatchNorm2d(channels[2]),
        #     nn.ReLU(),
        #     nn.Conv2d(channels[2], channels[2], 3, padding=1),
        # )
        self.layer3 = Bottleneck(channels[2], channels[2]//expansion, expansion=expansion, norm_layer=nn.BatchNorm2d)
    
    def forward(self, x):
        out1 = self.traspose(x)
        out1 = self.layer1(out1)

        out2 = self.transpose1(x)
        out2 = self.upsample(out2)
        out2 = self.layer2(out2)
        
        out3 = self.downsample(x)
        out3 = self.layer3(out3)
        return out2, out1, out3


class ViTDetector(nn.Module):
    def __init__(
            self,
            patch_size=16,
            out_channels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            expansion=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ):
        super().__init__()

        self.encoder = VisionTransformer(
            patch_size=patch_size,
            embed_dim=hidden_size,
            depth=num_hidden_layers,
            num_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer
        )

        self.layers = DetectionLayers(embed_dim=hidden_size, expansion=expansion)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.layers(x)
        return x

