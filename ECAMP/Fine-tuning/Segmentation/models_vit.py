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


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, sample_rate=4, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.sample_rate = sample_rate

        self.patch_size = self.patch_embed.patch_size[0]
        self.sample_v = int(math.pow(2, self.sample_rate))
        self.seg_head = nn.Linear(self.embed_dim, self.patch_size * self.patch_size * self.embed_dim // (self.sample_v ** 2))
        self.h = self.patch_size // self.sample_v
        self.w = self.patch_size // self.sample_v
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
        x = self.seg_head(x)
        x = einops.rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                             b=B, h=height, w=width, p1=self.h, p2=self.w, c=self.embed_dim)    # type: ignore
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x


class SegViT(nn.Module):
    def __init__(
            self,
            patch_size=16,
            out_channels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            decode_features=[512, 256, 128, 64],
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sample_rate=4
        ):
        super().__init__()
        self.features = decode_features
        self.encoder = VisionTransformer(
            patch_size=patch_size,
            embed_dim=hidden_size,
            depth=num_hidden_layers,
            num_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            sample_rate=sample_rate
        )
        self.decoder = Decoder(
            in_channels=hidden_size,
            out_channels=out_channels,
            features=self.features
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x