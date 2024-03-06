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

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from .bert_encoder import MultiModalBertEncoder


import ipdb


class InterpolateConvSuperResolution(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + residual
        x = self.relu(x)   # add skip connection

        return x


class ECAMP(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # image encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # image decoder and SR specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size)**2 * in_chans, bias=True)

        self.super_res = InterpolateConvSuperResolution(
            scale_factor=2,
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # --------------------------------------------------------------------------
        # Bert encoder
        self.bert_encoder = MultiModalBertEncoder()
        self.bert_mlp = nn.Linear(embed_dim, 768, bias=True)
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)


        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0] * 2
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep
    

    def mask_2_pixel(self, mask, column, row):
        """
        mask: [N, L], 1 is keep, 0 is remove, 
        pixel_mask: [N, 3, imgs_size, imgs_size], 1 is keep, 0 is remove
        """
        p = self.patch_embed.patch_size[0]
        super_p = p * 2

        mask = mask.reshape(shape=(mask.shape[0], int(mask.shape[1]**.5), int(mask.shape[1]**.5)))
        super_mask = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), device=mask.device)
        # ipdb.set_trace()
        for i in range(mask.shape[0]):
            super_mask[i, column[i]:column[i]+12, row[i]:row[i]+12] = 1
        # super_mask[:, column[:, None]:column[:, None]+12, row[:, None]:row[:, None]+12] = 1
        # ipdb.set_trace()
        pixel_mask = torch.kron(mask, torch.ones((p, p)).cuda())
        super_pixel_mask = torch.kron(super_mask, torch.ones((super_p, super_p)).cuda())
        pixel_mask = pixel_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        super_pixel_mask = super_pixel_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        return pixel_mask, super_pixel_mask


    def image_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, ids_keep
    

    def image_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predict the patch
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    

    def forward_report_decoder(self, latent, ids_keep, caption_ids, labels, attention_mask, token_type_ids, weights):
        latent = self.bert_mlp(latent)
        gap_token = latent[:, 1:, :].mean(dim=1)
        gap_token = gap_token.unsqueeze(1)
        latent = latent[:, 1:, :]
        outputs = self.bert_encoder(latent, gap_token, caption_ids, labels, attention_mask, token_type_ids, weights)
        return outputs.loss
    

    def forward_loss(self, imgs, big_imgs, pred, mask, column, row):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        pixel_mask, super_mask = self.mask_2_pixel(mask, column, row)

        # get back to image space for super resolution
        pred = self.unpatchify(pred)
        super_pred_imgs = self.super_res(pred)

        mask_pred_imgs = pred * pixel_mask
        mask_imgs = imgs * pixel_mask

        super_pred_imgs = super_pred_imgs * super_mask
        big_imgs = big_imgs * super_mask

        # loss = (pred_imgs - imgs) ** 2         # type: ignore
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # loss = (loss * pixel_mask).sum() / pixel_mask.sum()  # mean loss on removed patches
        mim_loss = F.mse_loss(mask_pred_imgs, mask_imgs, reduction='mean')
        res_loss = F.mse_loss(super_pred_imgs, big_imgs, reduction='mean')
        return mim_loss, res_loss


    def forward(self, batch, mask_ratio=0.75):
        big_imgs = batch["image"]
        
        ids, labels, attention_mask, type_ids = batch["ids"], batch["labels"], batch["attention_mask"], batch["type_ids"]
        weights = batch["weights"]

        column = batch["column"]
        row = batch["row"]

        big_imgs = big_imgs.cuda()
        ids = ids.cuda()
        labels = labels.cuda()
        attention_mask = attention_mask.cuda()
        type_ids = type_ids.cuda()
        weights = weights.cuda()
        imgs = torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC)(big_imgs)

        latent, img_mask, ids_restore, ids_keep = self.image_encoder(imgs, mask_ratio)                  # global feature
        pred_img = self.image_decoder(latent, ids_restore)  # [N, L, p*p*3]
        mim_loss, res_loss = self.forward_loss(imgs, big_imgs, pred_img, img_mask, column, row)

        mlm_loss = self.forward_report_decoder(latent, ids_keep, ids, labels, attention_mask, type_ids, weights)
        return mim_loss, res_loss, mlm_loss


def ecamp(**kwargs):
    model = ECAMP(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
