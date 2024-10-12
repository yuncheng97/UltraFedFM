# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            # Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

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
        p = self.patch_embed.patch_size[0]
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

        return x_masked, mask, ids_restore

    def preprocess_tensor(self, tensor):
        # 去除负值
        tensor = torch.clamp(tensor, min=0)
        
        # 处理 inf 和 nan
        tensor[torch.isinf(tensor)] = 0
        tensor[torch.isnan(tensor)] = 0
        
        # 计算每行的和
        row_sums = tensor.sum(dim=1, keepdim=True)
        
        # 找出和小于等于0的行
        invalid_rows = (row_sums <= 0).squeeze()
        
        # 对和小于等于0的行设置为均匀分布
        if invalid_rows.any():
            tensor[invalid_rows] = 1.0 / tensor.size(1)
        
        # 重新计算每行的和，防止出现0
        row_sums = tensor.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # 防止除以零
        
        # 归一化每一行，使其总和为1
        tensor = tensor / row_sums
        # 检查是否有负值
        if (tensor < 0).any():
            print("Tensor contains negative values.")
        # 检查是否有inf和nan
        if torch.isinf(tensor).any() or torch.isnan(tensor).any():
            print("Tensor contains inf or nan values.")
        # 检查每行的和是否大于0
        if (tensor.sum(dim=1) <= 0).any():
            print("Some rows have non-positive sums.")
        return tensor

    def attention_masking(self, x, a, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D   = x.shape  # batch, length, dim
        len_keep  = int(L * (1 - mask_ratio))
        max_value = a.max(dim=1, keepdim=True).values
        temp_a    = max_value - a
        temp_max_value = temp_a.max(dim=1, keepdim=True).values
        max_diff  = abs(max_value - temp_max_value)
        ins_a     = temp_a + max_diff
        ins_a     = self.preprocess_tensor(ins_a)
        # ins_a = temp_a
        noise     = torch.multinomial(ins_a, num_samples=L, replacement=False)
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

        return x_masked, mask, ids_restore

    def mix_attention_masking(self, x, x_aux, a, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D   = x.shape  # batch, length, dim


        len_keep  = int(L * (1 - mask_ratio))
        max_value = a.max(dim=1, keepdim=True).values
        temp_a    = max_value - a
        temp_max_value = temp_a.max(dim=1, keepdim=True).values
        max_diff  = abs(max_value - temp_max_value)
        ins_a     = temp_a + max_diff
        ins_a = self.preprocess_tensor(ins_a)
        noise     = torch.multinomial(ins_a, num_samples=L, replacement=False)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_restore_aux = torch.argsort(ids_shuffle, dim=1, descending=True)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_remove = ids_shuffle[:, len_keep:]
        # breakpoint()
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_unmasked = torch.gather(x_aux, dim=1, index=ids_remove.unsqueeze(-1).repeat(1, 1, D))
        x_masked = torch.concat((x_masked, x_unmasked), dim=1)
        # breakpoint()
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        mask_aux = torch.ones([N, L], device=x.device)
        mask_aux[:, :(L-len_keep)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask_aux = torch.gather(mask_aux, dim=1, index=ids_restore_aux)
        return x_masked, mask, mask_aux, ids_restore, len_keep

    def forward_encoder(self, x, a, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        x, mask, ids_restore = self.attention_masking(x, a, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_encoder_unmix(self, x, x_aux, a, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        x_aux = self.patch_embed(x_aux)
        # add pos embed w/o cls token
        x_aux = x_aux + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        x, mask, mask_aux, ids_restore, len_keep = self.mix_attention_masking(x, x_aux, a, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, mask_aux, ids_restore, len_keep

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # origin batch
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

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        
        return x

    def forward_decoder_unmix(self, x, ids_restore, len_keep):
        # embed tokens
        x = self.decoder_embed(x)

        # origin batch
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - len_keep, 1)
        x_ = torch.cat([x[:, 1:len_keep, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x_ori = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # auxiliary batch
        # append mask tokens to sequence
        mask_tokens_aux = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - (x.shape[1]-len_keep), 1)
        x_ = torch.cat([x[:, len_keep:, :], mask_tokens_aux], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x_aux = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token


        # add pos embed
        x_ori = x_ori + self.decoder_pos_embed
        x_aux = x_aux + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x_ori = blk(x_ori)
        x_ori = self.decoder_norm(x_ori)

        for blk in self.decoder_blocks:
            x_aux = blk(x_aux)
        x_aux = self.decoder_norm(x_aux)

        # predictor projection
        x_ori = self.decoder_pred(x_ori)
        x_aux = self.decoder_pred(x_aux)
        # remove cls token
        x_ori = x_ori[:, 1:, :]
        x_aux = x_aux[:, 1:, :]

        return x_ori, x_aux
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_unmix(self, imgs1, imgs2, pred1, pred2, mask1, mask2):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target1 = self.patchify(imgs1)
        if self.norm_pix_loss:
            mean = target1.mean(dim=-1, keepdim=True)
            var = target1.var(dim=-1, keepdim=True)
            target1 = (target1 - mean) / (var + 1.e-6)**.5

        target2 = self.patchify(imgs2)
        if self.norm_pix_loss:
            mean = target2.mean(dim=-1, keepdim=True)
            var = target2.var(dim=-1, keepdim=True)
            target2 = (target2 - mean) / (var + 1.e-6)**.5
        loss1 = (pred1 - target1) ** 2
        loss1 = loss1.mean(dim=-1)  # [N, L], mean loss per patch

        loss1 = (loss1 * mask1).sum() / mask1.sum()  # mean loss on removed patches

    
        loss2 = (pred2 - target2) ** 2
        loss2 = loss2.mean(dim=-1)  # [N, L], mean loss per patch

        loss2 = (loss2 * mask2).sum() / mask2.sum()  # mean loss on removed patches
        loss = loss1 + loss2
        return loss

    # def forward(self, imgs, attns, mask_ratio=0.75):
    #     N, C, H, W = imgs.shape
    #     indices = torch.randperm(N)
    #     # 使用随机索引对第一个维度进行打乱
    #     shuffled_imgs = imgs[indices]
    #     # breakpoint()
    #     latent, mask, mask_aux, ids_restore, len_keep = self.forward_encoder_unmix(imgs, shuffled_imgs, attns, mask_ratio)
    #     pred, pred_aux = self.forward_decoder_unmix(latent, ids_restore, len_keep)  # [N, L, p*p*3]
    #     loss = self.forward_loss_unmix(imgs, shuffled_imgs, pred, pred_aux, mask, mask_aux)
    #     return loss, pred, mask

    def forward(self, imgs, de_imgs, attns, mask_ratio=0.75):
        latent, mask, ids_restore= self.forward_encoder(de_imgs, attns, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks