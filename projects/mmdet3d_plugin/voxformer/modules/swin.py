# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial
from mmdet.models import BACKBONES

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .swin_transformer import SwinTransformer
# from .vision_transformer import VisionTransformer

@BACKBONES.register_module()
class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask=None):
        # x = x.unsqueeze(0)
        # B, cam, C, H, W = x.shape  # [2, 5, 3, 370, 1220]
        # ph, pw = 192, 192  # 目标 patch 大小
        # nh = H // ph + (1 if H % ph else 0)  # 370 // 192 + 1 = 2
        # nw = W // pw + (1 if W % pw else 0)  # 1220 // 192 + 1 = 7
        # patches = []

        # for b in range(B):
        #     for c in range(cam):
        #         for i in range(nh):
        #             for j in range(nw):
        #                 h_start = i * ph
        #                 h_end = min(h_start + ph, H)
        #                 w_start = j * pw
        #                 w_end = min(w_start + pw, W)
        #                 patch = x[b, c, :, h_start:h_end, w_start:w_end]
        #                 # 填充到 192×192
        #                 if patch.shape[1] < ph or patch.shape[2] < pw:
        #                     patch = F.pad(patch, (0, pw - patch.shape[2], 0, ph - patch.shape[1]))
        #                 patches.append(patch)

        # x = torch.stack(patches, dim=0)  # [B*cam*nh*nw, C, 192, 192]
        # x [b , cam , c, h ,w]
        x = self.patch_embed(x)

        B, L, _ = x.shape

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return [x]

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}



@BACKBONES.register_module()
class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        encoder_stride = 32
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)

    return model

# MODEL:
#   TYPE: swin
#   NAME: simmim_pretrain
#   DROP_PATH_RATE: 0.0
#   SWIN:
#     EMBED_DIM: 128
#     DEPTHS: [ 2, 2, 18, 2 ]
#     NUM_HEADS: [ 4, 8, 16, 32 ]
#     WINDOW_SIZE: 6
# DATA:
#   IMG_SIZE: 192
#   MASK_PATCH_SIZE: 32
#   MASK_RATIO: 0.6
# TRAIN:
#   EPOCHS: 800
#   WARMUP_EPOCHS: 10
#   BASE_LR: 1e-4
#   WARMUP_LR: 5e-7
#   WEIGHT_DECAY: 0.05
#   LR_SCHEDULER:
#     NAME: 'multistep'
#     GAMMA: 0.1
#     MULTISTEPS: [700,]
# PRINT_FREQ: 100
# SAVE_FREQ: 5
# TAG: simmim_pretrain__swin_base__img192_window6__800ep