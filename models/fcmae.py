# Copyright (c) Meta Platforms, Inc. and affiliates.
from argparse import Namespace
from typing import Tuple, Dict, AnyStr

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from torch import Tensor

from .convnextv2 import Block, ConvNeXtV2

from utils.norm_utils import LayerNorm


# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class FCMAE(nn.Module):
    """Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone"""

    def __init__(
        self,
        img_size: int = 112,
        in_chans: int = 3,
        depths: list[int] = None,
        dims: list[int] = None,
        decoder_depth: int = 1,
        decoder_embed_dim: int = 512,
        patch_size: float = 16,
        mask_ratio: float = 0.6,
        norm_pix_loss: bool = False,
        args: Namespace = None,
        loss_fn=None,
        sparse: bool = False,
        use_orig_stem: bool = False
    ):
        super().__init__()

        # configs
        self.args = args
        self.img_size = img_size
        if depths is None:  # set default value
            depths = [3, 3, 9, 3]
        self.depths = depths
        if dims is None:
            dims = [96, 192, 384, 768]
        self.dims = dims
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        self.loss_fn = loss_fn
        self.sparse = sparse
        self.in_chans = in_chans
        
        # encoder
        if sparse:
            pass
            #we discarded the sparse version of the implementation for simplicity.
            #also using the non-sparse version yielded better results.
        else:
            self.encoder = ConvNeXtV2(
                in_chans=self.in_chans,
                depths=depths,
                dims=dims,
                patch_size=patch_size,
                img_size=img_size,
                use_orig_stem=use_orig_stem,
            )
        self.proj = nn.Conv2d(
            in_channels=dims[-1], 
            out_channels=decoder_embed_dim, 
            kernel_size=1
        )

        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [
            Block(dim=decoder_embed_dim, drop_path=0.0) for _ in range(decoder_depth)
        ]

        self.decoder = nn.Sequential(*decoder)
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size**2 * self.in_chans,
            kernel_size=1,
        )


        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, imgs: Tensor) -> Tensor:
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
        return x

    def unpatchify(self, x: Tensor) -> Tensor:
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        print("shape of x:", x.shape)
        h = w = self.img_size // p
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def gen_random_mask(self, x: Tensor, mask_ratio: float) -> Tensor:
        N = x.shape[0]  # number of samples
        L = (x.shape[2] // self.patch_size) ** 2  # number of patches
        len_keep = int(L * (1 - mask_ratio))  # number of patches to keep

        # the following lines generate a mask with 0s and 1s at random locations
        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask  # (batch_size, no_patches**2)

    def upsample_mask(self, mask: Tensor, scale: float):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** 0.5)
        return (
            mask.reshape(-1, p, p)
            .repeat_interleave(scale, dim=1)
            .repeat_interleave(scale, dim=2)
        )

    def forward_encoder(self, imgs: Tensor, mask_ratio: float) -> Tuple[Tensor, Tensor]:
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask

    def forward_decoder(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.proj(x)
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1.0 - mask) + mask_token * mask

        x = self.decoder(x)
        
        pred = self.pred(x)
        return pred

    def forward_loss(
        self, imgs: Tensor, pred: Tensor, mask: Tensor
    ) -> Tensor:    

        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape  # [N, C, H, W]
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum("ncl->nlc", pred)
        target = self.patchify(imgs)

        if self.norm_pix_loss:  # we only compute the per-patch norm on sentinel2
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        
        loss = (pred - target) ** 2  # using mean squared error
        nan_mask = torch.isnan(loss)
        count = torch.count_nonzero(~nan_mask, dim=-1)
        loss[nan_mask] = 0
        loss = loss.sum(dim=-1) / count
        
        # uncomment the below line to compute the loss on the whole image - this results in better reconstructions, but
        # not better representations for downstream tasks
        # mask = torch.ones_like(mask)

        # counting the number of pixels where mask is 1 and loss is not nan. since we only compute the loss on these.
        # we create the nan mask again, since sometimes count can be 0.
        nan_mask = torch.isnan(loss * mask)
        tmp = loss * mask
        tmp[nan_mask] = 0
        sum_ = tmp.sum()
        
        count = torch.count_nonzero(tmp)
        loss = sum_ / count  # mean loss on removed patches
        
        return loss

    def forward(
        self, imgs: Tensor, labels=None, mask_ratio: float = 0.6
    ):

        x, mask = self.forward_encoder(imgs.clone(), mask_ratio) #included clone() to test

        pred = self.forward_decoder(x, mask)
        
        loss = self.forward_loss(
            imgs, pred, mask
        )
        
        return loss, pred, mask


def convnextv2_atto(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnextv2_pico(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = FCMAE(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = FCMAE(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_large(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model