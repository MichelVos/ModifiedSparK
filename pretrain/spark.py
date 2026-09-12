# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pprint import pformat
from typing import List

import sys
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import encoder
from decoder import LightDecoder
import matplotlib
matplotlib.use("QtAgg")  # "TkAgg" or "QtAgg",
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import torchvision
import torch.nn.functional as F
from utils import arg_util, misc, lamb
import torch
import math


class SparK(nn.Module):
    def __init__(
            self, sparse_encoder: encoder.SparseEncoder, dense_decoder: LightDecoder,
            mask_ratio=0.6, mask_type='random', mask_area='full', densify_norm='bn', sbn=False, exportDirectory=None, mean=None, std=None, device='cpu'
    ):
        super().__init__()
        input_size, downsample_raito = sparse_encoder.input_size, sparse_encoder.downsample_raito
        # self.downsample_raito = downsample_raito
        input_size_h, input_size_w = input_size
        self.downsample_ratio_h, self.downsample_ratio_w = downsample_raito
        self.fmap_h, self.fmap_w = input_size_h // self.downsample_ratio_h, input_size_w // self.downsample_ratio_w
        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_h * self.fmap_w * (1 - mask_ratio))
        self.exportDirectory = exportDirectory
        
        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder

        self._vis_dumped = False
        self.picture_number = 0
        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.mask_type = mask_type
        self.mask_area = mask_area
        
        # build the `densify` layers
        e_widths, d_width = self.sparse_encoder.enc_feat_map_chs, self.dense_decoder.width
        e_widths: List[int]
        for i in range(self.hierarchy): # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
            e_width = e_widths.pop()
            # create mask token
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)
            
            # create densify norm
            if self.densify_norm_str == 'bn':
                densify_norm = (encoder.SparseSyncBatchNorm2d if self.sbn else encoder.SparseBatchNorm2d)(e_width)
            elif self.densify_norm_str == 'ln':
                densify_norm = encoder.SparseConvNeXtLayerNorm(e_width, data_format='channels_first', sparse=True)
            else:
                densify_norm = nn.Identity()
            self.densify_norms.append(densify_norm)
            
            # create densify proj
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()    # todo: NOTE THAT CONVNEXT-S WOULD USE THIS, because it has a width of 768 that equals to the decoder's width 768
                print(f'[SparK.__init__, densify {i+1}/{self.hierarchy}]: use nn.Identity() as densify_proj')
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv2d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
                print(f'[SparK.__init__, densify {i+1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)')
            self.densify_projs.append(densify_proj)
            
            # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
            d_width //= 2
        
        print(f'[SparK.__init__] dims of mask_tokens={tuple(p.numel() for p in self.mask_tokens)}')
        
        # these are deprecated and would never be used; can be removed.
        self.register_buffer('imn_m', torch.empty(1, 3, 1, 1))
        self.register_buffer('imn_s', torch.empty(1, 3, 1, 1))

        self.register_buffer('norm_black', torch.zeros(1, 3, input_size_h, input_size_w))
        self.vis_active = self.vis_active_ex = self.vis_inp = self.vis_inp_mask = ...
    
    '''
    def mask(self, B: int, device, inp_bchw, generator=None):
        h, w = self.fmap_h, self.fmap_w
        idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(B, h * w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w)
    '''

    def valid_patches(self, inp_bchw):
        h, w = self.fmap_h, self.fmap_w
        x = inp_bchw * self.std + self.mean
        # Convert to grayscale if needed
        gray = x.mean(dim=1, keepdim=True)   # (B,H,W)
        # compute patch size
        ph = gray.shape[2] // h
        pw = gray.shape[3] // w

        # reshape into patches
        patches = gray.view(gray.shape[0], 1, h, ph, w, pw)

        # compute minimum per patch
        patch_min = patches.amin(dim=(3,5)).squeeze(1)  # (B,h,w)
        return patch_min
    
    def mask(self, B, device, inp_bchw):

        if self.mask_type == "random":
            return self.mask_random(B, device, inp_bchw)

        elif self.mask_type == "block":
            return self.mask_block(B, device, self.mask_ratio)

        elif self.mask_type == "hline": 
            return self.mask_lines(B, device)

        elif self.mask_type == "vline":
            return self.mask_vertical_lines(B, device)

        elif self.mask_type == "grid":
            return self.mask_grid(B, device)
        
        elif self.mask_type == "diagonal":
            return self.mask_diagonal(B, device)
    
    def mask_random(self, B, device, inp_bchw):
        #if self.mask_area == 'full':
        #    threshold = 2.0
        #else:
        #    threshold = 0.5 # what is the threshold with the READ dataset. That might be completely different. RIMES will be the same.
        bright_threshold = 0.75
        bright_fraction  = 0.5
        dark_threshold = 0.5            
        h, w = self.fmap_h, self.fmap_w
        x = inp_bchw * self.std + self.mean
        # Convert to grayscale if needed
        gray = x.mean(dim=1, keepdim=True)   # (B,H,W)
        # compute patch size
        ph = gray.shape[2] // h
        pw = gray.shape[3] // w

        # reshape into patches
        patches = gray.view(B, 1, h, ph, w, pw)

        bright_pixels = patches > bright_threshold
        dark_pixels = patches < dark_threshold

        # count bright pixels per patch
        bright_count = bright_pixels.sum(dim=(3,5))
        dark_count = dark_pixels.sum(dim=(3,5))

        # number of pixels per patch
        patch_area = ph * pw

        # fraction of bright pixels
        bright_fraction = bright_count.float() / patch_area
        dark_fraction = dark_count.float() / patch_area

        # valid patch if >= 50% bright pixels
        text_mask = ((bright_fraction >= 0.5) & (dark_fraction > 0.02)).squeeze(1)

        active = torch.ones(B, h*w, dtype=torch.bool, device=device)
        
        if self.mask_area == 'area':
            # determine the area that contains text, and only perform random masking within that area; the rest area would be fully visible (not masked)
            block = 2   # size of large patch
            mask = text_mask.float().unsqueeze(1)   # (B,1,h,w)

            large_mask = F.max_pool2d(mask, kernel_size=block, stride=block)

            large_mask = F.interpolate(
                large_mask,
                size=(h, w),
                mode="nearest"
            )

            # convert back to bool
            text_mask = large_mask.squeeze(1) > 0
            for b in range(B):
                valid_idx = torch.where(text_mask[b].flatten())[0]
                if len(valid_idx) == 0:
                    continue
                len_keep = int((self.mask_ratio) * len(valid_idx))
                perm = valid_idx[torch.randperm(len(valid_idx))]
                keep = perm[:min(len_keep, len(valid_idx))]
                active[b, keep] = False               
        elif self.mask_area == 'patches':
            for b in range(B):
                valid_idx = torch.where(text_mask[b].flatten())[0]
                if len(valid_idx) == 0:
                    continue
                len_keep = int((self.mask_ratio) * len(valid_idx))
                perm = valid_idx[torch.randperm(len(valid_idx))]
                keep = perm[:min(len_keep, len(valid_idx))]
                active[b, keep] = False
        else:   # full
            len_keep = int(self.mask_ratio * h * w)
            for b in range(B):
                perm = torch.randperm(h * w, device=device)
                keep = perm[:len_keep]
                active[b, keep] = False
        return active.view(B, 1, h, w)


    def mask_block(self, B, device, mask_ratio):
        h, w = self.fmap_h, self.fmap_w

        bh = 1   # block height
        bw = 4   # block width

        # --- effective region (divisible by block size) ---
        h_eff = (h // bh) * bh
        w_eff = (w // bw) * bw

        Gh = h_eff // bh
        Gw = w_eff // bw

        N_blocks = Gh * Gw
        N_mask = int((1 - mask_ratio) * N_blocks)

        # --- Step 1: sample blocks ---
        block_mask = torch.zeros(B, N_blocks, dtype=torch.bool, device=device)

        for b in range(B):
            idx = torch.randperm(N_blocks, device=device)[:N_mask]
            block_mask[b, idx] = True

        # --- Step 2: reshape to grid ---
        block_mask = block_mask.view(B, Gh, Gw)

        # --- Step 3: upscale to full resolution ---
        generated_mask = block_mask.repeat_interleave(bh, dim=1)\
                                .repeat_interleave(bw, dim=2)

        # --- Step 4: place into full mask ---
        mask = torch.zeros(B, h, w, dtype=torch.bool, device=device)
        mask[:, :h_eff, :w_eff] = generated_mask

        return mask.unsqueeze(1)

    def mask_lines(self, B, device):

        h, w = self.fmap_h, self.fmap_w
        mask = torch.zeros(B, h, w, dtype=torch.bool, device=device)

        num_keep = self.len_keep // w

        for b in range(B):
            rows = torch.randperm(h)[:num_keep]
            mask[b, rows, :] = True

        return mask.view(B,1,h,w)

    def mask_vertical_lines(self, B, device):

        h, w = self.fmap_h, self.fmap_w
        mask = torch.zeros(B, h, w, dtype=torch.bool, device=device)
        len_keep = int((self.mask_ratio) * w)
        num_keep = w - len_keep

        for b in range(B):
            column = torch.randperm(w)[:num_keep]
            mask[b, :, column] = True

        return mask.view(B,1,h,w)

    def mask_grid(self, B, device):
        h, w = self.fmap_h, self.fmap_w
        mask = torch.zeros(B, h, w, dtype=torch.bool, device=device)
        num_keep_cols = int(w * (1 - self.mask_ratio))

        for b in range(B):
            cols = torch.randperm(w)[:num_keep_cols]
            mask[b, :, cols] = True

        return mask.view(B,1,h,w)  

    def mask_diagonal(self, B, device):
        h, w = self.fmap_h, self.fmap_w
        mask = torch.zeros(B, h, w, dtype=torch.bool, device=device)
        num_keep_cols = int(w * (1 - self.mask_ratio))

        for b in range(B):
            cols = torch.randperm(w)[:num_keep_cols]
            for index in range(len(cols)):
                x = cols[index]
                for y in range(h):
                    if x < w:
                        mask[b, y, x] = True
                    x = x + 1

        return mask.view(B,1,h,w)

    def debug_image(self, x, name="img"):
        import torch
        import numpy as np

        print(f"\n{name}")
        print("type:", type(x))

        if isinstance(x, torch.Tensor):
            print("dtype:", x.dtype)
            print("shape:", tuple(x.shape))
            print("min:", x.min().item())
            print("max:", x.max().item())
            print("mean:", x.mean().item())
            print("std:", x.std().item())

        elif isinstance(x, np.ndarray):
            print("dtype:", x.dtype)
            print("shape:", x.shape)
            print("min:", x.min())
            print("max:", x.max())
            print("mean:", x.mean())
            print("std:", x.std())

    def forward(self, inp_bchw: torch.Tensor, active_b1ff=None, vis=False):
        # print(">>> ENTER SparK.forward", inp_bchw.shape)
        #assert inp_bchw.is_cuda, f"input on CPU: {inp_bchw.device}"
        #for p in self.parameters():
        #    assert p.is_cuda, "a model parameter is on CPU"
        # step1. Mask
        if active_b1ff is None:     # rand mask
            active_b1ff: torch.BoolTensor = self.mask(inp_bchw.shape[0], inp_bchw.device, inp_bchw)  # (B, 1, f, f)
        encoder._cur_active = active_b1ff    # (B, 1, f, f)
        active_b1hw = active_b1ff.repeat_interleave(self.downsample_ratio_h, 2).repeat_interleave(self.downsample_ratio_w, 3)  # (B, 1, H, W)
        masked_bchw = inp_bchw * active_b1hw
        B, C, H, W = inp_bchw.shape
        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchw)
        fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest
        #self.base_feat = next(f for f in fea_bcffs if f is not None)
        # step3. Densify: get hierarchical dense features for decoding
        cur_active = active_b1ff     # (B, 1, f, f)
        to_dec = []

        #factors = [(2,2), (2,2), (2,2), (2,2), (2,2)]  # the upsampling factors for dilating the mask; len(factors) should equal to self.hierarchy
        factors = [(4,1), (2,2), (2,2), (2,2)]  # the upsampling factors for dilating the mask; len(factors) should equal to self.hierarchy

        for i, bcff in enumerate(fea_bcffs):  # from the smallest feature map to the largest
            if bcff is not None:
                bcff = self.densify_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff)
                #print(f"Level {i}")
                #print("bcff:", bcff.shape if bcff is not None else None)
                #print("cur_active:", cur_active.shape)                
                #print(f"bcff.shape: {bcff.shape}, mask_tokens.shape: {mask_tokens.shape} curr_active.shape={cur_active.shape}")  # debug
                #print("bcff:", bcff.shape)
                #print("cur_active:", cur_active.shape)
                #print("mask_tokens:", mask_tokens.shape)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)   # fill in empty (non-active) positions with [mask] tokens
                bcff: torch.Tensor = self.densify_projs[i](bcff)
                
            to_dec.append(bcff)
            factor_h, factor_w = factors[i]
            cur_active = cur_active.repeat_interleave(factor_h, dim=2).repeat_interleave(factor_w, dim=3)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)
        
        # step4. Decode and reconstruct
        rec_bchw = self.dense_decoder(to_dec)

        #x = torch.randn(1, 3, H, W)
        #recon = self.unpatchify(self.patchify(x))
        #print((x - recon).abs().max())

        inp, rec = self.patchify(inp_bchw), self.patchify(rec_bchw)   # inp and rec: (B, L = f*f, N = C*downsample_raito**2)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
        inp = (inp - mean) / var
        #self.debug_image(inp[0].cpu(), name="inp_patchified")
        #self.debug_image(rec[0].cpu(), name="rec_patchified")


        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)    # (B, L, C) ==mean==> (B, L)
        #l1_loss = (rec - inp).abs().mean(dim=2)
        non_active = active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1)  # (B, 1, f, f) => (B, L)
        recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # loss only on masked (non-active) patches
        #recon_loss = l1_loss.mul(non_active).sum() / (non_active.sum() + 1e-8)  # loss only on masked (non-active) patches
        if vis:
            masked_bchw = inp_bchw * active_b1hw
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_b1hw, inp_bchw, rec_bchw)
            if not self._vis_dumped and self.exportDirectory is not None:
                self._vis_dumped = True  # dump only once


                # visualize first sample only
                x = inp_bchw[0].detach().cpu()
                m = active_b1hw[0].detach().cpu()
                xm = masked_bchw[0].detach().cpu()
                xr = rec_or_inp[0].detach().cpu()

                #print('xm', xm.min().item(), xm.max().item(), xm.mean().item())

                # undo mean/std normalization for visualization; note that this is not the same as the original image, but should be good enough for sanity check and visualization of the masking pattern; you can also directly visualize the rec_bchw or inp_bchw without unpatchify, but I find unpatchify gives better visualization effect.
                x = (x * self.std.detach().cpu() + self.mean.detach().cpu()) 
                xr = (xr * self.std.detach().cpu() + self.mean.detach().cpu())  
                xm = (xm - xm.min()) / (xm.max() - xm.min())

                
                # overlay mask for clarity
                overlay = x.clone()
                overlay[m.expand_as(x) == 0] = 0.5

                #for name, t in {
                #    "x": x,
                #    "xm": xm,
                #    "overlay": overlay,
                #    "xr": xr
                #}.items():
                #    print(name, t.min().item(), t.max().item(), t.mean().item())

                #self.debug_image(x, name="x")
                #self.debug_image(xm, name="xm")
                #self.debug_image(overlay, name="overlay")
                #self.debug_image(xr, name="xr")
                torchvision.utils.save_image(
                    [x, xm, overlay, xr],
                    f"{self.exportDirectory}/mask_debug_{self.picture_number}.png",
                    nrow=4,
                    #normalize=True
                )
                self.picture_number += 1
            #return inp_bchw, masked_bchw, rec_or_inp
            return recon_loss
        else:
            return recon_loss
    def patchify(self, bchw):
        """
        bchw: (B, C, H, W)
        h, w: base grid size (feature map size you patch against)

        Returns:
            (B, h*w, C * ph * pw)
        """
        h, w = self.fmap_h, self.fmap_w
        B, C, H, W = bchw.shape

        assert H % h == 0 and W % w == 0, "Input not divisible by grid"

        ph = H // h
        pw = W // w
        assert ph == self.downsample_ratio_h and pw == self.downsample_ratio_w, "Patch size does not match downsample ratio"
        # (B, C, h, ph, w, pw)
        x = bchw.view(B, C, h, ph, w, pw)

        # (B, h, w, ph, pw, C)
        x = x.permute(0, 2, 4, 3, 5, 1)

        # (B, h*w, ph*pw*C)
        patches = x.reshape(B, h * w, ph * pw * C)

        return patches   

    def unpatchify(self, bln):
        """
        bln: (B, h*w, C * ph * pw)

        Returns:
            (B, C, H, W)
        """
        ph=self.downsample_ratio_h
        pw=self.downsample_ratio_w
        h, w = self.fmap_h, self.fmap_w  
        B, C = bln.shape[0], bln.shape[-1] // (ph * pw)      
        assert bln.shape[-1] == C * ph * pw, "Input feature dimension does not match expected patch size"
        #B = bln.shape[0]

        # (B, h, w, ph, pw, C)
        x = bln.view(B, h, w, ph, pw, C)

        # (B, C, h, ph, w, pw)
        x = x.permute(0, 5, 1, 3, 2, 4)

        # (B, C, H, W)
        bchw = x.reshape(B, C, h * ph, w * pw)

        return bchw     

    def __repr__(self):
        return (
            f'\n'
            f'[SparK.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[SparK.structure]: {super(SparK, self).__repr__().replace(SparK.__name__, "")}'
        )
    
    def get_config(self):
        return {
            # self
            'mask_ratio': self.mask_ratio,
            'densify_norm_str': self.densify_norm_str,
            'sbn': self.sbn, 'hierarchy': self.hierarchy,
            
            # enc
            'sparse_encoder.input_size': self.sparse_encoder.input_size,
            # dec
            'dense_decoder.width': self.dense_decoder.width,
        }
    
    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(SparK, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop('config', None)
        incompatible_keys = super(SparK, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys

@torch.no_grad()
def show_triplet(model, img_bchw: torch.Tensor, active_b1ff=None, title_suffix=""):
    # Ensure batch of 1 for easy viewing
    assert img_bchw.ndim == 4 and img_bchw.shape[0] == 1, "Use a single image: shape (1, C, H, W)"
    model.eval()
    device = next(model.parameters()).device
    img_bchw = img_bchw.to(device)

    # Ask model for visualization tensors
    inp_bchw, masked_bchw, rec_or_inp = model(img_bchw, active_b1ff=active_b1ff, vis=True)

    def to_hwC(x):
        # x: (1, C, H, W) -> (H, W, C), clamp to [0,1] for viewing
        x = x[0].detach()
        # If your data is in [0,255], normalize for display:
        if x.max() > 1.0:
            x = x / 255.0
        return x.permute(1, 2, 0).cpu().clamp(0, 1)

    orig = to_hwC(inp_bchw)
    masked = to_hwC(masked_bchw)
    patched = to_hwC(rec_or_inp)

    # Plot
    plt.figure(figsize=(12, 4))
    for i, (im, title) in enumerate([
        (orig,   "Original"),
        (masked, "Masked (active regions kept)"),
        (patched,"Patched (masked regions reconstructed)"),
    ]):
        plt.subplot(1, 3, i+1)
        # If image is single-channel, drop the last dim for imshow
        if im.shape[-1] == 1:
            plt.imshow(im[..., 0], cmap="gray")
        else:
            plt.imshow(im)
        plt.axis("off")
        plt.title(f"{title} {title_suffix}".strip())
    plt.tight_layout()
    plt.show()

# Example usage:
# img_bchw = ...  # (1, C, H, W) tensor
# Optional: provide your own boolean mask at feature level: active_b1ff of shape (1,1,f,f)
# active_b1ff = ...  # torch.bool

def load_image_as_tensor(path: str, resize=None, to_gray=False):
    # Load image with PIL
    img = Image.open(path).convert("RGB")  # ensures 3 channels
    
    # Optional: convert to grayscale if your model expects 1 channel
    if to_gray:
        img = img.convert("L")
    
    # Build transform
    transforms = []
    if resize is not None:  # e.g., resize=(128,128)
        transforms.append(T.Resize(resize))
    transforms.append(T.ToTensor())  # converts to [0,1], shape (C, H, W)
    
    transform = T.Compose(transforms)
    tensor = transform(img)  # (C, H, W)
    
    # Add batch dimension: (1, C, H, W)
    return tensor.unsqueeze(0)

