import numpy as np
import matplotlib.pyplot as plt

import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import h5py
import fastmri
from fastmri.data import subsample, transforms, mri_data
from help_func import print_var_detail

def complex_abs_sq(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., 2]
    return: [...]
    """
    return x[..., 0] ** 2 + x[..., 1] ** 2


def sense_combine_torch(coil_imgs: torch.Tensor, maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    coil_imgs: [Nc, H, W, 2]
    maps:      [Nc, H, W, 2]
    return:    [H, W, 2]
    """
    num = fastmri.complex_mul(fastmri.complex_conj(maps), coil_imgs).sum(dim=0)   # [H,W,2]
    den = complex_abs_sq(maps).sum(dim=0).unsqueeze(-1) + eps                      # [H,W,1]
    return num / den
class DataTransform_Diffusion:
    def __init__(
            self,
            mask_func,
            img_size=320,
            combine_coil=True,
            flag_singlecoil=False,
            maps_root=None,
            map_key="s_maps",
            device=None,          # 新增：GPU设备
    ):
        self.mask_func = mask_func
        self.img_size = img_size
        self.combine_coil = combine_coil
        self.flag_singlecoil = flag_singlecoil
        self.maps_root = pathlib.Path(maps_root) if maps_root is not None else None
        self.map_key = map_key
        self.device = device

        if flag_singlecoil:
            self.combine_coil = True

    def _resolve_map_file(self, filename):
        if self.maps_root is None:
            return None

        fname = pathlib.Path(filename).name
        filename_str = str(filename)

        candidates = []

        # 优先按完整路径推断 split
        if "multicoil_train" in filename_str:
            candidates += [
                self.maps_root / "multicoil_train_knee" / "maps" / fname,
                self.maps_root / "multicoil_train" / "maps" / fname,
            ]
        if "multicoil_test" in filename_str:
            candidates += [
                self.maps_root / "multicoil_test" / "maps" / fname,
                self.maps_root / "multicoil_test_knee" / "maps" / fname,
            ]
        if "multicoil_val" in filename_str:
            candidates += [
                self.maps_root / "multicoil_val" / "maps" / fname,
                self.maps_root / "multicoil_val_knee" / "maps" / fname,
            ]

        # 兜底：全都试一遍
        candidates += [
            self.maps_root / "multicoil_train_knee" / "maps" / fname,
            self.maps_root / "multicoil_test" / "maps" / fname,
            self.maps_root / "multicoil_val" / "maps" / fname,
        ]

        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(
            f"Cannot find map file for {filename}. Tried under maps_root={self.maps_root}"
        )

    def _load_maps(self, filename, slice_num):
        """
        读取并处理 sensitivity maps，输出 [Nc, H, W, 2]
        """
        map_file = self._resolve_map_file(filename)
        if map_file is None:
            return None

        with h5py.File(map_file, "r") as hf:
            if self.map_key not in hf:
                raise KeyError(f"{self.map_key} not found in {map_file}")
            maps_np = hf[self.map_key][slice_num]

        maps = transforms.to_tensor(maps_np)  # [Nc,H,W,2] if complex
        if maps.ndim == 3:
            # 防御性处理：如果读出来是实数 [Nc,H,W]，补成复数格式
            maps = torch.stack([maps, torch.zeros_like(maps)], dim=-1)

        maps = transforms.complex_center_crop(maps, [320, 320])

        if self.img_size != 320:
            maps = torch.einsum('nhwc->nchw', maps)
            maps = T.Resize(size=self.img_size)(maps)
            maps = torch.einsum('nchw->nhwc', maps)

        return maps.float()

    def __call__(self, kspace, mask, target, data_attributes, filename, slice_num):
        if self.flag_singlecoil:
            kspace = kspace[None, ...]

        kspace = transforms.to_tensor(kspace)  # [Nc,H,W,2]



        image_full = fastmri.ifft2c(kspace)  # [Nc,H,W,2]
        image_full = transforms.complex_center_crop(image_full, [320, 320])

        if self.img_size != 320:
            image_full = torch.einsum('nhwc->nchw', image_full)
            image_full = T.Resize(size=self.img_size)(image_full)
            image_full = torch.einsum('nchw->nhwc', image_full)

        maps = None
        if (not self.flag_singlecoil) and (self.maps_root is not None):
            maps = self._load_maps(filename, slice_num)  # [Nc,H,W,2]


        mask, mask_fold = self.mask_func()  # mask: [1,H,W], mask_fold: [1,h,w]
        mask = torch.from_numpy(mask).float()
        mask_fold = torch.from_numpy(mask_fold).float()


        # ==========================================================
        # 分支1：map-aware coil combine -> 单张复图
        # ==========================================================
        if self.combine_coil:
            if maps is None:
                raise ValueError("combine_coil=True requires valid sensitivity maps.")

            # 用 maps 做 SENSE combine，得到单张复图 [H,W,2]
            image_comb = sense_combine_torch(image_full, maps)     # [H,W,2]

            # 转回单图 k-space [1,H,W,2]
            kspace_comb = fastmri.fft2c(image_comb.unsqueeze(0))   # [1,H,W,2]

            # 归一化：基于全采样图像的最大幅值
            image_full_abs = fastmri.complex_abs(image_comb)  # [H, W]
            max_val = torch.amax(image_full_abs)  # 标量
            scale_coeff = 1.0 / (max_val + 1e-8)
            kspace_comb = kspace_comb * scale_coeff

            return kspace_comb.float(), mask, mask_fold

        # ==========================================================
        # 分支2：原始 multicoil 版本（通常不会用到）
        # ==========================================================
        kspace = fastmri.fft2c(image_full)  # [Nc,H,W,2]

        mask_expand = mask[..., None].repeat(kspace.shape[0], 1, 1, 1)  # [Nc,H,W,1]
        masked_kspace = kspace * mask_expand

        image_masked = fastmri.ifft2c(masked_kspace)          # [Nc,H,W,2]
        image_masked_abs = fastmri.complex_abs(image_masked)  # [Nc,H,W]
        max_val = torch.amax(image_masked_abs, dim=(1, 2))    # [Nc]
        scale_coeff = 1.0 / (max_val + 1e-8)

        kspace = torch.einsum('nhwc,n->nhwc', kspace, scale_coeff)

        if maps is not None:
            return kspace.float(), mask, mask_fold, maps.float()
        else:
            return kspace.float(), mask, mask_fold

def apply_mask(data, mask_func):
    '''
    data: [Nc,H,W,2]
    mask_func: return [Nc(1),H,W]
    '''
    # mask, _ = mask_func()
    mask_return = mask_func()
    if len(mask_return) == 1:
        mask = mask_func()
    else:
        mask, _ = mask_func()
    mask = torch.from_numpy(mask)
    mask = mask[..., None]  # [Nc(1),H,W,1]
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    return masked_data, mask
