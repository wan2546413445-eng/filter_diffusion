import os
import torch
import h5py
import numpy as np
import pickle
from torch.utils.data import Dataset
from utils.utils import (
    get_all_files,
    fft2c_2d,
    ifft2c_2d,
    Emat_xyt_complex,
    normalize_complex,
    normalize_l2,
)
from .sample_mask import RandomMaskGaussianDiffusion, RandomMaskDiffusion, EquiSpaceMaskDiffusion

def crop_tensor(x, cropx, cropy):
    """
    x: torch tensor of shape (B, C, H, W) or (B, C, H, W, 2)
    """
    _, _, H, W = x.shape
    startx = W // 2 - cropx // 2
    starty = H // 2 - cropy // 2
    return x[:, :, starty:starty+cropy, startx:startx+cropx]

class FastMRIKneeDataSet(Dataset):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.mode = mode

        # 路径配置（请根据实际修改）
        if mode in ["train", "training"]:
            self.kspace_dir = os.path.join(config.data.data_root, 'multicoil_train_knee/kspace')
            self.maps_dir = os.path.join(config.data.maps_root, 'multicoil_train_knee/maps')
        elif mode in ["test", "sample"]:
            self.kspace_dir = os.path.join(config.data.data_root, 'multicoil_test/kspace')
            self.maps_dir = os.path.join(config.data.maps_root, 'multicoil_test/maps')
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.file_list = get_all_files(self.kspace_dir)
        # 读取切片数量（假设有 pkl 文件记录）
        with open(config.data.slice_info_pkl, 'rb') as f:
            slice_dict = pickle.load(f)
        self.num_slices = np.array([max(slice_dict[os.path.basename(f)] - 6, 1) for f in self.file_list])
        self.slice_mapper = np.cumsum(self.num_slices) - 1

        # 初始化掩码生成器
        size = (1, config.data.image_size, config.data.image_size)
        mask_type = config.data.mask_type
        if mask_type == 'gaussian_diffusion':
            self.mask_gen = RandomMaskGaussianDiffusion(
                acceleration=config.data.R,
                center_fraction=config.data.center_fraction,
                size=size,
                seed=config.data.seed,
                patch_size=config.data.patch_size
            )
        elif mask_type == 'random_diffusion':
            self.mask_gen = RandomMaskDiffusion(
                center_fraction=config.data.center_fraction,
                acceleration=config.data.R,
                size=size,
                seed=config.data.seed
            )
        elif mask_type == 'equispace_diffusion':
            self.mask_gen = EquiSpaceMaskDiffusion(
                center_fraction=config.data.center_fraction,
                acceleration=config.data.R,
                size=size,
                seed=config.data.seed
            )
        else:
            raise ValueError(f"Unsupported mask_type: {mask_type}")

    def __len__(self):
        return int(np.sum(self.num_slices))

    def __getitem__(self, idx):
        # 定位文件与切片索引
        scan_idx = np.where((self.slice_mapper - idx) >= 0)[0][0]
        if scan_idx == 0:
            slice_idx = idx
        else:
            slice_idx = idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1

        # 加载 maps（线圈敏感度图）
        maps_file = os.path.join(self.maps_dir, os.path.basename(self.file_list[scan_idx]))
        with h5py.File(maps_file, 'r') as f:
            maps = f['s_maps'][slice_idx]                     # (Nc, H, W) 复数？实际是实数
            maps = torch.from_numpy(maps).float()             # (Nc, H, W)
            maps = maps.unsqueeze(0)                          # (1, Nc, H, W)
            maps = crop_tensor(maps, 320, 320)                # crop 到 320x320
            maps = maps.squeeze(0)                            # (Nc, H, W)

        # 加载 kspace
        raw_file = os.path.join(self.kspace_dir, os.path.basename(self.file_list[scan_idx]))
        with h5py.File(raw_file, 'r') as f:
            kspace = f['kspace'][slice_idx]                   # (Nc, H, W) 复数（h5py 存储为复数）
            # 转为 torch 复数张量
            kspace = torch.from_numpy(kspace).cfloat()        # (Nc, H, W) complex64
            kspace = kspace.unsqueeze(0)                      # (1, Nc, H, W)

            # 先转换到图像域，裁剪，再回到 k 空间
            img = ifft2c_2d(kspace)                           # (1, Nc, H, W) 复数
            img = crop_tensor(img, 320, 320)                  # (1, Nc, H, W)
            kspace = fft2c_2d(img)                            # (1, Nc, H, W) 复数

            # 归一化（根据配置）
            if self.config.data.normalize_type == 'minmax':
                # 将复数张量转换为 (1, Nc, H, W, 2) 格式
                kspace_2c = torch.stack([kspace.real, kspace.imag], dim=-1)  # (1, Nc, H, W, 2)
                img_2c = Emat_xyt_complex(kspace_2c, True, maps, 1)          # (1, 1, H, W, 2)
                img_2c = self.config.data.normalize_coeff * normalize_complex(img_2c)
                kspace_2c = Emat_xyt_complex(img_2c, False, maps, 1)         # (1, Nc, H, W, 2)
                # 转回复数张量
                kspace = torch.complex(kspace_2c[...,0], kspace_2c[...,1])   # (1, Nc, H, W)
            elif self.config.data.normalize_type == 'std':
                std_val = torch.std(torch.abs(kspace))
                kspace = kspace / (self.config.data.normalize_coeff * std_val)
            elif self.config.data.normalize_type == 'img_std':
                img = ifft2c_2d(kspace)                       # (1, Nc, H, W) 复数
                img_abs = torch.abs(img)                      # (1, Nc, H, W)
                img_abs = normalize_l2(img_abs)               # 需要 torch 版本的 normalize_l2
                img = img * (img_abs / torch.abs(img))        # 保持相位
                kspace = fft2c_2d(img)

            # 最终 kspace 为复数张量 (1, Nc, H, W)
            kspace = kspace.squeeze(0)                        # (Nc, H, W)

        # 生成掩码（numpy 数组）
        mask, mask_fold = self.mask_gen()                     # mask: (1, H, W), mask_fold: (1, H//ps, W//ps)

        # 将 kspace 转换为 (Nc, H, W, 2) 实部虚部分离的形式
        kspace_2c = torch.stack([kspace.real, kspace.imag], dim=-1)  # (Nc, H, W, 2)

        # 转为 Tensor
        mask = torch.from_numpy(mask).float()
        mask_fold = torch.from_numpy(mask_fold).float()

        return kspace_2c, mask, mask_fold