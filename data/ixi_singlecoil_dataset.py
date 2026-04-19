from pathlib import Path
from typing import List
import torch
import nibabel as nib
import numpy as np
import fastmri
from torch.utils.data import Dataset
import torch.nn.functional as F


def center_crop_or_pad_2d(img: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    h, w = img.shape

    if h > out_h:
        top = (h - out_h) // 2
        img = img[top:top + out_h, :]
    elif h < out_h:
        pad_top = (out_h - h) // 2
        pad_bottom = out_h - h - pad_top
        img = F.pad(img, (0, 0, pad_top, pad_bottom))

    h, w = img.shape
    if w > out_w:
        left = (w - out_w) // 2
        img = img[:, left:left + out_w]
    elif w < out_w:
        pad_left = (out_w - w) // 2
        pad_right = out_w - w - pad_left
        img = F.pad(img, (pad_left, pad_right, 0, 0))

    return img


def list_ixi_files(root: str) -> List[Path]:
    root = Path(root)
    files = sorted(root.glob("*.nii.gz"))
    if len(files) == 0:
        raise ValueError(f"No .nii.gz files found in {root}")
    return files


class IXISinglecoilSliceDataset(Dataset):
    """
    IXI single-coil:
    nii.gz volume -> 2D slice -> single image -> FFT -> full k-space

    return:
        kspace:    [1, H, W, 2]
        mask:      [1, H, W]
        mask_fold: [1, h, w]
    """
    def __init__(
        self,
        root: str,
        mask_func,
        image_size: int = 256,
        num_skip_slice: int = 20,
        normalize_mode: str = "percentile",
        min_nonzero_fraction: float = 0.02,
    ):
        self.file_list = list_ixi_files(root)
        self.mask_func = mask_func
        self.image_size = int(image_size)
        self.num_skip_slice = int(num_skip_slice)
        self.normalize_mode = normalize_mode
        self.min_nonzero_fraction = float(min_nonzero_fraction)

        self.examples = []
        for f in self.file_list:
            nii = nib.load(str(f))
            shape = nii.shape
            if len(shape) < 3:
                continue

            num_slices = shape[2]
            start = self.num_skip_slice
            end = num_slices - self.num_skip_slice
            if end <= start:
                start = 0
                end = num_slices

            for s in range(start, end):
                self.examples.append((f, s))

    def __len__(self):
        return len(self.examples)

    def _normalize_slice(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize_mode == "minmax":
            mask = x > 0
            if mask.any():
                vals = x[mask]
                lo = vals.min()
                hi = vals.max()
                x = (x - lo) / (hi - lo + 1e-8)
            else:
                x = torch.zeros_like(x)

        elif self.normalize_mode == "percentile":
            mask = x > 0
            if mask.any():
                vals = x[mask]
                lo = torch.quantile(vals, 0.01)
                hi = torch.quantile(vals, 0.99)
                x = torch.clamp(x, lo, hi)
                x = (x - lo) / (hi - lo + 1e-8)
            else:
                x = torch.zeros_like(x)

        elif self.normalize_mode == "max":
            x = x / (x.abs().max() + 1e-8)

        elif self.normalize_mode == "img_std":
            # 新增：图像域 L2 归一化（保持零均值？MRI 图像非负，不强制减均值）
            # 计算幅值的均方根，并将图像缩放到 RMS=1
            rms = torch.sqrt(torch.mean(x ** 2) + 1e-8)
            x = x / rms

        else:
            raise ValueError(f"Unsupported normalize_mode: {self.normalize_mode}")

        return x

    def __getitem__(self, idx: int):
        fname, slice_idx = self.examples[idx]

        nii = nib.load(str(fname))
        slc = np.asarray(nii.dataobj[:, :, slice_idx], dtype=np.float32)
        img = torch.from_numpy(slc).float()

        img = center_crop_or_pad_2d(img, self.image_size, self.image_size)
        img = self._normalize_slice(img)

        nonzero_fraction = (img > 0).float().mean().item()
        if nonzero_fraction < self.min_nonzero_fraction:
            img = torch.zeros_like(img)

        # real image -> complex image [1,H,W,2]
        img_complex = torch.stack([img, torch.zeros_like(img)], dim=-1).unsqueeze(0)

        # full k-space [1,H,W,2]
        kspace = fastmri.fft2c(img_complex)

        mask, mask_fold = self.mask_func()
        mask = torch.from_numpy(mask).float()
        mask_fold = torch.from_numpy(mask_fold).float()

        return kspace.float(), mask, mask_fold