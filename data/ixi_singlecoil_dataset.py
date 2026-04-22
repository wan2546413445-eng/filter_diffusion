from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
import inspect

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

    return img.contiguous()



def list_ixi_files(root: str) -> List[Path]:
    root = Path(root)
    files = sorted(root.glob("*.nii.gz"))
    if len(files) == 0:
        raise ValueError(f"No .nii.gz files found in {root}")
    return files


class IXISinglecoilSliceDataset(Dataset):
    """
    IXI single-coil dataset:
        nii.gz volume -> 2D slice -> single image -> FFT -> full k-space

    Default return (compatible with many existing training loops):
        kspace:    [1, H, W, 2]
        mask:      [1, H, W]
        mask_fold: [1, h, w] or [1, H, W]

    Optional dict return (return_dict=True):
        {
            "kspace": [1, H, W, 2],
            "mask": [1, H, W],
            "mask_fold": [1, h, w] or [1, H, W],
            "target": [1, H, W],
            "fname": str,
            "slice_idx": int,
        }
    """

    def __init__(
        self,
        root: str,
        mask_func,
        image_size: int = 256,
        num_skip_slice: int = 20,
        normalize_mode: str = "percentile",
        min_nonzero_fraction: float = 0.02,
        filter_blank_slices: bool = True,
        return_dict: bool = False,
    ):
        self.file_list = list_ixi_files(root)
        self.mask_func = mask_func
        self.image_size = int(image_size)
        self.num_skip_slice = int(num_skip_slice)
        self.normalize_mode = normalize_mode
        self.min_nonzero_fraction = float(min_nonzero_fraction)
        self.filter_blank_slices = bool(filter_blank_slices)
        self.return_dict = bool(return_dict)

        self.examples: List[Tuple[Path, int]] = []
        self._vol_cache: Dict[str, np.ndarray] = {}

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

            # 只在初始化阶段读一次 volume，并提前过滤明显空白切片，
            # 避免训练时把切片置零后仍然参与训练。
            if self.filter_blank_slices:
                vol = np.asarray(nii.dataobj, dtype=np.float32)
                for s in range(start, end):
                    img = torch.from_numpy(vol[:, :, s]).float()
                    img = center_crop_or_pad_2d(img, self.image_size, self.image_size)
                    img = self._normalize_slice(img)
                    nonzero_fraction = (img > 0).float().mean().item()
                    if nonzero_fraction >= self.min_nonzero_fraction:
                        self.examples.append((f, s))
            else:
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
            rms = torch.sqrt(torch.mean(x ** 2) + 1e-8)
            x = x / rms

        else:
            raise ValueError(f"Unsupported normalize_mode: {self.normalize_mode}")

        return x.contiguous()

    def _load_slice(self, fname: Path, slice_idx: int) -> torch.Tensor:
        # Dataobj 本身有惰性读取能力；这里只取当前 slice，避免反复读取整 volume。
        nii = nib.load(str(fname))
        slc = np.asarray(nii.dataobj[:, :, slice_idx], dtype=np.float32)
        img = torch.from_numpy(slc).float()
        img = center_crop_or_pad_2d(img, self.image_size, self.image_size)
        img = self._normalize_slice(img)
        return img

    def _call_mask_func(self):
        # 优先把图像尺寸传进去，避免 mask_func 和 image_size 不一致。
        try:
            sig = inspect.signature(self.mask_func)
            if len(sig.parameters) >= 2:
                return self.mask_func(self.image_size, self.image_size)
            return self.mask_func()
        except (TypeError, ValueError):
            return self.mask_func()

    @staticmethod
    def _ensure_mask_shape(mask: Union[np.ndarray, torch.Tensor], h: int, w: int) -> torch.Tensor:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3:
            pass
        else:
            raise ValueError(f"mask ndim must be 2 or 3, got {mask.ndim}")

        if mask.shape[-2:] != (h, w):
            raise ValueError(f"mask spatial size mismatch: got {tuple(mask.shape[-2:])}, expected {(h, w)}")

        return mask.contiguous()

    def __getitem__(self, idx: int):
        fname, slice_idx = self.examples[idx]
        img = self._load_slice(fname, slice_idx)

        # real image -> complex image [1, H, W, 2]
        img_complex = torch.stack([img, torch.zeros_like(img)], dim=-1).unsqueeze(0)

        # full k-space [1, H, W, 2]
        kspace = fastmri.fft2c(img_complex).float()

        mask, mask_fold = self._call_mask_func()
        mask = self._ensure_mask_shape(mask, self.image_size, self.image_size)

        if isinstance(mask_fold, np.ndarray):
            mask_fold = torch.from_numpy(mask_fold).float()
        else:
            mask_fold = mask_fold.float()
        if mask_fold.ndim == 2:
            mask_fold = mask_fold.unsqueeze(0)
        elif mask_fold.ndim != 3:
            raise ValueError(f"mask_fold ndim must be 2 or 3, got {mask_fold.ndim}")
        mask_fold = mask_fold.contiguous()

        if self.return_dict:
            return {
                "kspace": kspace,
                "mask": mask,
                "mask_fold": mask_fold,
                "target": img.unsqueeze(0).float(),
                "fname": str(fname),
                "slice_idx": int(slice_idx),
            }

        return kspace, mask, mask_fold