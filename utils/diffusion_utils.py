import os
import glob
import random
import argparse
from itertools import cycle as itertools_cycle

import torch
import numpy as np
import torch.fft as FFT
import scipy.io as scio


# =========================
# 训练辅助
# =========================
def cycle(dl):
    """Endless dataloader iterator."""
    while True:
        for data in dl:
            yield data


class EMA:
    def __init__(self, beta: float):
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1.0 - self.beta) * new

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, new_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, new_weight)


def loss_backwards(fp16, loss, optimizer, **kwargs):
    """
    Compatibility wrapper.
    Current project uses standard fp32 path; keep fp16 flag for old trainer interface.
    """
    loss.backward(**kwargs)


# =========================
# 种子设置
# =========================
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================
# 文件操作
# =========================
def get_all_files(folder, pattern="*"):
    return sorted(glob.glob(os.path.join(folder, pattern)))


def create_path(path):
    os.makedirs(path, exist_ok=True)


# =========================
# 图像/数组操作
# =========================
def crop(img, cropx, cropy):
    nb, c, y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[:, :, starty:starty + cropy, startx:startx + cropx]


def normalize(img):
    img = img - torch.min(img)
    denom = torch.max(img).clamp_min(1e-12)
    img = img / denom
    return img


def normalize_np(img):
    img = img - np.min(img)
    denom = max(np.max(img), 1e-12)
    img = img / denom
    return img


def normalize_complex(img):
    abs_img = normalize(torch.abs(img))
    ang_img = normalize(torch.angle(img))
    return abs_img * torch.exp(1j * ang_img)


def normalize_l2(x):
    """Normalize by standard deviation (supports numpy array or torch tensor)."""
    if torch.is_tensor(x):
        return x / torch.std(x).clamp_min(1e-12)
    else:
        return x / max(np.std(x), 1e-12)


# 为了兼容 Emat_xyt_complex 中的调用
def ifft2c(x):
    return ifft2c_2d(x)


def fft2c(x):
    return fft2c_2d(x)


# =========================
# FFT 工具
# =========================
def ifftshift(x, axes=None):
    assert torch.is_tensor(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def fftshift(x, axes=None):
    assert torch.is_tensor(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)


def fft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.tensor([ny], device=device, dtype=torch.float32)
    nx = torch.tensor([nx], device=device, dtype=torch.float32)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.fft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.div(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(ny))
    return x


def ifft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.tensor([ny], device=device, dtype=torch.float32)
    nx = torch.tensor([nx], device=device, dtype=torch.float32)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.ifft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.mul(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(ny))
    return x


def FFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2) / np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.fft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3) / np.math.sqrt(ny)
    return x


def IFFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.ifft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2) * np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.ifft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3) * np.math.sqrt(ny)
    return x


# =========================
# 线圈敏感度图相关
# =========================
def Emat_xyt_complex(b, inv, csm, mask):
    if csm is None:
        if inv:
            b = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(b)
            else:
                x = ifft2c(b)
        else:
            if b.ndim == 4:
                x = fft2c_2d(b) * mask
            else:
                x = fft2c(b) * mask
    else:
        if inv:
            x = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c(x)
            x = x * torch.conj(csm)
            x = torch.sum(x, 1)
            x = torch.unsqueeze(x, 1)
        else:
            b = b * csm
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c(b)
            x = mask * b
    return x


# =========================
# 配置工具
# =========================
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace