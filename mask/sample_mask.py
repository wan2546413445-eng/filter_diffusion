import numpy as np
import torch
import torch.nn.functional as F


def apply_filter_schedule(t, schedule_type="dense"):
    """
    生成离散时间步的滤波器比例，根据论文中的设计。
    """
    # 这是你定义的离散比例表
    FILTER_RATIOS = [
        1.0, 0.92, 0.85, 0.79, 0.74, 0.69, 0.65, 0.60, 0.56, 0.52,
        0.48, 0.44, 0.41, 0.38, 0.35, 0.32, 0.30, 0.28, 0.26, 0.24
    ]

    if schedule_type == "dense":
        return FILTER_RATIOS[t]  # 返回对应时间步的滤波器比例
    elif schedule_type == "linear":
        return 1 - (t / 20)  # 线性下降
    elif schedule_type == "sparse":
        return (t / 20) ** 2  # 稀疏下降
    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}")


def generate_mask(kspace_shape, filter_ratio):
    """
    根据给定的滤波比例生成对应的采样掩码。
    kspace_shape: (B, C, H, W) -> kspace 张量的形状
    filter_ratio: 当前时间步的滤波器比例，决定了中心区域的大小

    返回：与 kspace_shape 同样大小的采样掩码
    """
    bsz, c, h, w = kspace_shape
    mask = torch.zeros((bsz, 1, h, w), dtype=torch.float32, device=kspace_shape.device)

    # 计算中心区域的大小
    center_h = int(h * filter_ratio)  # 中心区域的高度
    center_w = int(w * filter_ratio)  # 中心区域的宽度

    # 生成中心区域为 1 的 mask
    top = (h - center_h) // 2
    left = (w - center_w) // 2
    mask[:, :, top:top + center_h, left:left + center_w] = 1.0

    return mask


class RandomMaskFilterDiffusion:
    def __init__(self, size=(1, 256, 256), patch_size=4, seed=None):
        self.size = size
        self.patch_size = patch_size
        self.seed = seed

    def __call__(self, t, schedule_type="dense"):
        """
        生成一个根据时间步变化的采样掩码，且根据 t 和 schedule_type 调整采样比例。
        """
        filter_ratio = apply_filter_schedule(t, schedule_type)

        return random_mask_filter_diffusion(
            size=self.size,
            patch_size=self.patch_size,
            filter_ratio=filter_ratio  # 使用滤波器比例
        )


def random_mask_filter_diffusion(size=(16, 320, 320), patch_size=4, filter_ratio=0.5):
    """
    根据给定的滤波器比例生成随机掩码。
    """
    B, H, W = size
    if H != W:
        raise Exception("Different height and width of the mask setting")
    if H % patch_size != 0:
        raise Exception("Image dimension cannot be fully divided by patch size")

    # 生成 mask
    mask = torch.zeros((B, H, W), dtype=torch.float32)

    # 计算中心区域的大小
    center_h = int(H * filter_ratio)  # 中心区域的高度
    center_w = int(W * filter_ratio)  # 中心区域的宽度

    # 生成中心区域为 1 的 mask
    top = (H - center_h) // 2
    left = (W - center_w) // 2
    mask[:, top:top + center_h, left:left + center_w] = 1.0

    return mask