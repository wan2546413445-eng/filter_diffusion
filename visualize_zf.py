import os
import numpy as np
import torch
import fastmri
import matplotlib.pyplot as plt

from data.ixi_singlecoil_dataset import IXISinglecoilSliceDataset


def build_fixed_cartesian_mask_func(H=256, W=256, acc=4, acs=24, seed=1234):
    """
    返回一个固定笛卡尔欠采样 mask_func
    输出:
        mask:      [1, H, W]
        mask_fold: [1, H, W]
    """
    rng = np.random.RandomState(seed)

    center = W // 2
    left = center - acs // 2
    right = left + acs

    sampled_cols = set(range(left, right))

    # 目标保留列数，近似 R=acc
    num_keep = max(acs, W // acc)

    candidates = [i for i in range(W) if i not in sampled_cols]
    rng.shuffle(candidates)

    extra = max(0, num_keep - len(sampled_cols))
    sampled_cols.update(candidates[:extra])

    col_mask = np.zeros(W, dtype=np.float32)
    col_mask[list(sampled_cols)] = 1.0

    mask_2d = np.tile(col_mask[None, :], (H, 1))      # [H, W]
    mask = mask_2d[None, ...].astype(np.float32)      # [1, H, W]
    mask_fold = mask.copy()

    def mask_func():
        return mask.copy(), mask_fold.copy()

    return mask_func


def save_one_sample(dataset, idx, save_dir):
    kspace, mask, mask_fold = dataset[idx]

    # full image
    img_full = fastmri.ifft2c(kspace)                 # [1, H, W, 2]
    img_full = fastmri.complex_abs(img_full)[0]       # [H, W]

    # undersampled k-space
    kspace_us = kspace * mask.unsqueeze(-1)           # [1, H, W, 2]

    # zero-filled image
    img_zf = fastmri.ifft2c(kspace_us)
    img_zf = fastmri.complex_abs(img_zf)[0]           # [H, W]

    # 额外看一下 k-space 幅度（方便排错）
    k_full_mag = torch.sqrt(kspace[..., 0] ** 2 + kspace[..., 1] ** 2)[0]
    k_us_mag = torch.sqrt(kspace_us[..., 0] ** 2 + kspace_us[..., 1] ** 2)[0]

    # log 显示更清楚
    k_full_log = torch.log1p(k_full_mag)
    k_us_log = torch.log1p(k_us_mag)

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img_full.numpy(), cmap="gray")
    plt.title("Full Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(mask[0].numpy(), cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(img_zf.numpy(), cmap="gray")
    plt.title("ZF Image")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(k_full_log.numpy(), cmap="gray")
    plt.title("log |K_full|")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(k_us_log.numpy(), cmap="gray")
    plt.title("log |K_us|")
    plt.axis("off")

    diff = torch.abs(img_full - img_zf)
    plt.subplot(2, 3, 6)
    plt.imshow(diff.numpy(), cmap="gray")
    plt.title("|Full - ZF|")
    plt.axis("off")

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"sample_{idx:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[saved] {save_path}")
    print(
        f"idx={idx} | full max={img_full.max().item():.4f} | "
        f"zf max={img_zf.max().item():.4f} | "
        f"mask keep ratio={mask.mean().item():.4f}"
    )


if __name__ == "__main__":
    save_dir = "./debug_ixi_vis"
    os.makedirs(save_dir, exist_ok=True)

    mask_func = build_fixed_cartesian_mask_func(
        H=256,
        W=256,
        acc=4,
        acs=24,
        seed=1234,
    )

    dataset = IXISinglecoilSliceDataset(
        root="/mnt/SSD/wsy/data/train",
        mask_func=mask_func,
        image_size=256,
        num_skip_slice=20,
        normalize_mode="max",   # 先用 max，调试更快
    )

    print("dataset len =", len(dataset))

    test_indices = [0, 100, 1000, 5000]
    test_indices = [i for i in test_indices if i < len(dataset)]

    for idx in test_indices:
        save_one_sample(dataset, idx, save_dir)

    print("done.")