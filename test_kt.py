import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import fastmri

from data.ixi_singlecoil_dataset import IXISinglecoilSliceDataset
from utils.sample_mask import EquispacedCartesianMask, RandomMaskDiffusion, RandomMaskGaussianDiffusion
from diffusion.filter_schedule import CenterRectangleSchedule


def dict2obj(d):
    if isinstance(d, dict):
        return type("Cfg", (), {k: dict2obj(v) for k, v in d.items()})
    return d


def log_kspace(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(fastmri.complex_abs(x) + eps)


def normalize_for_show(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x.numpy()


def build_mask_func(config):
    size = (1, config.data.image_size, config.data.image_size)
    mask_type = config.data.mask_type
    if mask_type == 'equispaced_cartesian':
        return EquispacedCartesianMask(
            acceleration=config.data.R,
            center_fraction=1.0 / config.data.R,
            size=size,
            seed=config.data.seed,
        )
    elif mask_type == "random_diffusion":
        return RandomMaskDiffusion(
            center_fraction=config.data.center_fraction,
            acceleration=config.data.R,
            size=size,
            seed=config.data.seed,
        )
    elif mask_type == "gaussian_diffusion":
        return RandomMaskGaussianDiffusion(
            acceleration=config.data.R,
            center_fraction=config.data.center_fraction,
            size=size,
            seed=config.data.seed,
            patch_size=getattr(config.data, "patch_size", 4),
        )
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--t_list", type=int, nargs='+', help="List of timesteps. If not given, automatically sample 20 steps from 0 to T.")
    parser.add_argument("--nrows", type=int, default=4, help="Number of rows in subplot grid")
    parser.add_argument("--ncols", type=int, default=5, help="Number of columns in subplot grid")
    parser.add_argument("--save_dir", type=str, default="./debug_kt_grid")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = dict2obj(yaml.safe_load(f))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # mask & dataset
    mask_func = build_mask_func(cfg)
    if args.split == "train":
        root = cfg.data.train_root
    elif args.split == "val":
        root = cfg.data.val_root
    else:
        root = cfg.data.test_root

    dataset = IXISinglecoilSliceDataset(
        root=root,
        mask_func=mask_func,
        image_size=cfg.data.image_size,
        num_skip_slice=getattr(cfg.data, "num_skip_slice", 20),
        normalize_mode=getattr(cfg.data, "normalize_mode", "percentile"),
    )

    k0, acq_mask, _ = dataset[args.idx]
    k0 = k0.unsqueeze(0)               # [1,1,H,W,2]
    acq_mask = acq_mask.unsqueeze(0)   # [1,1,H,W]

    # schedule
    T = int(cfg.training.timesteps)
    center_core_size = getattr(
        cfg.training,
        "center_core_size",
        max(1, int(round(cfg.data.image_size * float(cfg.data.center_fraction))))
    )
    schedule = CenterRectangleSchedule(
        h=cfg.data.image_size,
        w=cfg.data.image_size,
        timesteps=T,
        center_core_size=center_core_size,
        schedule_type=getattr(cfg.training, "filter_schedule_type", "dense"),
    )

    # 确定要显示的 t 列表
    if args.t_list is not None:
        t_list = args.t_list
    else:
        # 自动从 0 到 T 均匀采样 nrows*ncols 个点
        n_subplots = args.nrows * args.ncols
        t_list = np.linspace(0, T, n_subplots, dtype=int).tolist()
        # 确保 T 包含在内（可能 linspace 的最后一个就是 T）
        if t_list[-1] != T:
            t_list[-1] = T

    # 创建子图网格
    fig, axes = plt.subplots(args.nrows, args.ncols, figsize=(args.ncols*3, args.nrows*3))
    axes = axes.flatten()  # 方便索引

    for i, t in enumerate(t_list):
        if i >= len(axes):
            break
        t_tensor = torch.tensor([t], dtype=torch.long)
        mt = schedule.get_by_t(t_tensor, device=k0.device, dtype=k0.dtype)
        kt = mt * k0
        kt_log = log_kspace(kt[0, 0])

        ax = axes[i]
        ax.imshow(normalize_for_show(kt_log), cmap="gray")
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis("off")

    # 隐藏多余的子图（如果 t_list 数量少于子图数量）
    for j in range(len(t_list), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig_path = save_dir / f"logkt_grid_{args.split}_idx{args.idx}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved grid figure: {fig_path} (showing {len(t_list)} timesteps)")


if __name__ == "__main__":
    main()