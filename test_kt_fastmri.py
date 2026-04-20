import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import fastmri

from data.mri_data import SliceDataset
from data.data_transform import DataTransform_Diffusion, sense_combine_torch
from utils.sample_mask import EquispacedCartesianMask, RandomMaskDiffusion, EquiSpaceMaskDiffusion
from diffusion.filter_schedule import CenterRectangleSchedule


def dict2obj(d):
    if isinstance(d, dict):
        return type("Cfg", (), {k: dict2obj(v) for k, v in d.items()})
    return d


def normalize_for_show(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x.numpy()


def log_kspace(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(fastmri.complex_abs(x) + eps)


def build_mask_func(cfg):
    size = (1, cfg.data.image_size, cfg.data.image_size)
    mask_type = cfg.data.mask_type

    if mask_type == "equispaced_cartesian":
        return EquispacedCartesianMask(
            acceleration=cfg.data.R,
            center_fraction=cfg.data.center_fraction,   # 和 train.py 保持一致
            size=size,
            seed=cfg.data.seed,
        )
    elif mask_type == "random_diffusion":
        return RandomMaskDiffusion(
            center_fraction=cfg.data.center_fraction,
            acceleration=cfg.data.R,
            size=size,
            seed=cfg.data.seed,
        )
    elif mask_type == "equispace_diffusion":
        return EquiSpaceMaskDiffusion(
            center_fraction=cfg.data.center_fraction,
            acceleration=cfg.data.R,
            size=size,
            seed=cfg.data.seed,
        )
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")


def get_root(cfg, split: str):
    if split == "train":
        return cfg.data.data_root
    elif split == "val":
        return cfg.data.val_root
    elif split == "test":
        return cfg.data.test_root
    else:
        raise ValueError(f"Unsupported split: {split}")


def build_dataset(cfg, split: str):
    mask_func = build_mask_func(cfg)
    combine_coil = getattr(cfg.data, "combine_coil", True)

    transform = DataTransform_Diffusion(
        mask_func=mask_func,
        img_size=cfg.data.image_size,
        combine_coil=combine_coil,
        flag_singlecoil=False,
        maps_root=getattr(cfg.data, "maps_root", None),
        map_key=getattr(cfg.data, "map_key", "s_maps"),
    )

    dataset = SliceDataset(
        root=Path(get_root(cfg, split)),
        transform=transform,
        challenge="multicoil",
        num_skip_slice=getattr(cfg.data, "num_skip_slice", 6),
    )
    return dataset


def to_2d_mask(mask: torch.Tensor) -> torch.Tensor:
    # [1,H,W] or [H,W]
    if mask.ndim == 3:
        return mask[0]
    return mask


def get_display_kspace(kspace: torch.Tensor, maps: torch.Tensor = None) -> torch.Tensor:
    """
    返回用于显示的单张 k-space：
    - 如果是单图 [1,H,W,2]，直接取第 0 张
    - 如果是多线圈 [Nc,H,W,2] 且有 maps，则先 SENSE combine 再 FFT
    - 如果是多线圈无 maps，则显示第一个 coil
    """
    if kspace.ndim != 4:
        raise ValueError(f"Expected [N,H,W,2], got {kspace.shape}")

    if kspace.shape[0] == 1:
        return kspace[0]

    coil_imgs = fastmri.ifft2c(kspace)  # [Nc,H,W,2]
    if maps is not None:
        img = sense_combine_torch(coil_imgs, maps)      # [H,W,2]
        k = fastmri.fft2c(img.unsqueeze(0))[0]          # [H,W,2]
        return k
    else:
        return kspace[0]


def summarize_mask(mask2d: torch.Tensor, name: str):
    cols = torch.where(mask2d.sum(dim=0) > 0)[0]
    if len(cols) == 0:
        print(f"{name}: no sampled columns")
        return
    width = cols.max().item() - cols.min().item() + 1
    print(f"{name}: sampled_cols={len(cols)}, span_width={width}, first={cols.min().item()}, last={cols.max().item()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--t_list", type=int, nargs="+", default=[0, 1, 5, 10, 19])
    parser.add_argument("--save_dir", type=str, default="./debug_fastmri_mask")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = dict2obj(yaml.safe_load(f))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg, args.split)
    sample = dataset[args.idx]

    if len(sample) == 3:
        k0, acq_mask, mask_fold = sample
        maps = None
    elif len(sample) == 4:
        k0, acq_mask, mask_fold, maps = sample
    else:
        raise ValueError(f"Unexpected sample length: {len(sample)}")

    # shapes:
    # k0: [1,H,W,2] or [Nc,H,W,2]
    # acq_mask: [1,H,W]
    # maps: [Nc,H,W,2] or None
    acq_mask_2d = to_2d_mask(acq_mask)

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

    # -------- terminal MT check --------
    tT = torch.tensor([T], dtype=torch.long)
    mT = schedule.get_by_t(tT, device=k0.device, dtype=k0.dtype)[0, 0, :, :, 0].float()

    # 检查 M_T 是否是 acq_mask 的子集
    violation = ((mT > 0.5) & (acq_mask_2d < 0.5)).sum().item()
    overlap = ((mT > 0.5) & (acq_mask_2d > 0.5)).sum().item()

    print("=" * 60)
    print(f"split={args.split}, idx={args.idx}")
    print(f"k0 shape      : {tuple(k0.shape)}")
    print(f"acq_mask shape: {tuple(acq_mask.shape)}")
    if maps is not None:
        print(f"maps shape    : {tuple(maps.shape)}")
    else:
        print("maps          : None")

    summarize_mask(acq_mask_2d, "acq_mask")
    summarize_mask(mT, "M_T")
    print(f"overlap pixels           : {overlap}")
    print(f"M_T outside acq_mask     : {violation}")
    print("subset check passed      :", violation == 0)
    print("=" * 60)

    # -------- construct kc and kT --------
    # acq_mask -> [1,H,W,1] for broadcasting
    acq_mask_bc = acq_mask.unsqueeze(-1) if acq_mask.ndim == 3 else acq_mask[None, ..., None]
    k_c = k0 * acq_mask_bc
    k_T = k0 * mT.unsqueeze(0).unsqueeze(-1)

    k0_show = get_display_kspace(k0, maps)
    kc_show = get_display_kspace(k_c, maps)
    kT_show = get_display_kspace(k_T, maps)

    # -------- summary figure --------
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(normalize_for_show(log_kspace(k0_show)), cmap="gray")
    axes[0, 0].set_title("log|k0|")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(normalize_for_show(acq_mask_2d), cmap="gray")
    axes[0, 1].set_title("acq_mask")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(normalize_for_show(log_kspace(kc_show)), cmap="gray")
    axes[0, 2].set_title("log|kc|")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(normalize_for_show(mT), cmap="gray")
    axes[1, 0].set_title(f"M_T (t={T})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(normalize_for_show(log_kspace(kT_show)), cmap="gray")
    axes[1, 1].set_title("log|kT|")
    axes[1, 1].axis("off")

    diff_mask = torch.abs(acq_mask_2d - mT)
    axes[1, 2].imshow(normalize_for_show(diff_mask), cmap="hot")
    axes[1, 2].set_title("|acq_mask - M_T|")
    axes[1, 2].axis("off")

    plt.tight_layout()
    fig_path = save_dir / f"summary_{args.split}_idx{args.idx}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------- kt grid --------
    n = len(args.t_list)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)

    for i, t in enumerate(args.t_list):
        t_tensor = torch.tensor([t], dtype=torch.long)
        mt = schedule.get_by_t(t_tensor, device=k0.device, dtype=k0.dtype)[0, 0, :, :, 0]
        kt = k0 * mt.unsqueeze(0).unsqueeze(-1)
        kt_show = get_display_kspace(kt, maps)

        axes[i].imshow(normalize_for_show(log_kspace(kt_show)), cmap="gray")
        axes[i].set_title(f"log|kt| @ t={t}")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig_path2 = save_dir / f"logkt_grid_{args.split}_idx{args.idx}.png"
    plt.savefig(fig_path2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {fig_path}")
    print(f"Saved: {fig_path2}")


if __name__ == "__main__":
    main()