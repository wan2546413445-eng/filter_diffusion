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


def complex_to_abs(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., 2]
    return: [...]
    """
    return fastmri.complex_abs(x)


def log_kspace(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x: [H, W, 2]
    return: [H, W]
    """
    return torch.log(complex_to_abs(x) + eps)


def normalize_for_show(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x.numpy()


def build_mask_func(config):
    size = (1, config.data.image_size, config.data.image_size)
    mask_type = config.data.mask_type

    if mask_type == 'equispaced_cartesian':  # 新增
        return EquispacedCartesianMask(
            acceleration=config.data.R,
            center_fraction=1.0 / config.data.R,  # 自动设为 1/acceleration
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
    parser.add_argument("--config", type=str, required=True, help="Path to ixi yaml config")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--idx", type=int, default=0, help="Slice index in dataset")
    parser.add_argument("--t", type=int, default=None, help="Diffusion timestep, default=T")
    parser.add_argument("--save_dir", type=str, default="./debug_unit_outputs_2")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = dict2obj(yaml.safe_load(f))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) 构造 acquisition mask（对应 kc）
    mask_func = build_mask_func(cfg)

    # 2) 构造 IXI dataset（返回 full k-space + acquisition mask）
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

    print("=" * 80)
    print(f"[INFO] split          : {args.split}")
    print(f"[INFO] dataset length : {len(dataset)}")
    print(f"[INFO] sample idx     : {args.idx}")
    print("=" * 80)

    k0, acq_mask, mask_fold = dataset[args.idx]
    # dataset output:
    # k0       : [1, H, W, 2]
    # acq_mask : [1, H, W]
    # mask_fold: [1, h, w] or [1,1,w] depending on mask type

    k0 = k0.unsqueeze(0)               # -> [B=1, Nc=1, H, W, 2]
    acq_mask = acq_mask.unsqueeze(0)   # -> [B=1, 1, H, W]

    B, Nc, H, W, C = k0.shape
    print(f"[SHAPE] k0       : {tuple(k0.shape)}")
    print(f"[SHAPE] acq_mask : {tuple(acq_mask.shape)}")
    print(f"[SHAPE] mask_fold: {tuple(mask_fold.shape)}")

    # 3) 条件输入 kc = k0 * acquisition mask
    kc = k0 * acq_mask.unsqueeze(-1)

    # 4) 构造扩散 schedule（对应 Mt）
    T = int(cfg.training.timesteps)
    t = T if args.t is None else int(args.t)

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

    t_tensor = torch.tensor([t], dtype=torch.long)
    mt = schedule.get_by_t(t_tensor, device=k0.device, dtype=k0.dtype)   # [1,1,H,W,1]
    mT = schedule.get_by_t(torch.tensor([T]), device=k0.device, dtype=k0.dtype)

    # 5) 扩散状态：k_t = M_t * k0
    kt = mt * k0
    kT = mT * k0

    # 6) 图像域观察
    x0 = fastmri.ifft2c(k0)
    xc = fastmri.ifft2c(kc)
    xt = fastmri.ifft2c(kt)
    xT = fastmri.ifft2c(kT)

    x0_abs = complex_to_abs(x0[0, 0])
    xc_abs = complex_to_abs(xc[0, 0])
    xt_abs = complex_to_abs(xt[0, 0])
    xT_abs = complex_to_abs(xT[0, 0])

    k0_log = log_kspace(k0[0, 0])
    kc_log = log_kspace(kc[0, 0])
    kt_log = log_kspace(kt[0, 0])
    kT_log = log_kspace(kT[0, 0])

    mt_show = mt[0, 0, :, :, 0]
    mT_show = mT[0, 0, :, :, 0]
    acq_show = acq_mask[0, 0]

    # 7) 打印关键统计
    print("\n[STAT] acquisition mask")
    print(f"  ones ratio      : {acq_show.float().mean().item():.6f}")

    print("[STAT] diffusion masks")
    print(f"  Mt ones ratio   : {mt_show.float().mean().item():.6f}  (t={t})")
    print(f"  MT ones ratio   : {mT_show.float().mean().item():.6f}  (T={T})")

    diff_kc_kT = torch.mean(torch.abs(kc - kT)).item()
    diff_k0_kT = torch.mean(torch.abs(k0 - kT)).item()
    diff_k0_kc = torch.mean(torch.abs(k0 - kc)).item()

    print("[STAT] k-space differences")
    print(f"  mean |kc - kT|  : {diff_kc_kT:.6e}")
    print(f"  mean |k0 - kT|  : {diff_k0_kT:.6e}")
    print(f"  mean |k0 - kc|  : {diff_k0_kc:.6e}")

    # 8) 简单逻辑判断
    print("\n[CHECK]")
    if diff_kc_kT < 1e-8:
        print("  [WARNING] kc 和 kT 基本一样，这通常不符合你现在理解的 FilterDiff 两套退化。")
    else:
        print("  [OK] kc 和 kT 不同：说明 acquisition mask 与 diffusion terminal mask 不是同一个东西。")

    if diff_k0_kT > 1e-8:
        print("  [OK] kT 确实是由 full k-space 退化得到，不是直接把 dataloader 输出当成终点。")
    else:
        print("  [WARNING] kT 和 k0 太接近，检查 center_core_size / timesteps / schedule_type 是否设置异常。")

    # 9) 保存图
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    axes[0, 0].imshow(normalize_for_show(x0_abs), cmap="gray")
    axes[0, 0].set_title("x0 from full k0")
    axes[0, 1].imshow(normalize_for_show(xc_abs), cmap="gray")
    axes[0, 1].set_title("xc from kc")
    axes[0, 2].imshow(normalize_for_show(xt_abs), cmap="gray")
    axes[0, 2].set_title(f"xt from kt (t={t})")
    axes[0, 3].imshow(normalize_for_show(xT_abs), cmap="gray")
    axes[0, 3].set_title(f"xT from kT (T={T})")

    axes[1, 0].imshow(normalize_for_show(k0_log), cmap="gray")
    axes[1, 0].set_title("log|k0|")
    axes[1, 1].imshow(normalize_for_show(kc_log), cmap="gray")
    axes[1, 1].set_title("log|kc|")
    axes[1, 2].imshow(normalize_for_show(kt_log), cmap="gray")
    axes[1, 2].set_title(f"log|kt| (t={t})")
    axes[1, 3].imshow(normalize_for_show(kT_log), cmap="gray")
    axes[1, 3].set_title(f"log|kT| (T={T})")

    axes[2, 0].imshow(normalize_for_show(acq_show), cmap="gray")
    axes[2, 0].set_title("acquisition mask")
    axes[2, 1].imshow(normalize_for_show(mt_show), cmap="gray")
    axes[2, 1].set_title(f"Mt (t={t})")
    axes[2, 2].imshow(normalize_for_show(mT_show), cmap="gray")
    axes[2, 2].set_title(f"MT (T={T})")
    axes[2, 3].imshow(normalize_for_show(torch.abs(acq_show - mT_show)), cmap="hot")
    axes[2, 3].set_title("|acq mask - MT|")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()

    fig_path = save_dir / f"debug_{args.split}_idx{args.idx}_t{t}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 10) 保存 tensor，便于你后续 MATLAB / Python 再查
    npz_path = save_dir / f"debug_{args.split}_idx{args.idx}_t{t}.npz"
    np.savez_compressed(
        npz_path,
        x0=x0_abs.detach().cpu().numpy(),
        xc=xc_abs.detach().cpu().numpy(),
        xt=xt_abs.detach().cpu().numpy(),
        xT=xT_abs.detach().cpu().numpy(),
        k0_log=k0_log.detach().cpu().numpy(),
        kc_log=kc_log.detach().cpu().numpy(),
        kt_log=kt_log.detach().cpu().numpy(),
        kT_log=kT_log.detach().cpu().numpy(),
        acq_mask=acq_show.detach().cpu().numpy(),
        mt=mt_show.detach().cpu().numpy(),
        mT=mT_show.detach().cpu().numpy(),
    )

    print("\n[SAVED]")
    print(f"  figure : {fig_path}")
    print(f"  npz    : {npz_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()