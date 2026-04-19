import argparse
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import fastmri

from utils.utils import dict2namespace, setup_seed
from utils.sample_mask import EquispacedCartesianMask, RandomMaskDiffusion, EquiSpaceMaskDiffusion
from data.ixi_singlecoil_dataset import IXISinglecoilSliceDataset
from diffusion.kspace_diffusion import KspaceDiffusion
from models.unet_diffusion import Unet
from models.restoration_net_filterdiff import build_filterdiff_restoration_net


def build_mask_func(config):
    size = (1, config.data.image_size, config.data.image_size)
    mask_type = config.data.mask_type
    seed = getattr(config.data, "seed", getattr(config, "seed", 42))

    if mask_type == 'equispaced_cartesian':
        return EquispacedCartesianMask(
            acceleration=config.data.R,
            center_fraction=config.data.center_fraction,
            size=size,
            seed=seed
        )
    elif mask_type == 'random_diffusion':
        return RandomMaskDiffusion(
            center_fraction=config.data.center_fraction,
            acceleration=config.data.R,
            size=size,
            seed=seed
        )
    elif mask_type == 'equispace_diffusion':
        return EquiSpaceMaskDiffusion(
            center_fraction=config.data.center_fraction,
            acceleration=config.data.R,
            size=size,
            seed=seed
        )
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")


def build_backbone(config, device):
    backbone = getattr(config.model, 'backbone', 'unet')

    if backbone == 'unet':
        return Unet(
            dim=config.model.dim,
            out_dim=2,
            channels=5,
            dim_mults=tuple(config.model.dim_mults),
            with_time_emb=True,
            residual=config.model.residual
        ).to(device)

    elif backbone == 'swin_dits':
        return build_filterdiff_restoration_net(
            img_size=config.data.image_size,
            patch_size=getattr(config.model, 'patch_size', 4),
            in_channels=5,
            out_channels=2,
            hidden_size=getattr(config.model, 'hidden_size', 384),
            depth=getattr(config.model, 'depth', 8),
            num_heads=getattr(config.model, 'num_heads', 8),
            window_size=getattr(config.model, 'window_size', 8),
            mlp_ratio=getattr(config.model, 'mlp_ratio', 4.0),
            with_time_emb=True,
        ).to(device)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


def build_model(config, device):
    denoise_fn = build_backbone(config, device)

    center_core_size = getattr(
        config.training,
        'center_core_size',
        max(1, int(round(config.data.image_size * float(config.data.center_fraction))))
    )

    model = KspaceDiffusion(
        denoise_fn=denoise_fn,
        image_size=config.data.image_size,
        device_of_kernel=str(device),
        channels=2,
        timesteps=config.training.timesteps,
        loss_type=config.training.loss_type,
        schedule_type=getattr(config.training, 'filter_schedule_type', 'dense'),
        center_core_size=center_core_size,
        lambda_img=float(getattr(config.training, 'lambda_img', 1.0)),
        use_explicit_dc=getattr(config.training, 'use_explicit_dc', False),
    ).to(device)
    return model


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_np(x: torch.Tensor):
    return x.detach().cpu().numpy()


def complex_abs_2d(x: torch.Tensor):
    if x.ndim == 5:
        x = x[0, 0]
    elif x.ndim == 4:
        x = x[0]
    return to_np(fastmri.complex_abs(x))


def complex_err_abs_2d(x: torch.Tensor, y: torch.Tensor):
    z = x - y
    return complex_abs_2d(z)


def kspace_logmag_2d(k: torch.Tensor, eps: float = 1e-8):
    if k.ndim == 5:
        k = k[0, 0]
    elif k.ndim == 4:
        k = k[0]
    mag = fastmri.complex_abs(k)
    return to_np(torch.log(mag + eps))


def norm01(img: np.ndarray):
    img = img.astype(np.float32)
    mn = img.min()
    mx = img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def save_gray(path: Path, img: np.ndarray, title: str = ""):
    plt.figure(figsize=(4, 4))
    plt.imshow(norm01(img), cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_hot(path: Path, img: np.ndarray, title: str = ""):
    plt.figure(figsize=(4, 4))
    plt.imshow(norm01(img), cmap="hot")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--t_list", type=str, default="0,1,5,10,19")
    parser.add_argument("--outdir", type=str, default="./debug_loss_terms_ixi")
    parser.add_argument("--use_ema", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    setup_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    mask_func = build_mask_func(config)

    if args.split == "train":
        root = config.data.train_root
    elif args.split == "val":
        root = config.data.val_root
    else:
        root = config.data.test_root

    dataset = IXISinglecoilSliceDataset(
        root=root,
        mask_func=mask_func,
        image_size=config.data.image_size,
        num_skip_slice=config.data.num_skip_slice,
        normalize_mode=config.data.normalize_mode,
    )

    model = build_model(config, device)
    ckpt = torch.load(args.ckpt, map_location=device)

    if args.use_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=True)
    else:
        model.load_state_dict(ckpt["model"], strict=True)

    model.eval()

    kspace, mask, mask_fold = dataset[args.idx]
    kspace = kspace.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    outdir = Path(args.outdir) / f"{args.split}_idx_{args.idx:04d}"
    ensure_dir(outdir)

    t_list = [int(x.strip()) for x in args.t_list.split(",") if x.strip() != ""]
    lines = []

    for t_scalar in t_list:
        t = torch.tensor([t_scalar], device=device, dtype=torch.long)
        dbg = model.debug_loss_terms(kspace, mask, t)

        step_dir = outdir / f"t_{t_scalar:02d}"
        ensure_dir(step_dir)

        msg = (
            f"[t={t_scalar:02d}] "
            f"loss_delta={dbg['loss_delta']:.6f} | "
            f"loss_img={dbg['loss_img']:.6f} | "
            f"total={dbg['total_loss']:.6f}"
        )
        print(msg)
        lines.append(msg)

        save_gray(step_dir / "x0_abs.png", complex_abs_2d(dbg["x0"]), "x0")
        save_gray(step_dir / "x0_pred_abs.png", complex_abs_2d(dbg["x0_pred"]), "x0_pred")
        save_hot(step_dir / "x0_err.png", complex_err_abs_2d(dbg["x0_pred"], dbg["x0"]), f"|x0_pred-x0| @ {t_scalar}")

        save_gray(step_dir / "k0_logmag.png", kspace_logmag_2d(dbg["k0"]), "log|k0|")
        save_gray(step_dir / "kc_logmag.png", kspace_logmag_2d(dbg["k_c"]), "log|kc|")
        save_gray(step_dir / "kt_logmag.png", kspace_logmag_2d(dbg["k_t"]), "log|kt|")
        save_gray(step_dir / "delta_gt_logmag.png", kspace_logmag_2d(dbg["delta_gt"]), "log|delta_gt|")
        save_gray(step_dir / "delta_pred_logmag.png", kspace_logmag_2d(dbg["delta_pred"]), "log|delta_pred|")
        save_hot(step_dir / "delta_err.png", complex_err_abs_2d(dbg["delta_pred"], dbg["delta_gt"]), f"|delta_pred-delta_gt| @ {t_scalar}")

        np.savez_compressed(
            step_dir / "tensors.npz",
            x0=to_np(dbg["x0"]),
            x0_pred=to_np(dbg["x0_pred"]),
            k0=to_np(dbg["k0"]),
            kc=to_np(dbg["k_c"]),
            kt=to_np(dbg["k_t"]),
            delta_gt=to_np(dbg["delta_gt"]),
            delta_pred=to_np(dbg["delta_pred"]),
            Mt=to_np(dbg["m_t"]),
            Mprev=to_np(dbg["m_t_minus_1"]),
            DeltaM=to_np(dbg["delta_m"]),
        )

    with open(outdir / "loss_terms_summary.txt", "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"[done] saved to {outdir}")


if __name__ == "__main__":
    main()