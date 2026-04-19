import argparse
import math
import yaml
from pathlib import Path

import numpy as np
import torch
import fastmri
import matplotlib.pyplot as plt

from utils.utils import dict2namespace, setup_seed
from utils.sample_mask import EquispacedCartesianMask, RandomMaskDiffusion, EquiSpaceMaskDiffusion
from data.ixi_singlecoil_dataset import IXISinglecoilSliceDataset
from diffusion.kspace_diffusion import KspaceDiffusion
from diffusion.degradation import apply_filter_degradation
from diffusion.delta_target import build_delta_target
from diffusion.dc import explicit_data_consistency
from models.unet_diffusion import Unet
from models.restoration_net_filterdiff import build_filterdiff_restoration_net


# =========================
# builders
# =========================
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


# =========================
# visualization helpers
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_np(x: torch.Tensor):
    return x.detach().cpu().numpy()


def complex_abs_2d(x: torch.Tensor):
    """
    x: [H,W,2] or [1,H,W,2] or [B,1,H,W,2]
    return: [H,W]
    """
    if x.ndim == 5:
        x = x[0, 0]
    elif x.ndim == 4:
        x = x[0]
    return to_np(fastmri.complex_abs(x))


def complex_err_abs_2d(x: torch.Tensor, y: torch.Tensor):
    """
    x, y: same complex tensor shape
    return |x-y| as [H,W]
    """
    z = x - y
    return complex_abs_2d(z)


def kspace_logmag_2d(k: torch.Tensor, eps: float = 1e-8):
    """
    k: [H,W,2] or [1,H,W,2] or [B,1,H,W,2]
    """
    if k.ndim == 5:
        k = k[0, 0]
    elif k.ndim == 4:
        k = k[0]
    mag = fastmri.complex_abs(k)
    mag = torch.log(mag + eps)
    return to_np(mag)


def mask_2d(m: torch.Tensor):
    """
    m: [B,1,H,W,1] or [B,1,H,W] or [1,H,W,1] or [1,H,W]
    """
    x = m
    while x.ndim > 2:
        x = x[0]
    return to_np(x.float())


def norm01(img: np.ndarray):
    img = img.astype(np.float32)
    mn = img.min()
    mx = img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def save_gray(path: Path, img: np.ndarray, title: str = ""):
    img = norm01(img)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_hot(path: Path, img: np.ndarray, title: str = ""):
    img = norm01(img)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="hot")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_npz(path: Path, **kwargs):
    arrs = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            arrs[k] = to_np(v)
        else:
            arrs[k] = v
    np.savez_compressed(path, **arrs)


def image_psnr_shared_gt(gt_img: torch.Tensor, pred_img: torch.Tensor, eps: float = 1e-8):
    """
    gt_img, pred_img: [B,1,H,W,2] or [1,H,W,2]
    共享 GT max 尺度，仅用于 debug 观察趋势
    """
    if gt_img.ndim == 5:
        gt_mag = fastmri.complex_abs(gt_img[0, 0])
        pr_mag = fastmri.complex_abs(pred_img[0, 0])
    else:
        gt_mag = fastmri.complex_abs(gt_img[0])
        pr_mag = fastmri.complex_abs(pred_img[0])

    scale = gt_mag.max() + eps
    gt_mag = gt_mag / scale
    pr_mag = pr_mag / scale
    mse = torch.mean((gt_mag - pr_mag) ** 2)
    psnr = -10.0 * torch.log10(mse + eps)
    return float(psnr.item())


# =========================
# debug main
# =========================
@torch.no_grad()
def debug_forward(model, kspace, mask, outdir: Path, t_list):
    """
    打印 forward:
      k0, kc, kt, k_{t-1}, delta_gt, x0_pred, delta_pred
    """
    ensure_dir(outdir)

    x0 = fastmri.ifft2c(kspace)
    k_c = model._build_conditional_kc(kspace, mask)
    T = model.num_timesteps

    # acquisition
    save_gray(outdir / "mask_acq.png", mask_2d(mask), "M_acq")
    save_gray(outdir / "x0_abs.png", complex_abs_2d(x0), "x0")
    save_gray(outdir / "xc_abs.png", complex_abs_2d(fastmri.ifft2c(k_c)), "xc")
    save_gray(outdir / "k0_logmag.png", kspace_logmag_2d(kspace), "log|k0|")
    save_gray(outdir / "kc_logmag.png", kspace_logmag_2d(k_c), "log|kc|")
    save_npz(outdir / "acq_tensors.npz", k0=kspace, x0=x0, kc=k_c, mask_acq=mask)

    summary_lines = []
    summary_lines.append(f"T = {T}")

    for t_scalar in t_list:
        t_scalar = int(t_scalar)
        t = torch.tensor([t_scalar], device=kspace.device, dtype=torch.long)
        t_prev = torch.clamp(t - 1, min=0)

        m_t = model.schedule.get_by_t(t, device=kspace.device, dtype=kspace.dtype)
        m_prev = model.schedule.get_by_t(t_prev, device=kspace.device, dtype=kspace.dtype)
        delta_m = m_prev - m_t

        k_t = apply_filter_degradation(kspace, m_t)
        k_prev = apply_filter_degradation(kspace, m_prev)
        x_t = fastmri.ifft2c(k_t)
        x_prev = fastmri.ifft2c(k_prev)

        delta_gt = build_delta_target(kspace, m_t, m_prev)

        step_dir = outdir / f"t_{t_scalar:02d}"
        ensure_dir(step_dir)

        save_gray(step_dir / "Mt.png", mask_2d(m_t), f"M_t @ {t_scalar}")
        save_gray(step_dir / "Mprev.png", mask_2d(m_prev), f"M_prev @ {int(t_prev.item())}")
        save_gray(step_dir / "DeltaM.png", mask_2d(delta_m.abs()), f"|DeltaM| @ {t_scalar}")

        save_gray(step_dir / "xt_abs.png", complex_abs_2d(x_t), f"x_t @ {t_scalar}")
        save_gray(step_dir / "xprev_abs.png", complex_abs_2d(x_prev), f"x_prev @ {int(t_prev.item())}")
        save_gray(step_dir / "kt_logmag.png", kspace_logmag_2d(k_t), f"log|k_t| @ {t_scalar}")
        save_gray(step_dir / "kprev_logmag.png", kspace_logmag_2d(k_prev), f"log|k_prev| @ {int(t_prev.item())}")
        save_gray(step_dir / "delta_gt_logmag.png", kspace_logmag_2d(delta_gt), f"log|delta_gt| @ {t_scalar}")

        # model one-step prediction
        x0_pred = model._run_backbone(k_t=k_t, k_c=k_c, m_t=m_t, t=t)
        k0_pred = fastmri.fft2c(x0_pred)
        delta_pred = delta_m * k0_pred

        x0_err = complex_err_abs_2d(x0_pred, x0)
        delta_err = complex_err_abs_2d(delta_pred, delta_gt)

        save_gray(step_dir / "x0_pred_abs.png", complex_abs_2d(x0_pred), f"x0_pred @ {t_scalar}")
        save_hot(step_dir / "x0_err.png", x0_err, f"|x0_pred-x0| @ {t_scalar}")
        save_gray(step_dir / "delta_pred_logmag.png", kspace_logmag_2d(delta_pred), f"log|delta_pred| @ {t_scalar}")
        save_hot(step_dir / "delta_err.png", delta_err, f"|delta_pred-delta_gt| @ {t_scalar}")

        psnr_x0 = image_psnr_shared_gt(x0, x0_pred)
        summary_lines.append(
            f"[forward t={t_scalar:02d}] "
            f"x0_pred_psnr_sharedGT={psnr_x0:.4f}"
        )

        save_npz(
            step_dir / "tensors.npz",
            k0=kspace,
            x0=x0,
            kc=k_c,
            Mt=m_t,
            Mprev=m_prev,
            DeltaM=delta_m,
            kt=k_t,
            kprev=k_prev,
            xt=x_t,
            xprev=x_prev,
            delta_gt=delta_gt,
            x0_pred=x0_pred,
            k0_pred=k0_pred,
            delta_pred=delta_pred,
        )

    with open(outdir / "forward_summary.txt", "w") as f:
        for line in summary_lines:
            f.write(line + "\n")


@torch.no_grad()
def debug_reverse(model, kspace, mask, outdir: Path):
    """
    打印 reverse:
      从 t=T 开始，每一步存
        cur_k vs gt_k_t
        x0_pred vs x0
        next_k vs gt_k_prev
        error map
    """
    ensure_dir(outdir)

    x0 = fastmri.ifft2c(kspace)
    k_c = model._build_conditional_kc(kspace, mask)

    T = model.num_timesteps
    t_T = torch.tensor([T], device=kspace.device, dtype=torch.long)
    m_T = model.schedule.get_by_t(t_T, device=kspace.device, dtype=kspace.dtype)

    # current implementation reverse init
    cur_k = m_T * k_c

    summary_lines = []
    summary_lines.append(f"reverse init uses k_T = M_T * k_c, T={T}")

    for t_scalar in range(T, 0, -1):
        t = torch.tensor([t_scalar], device=kspace.device, dtype=torch.long)
        t_prev = torch.clamp(t - 1, min=0)

        m_t = model.schedule.get_by_t(t, device=kspace.device, dtype=kspace.dtype)
        m_prev = model.schedule.get_by_t(t_prev, device=kspace.device, dtype=kspace.dtype)
        delta_m = m_prev - m_t

        gt_k_t = apply_filter_degradation(kspace, m_t)
        gt_k_prev = apply_filter_degradation(kspace, m_prev)

        cur_x = fastmri.ifft2c(cur_k)
        gt_x_t = fastmri.ifft2c(gt_k_t)
        gt_x_prev = fastmri.ifft2c(gt_k_prev)

        x0_pred = model._run_backbone(k_t=cur_k, k_c=k_c, m_t=m_t, t=t)
        delta_pred = delta_m * fastmri.fft2c(x0_pred)
        next_k = cur_k + delta_pred

        if model.use_explicit_dc:
            next_k_post = explicit_data_consistency(next_k, k_c, mask)
        else:
            next_k_post = next_k

        next_x = fastmri.ifft2c(next_k_post)

        step_dir = outdir / f"step_{t_scalar:02d}_to_{int(t_prev.item()):02d}"
        ensure_dir(step_dir)

        # save masks
        save_gray(step_dir / "Mt.png", mask_2d(m_t), f"M_t @ {t_scalar}")
        save_gray(step_dir / "Mprev.png", mask_2d(m_prev), f"M_prev @ {int(t_prev.item())}")
        save_gray(step_dir / "DeltaM.png", mask_2d(delta_m.abs()), f"|DeltaM|")

        # current state vs gt current
        save_gray(step_dir / "cur_x_abs.png", complex_abs_2d(cur_x), f"cur x_t @ {t_scalar}")
        save_gray(step_dir / "gt_xt_abs.png", complex_abs_2d(gt_x_t), f"gt x_t @ {t_scalar}")
        save_hot(step_dir / "cur_xt_err.png", complex_err_abs_2d(cur_x, gt_x_t), f"|cur_x-gt_x_t|")

        # x0 prediction
        save_gray(step_dir / "x0_abs.png", complex_abs_2d(x0), "x0")
        save_gray(step_dir / "x0_pred_abs.png", complex_abs_2d(x0_pred), "x0_pred")
        save_hot(step_dir / "x0_err.png", complex_err_abs_2d(x0_pred, x0), "|x0_pred-x0|")

        # delta
        delta_gt = build_delta_target(kspace, m_t, m_prev)
        save_gray(step_dir / "delta_gt_logmag.png", kspace_logmag_2d(delta_gt), "log|delta_gt|")
        save_gray(step_dir / "delta_pred_logmag.png", kspace_logmag_2d(delta_pred), "log|delta_pred|")
        save_hot(step_dir / "delta_err.png", complex_err_abs_2d(delta_pred, delta_gt), "|delta_pred-delta_gt|")

        # next state vs gt next
        save_gray(step_dir / "next_x_abs.png", complex_abs_2d(next_x), f"pred x_prev @ {int(t_prev.item())}")
        save_gray(step_dir / "gt_xprev_abs.png", complex_abs_2d(gt_x_prev), f"gt x_prev @ {int(t_prev.item())}")
        save_hot(step_dir / "next_err.png", complex_err_abs_2d(next_x, gt_x_prev), "|pred_xprev-gt_xprev|")

        psnr_cur = image_psnr_shared_gt(gt_x_t, cur_x)
        psnr_next = image_psnr_shared_gt(gt_x_prev, next_x)
        psnr_x0 = image_psnr_shared_gt(x0, x0_pred)

        summary_lines.append(
            f"[reverse {t_scalar:02d}->{int(t_prev.item()):02d}] "
            f"cur_psnr={psnr_cur:.4f} | next_psnr={psnr_next:.4f} | x0_pred_psnr={psnr_x0:.4f}"
        )

        save_npz(
            step_dir / "tensors.npz",
            k0=kspace,
            x0=x0,
            kc=k_c,
            Mt=m_t,
            Mprev=m_prev,
            DeltaM=delta_m,
            cur_k=cur_k,
            gt_k_t=gt_k_t,
            gt_k_prev=gt_k_prev,
            cur_x=cur_x,
            gt_x_t=gt_x_t,
            gt_x_prev=gt_x_prev,
            x0_pred=x0_pred,
            delta_gt=delta_gt,
            delta_pred=delta_pred,
            next_k=next_k_post,
            next_x=next_x,
        )

        cur_k = next_k_post

    final_x = fastmri.ifft2c(cur_k)
    save_gray(outdir / "final_recon_abs.png", complex_abs_2d(final_x), "final recon")
    save_gray(outdir / "gt_x0_abs.png", complex_abs_2d(x0), "gt x0")
    save_hot(outdir / "final_err.png", complex_err_abs_2d(final_x, x0), "|final-gt|")

    final_psnr = image_psnr_shared_gt(x0, final_x)
    summary_lines.append(f"[reverse final] final_psnr_sharedGT={final_psnr:.4f}")

    with open(outdir / "reverse_summary.txt", "w") as f:
        for line in summary_lines:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to ixi_swin.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path, e.g. best.pt")
    parser.add_argument("--idx", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--outdir", type=str, default="./debug_filterdiff_ixi")
    parser.add_argument("--t_list", type=str, default="0,1,5,10,19")
    parser.add_argument("--use_ema", action="store_true", help="Load ema weights if present")
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
        print("[load] using EMA weights")
        model.load_state_dict(ckpt["ema"], strict=True)
    elif "model" in ckpt:
        print("[load] using model weights")
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        raise KeyError("Checkpoint does not contain 'model' or 'ema'")

    model.eval()

    kspace, mask, mask_fold = dataset[args.idx]
    kspace = kspace.unsqueeze(0).to(device)       # [B,1,H,W,2]
    mask = mask.unsqueeze(0).to(device)           # [B,1,H,W]
    mask_fold = mask_fold.unsqueeze(0).to(device)

    outdir = Path(args.outdir) / f"{args.split}_idx_{args.idx:04d}"
    ensure_dir(outdir)

    print(f"[debug] save to: {outdir}")
    print(f"[debug] split={args.split}, idx={args.idx}")

    t_list = [int(x.strip()) for x in args.t_list.split(",") if x.strip() != ""]
    t_list = sorted(set(t_list))

    debug_forward(model, kspace, mask, outdir / "forward", t_list=t_list)
    debug_reverse(model, kspace, mask, outdir / "reverse")

    print("[done] forward and reverse debug saved.")


if __name__ == "__main__":
    main()