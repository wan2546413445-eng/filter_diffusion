#!/usr/bin/env python3
"""Phase-0 diagnostics for FilterDiff MRI pipeline.

Usage:
  python stage0_diagnostics.py --config configs/ixi_swin.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml
import fastmri

from utils.utils import dict2namespace
from utils.sample_mask import EquispacedCartesianMask
from data.ixi_singlecoil_dataset import IXISinglecoilSliceDataset
from diffusion.filter_schedule import CenterRectangleSchedule, _ratio_at_t
from diffusion.kspace_diffusion import KspaceDiffusion
from models.unet_diffusion import Unet


try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def _load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return dict2namespace(cfg)


def _count_nii(root: str) -> int:
    p = Path(root)
    if not p.exists():
        return -1
    return len(sorted(p.glob("*.nii.gz")))


def _is_t1_name(path: Path) -> bool:
    name = path.name.lower()
    return ("t1" in name) or ("mprage" in name)


def check_data_pipeline(cfg) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    train_root = getattr(cfg.data, "train_root", None)
    val_root = getattr(cfg.data, "val_root", None)
    test_root = getattr(cfg.data, "test_root", None)

    out["train_count"] = _count_nii(train_root) if train_root else -1
    out["val_count"] = _count_nii(val_root) if val_root else -1
    out["test_count"] = _count_nii(test_root) if test_root else -1

    out["split_ok_500_37_40"] = (
        out["train_count"] == 500 and out["val_count"] == 37 and out["test_count"] == 40
    )

    t1_hits = 0
    if train_root and Path(train_root).exists():
        train_files = sorted(Path(train_root).glob("*.nii.gz"))
        t1_hits = sum(_is_t1_name(p) for p in train_files[: min(20, len(train_files))])
    out["t1_name_hits_in_first20_train"] = t1_hits

    # mask behavior check (4x / 8x)
    H = W = int(cfg.data.image_size)
    center_fraction = float(cfg.data.center_fraction)
    mask_stats = {}
    for acc in (4, 8):
        mk = EquispacedCartesianMask(
            acceleration=acc,
            center_fraction=center_fraction,
            size=(1, H, W),
            seed=getattr(cfg.data, "seed", 42),
        )
        mask, _ = mk()
        mask = torch.from_numpy(mask)
        center_cols = int(round(W * center_fraction))
        mask_stats[f"acc{acc}"] = {
            "shape": tuple(mask.shape),
            "sum": float(mask.sum().item()),
            "center_keep_cols": center_cols,
            "effective_rate": float(mask.mean().item()),
        }
    out["mask_stats"] = mask_stats

    # FFT norm check from fastmri implementation behavior (ortho)
    x = torch.randn(1, H, W, 2)
    k = fastmri.fft2c(x)
    x_back = fastmri.ifft2c(k)
    out["fft_roundtrip_maxerr_fastmri"] = float((x_back - x).abs().max().item())

    # sample-level check when dataset exists
    if train_root and Path(train_root).exists() and out["train_count"] > 0:
        mask_func = EquispacedCartesianMask(
            acceleration=int(cfg.data.R),
            center_fraction=float(cfg.data.center_fraction),
            size=(1, H, W),
            seed=getattr(cfg.data, "seed", 42),
        )
        ds = IXISinglecoilSliceDataset(
            root=train_root,
            mask_func=mask_func,
            image_size=H,
            num_skip_slice=int(cfg.data.num_skip_slice),
            normalize_mode=str(cfg.data.normalize_mode),
            return_dict=True,
        )
        s = ds[0]
        k0 = s["kspace"]
        mask = s["mask"].unsqueeze(-1)
        kc = k0 * mask

        out["sample_stats"] = {
            "k0_shape": tuple(k0.shape),
            "mask_shape": tuple(s["mask"].shape),
            "target_shape": tuple(s["target"].shape),
            "k0_min": float(k0.min().item()),
            "k0_max": float(k0.max().item()),
            "kc_vs_k0mask_maxdiff": float((kc - k0 * mask).abs().max().item()),
            "fft_ifft_roundtrip_img_maxdiff": float(
                (fastmri.ifft2c(fastmri.fft2c(torch.stack([s["target"], torch.zeros_like(s["target"])], dim=-1)))
                 - torch.stack([s["target"], torch.zeros_like(s["target"])], dim=-1)).abs().max().item()
            ),
        }

        # degradation demo
        m4, _ = EquispacedCartesianMask(4, center_fraction, size=(1, H, W), seed=42)()
        m4 = torch.from_numpy(m4).unsqueeze(-1)
        x0 = torch.stack([s["target"], torch.zeros_like(s["target"])], dim=-1)
        k0_full = fastmri.fft2c(x0)
        x_deg = fastmri.ifft2c(k0_full * m4)

        if HAS_MPL:
            vis_dir = Path("diagnostics")
            vis_dir.mkdir(exist_ok=True)
            fig = plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(s["target"][0].cpu().numpy(), cmap="gray")
            plt.title("x0")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(fastmri.complex_abs(x_deg)[0].cpu().numpy(), cmap="gray")
            plt.title("iFFT(M*k0)")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            err = (s["target"] - fastmri.complex_abs(x_deg)).abs()[0].cpu().numpy()
            plt.imshow(err, cmap="magma")
            plt.title("abs error")
            plt.axis("off")
            out_png = vis_dir / "stage0_degrade_check.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            out["degrade_vis"] = str(out_png)

    return out


def check_schedule(cfg) -> Dict[str, Any]:
    H = W = int(cfg.data.image_size)
    T = int(cfg.training.timesteps)
    core = int(getattr(cfg.training, "center_core_size", round(W * float(cfg.data.center_fraction))))
    sch = CenterRectangleSchedule(H, W, T, core, getattr(cfg.training, "filter_schedule_type", "dense"))

    out: Dict[str, Any] = {"rows": []}
    rt_vals = []
    for t in range(T + 1):
        mt = sch.get_by_t(torch.tensor([t]))
        r_t = float(mt.sum().item() / mt.numel())
        rt_vals.append(r_t)
        out["rows"].append((t, r_t, float(mt.sum().item())))

    delta_rows = []
    nonneg = True
    for t in range(1, T + 1):
        mt = sch.get_by_t(torch.tensor([t]))
        mt_prev = sch.get_by_t(torch.tensor([t - 1]))
        d = mt_prev - mt
        nonneg = nonneg and bool((d >= 0).all().item())
        delta_rows.append((t, float(d.sum().item()), float(d.min().item()), float(d.max().item())))

    out["delta_rows"] = delta_rows
    out["delta_nonneg"] = nonneg

    # expected profile curve
    ts = np.arange(0, T + 1)
    r_min = core / float(W)
    profile = np.array([_ratio_at_t(t, T, getattr(cfg.training, "filter_schedule_type", "dense"), r_min) for t in ts])

    if HAS_MPL:
        vis_dir = Path("diagnostics")
        vis_dir.mkdir(exist_ok=True)
        fig = plt.figure(figsize=(5, 3))
        plt.plot(ts, rt_vals, marker="o", label="mask ratio from M_t")
        plt.plot(ts, profile, linestyle="--", label="_ratio_at_t")
        plt.xlabel("t")
        plt.ylabel("r_t")
        plt.title("Schedule ratio curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_png = vis_dir / "stage0_schedule_curve.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        out["curve_png"] = str(out_png)
    else:
        out["curve_png"] = "N/A (matplotlib not installed)"
    return out


def build_tiny_model(cfg, device):
    net = Unet(dim=16, out_dim=2, channels=5, dim_mults=(1, 2), with_time_emb=True, residual=False).to(device)
    model = KspaceDiffusion(
        denoise_fn=net,
        image_size=int(cfg.data.image_size),
        device_of_kernel=str(device),
        channels=2,
        timesteps=int(cfg.training.timesteps),
        loss_type="l1",
        schedule_type=getattr(cfg.training, "filter_schedule_type", "dense"),
        center_core_size=int(getattr(cfg.training, "center_core_size", round(int(cfg.data.image_size) * float(cfg.data.center_fraction)))),
        lambda_img=1.0,
        image_loss_mode="complex",
    ).to(device)
    return model


def check_single_step_and_reverse(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, Nc = 2, 1
    H = W = int(cfg.data.image_size)
    T = int(cfg.training.timesteps)

    model = build_tiny_model(cfg, device)
    model.train()

    k0 = torch.randn(B, Nc, H, W, 2, device=device)
    mask = (torch.rand(B, 1, H, W, device=device) > 0.75).float()

    t = torch.randint(1, T + 1, (B,), device=device)
    mt = model.schedule.get_by_t(t, device=device, dtype=k0.dtype)
    mt_prev = model.schedule.get_by_t(torch.clamp(t - 1, min=0), device=device, dtype=k0.dtype)
    delta_mask = mt_prev - mt

    kc = k0 * mask.unsqueeze(-1)
    kt = k0 * mt
    x0 = fastmri.ifft2c(k0)
    x0_pred = model._run_backbone(kt, kc, mt, t)

    k0_pred = fastmri.fft2c(x0_pred)
    delta_pred = delta_mask * k0_pred
    delta_gt = delta_mask * k0

    loss_k = torch.mean(torch.abs(delta_pred - delta_gt))
    loss_x = torch.mean(torch.abs(x0_pred - x0))

    reverse_stats = []
    cur_k = model.schedule.get_by_t(torch.full((B,), T, dtype=torch.long, device=device), device=device, dtype=k0.dtype) * kc
    ok = True
    for ts in range(T, 0, -1):
        mt_s = model.schedule.get_by_t(torch.full((B,), ts, dtype=torch.long, device=device), device=device, dtype=k0.dtype)
        x_pred = model._run_backbone(cur_k, kc, mt_s, torch.full((B,), ts, dtype=torch.long, device=device))
        delta = (model.schedule.get_by_t(torch.full((B,), ts - 1, dtype=torch.long, device=device), device=device, dtype=k0.dtype) - mt_s) * fastmri.fft2c(x_pred)
        cur_k = cur_k + delta
        vmin, vmax = float(cur_k.min().item()), float(cur_k.max().item())
        reverse_stats.append((ts, vmin, vmax))
        if not torch.isfinite(cur_k).all():
            ok = False
            break

    out_img = fastmri.ifft2c(cur_k)

    return {
        "kt_shape": tuple(kt.shape),
        "kt_min": float(kt.min().item()),
        "kt_max": float(kt.max().item()),
        "x0_pred_shape": tuple(x0_pred.shape),
        "x0_pred_min": float(x0_pred.min().item()),
        "x0_pred_max": float(x0_pred.max().item()),
        "k0_pred_min": float(k0_pred.min().item()),
        "k0_pred_max": float(k0_pred.max().item()),
        "delta_pred_min": float(delta_pred.min().item()),
        "delta_pred_max": float(delta_pred.max().item()),
        "delta_gt_min": float(delta_gt.min().item()),
        "delta_gt_max": float(delta_gt.max().item()),
        "loss_k": float(loss_k.item()),
        "loss_x": float(loss_x.item()),
        "loss_ratio": float(loss_k.item() / (loss_x.item() + 1e-10)),
        "loss_finite": bool(torch.isfinite(loss_k) and torch.isfinite(loss_x)),
        "reverse_ok": ok,
        "reverse_stats": reverse_stats,
        "final_img_shape": tuple(out_img.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ixi_swin.yaml")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    print("[阶段0.1 数据管线检查]")
    d = check_data_pipeline(cfg)
    for k, v in d.items():
        print(f"- {k}: {v}")

    print("\n[阶段0.2 掩膜调度检查]")
    s = check_schedule(cfg)
    for t, r_t, ssum in s["rows"]:
        print(f"t={t:2d}, r_t={r_t:.6f}, mask_sum={ssum:.0f}")
    for t, dsum, dmin, dmax in s["delta_rows"]:
        print(f"ΔM t={t:2d}: sum={dsum:.0f}, min={dmin:.3f}, max={dmax:.3f}")
    print(f"delta_nonneg={s['delta_nonneg']}")
    print(f"curve_png={s['curve_png']}")

    print("\n[阶段0.3/0.4 单步loss + reverse检查]")
    l = check_single_step_and_reverse(cfg)
    for k, v in l.items():
        if k != "reverse_stats":
            print(f"- {k}: {v}")
    print("- reverse per-step ranges:")
    for ts, vmin, vmax in l["reverse_stats"]:
        print(f"  t={ts:2d}: [{vmin:.6f}, {vmax:.6f}]")


if __name__ == "__main__":
    main()
