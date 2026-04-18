import argparse
import csv
import pathlib
import re
import sys

import torch
import yaml
from tqdm import tqdm

import fastmri

from utils.utils import dict2namespace
from utils.mri_data import SliceDataset
from utils.data_transform import DataTransform_Diffusion
from utils.sample_mask import (
    RandomMaskGaussianDiffusion,
    RandomMaskDiffusion,
    EquiSpaceMaskDiffusion,
)
from diffusion.kspace_diffusion import KspaceDiffusion
from models.unet_diffusion import Unet
from models.restoration_net_filterdiff import build_filterdiff_restoration_net
from trainer import eval_image_from_multicoil
from utils.evaluation import calc_nmse_tensor, calc_psnr_tensor, calc_ssim_tensor

def build_model(config, device):
    backbone = getattr(config.model, "backbone", "unet")

    if backbone == "unet":
        denoise_fn = Unet(
            dim=config.model.dim,
            out_dim=2,
            channels=5,  # [kt, kc, mt] -> 2 + 2 + 1 = 5
            dim_mults=tuple(config.model.dim_mults),
            with_time_emb=True,
            residual=config.model.residual,
        ).to(device)

    elif backbone == "swin_dits":
        denoise_fn = build_filterdiff_restoration_net(
            img_size=config.data.image_size,
            patch_size=getattr(config.model, "patch_size", 4),
            in_channels=5,
            out_channels=2,
            hidden_size=getattr(config.model, "hidden_size", 384),
            depth=getattr(config.model, "depth", 8),
            num_heads=getattr(config.model, "num_heads", 8),
            window_size=getattr(config.model, "window_size", 8),
            mlp_ratio=getattr(config.model, "mlp_ratio", 4.0),
            with_time_emb=True,
        ).to(device)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    center_core_size = getattr(
        config.training,
        "center_core_size",
        [config.data.image_size, max(1, int(round(config.data.image_size * float(config.data.center_fraction))))]
    )

    model = KspaceDiffusion(
        denoise_fn=denoise_fn,
        image_size=config.data.image_size,
        device_of_kernel=str(device),
        channels=2,
        timesteps=config.training.timesteps,
        loss_type=config.training.loss_type,
        schedule_type=getattr(config.training, "filter_schedule_type", "dense"),
        center_core_size=center_core_size,
        lambda_img=getattr(config.training, "lambda_img", 1.0),
        use_explicit_dc=getattr(config.training, "use_explicit_dc", False),
    ).to(device)

    return model

def build_dataset(config, data_root):
    size = (1, config.data.image_size, config.data.image_size)

    mask_type = config.data.mask_type
    if mask_type == "gaussian_diffusion":
        mask_func = RandomMaskGaussianDiffusion(
            acceleration=config.data.R,
            center_fraction=config.data.center_fraction,
            size=size,
            seed=config.data.seed,
            patch_size=config.data.patch_size,
        )
    elif mask_type == "random_diffusion":
        mask_func = RandomMaskDiffusion(
            center_fraction=config.data.center_fraction,
            acceleration=config.data.R,
            size=size,
            seed=config.data.seed,
        )
    elif mask_type == "equispace_diffusion":
        mask_func = EquiSpaceMaskDiffusion(
            center_fraction=config.data.center_fraction,
            acceleration=config.data.R,
            size=size,
            seed=config.data.seed,
        )
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")

    combine_coil = getattr(config.data, "combine_coil", True)

    data_transform = DataTransform_Diffusion(
        mask_func,
        img_size=config.data.image_size,
        combine_coil=combine_coil,
        flag_singlecoil=False,
        maps_root=getattr(config.data, "maps_root", None),
        map_key=getattr(config.data, "map_key", "s_maps"),
    )

    dataset = SliceDataset(
        root=pathlib.Path(data_root),
        transform=data_transform,
        challenge="multicoil",
        num_skip_slice=getattr(config.data, "num_skip_slice", 6),
    )
    return dataset, combine_coil

def evaluate_one_checkpoint(
    model,
    ckpt_path,
    dataset,
    combine_coil,
    device,
    t,
    max_cases=None,
):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        raise KeyError(f"Neither 'ema' nor 'model' found in checkpoint: {ckpt_path}")

    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_nmse = 0.0
    total_count = 0

    n_cases = len(dataset) if max_cases is None else min(len(dataset), max_cases)

    with torch.no_grad():
        for idx in tqdm(range(n_cases), desc=f"Eval {ckpt_path.name}", leave=False):
            sample = dataset[idx]
            if len(sample) == 3:
                kspace, mask, mask_fold = sample
                maps = None
            elif len(sample) == 4:
                kspace, mask, mask_fold, maps = sample
            else:
                raise ValueError(f"Unexpected dataset item length: {len(sample)}")

            kspace = kspace.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            mask_fold = mask_fold.unsqueeze(0).to(device)
            if maps is not None:
                maps = maps.unsqueeze(0).to(device)

            # sample 输入必须是 k_c
            k_c = kspace * mask.unsqueeze(-1)
            _, _, sample_imgs = model.sample(k_c, mask, mask_fold, t=t)

            gt_imgs = fastmri.ifft2c(kspace)

            gt_abs = eval_image_from_multicoil(gt_imgs, maps)         # [B,H,W]
            recon_abs = eval_image_from_multicoil(sample_imgs, maps)  # [B,H,W]

            B = gt_abs.shape[0]
            for i in range(B):
                total_nmse += float(calc_nmse_tensor(gt_abs[i], recon_abs[i]))
                total_psnr += float(calc_psnr_tensor(gt_abs[i], recon_abs[i]))
                total_ssim += float(calc_ssim_tensor(gt_abs[i], recon_abs[i]))
                total_count += 1

    return {
        "avg_psnr": total_psnr / total_count,
        "avg_ssim": total_ssim / total_count,
        "avg_nmse": total_nmse / total_count,
        "count": total_count,
    }


def extract_step(path_obj: pathlib.Path):
    m = re.search(r"(\d+)", path_obj.stem)
    if m is None:
        return -1
    return int(m.group(1))


def main():
    class DebugArgs:
        config = "configs/base.yaml"
        ckpt_dir = "/mnt/SSD/wsy/projects/filter_diffusion/results_filterdiff_dc"
        val_root = "/mnt/SSD/wsy/projects/HFS-SDE-master/data/multicoil_val/kspace"
        t = 20
        select_by = "psnr"
        max_cases = 20
        save_csv = "filterdiff_ckpt_scan.csv"
        pattern = "model_*.pt"
        stride = 100
        start_step = 100
        end_step = 10000

    if len(sys.argv) == 1:
        args = DebugArgs()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument("--ckpt_dir", type=str, required=True)
        parser.add_argument("--val_root", type=str, required=True)
        parser.add_argument("--t", type=int, default=20)
        parser.add_argument("--select_by", type=str, default="psnr", choices=["psnr", "ssim"])
        parser.add_argument("--max_cases", type=int, default=None)
        parser.add_argument("--save_csv", type=str, default="val_summary.csv")
        parser.add_argument("--pattern", type=str, default="model_*.pt")
        parser.add_argument("--stride", type=int, default=200)
        parser.add_argument("--start_step", type=int, default=200)
        parser.add_argument("--end_step", type=int, default=10 ** 9)
        args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, combine_coil = build_dataset(config, args.val_root)
    model = build_model(config, device)

    ckpt_dir = pathlib.Path(args.ckpt_dir)
    ckpt_paths_all = sorted(ckpt_dir.glob(args.pattern), key=extract_step)

    ckpt_paths = []
    for p in ckpt_paths_all:
        step = extract_step(p)
        if step < 0:
            continue
        if step < args.start_step or step > args.end_step:
            continue
        if step % args.stride != 0:
            continue
        ckpt_paths.append(p)

    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir} with pattern {args.pattern}")

    print(f"Found {len(ckpt_paths)} checkpoints.")
    print(f"Validation cases: {len(dataset) if args.max_cases is None else min(len(dataset), args.max_cases)}")

    rows = []
    best_metric = None
    best_row = None

    for ckpt_path in ckpt_paths:
        stats = evaluate_one_checkpoint(
            model=model,
            ckpt_path=ckpt_path,
            dataset=dataset,
            combine_coil=combine_coil,
            device=device,
            t=args.t,
            max_cases=args.max_cases,
        )

        row = {
            "ckpt": ckpt_path.name,
            "step": extract_step(ckpt_path),
            "avg_psnr": stats["avg_psnr"],
            "avg_ssim": stats["avg_ssim"],
            "avg_nmse": stats["avg_nmse"],
            "count": stats["count"],
        }
        rows.append(row)

        current_metric = row["avg_psnr"] if args.select_by == "psnr" else row["avg_ssim"]
        if best_metric is None or current_metric > best_metric:
            best_metric = current_metric
            best_row = row

        print(
            f"[{ckpt_path.name}] "
            f"PSNR={row['avg_psnr']:.4f}, "
            f"SSIM={row['avg_ssim']:.4f}, "
            f"NMSE={row['avg_nmse']:.6f}"
        )

    save_csv_path = pathlib.Path(args.save_csv)
    with open(save_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ckpt", "step", "avg_psnr", "avg_ssim", "avg_nmse", "count"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 80)
    print(f"Best checkpoint selected by {args.select_by.upper()}:")
    print(best_row)
    print(f"Saved summary to: {save_csv_path}")


if __name__ == "__main__":
    main()