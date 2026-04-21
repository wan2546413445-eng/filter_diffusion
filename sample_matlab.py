import argparse
import pathlib

import fastmri
import numpy as np
import scipy.io as scio
import torch
import yaml
from tqdm import tqdm

from diffusion.kspace_diffusion import KspaceDiffusion
from models.unet_diffusion import Unet
from models.restoration_net_filterdiff import build_filterdiff_restoration_net
from data.data_transform import DataTransform_Diffusion
from data.mri_data import SliceDataset
from utils.sample_mask import (
    RandomMaskDiffusion,
    EquiSpaceMaskDiffusion,
    EquispacedCartesianMask,
)
from utils.utils import dict2namespace
from utils.evaluation import calc_nmse_tensor, calc_psnr_tensor, calc_ssim_tensor


def complex_abs_sq(x: torch.Tensor) -> torch.Tensor:
    return x[..., 0] ** 2 + x[..., 1] ** 2


def sense_combine(coil_imgs: torch.Tensor, maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    coil_imgs: [B, Nc, H, W, 2]
    maps:      [B, Nc, H, W, 2]
    return:    [B, H, W, 2]
    """
    num = fastmri.complex_mul(fastmri.complex_conj(maps), coil_imgs).sum(dim=1)
    den = complex_abs_sq(maps).sum(dim=1).unsqueeze(-1) + eps
    return num / den


def eval_image_from_multicoil(coil_imgs: torch.Tensor, maps: torch.Tensor = None) -> torch.Tensor:
    """
    与 trainer.py 评估口径保持一致：
    1) 多线圈 + maps: sense combine -> magnitude
    2) 多线圈无 maps: RSS magnitude
    3) 单图复数: magnitude
    return: [B, H, W]
    """
    if maps is not None:
        img = sense_combine(coil_imgs, maps)   # [B,H,W,2]
        img_abs = fastmri.complex_abs(img)     # [B,H,W]
    else:
        img_abs = fastmri.complex_abs(coil_imgs)  # [B,Nc,H,W]
        if img_abs.shape[1] == 1:
            img_abs = img_abs[:, 0]
        else:
            img_abs = fastmri.rss(img_abs, dim=1)
    return img_abs


def magnitude_image_from_multicoil(coil_imgs: torch.Tensor, maps: torch.Tensor = None) -> np.ndarray:
    """
    返回 [H,W] 的 magnitude 图，评估口径与 trainer.py 一致
    """
    abs_img = eval_image_from_multicoil(coil_imgs, maps)
    return abs_img[0].detach().cpu().numpy().astype(np.float32)


def apply_sampling_mask(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_to_use = mask
    while mask_to_use.ndim < kspace.ndim - 1:
        mask_to_use = mask_to_use.unsqueeze(-1)
    if mask_to_use.ndim == kspace.ndim - 1:
        mask_to_use = mask_to_use.unsqueeze(-1)
    return kspace * mask_to_use


def unpack_sample(sample):
    """
    兼容：
    1) kspace, mask, mask_fold
    2) kspace, mask, mask_fold, maps
    """
    if len(sample) == 3:
        kspace, mask, mask_fold = sample
        maps = None
    elif len(sample) == 4:
        kspace, mask, mask_fold, maps = sample
    else:
        raise ValueError(f"Unexpected sample length: {len(sample)}")
    return kspace, mask, mask_fold, maps


def build_mask_func(config):
    size = (1, config.data.image_size, config.data.image_size)
    mask_type = config.data.mask_type

    if mask_type == "equispaced_cartesian":
        return EquispacedCartesianMask(
            acceleration=config.data.R,
            center_fraction=config.data.center_fraction,
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
    elif mask_type == "equispace_diffusion":
        return EquiSpaceMaskDiffusion(
            center_fraction=config.data.center_fraction,
            acceleration=config.data.R,
            size=size,
            seed=config.data.seed,
        )
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")


def build_backbone(config, device):
    backbone = getattr(config.model, "backbone", "unet")

    if backbone == "unet":
        denoise_fn = Unet(
            dim=config.model.dim,
            out_dim=2,
            channels=5,
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

    return denoise_fn


def resolve_sample_root(config, split: str):
    if split == "val":
        root = getattr(config.data, "val_root", None)
        if root is None:
            raise ValueError("Requested split=val, but config.data.val_root is missing.")
        return root

    if split == "test":
        root = getattr(config.data, "test_root", None)
        if root is None:
            raise ValueError("Requested split=test, but config.data.test_root is missing.")
        return root

    if split == "sample":
        root = getattr(config.data, "sample_root", None)
        if root is None:
            raise ValueError("Requested split=sample, but config.data.sample_root is missing.")
        return root

    # auto: 优先 val_root，方便直接看验证结果
    for key in ["val_root", "sample_root", "test_root"]:
        root = getattr(config.data, key, None)
        if root is not None:
            return root

    raise ValueError("config.data.val_root / sample_root / test_root are all missing.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./sample_results_matlab")
    parser.add_argument("--t", type=int, default=None, help="sampling steps; default uses config.training.timesteps")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "val", "test", "sample"],
        help="auto will prefer val_root first",
    )
    parser.add_argument(
        "--keep_all_slices",
        action="store_true",
        help="if set, num_skip_slice=0; otherwise align with train/val setting",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = device

    mask_func = build_mask_func(config)
    combine_coil = getattr(config.data, "combine_coil", True)

    data_transform = DataTransform_Diffusion(
        mask_func,
        img_size=config.data.image_size,
        combine_coil=combine_coil,
        flag_singlecoil=False,
        maps_root=getattr(config.data, "maps_root", None),
        map_key=getattr(config.data, "map_key", "s_maps"),
    )

    sample_root = resolve_sample_root(config, args.split)

    if args.keep_all_slices:
        num_skip_slice = 0
    else:
        num_skip_slice = config.data.num_skip_slice if hasattr(config.data, "num_skip_slice") else 6

    sample_dataset = SliceDataset(
        root=pathlib.Path(sample_root),
        transform=data_transform,
        challenge="multicoil",
        num_skip_slice=num_skip_slice,
    )

    denoise_fn = build_backbone(config, device)

    center_core_size = getattr(
        config.training,
        "center_core_size",
        config.data.image_size // config.data.R
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
        image_loss_mode=str(getattr(config.training, "image_loss_mode", "complex")),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("[INFO] load ema weights")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("[INFO] load model weights")
    else:
        raise KeyError("Checkpoint must contain key 'ema' or 'model'.")

    model.eval()

    save_path = pathlib.Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    case_names = []
    case_metrics = []

    num_cases = len(sample_dataset) if args.max_cases is None else min(len(sample_dataset), args.max_cases)
    sampling_t = config.training.timesteps if args.t is None else args.t

    with torch.no_grad():
        for idx in tqdm(range(num_cases), desc="Inference only"):
            fname, slice_idx, _ = sample_dataset.raw_samples[idx]
            fname = pathlib.Path(fname).stem
            case_name = f"{fname}_slice{slice_idx:03d}"

            sample = sample_dataset[idx]
            kspace, mask, mask_fold, maps = unpack_sample(sample)

            kspace = kspace.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            mask_fold = mask_fold.unsqueeze(0).to(device)
            if maps is not None:
                maps = maps.unsqueeze(0).to(device)

            # sample 输入必须是欠采样 k-space
            k_c = apply_sampling_mask(kspace, mask)
            _, _, sample_imgs = model.sample(k_c, mask, mask_fold, t=sampling_t)

            gt_imgs = fastmri.ifft2c(kspace)
            zf_imgs = fastmri.ifft2c(k_c)

            gt_abs_torch = eval_image_from_multicoil(gt_imgs, maps)
            recon_abs_torch = eval_image_from_multicoil(sample_imgs, maps)
            zf_abs_torch = eval_image_from_multicoil(zf_imgs, maps)

            gt_abs = gt_abs_torch[0].detach().cpu().numpy().astype(np.float32)
            recon_abs = recon_abs_torch[0].detach().cpu().numpy().astype(np.float32)
            zf_abs = zf_abs_torch[0].detach().cpu().numpy().astype(np.float32)

            psnr = float(calc_psnr_tensor(gt_abs_torch[0], recon_abs_torch[0]))
            ssim = float(calc_ssim_tensor(gt_abs_torch[0], recon_abs_torch[0]))
            nmse = float(calc_nmse_tensor(gt_abs_torch[0], recon_abs_torch[0]))

            mask_np = mask.squeeze().detach().cpu().numpy().astype(np.float32)

            save_dict = {
                "file_name": fname,
                "slice_idx": int(slice_idx),
                "zf": zf_abs,
                "recon": recon_abs,
                "gt": gt_abs,
                "mask": mask_np,
                "gt_max": np.array([gt_abs.max()], dtype=np.float32),
                "recon_max": np.array([recon_abs.max()], dtype=np.float32),
                "zf_max": np.array([zf_abs.max()], dtype=np.float32),
                "sampling_step": np.array([sampling_t], dtype=np.int32),
                "psnr": np.array([psnr], dtype=np.float32),
                "ssim": np.array([ssim], dtype=np.float32),
                "nmse": np.array([nmse], dtype=np.float32),
            }

            save_dict["combine_mode"] = "sense" if maps is not None else "rss"

            scio.savemat(save_path / f"{case_name}.mat", save_dict)
            case_names.append(case_name)
            case_metrics.append([psnr, ssim, nmse])

            print(
                f"[{idx + 1:03d}] saved {case_name}.mat | "
                f"PSNR={psnr:.4f} | SSIM={ssim:.4f} | NMSE={nmse:.6f}"
            )

    case_metrics = np.array(case_metrics, dtype=np.float32) if len(case_metrics) > 0 else np.zeros((0, 3), dtype=np.float32)

    scio.savemat(
        save_path / "manifest.mat",
        {
            "case_names": np.array(case_names, dtype=object),
            "num_cases": np.array([len(case_names)], dtype=np.int32),
            "sampling_step": np.array([sampling_t], dtype=np.int32),
            "split": np.array([args.split], dtype=object),
            "root_used": np.array([str(sample_root)], dtype=object),
            "num_skip_slice": np.array([num_skip_slice], dtype=np.int32),
            "metrics": case_metrics,  # columns: [psnr, ssim, nmse]
        },
    )

    print(f"\nSaved {len(case_names)} cases to {save_path}")
    print(f"Manifest saved to {save_path / 'manifest.mat'}")


if __name__ == "__main__":
    main()