import argparse
import pathlib

import fastmri
import numpy as np
import scipy.io as scio
import torch
import yaml
from tqdm import tqdm

from models.kspace_diffusion import KspaceDiffusion
from models.unet_diffusion import Unet
from utils.data_transform import DataTransform_Diffusion
from utils.mri_data import SliceDataset
from utils.sample_mask import RandomMaskGaussianDiffusion
from utils.utils import dict2namespace


def complex_abs_sq(x: torch.Tensor) -> torch.Tensor:
    return x[..., 0] ** 2 + x[..., 1] ** 2


def sense_combine(coil_imgs: torch.Tensor, maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    coil_imgs: [B, Nc, H, W, 2]
    maps:      [B, Nc, H, W, 2]
    return:    [B, H, W, 2]
    """
    num = fastmri.complex_mul(fastmri.complex_conj(maps), coil_imgs).sum(dim=1)  # [B,H,W,2]
    den = complex_abs_sq(maps).sum(dim=1).unsqueeze(-1) + eps                    # [B,H,W,1]
    return num / den


def magnitude_image_from_multicoil(
    coil_imgs: torch.Tensor,
    maps: torch.Tensor = None,
    fallback_mode: str = "mean",
) -> np.ndarray:
    """
    返回 [H,W] 的 magnitude 图。

    优先逻辑：
    - maps 不为 None：sensitivity combine
    - maps 为 None：
        - fallback_mode == 'mean' -> abs 后 coil 平均
        - fallback_mode == 'rss'  -> abs 后 RSS
    """
    if maps is not None:
        img = sense_combine(coil_imgs, maps)         # [B,H,W,2]
        abs_img = fastmri.complex_abs(img)           # [B,H,W]
    else:
        abs_coils = fastmri.complex_abs(coil_imgs)   # [B,Nc,H,W]
        if fallback_mode == "rss":
            abs_img = fastmri.rss(abs_coils, dim=1)  # [B,H,W]
        else:
            abs_img = torch.mean(abs_coils, dim=1)   # [B,H,W]

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./sample_results_matlab')
    parser.add_argument('--t', type=int, default=100)
    parser.add_argument('--fallback_mode', type=str, default='mean', choices=['mean', 'rss'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    size = (1, config.data.image_size, config.data.image_size)
    mask_func = RandomMaskGaussianDiffusion(
        acceleration=config.data.R,
        center_fraction=config.data.center_fraction,
        size=size,
        seed=config.data.seed,
        patch_size=config.data.patch_size,
    )

    combine_coil = getattr(config.data, 'combine_coil', True)

    data_transform = DataTransform_Diffusion(
        mask_func,
        img_size=config.data.image_size,
        combine_coil=combine_coil,
        flag_singlecoil=False,
        maps_root=getattr(config.data, 'maps_root', None),
        map_key=getattr(config.data, 'map_key', 's_maps'),
    )
    sample_dataset = SliceDataset(
        root=pathlib.Path(config.data.sample_root),
        transform=data_transform,
        challenge='multicoil',
        num_skip_slice=0,  # sample 时保留全部切片
    )

    denoise_fn = Unet(
        dim=config.model.dim,
        out_dim=2,
        channels=2,
        dim_mults=tuple(config.model.dim_mults),
        with_time_emb=True,
        residual=config.model.residual,
    ).to(device)

    model = KspaceDiffusion(
        denoise_fn=denoise_fn,
        image_size=config.data.image_size,
        device_of_kernel=str(device),
        channels=2,
        timesteps=config.training.timesteps,
        loss_type=config.training.loss_type,
        blur_routine=config.training.blur_routine,
        train_routine=config.training.train_routine,
        sampling_routine=config.training.sampling_routine,
        discrete=config.training.discrete,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['ema'])
    model.eval()

    save_path = pathlib.Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    case_names = []

    with torch.no_grad():
        for idx in tqdm(range(len(sample_dataset)), desc='Inference only'):
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

            _, _, sample_imgs = model.sample(kspace, mask, mask_fold, t=args.t)

            gt_abs = magnitude_image_from_multicoil(
                fastmri.ifft2c(kspace), maps, fallback_mode=args.fallback_mode
            )
            recon_abs = magnitude_image_from_multicoil(
                sample_imgs, maps, fallback_mode=args.fallback_mode
            )

            masked_kspace = apply_sampling_mask(kspace, mask)
            zf_abs = magnitude_image_from_multicoil(
                fastmri.ifft2c(masked_kspace), maps, fallback_mode=args.fallback_mode
            )

            mask_np = mask.squeeze().detach().cpu().numpy().astype(np.float32)

            save_dict = {
                'file_name': fname,
                'slice_idx': int(slice_idx),
                'zf': zf_abs.astype(np.float32),
                'recon': recon_abs.astype(np.float32),
                'gt': gt_abs.astype(np.float32),
                'mask': mask_np,
                'gt_max': np.array([gt_abs.max()], dtype=np.float32),
                'sampling_step': np.array([args.t], dtype=np.int32),
            }

            # 方便后面核对当前到底是用 maps combine 还是 fallback
            if maps is not None:
                save_dict['combine_mode'] = 'sense'
            else:
                save_dict['combine_mode'] = args.fallback_mode

            scio.savemat(save_path / f"{case_name}.mat", save_dict)

            case_names.append(case_name)
            print(f"[{idx + 1:03d}] saved {case_name}.mat")

    scio.savemat(
        save_path / 'manifest.mat',
        {
            'case_names': np.array(case_names, dtype=object),
            'num_cases': np.array([len(case_names)], dtype=np.int32),
            'sampling_step': np.array([args.t], dtype=np.int32),
        },
    )

    print(f"\nSaved {len(case_names)} cases to {save_path}")
    print(f"Manifest saved to {save_path / 'manifest.mat'}")


if __name__ == '__main__':
    main()