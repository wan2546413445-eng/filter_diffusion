import argparse
import yaml
import torch
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from utils.utils import dict2namespace
from utils.mri_data import SliceDataset
from utils.data_transform import DataTransform_Diffusion
from utils.sample_mask import RandomMaskGaussianDiffusion
from models.kspace_diffusion import KspaceDiffusion
from models.unet_diffusion import Unet
import fastmri

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--idx', type=int, default=0, help='Index of test sample')
    parser.add_argument('--t', type=int, default=100, help='Diffusion steps')
    parser.add_argument('--save_dir', type=str, default='./vis')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    size = (1, config.data.image_size, config.data.image_size)
    mask_func = RandomMaskGaussianDiffusion(
        acceleration=config.data.R,
        center_fraction=config.data.center_fraction,
        size=size,
        seed=config.data.seed,
        patch_size=config.data.patch_size
    )
    combine_coil = getattr(config.data, 'combine_coil', True)
    data_transform = DataTransform_Diffusion(
        mask_func,
        img_size=config.data.image_size,
        combine_coil=combine_coil,
        flag_singlecoil=False,
    )
    test_dataset = SliceDataset(
        root=pathlib.Path(config.data.test_root),
        transform=data_transform,
        challenge='multicoil',
        num_skip_slice=getattr(config.data, 'num_skip_slice', 5),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 构建模型
    denoise_fn = Unet(
        dim=config.model.dim,
        out_dim=2,
        channels=2,
        dim_mults=tuple(config.model.dim_mults),
        with_time_emb=True,
        residual=config.model.residual
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
        discrete=config.training.discrete
    ).to(device)

    # 加载权重（使用 EMA 模型）
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['ema'])
    model.eval()

    # 获取指定样本
    for i, (kspace, mask, mask_fold) in enumerate(test_loader):
        if i != args.idx:
            continue
        kspace = kspace.to(device)
        mask = mask.to(device)
        mask_fold = mask_fold.to(device)

        with torch.no_grad():
            xt, direct_recons, sample_imgs = model.sample(kspace, mask, mask_fold, t=args.t)

        # 计算幅值图像
        gt_abs = fastmri.complex_abs(fastmri.ifft2c(kspace))          # [1, Nc, H, W]
        if combine_coil:
            gt_abs = fastmri.rss(gt_abs, dim=1)                       # [1, H, W]
        else:
            gt_abs = gt_abs[0, 0]                                     # 取第一个线圈
        gt_abs = gt_abs[0].cpu().numpy()

        recon_abs = fastmri.complex_abs(sample_imgs)                   # [1, Nc, H, W]
        if combine_coil:
            recon_abs = fastmri.rss(recon_abs, dim=1)
        else:
            recon_abs = recon_abs[0, 0]
        recon_abs = recon_abs[0].cpu().numpy()

        zero_filled = fastmri.complex_abs(fastmri.ifft2c(kspace * mask[..., None]))
        if combine_coil:
            zero_filled = fastmri.rss(zero_filled, dim=1)
        else:
            zero_filled = zero_filled[0, 0]
        zero_filled = zero_filled[0].cpu().numpy()

        # 归一化显示
        def norm(x):
            x = x - x.min()
            x = x / (x.max() + 1e-8)
            return x

        gt_norm = norm(gt_abs)
        zero_norm = norm(zero_filled)
        recon_norm = norm(recon_abs)

        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gt_norm, cmap='gray')
        axes[0].set_title('Fully Sampled')
        axes[1].imshow(zero_norm, cmap='gray')
        axes[1].set_title('Zero-filled')
        axes[2].imshow(recon_norm, cmap='gray')
        axes[2].set_title('Cold Diffusion')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        save_path = pathlib.Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        out_file = save_path / f'recon_sample_{args.idx}.png'
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"Saved {out_file}")
        break

if __name__ == '__main__':
    main()
