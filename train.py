# train.py
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import pathlib
from utils.utils import dict2namespace, setup_seed
from utils.mri_data import SliceDataset
from utils.data_transform import DataTransform_Diffusion
from utils.sample_mask import RandomMaskGaussianDiffusion, RandomMaskDiffusion, EquiSpaceMaskDiffusion
from diffusion.kspace_diffusion import KspaceDiffusion
from models.unet_diffusion import Unet
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path for testing')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)  # 读配置文件

    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = device

    # 固定随机种子
    setup_seed(config.seed)

    # ====== 初始化掩码生成器 ======
    size = (1, config.data.image_size, config.data.image_size)
    mask_type = config.data.mask_type
    if mask_type == 'gaussian_diffusion':
        mask_func = RandomMaskGaussianDiffusion(
            acceleration=config.data.R,
            center_fraction=config.data.center_fraction,
            size=size,
            seed=config.data.seed,
            patch_size=config.data.patch_size
        )
    elif mask_type == 'random_diffusion':
        mask_func = RandomMaskDiffusion(
            center_fraction=config.data.center_fraction,
            acceleration=config.data.R,
            size=size,
            seed=config.data.seed
        )
    elif mask_type == 'equispace_diffusion':
        mask_func = EquiSpaceMaskDiffusion(
            center_fraction=config.data.center_fraction,
            acceleration=config.data.R,
            size=size,
            seed=config.data.seed
        )
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")

    # ====== 数据变换 ======
    combine_coil = config.data.combine_coil if hasattr(config.data, 'combine_coil') else True
    data_transform = DataTransform_Diffusion(
        mask_func,
        img_size=config.data.image_size,
        combine_coil=combine_coil,
        flag_singlecoil=False,  # 多线圈数据
        maps_root=getattr(config.data, 'maps_root', None),
        map_key=getattr(config.data, 'map_key', 's_maps'),
    )
    # ====== 训练数据集 ======
    num_skip_slice = config.data.num_skip_slice if hasattr(config.data, 'num_skip_slice') else 6
    train_dataset = SliceDataset(
        root=pathlib.Path(config.data.data_root),
        transform=data_transform,
        challenge='multicoil',
        num_skip_slice=num_skip_slice,  # 一般train和test做跳过前六操作默认设置，配置文件中num_skip_slice 可控制除了6以外的其他跳过切片数量
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=False,
    )

    # ====== 测试数据集 ======
    if hasattr(config.data, 'test_root') and config.data.test_root:
        test_dataset = SliceDataset(
            root=pathlib.Path(config.data.test_root),
            transform=data_transform,
            challenge='multicoil',
            num_skip_slice=num_skip_slice,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=False,
        )
    else:
        test_loader = None

    # 构建去噪网络
    denoise_fn = Unet(
        dim=config.model.dim,
        out_dim=2,
        channels=5,
        dim_mults=tuple(config.model.dim_mults),
        with_time_emb=True,
        residual=config.model.residual
    ).to(device)

    '''
    denoise_fn = build_filterdiff_restoration_net(
        img_size=config.data.image_size,
        patch_size=4,
        in_channels=5,
        out_channels=2,
        hidden_size=384,
        depth=8,
        num_heads=8,
        window_size=8,
        mlp_ratio=4.0,
        with_time_emb=True,
    ).to(device)
    '''
    # 构建扩散模型
    # r_min should align with acquisition center_fraction by default.
    center_core_size = getattr(
        config.training,
        'center_core_size',
        [config.data.image_size, max(1, int(round(config.data.image_size * float(config.data.center_fraction))))]
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
        lambda_img=getattr(config.training, 'lambda_img', 1.0),
        use_explicit_dc=getattr(config.training, 'use_explicit_dc', False),
    ).to(device)
    # 网络结构不等于扩散逻辑！！u-net作为去噪/预测网络，可以修改，KspaceDiffusion是扩散框架
    # 创建 Trainer
    trainer = Trainer(
        diffusion_model=model,
        ema_decay=config.training.ema_decay,
        image_size=config.data.image_size,
        train_batch_size=config.training.batch_size,
        train_lr=float(config.training.lr),
        train_num_steps=int(config.training.train_num_steps),
        gradient_accumulate_every=int(config.training.gradient_accumulate_every),
        fp16=config.training.fp16,
        step_start_ema=int(config.training.step_start_ema),
        update_ema_every=int(config.training.update_ema_every),
        save_and_sample_every=int(config.training.save_and_sample_every),
        results_folder=config.training.results_folder,
        load_path=args.ckpt if args.mode == 'test' else None,
        dataloader_train=train_loader,
        dataloader_test=test_loader,
        val_every=int(getattr(config.training, 'val_every', 500)),
        early_stop_patience=int(getattr(config.training, 'early_stop_patience', 10)),
        early_stop_min_delta=float(getattr(config.training, 'early_stop_min_delta', 1e-4)),
        monitor_metric=str(getattr(config.training, 'monitor_metric', 'psnr')),
    )
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        if test_loader is None:
            print("No test dataloader provided. Please set data.test_root in config.")
        else:
            trainer.test(t=config.training.timesteps, num_samples=1)


if __name__ == "__main__":
    main()