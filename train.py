# train.py
import argparse
import yaml
import torch
from data.cached_mri_data import CachedSliceDataset
from torch.utils.data import DataLoader
import pathlib
from utils.utils import dict2namespace, setup_seed
from data.mri_data import SliceDataset
from data.data_transform import DataTransform_Diffusion

from diffusion.kspace_diffusion import KspaceDiffusion
from models.unet_diffusion import Unet
from trainer import Trainer
from models.restoration_net_filterdiff import build_filterdiff_restoration_net

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


    acq_mask_path = getattr(config.data, 'acq_mask_path', None)
    acq_mask_key = getattr(config.data, 'acq_mask_key', 'mask')
    acq_mask_fold_path = getattr(config.data, 'acq_mask_fold_path', None)
    acq_mask_fold_key = getattr(config.data, 'acq_mask_fold_key', 'mask_fold')

    # ====== 数据变换 ======多线圈sense成单线圈
    combine_coil = config.data.combine_coil if hasattr(config.data, 'combine_coil') else True
    data_transform = DataTransform_Diffusion(
        img_size=config.data.image_size,
        combine_coil=combine_coil,
        flag_singlecoil=False,
        maps_root=getattr(config.data, 'maps_root', None),
        map_key=getattr(config.data, 'map_key', 's_maps'),
        device=None,
        fixed_mask_path=acq_mask_path,
        fixed_mask_key=acq_mask_key,
        fixed_mask_fold_path=acq_mask_fold_path,
        fixed_mask_fold_key=acq_mask_fold_key,
    )
    # ====== 训练数据集 ======
    num_skip_slice = config.data.num_skip_slice if hasattr(config.data, 'num_skip_slice') else 6
    raw_train_dataset = SliceDataset(
        root=pathlib.Path(config.data.data_root),
        transform=data_transform,
        challenge='multicoil',
        num_skip_slice=num_skip_slice,
    )

    train_dataset = CachedSliceDataset(
        original_dataset=raw_train_dataset,
        cache_root="/mnt/SSD/wsy/projects/filter_diffusion/cache/train_cache_24",
        slice_info_pkl="/mnt/SSD/wsy/projects/HFS-SDE-master/data/data_slice.pkl",
        force_rebuild=True,
        num_skip_slice=num_skip_slice,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    # ====== 验证数据集 ======
    # 验证数据集
    if hasattr(config.data, 'val_root') and config.data.val_root:
        raw_val_dataset = SliceDataset(
            root=pathlib.Path(config.data.val_root),
            transform=data_transform,
            challenge='multicoil',
            num_skip_slice=num_skip_slice,
        )
        val_dataset = CachedSliceDataset(
            original_dataset=raw_val_dataset,
            cache_root="/mnt/SSD/wsy/projects/filter_diffusion/cache/val_cache",
            slice_info_pkl="/mnt/SSD/wsy/projects/HFS-SDE-master/data/data_slice.pkl",  # 如果有独立的 val pkl，替换路径
            force_rebuild=True,  # 首次运行会自动构建，后续直接加载
            num_skip_slice=num_skip_slice,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # 缓存后单进程足够快
            pin_memory=False,
        )
    else:
        val_loader = None

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
            persistent_workers=True if config.data.num_workers > 0 else False,
        )
    else:
        test_loader = None
    # 构建去噪网络
    backbone = getattr(config.model, 'backbone', 'unet')

    if backbone == 'unet':
        denoise_fn = Unet(
            dim=config.model.dim,
            out_dim=2,
            channels=5,
            dim_mults=tuple(config.model.dim_mults),
            with_time_emb=True,
            residual=config.model.residual
        ).to(device)

    elif backbone == 'swin_dits':
        denoise_fn = build_filterdiff_restoration_net(
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


    # 构建扩散模型
    # r_min should align with acquisition center_fraction by default.
    center_core_size = getattr(
        config.training,
        'center_core_size',
        config.data.image_size // config.data.R  # 例如 256 // 4 = 64
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


        image_loss_mode=str(getattr(config.training, 'image_loss_mode', 'complex')),
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
        dataloader_test=val_loader,
        val_every=int(getattr(config.training, 'val_every', 500)),
        early_stop_patience=int(getattr(config.training, 'early_stop_patience', 10)),
        early_stop_min_delta=float(getattr(config.training, 'early_stop_min_delta', 1e-4)),
        monitor_metric=str(getattr(config.training, 'monitor_metric', 'psnr')),
        max_val_batches=int(getattr(config.training, 'max_val_batches', 20)),
        lr_scheduler_type=str(getattr(config.training, 'lr_scheduler_type', 'none')),
        warmup_steps=int(getattr(config.training, 'warmup_steps', 0)),
        min_lr=float(getattr(config.training, 'min_lr', 0.0)),
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