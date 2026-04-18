import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from utils.utils import dict2namespace, setup_seed
from utils.sample_mask import RandomMaskGaussianDiffusion, RandomMaskDiffusion, EquiSpaceMaskDiffusion
from utils.diffusion_utils import cycle

from data.ixi_singlecoil_dataset import IXISinglecoilSliceDataset
from diffusion.kspace_diffusion import KspaceDiffusion
from models.unet_diffusion import Unet
from models.restoration_net_filterdiff import build_filterdiff_restoration_net
from trainer import Trainer


def build_mask_func(config):
    size = (1, config.data.image_size, config.data.image_size)
    mask_type = config.data.mask_type
    seed = getattr(config.data, "seed", getattr(config, "seed", 42))

    if mask_type == 'gaussian_diffusion':
        return RandomMaskGaussianDiffusion(
            acceleration=config.data.R,
            center_fraction=config.data.center_fraction,
            size=size,
            seed=seed,
            patch_size=config.data.patch_size
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = device
    setup_seed(config.seed)

    mask_func = build_mask_func(config)

    train_dataset = IXISinglecoilSliceDataset(
        root=config.data.train_root,
        mask_func=mask_func,
        image_size=config.data.image_size,
        num_skip_slice=config.data.num_skip_slice,
        normalize_mode=config.data.normalize_mode,
    )
    val_dataset = IXISinglecoilSliceDataset(
        root=config.data.val_root,
        mask_func=mask_func,
        image_size=config.data.image_size,
        num_skip_slice=config.data.num_skip_slice,
        normalize_mode=config.data.normalize_mode,
    )
    test_dataset = IXISinglecoilSliceDataset(
        root=config.data.test_root,
        mask_func=mask_func,
        image_size=config.data.image_size,
        num_skip_slice=config.data.num_skip_slice,
        normalize_mode=config.data.normalize_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=False,
    )

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
        center_fraction=getattr(config.data, 'center_fraction', None),
        use_explicit_dc=getattr(config.training, 'use_explicit_dc', False),
    ).to(device)

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
    )

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.dataloader_test = test_loader
        trainer.dl_test = cycle(test_loader)
        trainer.test(t=config.training.timesteps, num_samples=1)


if __name__ == "__main__":
    main()