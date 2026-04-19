# trainer.py
import math
import copy
import torch
import torch.nn.functional as F
from functools import partial
from pathlib import Path
from torch.optim import AdamW
from tqdm import tqdm
import fastmri
import os
import errno
from collections import OrderedDict
import sys
from utils.diffusion_utils import cycle, EMA, loss_backwards
from utils.evaluation import calc_nmse_tensor, calc_psnr_tensor, calc_ssim_tensor


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        name = k.replace('.module', '')
        new_state_dict[name] = v
    return new_state_dict


def adjust_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        name = k.replace('denoise_fn.module', 'module.denoise_fn')
        new_state_dict[name] = v
    return new_state_dict


def complex_abs_sq(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., 2]
    return: [...]
    """
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

def eval_image_from_multicoil(coil_imgs: torch.Tensor, maps: torch.Tensor = None) -> torch.Tensor:
    """
    统一评估口径，兼容：
    1) 多线圈 + maps: [B, Nc, H, W, 2]
    2) 多线圈无 maps: [B, Nc, H, W, 2]
    3) 单图复数:      [B, 1, H, W, 2]

    return:
        [B, H, W] magnitude image
    """
    if maps is not None:
        img = sense_combine(coil_imgs, maps)          # [B,H,W,2]
        img_abs = fastmri.complex_abs(img)            # [B,H,W]
    else:
        img_abs = fastmri.complex_abs(coil_imgs)      # [B,Nc,H,W]
        if img_abs.shape[1] == 1:
            img_abs = img_abs[:, 0]                   # [B,H,W]
        else:
            img_abs = fastmri.rss(img_abs, dim=1)     # [B,H,W]
    return img_abs

def unpack_batch(batch):
    """
    兼容两种 dataloader 输出：
    1) kspace, mask, mask_fold
    2) kspace, mask, mask_fold, maps
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            kspace, mask, mask_fold = batch
            maps = None
        elif len(batch) == 4:
            kspace, mask, mask_fold, maps = batch
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
    else:
        raise TypeError("Batch must be list/tuple.")
    return kspace, mask, mask_fold, maps


class Trainer:
    def __init__(
            self,
            diffusion_model,
            *,
            ema_decay=0.995,
            image_size=128,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=200,
            results_folder='./results',
            load_path=None,
            dataloader_train=None,
            dataloader_test=None,
            val_every=500,
            early_stop_patience=10,
            early_stop_min_delta=1e-4,
            monitor_metric='psnr',
            max_val_batches=None,
            lr_scheduler_type='none',
            warmup_steps=0,
            min_lr=0.0,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.dl = cycle(dataloader_train)
        self.dataloader_test = dataloader_test
        self.dl_test = cycle(dataloader_test) if dataloader_test is not None else None

        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr)


        self.base_lr = float(train_lr)
        self.lr_scheduler_type = str(lr_scheduler_type).lower()
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.fp16 = fp16

        self.val_every = int(val_every)
        self.early_stop_patience = int(early_stop_patience)
        self.early_stop_min_delta = float(early_stop_min_delta)
        self.monitor_metric = monitor_metric.lower()
        self.max_val_batches = None if max_val_batches is None else int(max_val_batches)

        if self.monitor_metric not in ['psnr', 'ssim', 'nmse']:
            raise ValueError(f"Unsupported monitor_metric: {self.monitor_metric}")

        self.best_metric = -float('inf') if self.monitor_metric in ['psnr', 'ssim'] else float('inf')
        self.no_improve_count = 0

        self.reset_parameters()
        self.model = diffusion_model
        self.device = next(self.model.parameters()).device
        self.ema = EMA(ema_decay)

        if load_path is not None:
            self.load(load_path)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, itrs=None):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / 'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def _compute_lr(self, step: int) -> float:
        if self.lr_scheduler_type in ['none', 'constant', 'off', 'false']:
            return self.base_lr

        if self.lr_scheduler_type != 'cosine':
            raise ValueError(f"Unsupported lr_scheduler_type: {self.lr_scheduler_type}")

        warmup_steps = max(0, self.warmup_steps)
        min_lr = max(0.0, self.min_lr)

        if warmup_steps > 0 and step < warmup_steps:
            return self.base_lr * float(step + 1) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, self.train_num_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (self.base_lr - min_lr) * cosine

    def _set_lr(self, lr: float):
        for param_group in self.opt.param_groups:
            param_group['lr'] = float(lr)

    def _log(self, msg: str):
        tqdm.write(msg, file=sys.stdout)

    @torch.no_grad()
    def validate(self, t):
        if self.dataloader_test is None:
            return None

        self.ema_model.eval()
        self.ema_model.training = False

        nmse = 0.0
        psnr = 0.0
        ssim = 0.0
        num_eval_batches = 0

        total_eval = len(self.dataloader_test)
        if self.max_val_batches is not None:
            total_eval = min(total_eval, self.max_val_batches)

        # 不再给 validate 单独开 tqdm，避免和训练条抢输出
        for batch_idx, batch in enumerate(self.dataloader_test):
            if self.max_val_batches is not None and batch_idx >= self.max_val_batches:
                break

            kspace, mask, mask_fold, maps = unpack_batch(batch)

            kspace = kspace.to(self.device)
            mask = mask.to(self.device)
            mask_fold = mask_fold.to(self.device)
            if maps is not None:
                maps = maps.to(self.device)

            B, Nc, H, W, C = kspace.shape
            gt_imgs = fastmri.ifft2c(kspace)
            k_c = kspace * mask.unsqueeze(-1)

            _, _, sample_imgs = self.ema_model.sample(k_c, mask, mask_fold, t=t)

            gt_imgs_abs = eval_image_from_multicoil(gt_imgs, maps)
            sample_imgs_abs = eval_image_from_multicoil(sample_imgs, maps)

            nmseb = 0.0
            psnrb = 0.0
            ssimb = 0.0
            for i in range(B):
                nmseb += calc_nmse_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                psnrb += calc_psnr_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                ssimb += calc_ssim_tensor(gt_imgs_abs[i], sample_imgs_abs[i])

            nmse += nmseb / B
            psnr += psnrb / B
            ssim += ssimb / B
            num_eval_batches += 1

        if num_eval_batches == 0:
            self.ema_model.train()
            return None

        nmse /= num_eval_batches
        psnr /= num_eval_batches
        ssim /= num_eval_batches

        self.ema_model.train()
        return {
            'nmse': float(nmse),
            'psnr': float(psnr),
            'ssim': float(ssim),
        }

    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0.0
        pbar = tqdm(
            range(self.train_num_steps),
            desc='Loss=0.000000',
            ascii=True,
            dynamic_ncols=False,
            ncols=100,
            mininterval=0.5,
            leave=True,
            file=sys.stdout
        )

        for step in pbar:
            self.step = step
            self._set_lr(self._compute_lr(step))
            u_loss = 0.0

            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dl)
                kspace, mask, mask_fold, maps = unpack_batch(batch)

                kspace = kspace.to(self.device)
                mask = mask.to(self.device)
                mask_fold = mask_fold.to(self.device)

                if maps is not None:
                    maps = maps.to(self.device)

                loss = self.model(kspace, mask, mask_fold)
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            train_loss = u_loss / self.gradient_accumulate_every
            acc_loss += train_loss

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # 保留你原来喜欢的显示风格：左边直接显示 Loss=...
            current_lr = self.opt.param_groups[0]['lr']
            pbar.set_description(f"Loss={train_loss:.6f} | LR={current_lr:.6e}")

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                mean_loss = acc_loss / (self.save_and_sample_every + 1)
                self._log(f"Mean LOSS of last {self.step}: {mean_loss:.6f}")
                acc_loss = 0.0
                self.save(self.step)

            if self.dataloader_test is not None and self.step != 0 and self.step % self.val_every == 0:
                metrics = self.validate(t=self.model.num_timesteps)
                if metrics is None:
                    self._log(f"[VAL @ step {self.step}] skipped (no validation batches)")
                    continue

                cur_metric = metrics[self.monitor_metric]

                self._log(
                    f"[VAL @ step {self.step}] "
                    f"PSNR={metrics['psnr']:.6f} | "
                    f"SSIM={metrics['ssim']:.6f} | "
                    f"NMSE={metrics['nmse']:.6f}"
                )

                if self.monitor_metric in ['psnr', 'ssim']:
                    improved = cur_metric > (self.best_metric + self.early_stop_min_delta)
                else:
                    improved = cur_metric < (self.best_metric - self.early_stop_min_delta)

                if improved:
                    self.best_metric = cur_metric
                    self.no_improve_count = 0

                    data = {
                        'step': self.step,
                        'model': self.model.state_dict(),
                        'ema': self.ema_model.state_dict(),
                        'best_metric': self.best_metric,
                        'monitor_metric': self.monitor_metric,
                        'val_metrics': metrics,
                    }
                    torch.save(data, str(self.results_folder / 'best.pt'))
                    self._log(f"[BEST] step={self.step} | {self.monitor_metric}={self.best_metric:.6f}")
                else:
                    self.no_improve_count += 1
                    self._log(
                        f"[NO IMPROVEMENT] count={self.no_improve_count}/"
                        f"{self.early_stop_patience}"
                    )

                    if self.no_improve_count >= self.early_stop_patience:
                        self._log(
                            f"[EARLY STOP] step={self.step} | "
                            f"best {self.monitor_metric}={self.best_metric:.6f}"
                        )
                        self.save(self.step)
                        return

        self.save(self.step + 1)
        self._log('training completed')

    def test(self, t, num_samples=1):
        if self.dataloader_test is None:
            self._log("No test dataloader provided.")
            return None, None, None, None

        torch.set_grad_enabled(False)
        sample_imgs_list = []
        gt_imgs_list = []
        xt_list = []
        direct_recons_list = []

        nmse = 0.0
        psnr = 0.0
        ssim = 0.0

        self._log('\nEvaluation:')
        self.ema_model.eval()
        self.ema_model.training = False

        with torch.no_grad():
            pbar = tqdm(
                range(len(self.dataloader_test)),
                desc='TEST',
                ascii=True,
                dynamic_ncols=False,
                ncols=100,
                mininterval=0.5,
                leave=True,
                file=sys.stdout
            )

            for idx in pbar:
                batch = next(self.dl_test)
                kspace, mask, mask_fold, maps = unpack_batch(batch)

                kspace = kspace.to(self.device)
                mask = mask.to(self.device)
                mask_fold = mask_fold.to(self.device)
                if maps is not None:
                    maps = maps.to(self.device)

                B, Nc, H, W, C = kspace.shape
                gt_imgs = fastmri.ifft2c(kspace)
                k_c = kspace * mask.unsqueeze(-1)

                if num_samples == 1:
                    xt, direct_recons, sample_imgs = self.ema_model.sample(k_c, mask, mask_fold, t=t)
                else:
                    for i_sample in range(num_samples):
                        xti, direct_reconsi, sample_imgsi = self.ema_model.sample(k_c, mask, mask_fold, t=t)
                        if i_sample == 0:
                            xt = xti
                            direct_recons = direct_reconsi
                            sample_imgs = sample_imgsi
                        else:
                            xt = xti
                            direct_recons = torch.cat((direct_recons, direct_reconsi), dim=1)
                            sample_imgs = torch.cat((sample_imgs, sample_imgsi), dim=1)

                gt_imgs_abs = eval_image_from_multicoil(gt_imgs, maps)
                sample_imgs_abs = eval_image_from_multicoil(sample_imgs, maps)

                nmseb = 0.0
                psnrb = 0.0
                ssimb = 0.0
                for i in range(B):
                    nmseb += calc_nmse_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                    psnrb += calc_psnr_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                    ssimb += calc_ssim_tensor(gt_imgs_abs[i], sample_imgs_abs[i])

                nmseb /= B
                psnrb /= B
                ssimb /= B

                nmse += nmseb
                psnr += psnrb
                ssim += ssimb

                if idx == 0:
                    self._log(f'Batch PSNR: {psnrb:.5f} || SSIM: {ssimb:.5f}')

                sample_imgs_list.append(sample_imgs)
                gt_imgs_list.append(gt_imgs)
                xt_list.append(xt)
                direct_recons_list.append(direct_recons)

            nmse = nmse / len(self.dataloader_test)
            psnr = psnr / len(self.dataloader_test)
            ssim = ssim / len(self.dataloader_test)

            self._log(f'### NMSE: {nmse:.6f} || PSNR: {psnr:.6f} || SSIM: {ssim:.6f}')
            self._log('----------------------------------------------------------------------')
            torch.set_grad_enabled(True)

        return sample_imgs_list, gt_imgs_list, xt_list, direct_recons_list

    def recon_slice(self, t, idx_case, num_samples=1):
        torch.set_grad_enabled(False)

        self._log('\nEvaluation:')
        self.ema_model.eval()
        self.ema_model.training = False

        sample_imgs = None
        gt_imgs = None
        xt = None
        direct_recons = None

        for idx, batch in enumerate(self.dataloader_test):
            if idx != idx_case:
                continue

            kspace, mask, mask_fold, maps = unpack_batch(batch)

            kspace = kspace.to(self.device)
            mask = mask.to(self.device)
            mask_fold = mask_fold.to(self.device)
            if maps is not None:
                maps = maps.to(self.device)

            B, Nc, H, W, C = kspace.shape
            gt_imgs = fastmri.ifft2c(kspace)
            k_c = kspace * mask.unsqueeze(-1)

            if num_samples == 1:
                xt, direct_recons, sample_imgs = self.ema_model.sample(k_c, mask, mask_fold, t=t)
            else:
                for i_sample in range(num_samples):
                    xti, direct_reconsi, sample_imgsi = self.ema_model.sample(k_c, mask, mask_fold, t=t)
                    if i_sample == 0:
                        xt = xti
                        direct_recons = direct_reconsi
                        sample_imgs = sample_imgsi
                    else:
                        xt = xti
                        direct_recons = torch.cat((direct_recons, direct_reconsi), dim=1)
                        sample_imgs = torch.cat((sample_imgs, sample_imgsi), dim=1)

            gt_imgs_abs = eval_image_from_multicoil(gt_imgs, maps)
            sample_imgs_abs = eval_image_from_multicoil(sample_imgs, maps)

            nmseb = 0.0
            psnrb = 0.0
            ssimb = 0.0
            for i in range(B):
                nmseb += calc_nmse_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                psnrb += calc_psnr_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                ssimb += calc_ssim_tensor(gt_imgs_abs[i], sample_imgs_abs[i])

            nmseb /= B
            psnrb /= B
            ssimb /= B

            self._log(f'### NMSE: {nmseb:.6f} || PSNR: {psnrb:.6f} || SSIM: {ssimb:.6f}')
            self._log('----------------------------------------------------------------------')
            torch.set_grad_enabled(True)
            break

        if sample_imgs is None:
            torch.set_grad_enabled(True)
            raise IndexError(
                f'idx_case={idx_case} is out of range for dataloader_test with len={len(self.dataloader_test)}')

        return sample_imgs, gt_imgs, xt, direct_recons