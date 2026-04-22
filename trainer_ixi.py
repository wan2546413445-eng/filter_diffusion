import math
import copy
import sys
from functools import partial
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm
import fastmri

from utils.diffusion_utils import cycle, EMA, loss_backwards
from utils.evaluation import calc_nmse_tensor, calc_psnr_tensor, calc_ssim_tensor


def unpack_batch_ixi(batch):
    if not isinstance(batch, (list, tuple)):
        raise TypeError('Batch must be list/tuple.')
    if len(batch) != 3:
        raise ValueError(f'IXI batch must be (kspace, mask, mask_fold), got len={len(batch)}')
    kspace, mask, mask_fold = batch
    return kspace, mask, mask_fold


def eval_image_from_ixi(imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: [B, 1, H, W, 2]
    return: [B, H, W]
    """
    if imgs.ndim != 5 or imgs.shape[1] != 1 or imgs.shape[-1] != 2:
        raise ValueError(f'Expected [B,1,H,W,2], got {tuple(imgs.shape)}')
    return fastmri.complex_abs(imgs)[:, 0]


def tensor_stats(x: torch.Tensor):
    return {
        'min': float(x.min().item()),
        'max': float(x.max().item()),
        'mean': float(x.mean().item()),
        'std': float(x.std().item()),
    }


def grad_stats(model: torch.nn.Module):
    grad_mean_sum = 0.0
    grad_max = 0.0
    num = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        grad_mean_sum += float(g.abs().mean().item())
        grad_max = max(grad_max, float(g.abs().max().item()))
        num += 1
    if num == 0:
        return {'grad_mean_sum': 0.0, 'grad_max': 0.0}
    return {'grad_mean_sum': grad_mean_sum, 'grad_max': grad_max}


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
        debug_every=200,
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
        self.debug_every = int(debug_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.fp16 = fp16
        self.val_every = int(val_every)
        self.early_stop_patience = int(early_stop_patience)
        self.early_stop_min_delta = float(early_stop_min_delta)
        self.monitor_metric = monitor_metric.lower()
        self.max_val_batches = None if max_val_batches is None else int(max_val_batches)

        if self.monitor_metric not in ['psnr', 'ssim', 'nmse']:
            raise ValueError(f'Unsupported monitor_metric: {self.monitor_metric}')

        self.best_metric = -float('inf') if self.monitor_metric in ['psnr', 'ssim'] else float('inf')
        self.no_improve_count = 0

        self.reset_parameters()
        self.device = next(self.model.parameters()).device

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
            'ema': self.ema_model.state_dict(),
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / 'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))

    def load(self, load_path):
        print('Loading : ', load_path)
        data = torch.load(load_path, map_location=self.device)
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def _compute_lr(self, step: int) -> float:
        if self.lr_scheduler_type in ['none', 'constant', 'off', 'false']:
            return self.base_lr
        if self.lr_scheduler_type != 'cosine':
            raise ValueError(f'Unsupported lr_scheduler_type: {self.lr_scheduler_type}')

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

        for batch_idx, batch in enumerate(self.dataloader_test):
            if self.max_val_batches is not None and batch_idx >= self.max_val_batches:
                break

            kspace, mask, mask_fold = unpack_batch_ixi(batch)
            kspace = kspace.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            mask_fold = mask_fold.to(self.device, non_blocking=True)

            gt_imgs = fastmri.ifft2c(kspace)
            k_c = kspace * mask.unsqueeze(-1)
            zf_imgs = fastmri.ifft2c(k_c)
            _, direct_recons, sample_imgs = self.ema_model.sample(k_c, mask, mask_fold, t=t)

            gt_imgs_abs = eval_image_from_ixi(gt_imgs)
            zf_imgs_abs = eval_image_from_ixi(zf_imgs)
            sample_imgs_abs = eval_image_from_ixi(sample_imgs)

            if batch_idx == 0:
                gt_s = tensor_stats(gt_imgs_abs)
                zf_s = tensor_stats(zf_imgs_abs)
                sample_s = tensor_stats(sample_imgs_abs)
                if direct_recons is not None:
                    direct_imgs_abs = eval_image_from_ixi(direct_recons)
                    direct_s = tensor_stats(direct_imgs_abs)
                    self._log(
                        '[VAL-DBG] '
                        f"gt_max={gt_s['max']:.4f} gt_mean={gt_s['mean']:.4f} | "
                        f"zf_max={zf_s['max']:.4f} zf_mean={zf_s['mean']:.4f} | "
                        f"direct_max={direct_s['max']:.4f} direct_mean={direct_s['mean']:.4f} | "
                        f"sample_max={sample_s['max']:.4f} sample_mean={sample_s['mean']:.4f}"
                    )
                else:
                    self._log(
                        '[VAL-DBG] '
                        f"gt_max={gt_s['max']:.4f} gt_mean={gt_s['mean']:.4f} | "
                        f"zf_max={zf_s['max']:.4f} zf_mean={zf_s['mean']:.4f} | "
                        f"sample_max={sample_s['max']:.4f} sample_mean={sample_s['mean']:.4f}"
                    )

            bsz = gt_imgs_abs.shape[0]
            nmseb = 0.0
            psnrb = 0.0
            ssimb = 0.0
            for i in range(bsz):
                nmseb += calc_nmse_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                psnrb += calc_psnr_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                ssimb += calc_ssim_tensor(gt_imgs_abs[i], sample_imgs_abs[i])

            nmse += nmseb / bsz
            psnr += psnrb / bsz
            ssim += ssimb / bsz
            num_eval_batches += 1

        if num_eval_batches == 0:
            self.ema_model.train()
            return None

        nmse /= num_eval_batches
        psnr /= num_eval_batches
        ssim /= num_eval_batches

        self.ema_model.train()
        return {'nmse': float(nmse), 'psnr': float(psnr), 'ssim': float(ssim)}

    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        acc_loss = 0.0

        pbar = tqdm(
            total=self.train_num_steps,
            ascii=True,
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
            file=sys.stdout,
        )

        for step in range(self.train_num_steps):
            self.step = step
            self._set_lr(self._compute_lr(step))
            u_loss = 0.0
            last_grad = {'grad_mean_sum': 0.0, 'grad_max': 0.0}

            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dl)
                kspace, mask, mask_fold = unpack_batch_ixi(batch)

                kspace = kspace.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                mask_fold = mask_fold.to(self.device, non_blocking=True)

                out = self.model(kspace, mask, mask_fold, return_components=True)
                if isinstance(out, dict):
                    loss = out["loss_total"]
                    loss_delta = float(out["loss_delta"].item())
                    loss_img = float(out["loss_img"].item())
                    delta_support_ratio = float(out["delta_support_ratio"].item())
                else:
                    loss = out
                    loss_delta = float("nan")
                    loss_img = float("nan")
                    delta_support_ratio = float("nan")

                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)
                last_grad = grad_stats(self.model)

            train_loss = u_loss / self.gradient_accumulate_every
            acc_loss += train_loss

            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            current_lr = self.opt.param_groups[0]['lr']
            pbar.set_postfix(loss=f'{train_loss:.6f}', lr=f'{current_lr:.6e}')
            pbar.update(1)

            if self.step == 0 or (self.debug_every > 0 and self.step % self.debug_every == 0):
                self._log(
                    f'[TRAIN-DBG] step={self.step} | '
                    f'loss={train_loss:.6f} | '
                    f'loss_k={loss_delta:.6f} | '
                    f'loss_x={loss_img:.6f} | '
                    f'delta_support={delta_support_ratio:.6f} | '
                    f'grad_mean_sum={last_grad["grad_mean_sum"]:.6e} | '
                    f'grad_max={last_grad["grad_max"]:.6e} | '
                    f'lr={current_lr:.6e}'
                )

            if self.step != 0 and self.step % 1000 == 0:
                self._log(
                    f'[TRAIN] step={self.step}/{self.train_num_steps} | '
                    f'loss={train_loss:.6f} | lr={current_lr:.6e}'
                )

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                mean_loss = acc_loss / (self.save_and_sample_every + 1)
                self._log(f'Mean LOSS of last {self.step}: {mean_loss:.6f}')
                acc_loss = 0.0
                self.save(self.step)

            if self.dataloader_test is not None and self.step != 0 and self.step % self.val_every == 0:
                metrics = self.validate(t=self.model.num_timesteps)
                if metrics is None:
                    self._log(f'[VAL @ step {self.step}] skipped (no validation batches)')
                    continue

                cur_metric = metrics[self.monitor_metric]
                self._log(
                    f'[VAL @ step {self.step}] '
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
                    self._log(f'[BEST] step={self.step} | {self.monitor_metric}={self.best_metric:.6f}')
                else:
                    self.no_improve_count += 1
                    self._log(f'[NO IMPROVEMENT] count={self.no_improve_count}/{self.early_stop_patience}')
                    if self.no_improve_count >= self.early_stop_patience:
                        self._log(
                            f'[EARLY STOP] step={self.step} | '
                            f'best {self.monitor_metric}={self.best_metric:.6f}'
                        )
                        self.save(self.step)
                        pbar.close()
                        return

        pbar.close()
        self.save(self.step + 1)
        self._log('training completed')

    def test(self, t, num_samples=1):
        if self.dataloader_test is None:
            self._log('No test dataloader provided.')
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
                file=sys.stdout,
            )

            for idx in pbar:
                batch = next(self.dl_test)
                kspace, mask, mask_fold = unpack_batch_ixi(batch)

                kspace = kspace.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                mask_fold = mask_fold.to(self.device, non_blocking=True)

                gt_imgs = fastmri.ifft2c(kspace)
                k_c = kspace * mask.unsqueeze(-1)
                zf_imgs = fastmri.ifft2c(k_c)

                if num_samples == 1:
                    xt, direct_recons, sample_imgs = self.ema_model.sample(k_c, mask, mask_fold, t=t)
                else:
                    xt = None
                    direct_recons = None
                    sample_imgs = None
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

                gt_imgs_abs = eval_image_from_ixi(gt_imgs)
                zf_imgs_abs = eval_image_from_ixi(zf_imgs)
                sample_imgs_abs = eval_image_from_ixi(sample_imgs)

                if idx == 0:
                    gt_s = tensor_stats(gt_imgs_abs)
                    zf_s = tensor_stats(zf_imgs_abs)
                    sample_s = tensor_stats(sample_imgs_abs)
                    if direct_recons is not None:
                        direct_s = tensor_stats(eval_image_from_ixi(direct_recons))
                        self._log(
                            '[TEST-DBG] '
                            f"gt_max={gt_s['max']:.4f} gt_mean={gt_s['mean']:.4f} | "
                            f"zf_max={zf_s['max']:.4f} zf_mean={zf_s['mean']:.4f} | "
                            f"direct_max={direct_s['max']:.4f} direct_mean={direct_s['mean']:.4f} | "
                            f"sample_max={sample_s['max']:.4f} sample_mean={sample_s['mean']:.4f}"
                        )
                    else:
                        self._log(
                            '[TEST-DBG] '
                            f"gt_max={gt_s['max']:.4f} gt_mean={gt_s['mean']:.4f} | "
                            f"zf_max={zf_s['max']:.4f} zf_mean={zf_s['mean']:.4f} | "
                            f"sample_max={sample_s['max']:.4f} sample_mean={sample_s['mean']:.4f}"
                        )

                bsz = gt_imgs_abs.shape[0]
                nmseb = 0.0
                psnrb = 0.0
                ssimb = 0.0
                for i in range(bsz):
                    nmseb += calc_nmse_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                    psnrb += calc_psnr_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                    ssimb += calc_ssim_tensor(gt_imgs_abs[i], sample_imgs_abs[i])

                nmseb /= bsz
                psnrb /= bsz
                ssimb /= bsz

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

            kspace, mask, mask_fold = unpack_batch_ixi(batch)
            kspace = kspace.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            mask_fold = mask_fold.to(self.device, non_blocking=True)

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

            gt_imgs_abs = eval_image_from_ixi(gt_imgs)
            sample_imgs_abs = eval_image_from_ixi(sample_imgs)

            bsz = gt_imgs_abs.shape[0]
            nmseb = 0.0
            psnrb = 0.0
            ssimb = 0.0
            for i in range(bsz):
                nmseb += calc_nmse_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                psnrb += calc_psnr_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                ssimb += calc_ssim_tensor(gt_imgs_abs[i], sample_imgs_abs[i])

            nmseb /= bsz
            psnrb /= bsz
            ssimb /= bsz

            self._log(f'### NMSE: {nmseb:.6f} || PSNR: {psnrb:.6f} || SSIM: {ssimb:.6f}')
            self._log('----------------------------------------------------------------------')
            torch.set_grad_enabled(True)
            break

        if sample_imgs is None:
            torch.set_grad_enabled(True)
            raise IndexError(
                f'idx_case={idx_case} is out of range for dataloader_test with len={len(self.dataloader_test)}'
            )

        return sample_imgs, gt_imgs, xt, direct_recons
