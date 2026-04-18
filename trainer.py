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
            img_abs = torch.mean(img_abs, dim=1)      # [B,H,W]
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
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.fp16 = fp16

        self.val_every = val_every
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.monitor_metric = monitor_metric.lower()

        if self.monitor_metric not in ['psnr', 'ssim', 'nmse']:
            raise ValueError(f"Unsupported monitor_metric: {self.monitor_metric}")

        self.best_metric = -float('inf') if self.monitor_metric in ['psnr', 'ssim'] else float('inf')
        self.no_improve_count = 0

        self.reset_parameters()

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
    @torch.no_grad()
    def validate(self, t):
        if self.dataloader_test is None:
            return None

        self.ema_model.eval()
        self.ema_model.training = False

        nmse = 0.0
        psnr = 0.0
        ssim = 0.0

        for batch in self.dataloader_test:
            kspace, mask, mask_fold, maps = unpack_batch(batch)

            kspace = kspace.cuda()
            mask = mask.cuda()
            mask_fold = mask_fold.cuda()
            if maps is not None:
                maps = maps.cuda()

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

        nmse /= len(self.dataloader_test)
        psnr /= len(self.dataloader_test)
        ssim /= len(self.dataloader_test)

        self.ema_model.train()
        return {
            'nmse': float(nmse),
            'psnr': float(psnr),
            'ssim': float(ssim),
        }

    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0.0
        pbar = tqdm(range(self.train_num_steps), desc='LOSS')

        for step in pbar:
            self.step = step
            u_loss = 0.0

            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dl)
                kspace, mask, mask_fold, maps = unpack_batch(batch)

                kspace = kspace.cuda()
                mask = mask.cuda()
                mask_fold = mask_fold.cuda()

                if maps is not None:
                    maps = maps.cuda()

                loss = self.model(kspace, mask, mask_fold)
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            train_loss = u_loss / self.gradient_accumulate_every
            pbar.set_description(f"Loss={train_loss:.6f}")
            acc_loss += train_loss

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # 常规保存
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                mean_loss = acc_loss / (self.save_and_sample_every + 1)
                print(f"Mean LOSS of last {self.step}: {mean_loss:.6f}")
                acc_loss = 0.0
                self.save(self.step)

            # 在线验证 + best ckpt + early stopping
            if self.dataloader_test is not None and self.step != 0 and self.step % self.val_every == 0:
                metrics = self.validate(t=self.model.num_timesteps)
                cur_metric = metrics[self.monitor_metric]

                print(
                    f"[VAL @ step {self.step}] "
                    f"PSNR={metrics['psnr']:.6f} | "
                    f"SSIM={metrics['ssim']:.6f} | "
                    f"NMSE={metrics['nmse']:.6f}"
                )

                improved = False
                if self.monitor_metric in ['psnr', 'ssim']:
                    improved = cur_metric > (self.best_metric + self.early_stop_min_delta)
                else:  # nmse 越小越好
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
                    print(f"[BEST] step={self.step} | {self.monitor_metric}={self.best_metric:.6f}")
                else:
                    self.no_improve_count += 1
                    print(
                        f"[NO IMPROVEMENT] count={self.no_improve_count}/"
                        f"{self.early_stop_patience}"
                    )

                    if self.no_improve_count >= self.early_stop_patience:
                        print(
                            f"[EARLY STOP] step={self.step} | "
                            f"best {self.monitor_metric}={self.best_metric:.6f}"
                        )
                        self.save(self.step)
                        return

        self.save(self.step + 1)
        print('training completed')
    def test(self, t, num_samples=1):
        if self.dataloader_test is None:
            print("No test dataloader provided.")
            return None, None, None, None

        torch.set_grad_enabled(False)
        sample_imgs_list = []
        gt_imgs_list = []
        xt_list = []
        direct_recons_list = []

        nmse = 0
        psnr = 0
        ssim = 0

        print('\nEvaluation:')
        self.ema_model.eval()
        self.ema_model.training = False

        with torch.no_grad():
            pbar = tqdm(range(len(self.dataloader_test)), desc='TEST')
            for idx in pbar:
                batch = next(self.dl_test)
                kspace, mask, mask_fold, maps = unpack_batch(batch)

                kspace = kspace.cuda()
                mask = mask.cuda()
                mask_fold = mask_fold.cuda()
                if maps is not None:
                    maps = maps.cuda()

                B, Nc, H, W, C = kspace.shape
                gt_imgs = fastmri.ifft2c(kspace)  # [B,Nc,H,W,2]
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

                gt_imgs_abs = eval_image_from_multicoil(gt_imgs, maps)           # [B,H,W]
                sample_imgs_abs = eval_image_from_multicoil(sample_imgs, maps)   # [B,H,W]

                nmseb = 0
                psnrb = 0
                ssimb = 0
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
                    print(f'Batch PSNR: {psnrb:.5f} || SSIM: {ssimb:.5f}')

                sample_imgs_list.append(sample_imgs)
                gt_imgs_list.append(gt_imgs)
                xt_list.append(xt)
                direct_recons_list.append(direct_recons)

            nmse = nmse / len(self.dataloader_test)
            psnr = psnr / len(self.dataloader_test)
            ssim = ssim / len(self.dataloader_test)

            print(f'### NMSE: {nmse:.6f} || PSNR: {psnr:.6f} || SSIM: {ssim:.6f}')
            print('----------------------------------------------------------------------')
            torch.set_grad_enabled(True)

        return sample_imgs_list, gt_imgs_list, xt_list, direct_recons_list

    def recon_slice(self, t, idx_case, num_samples=1):
        torch.set_grad_enabled(False)

        print('\nEvaluation:')
        self.ema_model.eval()
        self.ema_model.training = False

        for idx, batch in enumerate(self.dataloader_test):
            if idx != idx_case:
                continue

            kspace, mask, mask_fold, maps = unpack_batch(batch)

            kspace = kspace.cuda()
            mask = mask.cuda()
            mask_fold = mask_fold.cuda()
            if maps is not None:
                maps = maps.cuda()

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

            nmseb = 0
            psnrb = 0
            ssimb = 0
            for i in range(B):
                nmseb += calc_nmse_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                psnrb += calc_psnr_tensor(gt_imgs_abs[i], sample_imgs_abs[i])
                ssimb += calc_ssim_tensor(gt_imgs_abs[i], sample_imgs_abs[i])

            nmseb /= B
            psnrb /= B
            ssimb /= B

            print(f'### NMSE: {nmseb:.6f} || PSNR: {psnrb:.6f} || SSIM: {ssimb:.6f}')
            print('----------------------------------------------------------------------')
            torch.set_grad_enabled(True)
            break

        return sample_imgs, gt_imgs, xt, direct_recons