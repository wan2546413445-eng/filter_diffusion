import torch
from torch import nn
import torch.nn.functional as F

import fastmri

from .filter_schedule import CenterRectangleSchedule
from .degradation import apply_filter_degradation
from .delta_target import build_delta_target
from .reverse_loop import run_reverse_loop


class KspaceDiffusion(nn.Module):
    """
    FilterDiff-style lightweight supervised baseline:
    - forward degradation: k_t = M_t * k_0
    - target: delta_gt = (M_{t-1} - M_t) * k_0
    - reverse: predict step-wise delta in k-space with explicit conditional kc + DC
    """

    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            device_of_kernel,
            channels=2,
            timesteps=100,
            loss_type='l1',
            schedule_type='dense',
            center_core_size=32,
            use_explicit_dc=False,
            **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device_of_kernel = device_of_kernel

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.lambda_img = 1.0
        # paper-mainline default: no explicit DC projection in reverse loop
        self.use_explicit_dc = use_explicit_dc

        self.schedule = CenterRectangleSchedule(
            h=image_size,
            w=image_size,
            timesteps=self.num_timesteps,
            center_core_size=center_core_size,
            schedule_type=schedule_type,
        )

    def _build_conditional_kc(self, kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        kspace: [B,Nc,H,W,2]
        mask:   [B,1,H,W]
        """
        return kspace * mask.unsqueeze(-1)

    def p_losses(self, kspace: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz, ncoil, h, w, _ = kspace.shape

        m_t = self.schedule.get_by_t(t, device=kspace.device, dtype=kspace.dtype)
        t_minus_1 = torch.clamp(t - 1, min=0)
        m_t_minus_1 = self.schedule.get_by_t(t_minus_1, device=kspace.device, dtype=kspace.dtype)

        k_t = apply_filter_degradation(kspace, m_t)
        k_c = self._build_conditional_kc(kspace, mask)
        delta_gt = build_delta_target(kspace, m_t, m_t_minus_1)

        m_t_ch = m_t.expand(-1, ncoil, -1, -1, -1)

        kt_in = k_t.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)
        kc_in = k_c.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)
        mt_in = m_t_ch.reshape(bsz * ncoil, h, w, 1).permute(0, 3, 1, 2)

        model_in = torch.cat([kt_in, kc_in, mt_in], dim=1)  # [B*Nc,5,H,W]
        t_in = t.repeat_interleave(ncoil)

        delta_pred = self.denoise_fn(model_in, t_in).permute(0, 2, 3, 1).reshape(bsz, ncoil, h, w, 2)

        # Paper-aligned minimal supervised objective:
        # L = ||R_theta - delta_gt|| + lambda * ||phi_theta - x0||, lambda=1
        # We reuse a single UNet head and derive phi_theta as image converted from
        # k_{t-1} prediction: k_pred = k_t + delta_pred.
        k_pred = k_t + delta_pred
        x0 = fastmri.ifft2c(kspace)
        x_pred = fastmri.ifft2c(k_pred)

        if self.loss_type == 'l1':
            loss_delta = F.l1_loss(delta_pred, delta_gt)
            loss_img = F.l1_loss(x_pred, x0)
        elif self.loss_type == 'l2':
            loss_delta = F.mse_loss(delta_pred, delta_gt)
            loss_img = F.mse_loss(x_pred, x0)
        else:
            raise NotImplementedError(f"Unsupported loss type: {self.loss_type}")

        return loss_delta + self.lambda_img * loss_img

    @torch.no_grad()
    def sample(self, k_c: torch.Tensor, mask: torch.Tensor, mask_fold=None, t=None):
        """
        kc-only inference mode:
        - Input `k_c` is the under-sampled conditional k-space
        - Reverse is initialized directly from k_T = k_c

        No full-sampled k0 is used inside sampling.
        """
        self.denoise_fn.eval()

        if t is None:
            t = self.num_timesteps

        # `k_c` is already the under-sampled conditional k-space produced by the
        # caller. Do not apply the acquisition mask a second time here.
        k_t = k_c

        k_rec, direct_k = run_reverse_loop(
            model=self.denoise_fn,
            k_t=k_t,
            k_c=k_c,
            acq_mask=mask,
            schedule=self.schedule,
            timesteps=t,
            use_explicit_dc=self.use_explicit_dc,
        )

        xt = fastmri.ifft2c(k_t)
        direct_recons = fastmri.ifft2c(direct_k) if direct_k is not None else None
        img = fastmri.ifft2c(k_rec)

        self.denoise_fn.train()
        return xt, direct_recons, img

    def forward(self, kspace, mask, mask_fold=None, *args, **kwargs):
        bsz, _, h, w, _ = kspace.shape
        assert h == self.image_size and w == self.image_size, (
            f'height and width of image must be {self.image_size}'
        )

        t = torch.randint(1, self.num_timesteps + 1, (bsz,), device=kspace.device).long()
        return self.p_losses(kspace, mask, t)
