import torch
from torch import nn
import torch.nn.functional as F

import fastmri

from diffusion.filter_schedule import CenterRectangleSchedule
from diffusion.delta_target import build_delta_target
from diffusion.degradation import apply_filter_degradation
from diffusion.reverse_loop import run_reverse_loop


class KspaceDiffusion(nn.Module):
    """
    FilterDiff-aligned implementation (except restoration network architecture):
      Eq.6:  k_t = M_t ⊙ k_0
      Eq.8:  L = ||Δk_pred - Δk_gt|| + λ ||x0_pred - x0||, λ=1
             where Δk_pred = ΔM_{t-1} ⊙ FFT(x0_pred)
      Eq.9:  k_{t-1} = k_t + Δk_t, deterministic reverse loop.

    Network input channels are [k_t(2), k_c(2), M_t(1)] = 5 channels per coil.
    """

    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            device_of_kernel,
            channels=2,
            timesteps=20,
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
        self.lambda_img = float(kwargs.get("lambda_img", 1.0))
        self.use_explicit_dc = bool(use_explicit_dc)

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
        mask:   [B,1,H,W] or [B,H,W]
        """
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        # -> [B,1,H,W,1], broadcast on coil/channel dimensions
        mask = mask.unsqueeze(-1)
        return kspace * mask

    def _run_backbone(self, k_t: torch.Tensor, k_c: torch.Tensor, m_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return x0_pred in image domain, shape [B,Nc,H,W,2]."""
        bsz, ncoil, h, w, _ = k_t.shape

        m_t_ch = m_t.expand(-1, ncoil, -1, -1, -1)

        kt_in = k_t.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)
        kc_in = k_c.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)
        mt_in = m_t_ch.reshape(bsz * ncoil, h, w, 1).permute(0, 3, 1, 2)

        model_in = torch.cat([kt_in, kc_in, mt_in], dim=1)  # [B*Nc,5,H,W]
        t_in = t.repeat_interleave(ncoil)

        x0_pred = self.denoise_fn(model_in, t_in)
        x0_pred = x0_pred.permute(0, 2, 3, 1).reshape(bsz, ncoil, h, w, 2)
        return x0_pred

    def p_losses(self, kspace: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0 = fastmri.ifft2c(kspace)

        m_t = self.schedule.get_by_t(t, device=kspace.device, dtype=kspace.dtype)
        t_minus_1 = torch.clamp(t - 1, min=0)
        m_t_minus_1 = self.schedule.get_by_t(t_minus_1, device=kspace.device, dtype=kspace.dtype)

        delta_m = m_t_minus_1 - m_t

        # Eq.6 forward degradation: k_t = M_t ⊙ k_0
        k_t = apply_filter_degradation(kspace, m_t)

        # Conditional input k_c (fixed under-sampled k-space)
        k_c = self._build_conditional_kc(kspace, mask)

        # Eq.8 Δk ground truth: ΔM_{t-1} ⊙ k_0
        delta_gt = build_delta_target(kspace, m_t, m_t_minus_1)

        # φθ output: x0_pred (image domain)
        x0_pred = self._run_backbone(k_t=k_t, k_c=k_c, m_t=m_t, t=t)

        # Eq.8 Rθ: Δk_pred = ΔM_{t-1} ⊙ FFT(x0_pred)
        k0_pred = fastmri.fft2c(x0_pred)
        delta_pred = delta_m * k0_pred

        if self.loss_type == 'l1':
            loss_delta = F.l1_loss(delta_pred, delta_gt)
            loss_img = F.l1_loss(x0_pred, x0)
        elif self.loss_type == 'l2':
            loss_delta = F.mse_loss(delta_pred, delta_gt)
            loss_img = F.mse_loss(x0_pred, x0)
        else:
            raise NotImplementedError(f"Unsupported loss type: {self.loss_type}")

        return loss_delta + self.lambda_img * loss_img

    @torch.no_grad()
    def sample(self, k_c: torch.Tensor, mask: torch.Tensor, mask_fold=None, t=None):
        """
        Input: conditional under-sampled k-space k_c and acquisition mask.
        Eq.9 initialization: k_T = M_T ⊙ k_c.
        Returns final reconstruction image from k_0.
        """
        self.denoise_fn.eval()

        if t is None:
            t = self.num_timesteps

        bsz = k_c.shape[0]
        t_tensor = torch.full((bsz,), t, dtype=torch.long, device=k_c.device)
        m_t = self.schedule.get_by_t(t_tensor, device=k_c.device, dtype=k_c.dtype)

        # Eq.9 start from center-preserved k_T, no Gaussian noise.
        k_t = m_t * k_c

        k0_rec, direct_k = run_reverse_loop(
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
        img = fastmri.ifft2c(k0_rec)

        self.denoise_fn.train()
        return xt, direct_recons, img

    def forward(self, kspace, mask, mask_fold=None, *args, **kwargs):
        bsz, _, h, w, _ = kspace.shape
        assert h == self.image_size and w == self.image_size, (
            f'height and width of image must be {self.image_size}'
        )

        # sample t in [1, T]
        t = torch.randint(1, self.num_timesteps + 1, (bsz,), device=kspace.device).long()
        return self.p_losses(kspace, mask, t)