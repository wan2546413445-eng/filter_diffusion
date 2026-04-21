import torch
import torch.nn.functional as F
import fastmri
from torch import nn

from diffusion.filter_schedule import CenterRectangleSchedule
from diffusion.delta_target import build_delta_target
from diffusion.degradation import apply_filter_degradation
from diffusion.reverse_loop import run_reverse_loop


class KspaceDiffusion(nn.Module):
    """
    Strictly following Eq.(7)(8)(9) in FilterDiff:

      Eq.6:  k_t = M_t ⊙ k_0

      Eq.7:  Δk̄_{t-1} = Rθ(Cond)
                       = ΔM_{t-1} ⊙ FFT( φθ(Cond) )
             where Cond = (M_t, k_t, k_c, t)

      Eq.8:  L = || Rθ(M_t, k_t, k_c, t) - Δk_{t-1} ||
                + λ || φθ(M_t, k_t, k_c, t) - x_0 ||

      Eq.9:  k̄_{t-1} = k̄_t + Rθ(M_t, k̄_t, k_c, t)

    Important:
    - We keep image-domain supervision on φθ output (x0_pred vs x0)
    - We do NOT ifft k_t / k_c before feeding the backbone, because
      the formula defines Cond using (M_t, k_t, k_c, t)
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
            center_core_size=64,

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
        self.image_loss_mode = str(kwargs.get("image_loss_mode", "complex")).lower()

        # --- in __init__ ---
        self.input_representation = str(kwargs.get("input_representation", "kspace_raw")).lower()

        if self.image_loss_mode not in ["complex", "real", "magnitude"]:
            raise ValueError(f"Unsupported image_loss_mode: {self.image_loss_mode}")

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
        return: k_c = mask ⊙ k_0
        """
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # [B,1,H,W,1]
        return kspace * mask

    def _run_backbone(self, k_t, k_c, m_t, t):
        """
        Cond input channels:
            k_t(2) + k_c(2) + M_t(1) = 5
        """
        bsz, ncoil, h, w, _ = k_t.shape

        # frequency-domain aware preprocessing

        # 直接使用原始 K 空间，不做额外变换
        k_t_in = k_t.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)
        k_c_in = k_c.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)

        m_t_ch = m_t.expand(-1, ncoil, -1, -1, -1)
        m_t_in = m_t_ch.reshape(bsz * ncoil, h, w, 1).permute(0, 3, 1, 2)

        model_in = torch.cat([k_t_in, k_c_in, m_t_in], dim=1)
        t_in = t.repeat_interleave(ncoil)

        x0_pred = self.denoise_fn(model_in, t_in)
        return x0_pred.permute(0, 2, 3, 1).reshape(bsz, ncoil, h, w, 2)

    def p_losses(self, kspace: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Training loss exactly following Eq.(8).
        """
        # x0 in image domain for supervision
        x0 = fastmri.ifft2c(kspace)

        # masks
        m_t = self.schedule.get_by_t(t, device=kspace.device, dtype=kspace.dtype)
        t_minus_1 = torch.clamp(t - 1, min=0)
        m_t_minus_1 = self.schedule.get_by_t(t_minus_1, device=kspace.device, dtype=kspace.dtype)
        delta_m = m_t_minus_1 - m_t

        # Eq.6: k_t = M_t ⊙ k_0
        k_t = apply_filter_degradation(kspace, m_t)

        # fixed conditional observation k_c
        k_c = self._build_conditional_kc(kspace, mask)

        # ground-truth Δk_{t-1} = (M_{t-1} - M_t) ⊙ k_0
        delta_gt = build_delta_target(kspace, m_t, m_t_minus_1)

        # φθ(Cond)
        x0_pred = self._run_backbone(k_t=k_t, k_c=k_c, m_t=m_t, t=t)

        # Eq.7: Rθ = ΔM_{t-1} ⊙ FFT(φθ)
        k0_pred = fastmri.fft2c(x0_pred)
        delta_pred = delta_m * k0_pred

        # Eq.8 losses
        # 只在 DeltaM support 上计算 delta loss，避免被整幅图均值稀释
        delta_mask = delta_m.abs()  # [B,1,H,W,1]
        delta_mask = delta_mask.expand(-1, delta_pred.shape[1], -1, -1, delta_pred.shape[-1])  # [B,Nc,H,W,2]

        if self.loss_type == 'l1':
            delta_abs = torch.abs(delta_pred - delta_gt)
            loss_delta = (delta_abs * delta_mask).sum() / (delta_mask.sum() + 1e-8)


        elif self.loss_type == 'l2':
            delta_sq = (delta_pred - delta_gt) ** 2
            loss_delta = (delta_sq * delta_mask).sum() / (delta_mask.sum() + 1e-8)
        else:
            raise NotImplementedError(f"Unsupported loss type: {self.loss_type}")
        loss_img, loss_img_real, loss_img_imag, loss_img_mag = self._compute_image_losses(x0_pred, x0)

        return loss_delta + self.lambda_img * loss_img

    def _compute_image_losses(self, x0_pred: torch.Tensor, x0: torch.Tensor):
        """
        返回：
          loss_img      : 当前训练真正使用的图像域 loss
          loss_img_real : 仅 real 部分的 loss（debug 用）
          loss_img_imag : 仅 imag 部分的 loss（debug 用）
          loss_img_mag  : magnitude loss（debug 用）
        """
        if self.loss_type == 'l1':
            loss_img_real = F.l1_loss(x0_pred[..., 0], x0[..., 0])
            loss_img_imag = F.l1_loss(x0_pred[..., 1], x0[..., 1])
            loss_img_mag = F.l1_loss(fastmri.complex_abs(x0_pred), fastmri.complex_abs(x0))
        elif self.loss_type == 'l2':
            loss_img_real = F.mse_loss(x0_pred[..., 0], x0[..., 0])
            loss_img_imag = F.mse_loss(x0_pred[..., 1], x0[..., 1])
            loss_img_mag = F.mse_loss(fastmri.complex_abs(x0_pred), fastmri.complex_abs(x0))
        else:
            raise NotImplementedError(f"Unsupported loss type: {self.loss_type}")

        if self.image_loss_mode == "complex":
            if self.loss_type == 'l1':
                loss_img = F.l1_loss(x0_pred, x0)
            else:
                loss_img = F.mse_loss(x0_pred, x0)

        elif self.image_loss_mode == "real":
            loss_img = loss_img_real

        elif self.image_loss_mode == "magnitude":
            loss_img = loss_img_mag

        else:
            raise NotImplementedError(f"Unsupported image_loss_mode: {self.image_loss_mode}")

        return loss_img, loss_img_real, loss_img_imag, loss_img_mag
    @torch.no_grad()
    def sample(self, k_c: torch.Tensor, mask: torch.Tensor, mask_fold=None, t=None):
        """
        Reverse process following Eq.(9).

        Initialization keeps the original FilterDiff logic:
            k_T = M_T ⊙ k_c
        """
        self.denoise_fn.eval()

        if t is None:
            t = self.num_timesteps

        bsz = k_c.shape[0]
        t_tensor = torch.full((bsz,), t, dtype=torch.long, device=k_c.device)
        m_t = self.schedule.get_by_t(t_tensor, device=k_c.device, dtype=k_c.dtype)

        # reverse start from k_T
        k_t = m_t * k_c

        k0_rec, direct_k = run_reverse_loop(
            model=self.denoise_fn,
            k_t=k_t,
            k_c=k_c,
            acq_mask=mask,
            schedule=self.schedule,
            timesteps=t,

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