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
            schedule_type='linear',
            center_core_size=32,
            **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device_of_kernel = device_of_kernel

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

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

        # 1. 获取当前时刻 t 和前一刻 t-1 的滤波掩码
        m_t = self.schedule.get_by_t(t, device=kspace.device, dtype=kspace.dtype)
        t_minus_1 = torch.clamp(t - 1, min=0)
        m_t_minus_1 = self.schedule.get_by_t(t_minus_1, device=kspace.device, dtype=kspace.dtype)

        # 2. 前向退化：k_t = M_t * k_0
        k_t = apply_filter_degradation(kspace, m_t)
        # 3. 条件部分：k_c = M_acq * k_0（采集到的低频/中心部分）
        k_c = self._build_conditional_kc(kspace, mask)
        # 4. 真实目标残差：delta_gt = (M_{t-1} - M_t) * k_0
        delta_gt = build_delta_target(kspace, m_t, m_t_minus_1)

        # 5. 准备网络输入：将 k_t, k_c, m_t 沿通道拼接
        m_t_ch = m_t.expand(-1, ncoil, -1, -1, -1)  # 扩展到多线圈

        # 将 [B,Nc,H,W,2] 展平为 [B*Nc,H,W,2]，然后 permute 为 [B*Nc,2,H,W]
        kt_in = k_t.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)
        kc_in = k_c.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)
        mt_in = m_t_ch.reshape(bsz * ncoil, h, w, 1).permute(0, 3, 1, 2)

        # 拼接：最终输入形状 [B*Nc, 5, H, W]（2+2+1=5）
        model_in = torch.cat([kt_in, kc_in, mt_in], dim=1)
        t_in = t.repeat_interleave(ncoil)  # 时间步复制到每个线圈

        # 6. 网络预测残差
        delta_pred = self.denoise_fn(model_in, t_in).permute(0, 2, 3, 1).reshape(bsz, ncoil, h, w, 2)

        # 7. 计算损失
        if self.loss_type == 'l1':
            loss = F.l1_loss(delta_pred, delta_gt)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(delta_pred, delta_gt)
        else:
            raise NotImplementedError(f"Unsupported loss type: {self.loss_type}")

        return loss

    @torch.no_grad()
    def sample(self, kspace: torch.Tensor, mask: torch.Tensor, mask_fold=None, t=None):
        """
        Paired validation mode:
        - `kspace` is full-sampled k0 from dataloader
        - k_T is initialized by degrading k0 with M_t

        This is NOT kc-only deployment inference.
        """
        self.denoise_fn.eval()

        if t is None:
            t = self.num_timesteps

        bsz = kspace.shape[0]
        t_init = torch.full((bsz,), t, dtype=torch.long, device=kspace.device)
        m_t = self.schedule.get_by_t(t_init, device=kspace.device, dtype=kspace.dtype)

        k_t = apply_filter_degradation(kspace, m_t)
        k_c = self._build_conditional_kc(kspace, mask)

        k_rec, direct_k = run_reverse_loop(
            model=self.denoise_fn,
            k_t=k_t,
            k_c=k_c,
            acq_mask=mask,
            schedule=self.schedule,
            timesteps=t,
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