import sys
import unittest
import torch

sys.path.append(".")
from diffusion.kspace_diffusion import KspaceDiffusion
from diffusion.filter_schedule import CenterRectangleSchedule


class DummyDenoiseFn(torch.nn.Module):
    """模拟恢复网络，接受 5 通道输入，输出 2 通道"""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 2, kernel_size=1)

    def forward(self, x, t):
        # x: (B,5,H,W), t: (B,)
        return self.conv(x)


class TestKspaceDiffusionComponents(unittest.TestCase):

    def setUp(self):
        self.image_size = 256
        self.channels = 2
        self.timesteps = 20
        self.center_core_size = 32
        self.device = torch.device('cpu')
        self.dtype = torch.float32

        self.denoise_fn = DummyDenoiseFn()
        self.diffusion = KspaceDiffusion(
            denoise_fn=self.denoise_fn,
            image_size=self.image_size,
            device_of_kernel=self.device,
            channels=self.channels,
            timesteps=self.timesteps,
            schedule_type='dense',
            center_core_size=self.center_core_size,
        )

    def test_forward_diffusion_eq6(self):
        """测试公式 (6): k_t = M_t * k_0"""
        B, Nc, H, W = 2, 1, self.image_size, self.image_size
        kspace = torch.randn(B, Nc, H, W, 2, device=self.device)
        t = torch.randint(1, self.timesteps, (B,), device=self.device).long()
        m_t = self.diffusion.schedule.get_by_t(t, device=self.device, dtype=self.dtype)

        from diffusion.degradation import apply_filter_degradation
        k_t = apply_filter_degradation(kspace, m_t)

        # 验证非零区域应与 m_t 一致
        # m_t 形状可能为 (B,1,H,W,1) 或 (B,H,W)，需统一处理
        if m_t.dim() == 5:
            m_t_squeezed = m_t.squeeze(1).squeeze(-1)  # -> (B,H,W)
        else:
            m_t_squeezed = m_t.squeeze(1)  # 假设 (B,1,H,W)
        m_t_bool = m_t_squeezed > 0.5
        k_t_mag = torch.sqrt(k_t[..., 0]**2 + k_t[..., 1]**2)
        # 对于每个 batch，m_t 为 0 的位置，k_t 应该接近 0
        for b in range(B):
            mask_zero = ~m_t_bool[b]
            self.assertTrue(torch.all(k_t_mag[b, 0][mask_zero] < 1e-5),
                            f"Batch {b}: Regions where M_t=0 should have zero k-space")

    def test_delta_gt_calculation(self):
        """测试 Δk_gt = (M_{t-1} - M_t) * k_0"""
        B, Nc, H, W = 1, 1, self.image_size, self.image_size
        kspace = torch.randn(B, Nc, H, W, 2, device=self.device)
        t = torch.randint(1, self.timesteps, (B,), device=self.device).long()
        m_t = self.diffusion.schedule.get_by_t(t, device=self.device, dtype=self.dtype)
        t_minus_1 = torch.clamp(t - 1, min=0)
        m_t_minus_1 = self.diffusion.schedule.get_by_t(t_minus_1, device=self.device, dtype=self.dtype)

        from diffusion.delta_target import build_delta_target
        delta_gt = build_delta_target(kspace, m_t, m_t_minus_1)

        # 手动计算
        delta_m = m_t_minus_1 - m_t
        delta_gt_manual = delta_m * kspace

        self.assertTrue(torch.allclose(delta_gt, delta_gt_manual, atol=1e-5),
                        "Δk_gt calculation mismatch")

    def test_network_input_output_shapes(self):
        """测试 _run_backbone 的输入输出形状正确"""
        B, Nc, H, W = 2, 1, self.image_size, self.image_size
        kspace = torch.randn(B, Nc, H, W, 2, device=self.device)
        mask = torch.ones(B, 1, H, W, device=self.device)

        t = torch.randint(1, self.timesteps, (B,), device=self.device).long()
        m_t = self.diffusion.schedule.get_by_t(t, device=self.device, dtype=self.dtype)

        k_c = self.diffusion._build_conditional_kc(kspace, mask)
        k_t = m_t * kspace  # 简化前向

        x0_pred = self.diffusion._run_backbone(k_t, k_c, m_t, t)

        expected_shape = (B, Nc, H, W, 2)
        self.assertEqual(x0_pred.shape, expected_shape,
                         f"x0_pred shape should be {expected_shape}, got {x0_pred.shape}")

    def test_loss_computation(self):
        """测试 p_losses 能够正常计算并返回标量"""
        B, Nc, H, W = 1, 1, self.image_size, self.image_size
        kspace = torch.randn(B, Nc, H, W, 2, device=self.device)
        mask = torch.ones(B, 1, H, W, device=self.device)
        t = torch.randint(1, self.timesteps, (B,), device=self.device).long()

        loss = self.diffusion.p_losses(kspace, mask, t)
        self.assertEqual(loss.dim(), 0, "Loss should be a scalar")
        self.assertTrue(torch.isfinite(loss), "Loss should be finite")


if __name__ == "__main__":
    unittest.main()