import sys
import unittest
import torch
import yaml
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mri_data import SliceDataset
from data.data_transform import DataTransform_Diffusion
from utils.sample_mask import EquispacedCartesianMask
from diffusion.kspace_diffusion import KspaceDiffusion
from diffusion.filter_schedule import CenterRectangleSchedule
from models.restoration_net_filterdiff import build_filterdiff_restoration_net
from utils.utils import dict2namespace
import fastmri


class TestFilterDiffFastMRI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """加载配置并初始化所有组件"""
        config_path = '/mnt/SSD/wsy/projects/filter_diffusion/configs/swin.yaml'
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        cls.config = dict2namespace(config_dict)
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 掩码生成器
        size = (1, cls.config.data.image_size, cls.config.data.image_size)
        cls.mask_func = EquispacedCartesianMask(
            acceleration=cls.config.data.R,
            center_fraction=cls.config.data.center_fraction,
            size=size,
            seed=cls.config.data.seed
        )

        # 2. 数据变换
        cls.data_transform = DataTransform_Diffusion(
            cls.mask_func,
            img_size=cls.config.data.image_size,
            combine_coil=cls.config.data.combine_coil,
            flag_singlecoil=False,
            maps_root=cls.config.data.maps_root,
            map_key=cls.config.data.map_key,
        )

        # 3. 数据集
        cls.dataset = SliceDataset(
            root=Path(cls.config.data.data_root),
            transform=cls.data_transform,
            challenge='multicoil',
            num_skip_slice=cls.config.data.num_skip_slice,
        )

        # 4. 恢复网络
        cls.denoise_fn = build_filterdiff_restoration_net(
            img_size=cls.config.data.image_size,
            patch_size=cls.config.model.patch_size,
            in_channels=5,
            out_channels=2,
            hidden_size=cls.config.model.hidden_size,
            depth=cls.config.model.depth,
            num_heads=cls.config.model.num_heads,
            window_size=cls.config.model.window_size,
            mlp_ratio=cls.config.model.mlp_ratio,
            with_time_emb=True,
        ).to(cls.device)

        # 5. 扩散模型
        center_core_size = getattr(
            cls.config.training,
            'center_core_size',
            cls.config.data.image_size // cls.config.data.R
        )
        cls.diffusion = KspaceDiffusion(
            denoise_fn=cls.denoise_fn,
            image_size=cls.config.data.image_size,
            device_of_kernel=str(cls.device),
            channels=2,
            timesteps=cls.config.training.timesteps,
            loss_type=cls.config.training.loss_type,
            schedule_type=cls.config.training.filter_schedule_type,
            center_core_size=center_core_size,
            lambda_img=1.0,
        ).to(cls.device)

    def test_01_data_loading(self):
        """测试数据加载：形状、数值范围、掩码正确性"""
        kspace, mask, mask_fold = self.dataset[0]
        kspace = kspace.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        H, W = self.config.data.image_size, self.config.data.image_size
        self.assertEqual(kspace.shape, (1, 1, H, W, 2))
        self.assertEqual(mask.shape, (1, 1, H, W))

        # 验证图像域归一化
        img = fastmri.complex_abs(fastmri.ifft2c(kspace))
        img_max = img.max().item()
        self.assertLess(img_max, 1.5)
        self.assertGreater(img_max, 0.5)

        # 掩码总采样数校验（中心全宽 + 外围等间隔）
        center_width = int(W * self.config.data.center_fraction)
        center_points = H * center_width
        outer_points = H * (W - center_width) // self.config.data.R
        expected_total = center_points + outer_points

        mask_sum = mask.sum().item()
        self.assertEqual(mask_sum, expected_total)

        # 中心区域应为全1矩形
        center_start = (W - center_width) // 2
        center_region = mask[0, 0, :, center_start:center_start + center_width]
        self.assertTrue(torch.all(center_region == 1.0))

        print(f"✅ 数据加载测试通过：图像域最大幅值={img_max:.4f}，掩码采样数={mask_sum}")

    def test_02_mask_schedule(self):
        """测试扩散掩码序列"""
        schedule = self.diffusion.schedule
        T = self.config.training.timesteps
        H, W = self.config.data.image_size, self.config.data.image_size

        # M_0 应全1
        m0 = schedule.get_by_t(torch.tensor([0], device=self.device))
        self.assertTrue(torch.all(m0 == 1.0))

        # M_T 宽度应为 center_core_size
        mT = schedule.get_by_t(torch.tensor([T], device=self.device))
        non_zero_cols = torch.where(mT[0, 0, :, :, 0].sum(dim=0) > 0)[0]
        width = non_zero_cols.max().item() - non_zero_cols.min().item() + 1
        expected_width = self.config.training.center_core_size
        self.assertEqual(width, expected_width)

        # 单调递减
        prev_sum = None
        for t in range(T + 1):
            mt = schedule.get_by_t(torch.tensor([t], device=self.device))
            curr_sum = mt.sum().item()
            if prev_sum is not None:
                self.assertGreaterEqual(prev_sum, curr_sum)
            prev_sum = curr_sum

        print(f"✅ 扩散掩码序列测试通过：M_0 全1，M_{T} 宽度={width}")

    def test_03_forward_diffusion(self):
        """测试前向扩散 k_t = M_t ⊙ k_0"""
        kspace, mask, _ = self.dataset[0]
        kspace = kspace.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        t = torch.tensor([10], device=self.device).long()
        m_t = self.diffusion.schedule.get_by_t(t, device=self.device)
        k_t = m_t * kspace

        # 掩码为0的区域k_t应为0
        m_t_bool = m_t.squeeze() > 0.5
        k_t_mag = fastmri.complex_abs(k_t)
        zero_region = k_t_mag[..., ~m_t_bool]
        self.assertTrue(torch.all(zero_region < 1e-5))

        print("✅ 前向扩散测试通过")

    def test_04_conditional_kc(self):
        """测试条件输入 k_c = M_full ⊙ k_0"""
        kspace, mask, _ = self.dataset[0]
        kspace = kspace.unsqueeze(0).to(self.device)  # [1, 1, H, W, 2]
        mask = mask.unsqueeze(0).to(self.device)  # [1, 1, H, W]

        k_c = self.diffusion._build_conditional_kc(kspace, mask)

        # 获取掩码布尔值，形状 [1, 1, H, W]
        mask_bool = mask > 0.5  # 保持四维以便直接索引
        k_c_mag = fastmri.complex_abs(k_c)

        # 验证掩码为0的位置k_c幅值接近0
        self.assertTrue(torch.all(k_c_mag[~mask_bool] < 1e-5))

        # 验证掩码为1的位置k_c与kspace幅值相等（使用幅值比较避免复数相位差异）
        kspace_mag = fastmri.complex_abs(kspace)
        self.assertTrue(torch.allclose(k_c_mag[mask_bool], kspace_mag[mask_bool], atol=1e-6))

        print("✅ 条件输入k_c测试通过")

    def test_05_network_forward(self):
        """测试恢复网络前向传播"""
        kspace, mask, _ = self.dataset[0]
        kspace = kspace.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        t = torch.tensor([5], device=self.device).long()
        m_t = self.diffusion.schedule.get_by_t(t, device=self.device)
        k_t = m_t * kspace
        k_c = self.diffusion._build_conditional_kc(kspace, mask)

        x0_pred = self.diffusion._run_backbone(k_t, k_c, m_t, t.expand(1))

        H, W = self.config.data.image_size, self.config.data.image_size
        self.assertEqual(x0_pred.shape, (1, 1, H, W, 2))
        self.assertTrue(torch.isfinite(x0_pred).all())

        print(f"✅ 网络前向测试通过：输出形状={x0_pred.shape}")

    def test_06_loss_computation(self):
        """测试损失计算"""
        kspace, mask, _ = self.dataset[0]
        kspace = kspace.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        t = torch.tensor([8], device=self.device).long()
        loss = self.diffusion.p_losses(kspace, mask, t)

        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        print(f"✅ 损失计算测试通过：loss={loss.item():.6f}")

    def test_07_reverse_sampling(self):
        """测试反向采样与确定性"""
        kspace, mask, _ = self.dataset[0]
        kspace = kspace.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        k_c = self.diffusion._build_conditional_kc(kspace, mask)

        self.diffusion.eval()
        with torch.no_grad():
            xt, direct_recons, img = self.diffusion.sample(k_c, mask, t=20)

        H, W = self.config.data.image_size, self.config.data.image_size
        self.assertEqual(img.shape, (1, 1, H, W, 2))
        self.assertTrue(torch.isfinite(img).all())

        # 确定性：两次采样应完全相同
        with torch.no_grad():
            _, _, img2 = self.diffusion.sample(k_c, mask, t=20)
        self.assertTrue(torch.allclose(img, img2, atol=1e-5))

        print(f"✅ 反向采样测试通过：输出形状={img.shape}，确定性成立")

    def test_08_end_to_end_gradient(self):
        """测试端到端梯度流"""
        kspace, mask, _ = self.dataset[0]
        kspace = kspace.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)
        kspace.requires_grad = False
        mask.requires_grad = False

        t = torch.tensor([10], device=self.device).long()
        loss = self.diffusion.p_losses(kspace, mask, t)
        loss.backward()

        # 检查是否有参数获得了梯度
        has_grad = False
        for name, param in self.diffusion.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        self.assertTrue(has_grad, "No parameter received gradient")

        print("✅ 端到端梯度流测试通过")


if __name__ == "__main__":
    # 设置测试顺序，避免随机性导致的不稳定
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTest(TestFilterDiffFastMRI('test_01_data_loading'))
    suite.addTest(TestFilterDiffFastMRI('test_02_mask_schedule'))
    suite.addTest(TestFilterDiffFastMRI('test_03_forward_diffusion'))
    suite.addTest(TestFilterDiffFastMRI('test_04_conditional_kc'))
    suite.addTest(TestFilterDiffFastMRI('test_05_network_forward'))
    suite.addTest(TestFilterDiffFastMRI('test_06_loss_computation'))
    suite.addTest(TestFilterDiffFastMRI('test_07_reverse_sampling'))
    suite.addTest(TestFilterDiffFastMRI('test_08_end_to_end_gradient'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)