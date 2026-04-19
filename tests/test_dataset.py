import sys
import unittest
from pathlib import Path
from data.ixi_singlecoil_dataset import IXISinglecoilSliceDataset
import torch


class TestIXIDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """设置整个测试类共享的变量"""
        # 请修改为你本地的 IXI 数据根目录
        cls.data_root = "/mnt/SSD/wsy/data/train"

        # 临时构造一个简单的 mask_func（如果还没有实现 mask 模块）
        # 请根据你实际的 mask_func 替换这个 lambda
        def dummy_mask_func():
            # 返回两个占位张量：mask [1, H, W] 和 mask_fold [1, h, w]
            # 注意：这只是为了让测试能跑通，后续你要换成真实的 mask_func
            mask = torch.ones(1, 256, 256)
            mask_fold = torch.ones(1, 64, 64)  # 假设某种折叠尺寸
            return mask.numpy(), mask_fold.numpy()

        cls.mask_func = dummy_mask_func
        cls.image_size = 256

    def test_dataset_length(self):
        """测试数据集能否正常初始化并返回长度"""
        dataset = IXISinglecoilSliceDataset(
            root=self.data_root,
            mask_func=self.mask_func,
            image_size=self.image_size
        )
        self.assertGreater(len(dataset), 0, "数据集长度应为正数")

    def test_getitem_shape_and_type(self):
        """测试 __getitem__ 返回的张量形状和数据类型"""
        dataset = IXISinglecoilSliceDataset(
            root=self.data_root,
            mask_func=self.mask_func,
            image_size=self.image_size
        )
        kspace, mask, mask_fold = dataset[0]

        # 检查形状
        self.assertEqual(kspace.shape, (1, 256, 256, 2),
                         f"kspace shape 应为 (1, 256, 256, 2)，实际为 {kspace.shape}")
        self.assertEqual(mask.shape, (1, 256, 256),
                         f"mask shape 应为 (1, 256, 256)，实际为 {mask.shape}")
        # mask_fold 的形状取决于你的实现，这里不强制断言，但可检查是二维或三维

        # 检查数据类型
        self.assertEqual(kspace.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask_fold.dtype, torch.float32)

    def test_kspace_shift(self):
        """验证 k-space 是否经过了正确的 fftshift，即中心应为低频高能量"""
        dataset = IXISinglecoilSliceDataset(
            root=self.data_root,
            mask_func=self.mask_func,
            image_size=self.image_size
        )
        kspace, _, _ = dataset[0]
        k_mag = torch.sqrt(kspace[..., 0] ** 2 + kspace[..., 1] ** 2)
        center_val = k_mag[0, 128, 128].item()
        corner_val = k_mag[0, 0, 0].item()
        self.assertGreater(center_val, corner_val * 5,
                           "k-space 中心能量应显著大于角落能量，可能未正确 fftshift")

    def test_mask_binary(self):
        """检查 mask 是否只包含 0 和 1"""
        dataset = IXISinglecoilSliceDataset(
            root=self.data_root,
            mask_func=self.mask_func,
            image_size=self.image_size
        )
        _, mask, _ = dataset[0]
        unique_vals = torch.unique(mask)
        self.assertTrue(torch.all((unique_vals == 0) | (unique_vals == 1)),
                        "mask 应只包含 0 和 1")

    def test_center_region_fully_sampled(self):
        """
        检查掩码的中心低频区域是否全为 1（矩形连续区域）
        注意：这个测试依赖于真实的 mask_func，如果你的 dummy 是全 1 则测试无意义。
        请在替换为真实 mask_func 后运行。
        """
        dataset = IXISinglecoilSliceDataset(
            root=self.data_root,
            mask_func=self.mask_func,
            image_size=self.image_size
        )
        _, mask, _ = dataset[0]
        # 假设加速倍数为 4，中心宽度应为 256 / 4 = 64
        acc = 4
        expected_width = self.image_size // acc
        center_start = self.image_size // 2 - expected_width // 2
        center_end = center_start + expected_width
        center_region = mask[0, :, center_start:center_end]
        self.assertTrue(torch.all(center_region == 1.0),
                        "中心低频区域（矩形）应全为 1，请检查 mask_func 实现")

    def test_normalization_deterministic(self):
        """测试 _normalize_slice 方法在相同输入下输出一致（无随机性）"""
        dataset = IXISinglecoilSliceDataset(
            root=self.data_root,
            mask_func=self.mask_func,
            image_size=self.image_size
        )
        # 构造一个随机图像
        x = torch.rand(256, 256) * 100
        norm1 = dataset._normalize_slice(x.clone())
        norm2 = dataset._normalize_slice(x.clone())
        self.assertTrue(torch.allclose(norm1, norm2),
                        "归一化方法应具有确定性，但两次结果不一致")


if __name__ == '__main__':
    unittest.main()
