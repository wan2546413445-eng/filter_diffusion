
import sys
import unittest
import torch

# 假设你的项目根目录在 sys.path 中，或者使用相对导入
# 请根据你的实际包结构调整导入路径
sys.path.append(".")  # 或 sys.path.append("..")
from diffusion.filter_schedule import CenterRectangleSchedule


class TestCenterRectangleSchedule(unittest.TestCase):

    def setUp(self):
        """设置测试用的基本参数"""
        self.h = 256
        self.w = 256
        self.T = 20
        self.center_core_size = 32  # 注意：论文要求 4× 加速时应为 64
        self.schedule_type = 'dense'

    def test_initialization(self):
        """测试 schedule 对象能否正常初始化"""
        schedule = CenterRectangleSchedule(
            h=self.h,
            w=self.w,
            timesteps=self.T,
            center_core_size=self.center_core_size,
            schedule_type=self.schedule_type
        )
        self.assertIsNotNone(schedule)

    def test_endpoint_masks(self):
        """测试 M_0 应为全 1，M_T 应只有中心区域为 1"""
        schedule = CenterRectangleSchedule(
            h=self.h,
            w=self.w,
            timesteps=self.T,
            center_core_size=self.center_core_size,
            schedule_type=self.schedule_type
        )
        device = torch.device('cpu')
        dtype = torch.float32

        # M_0
        t0 = torch.tensor([0], device=device)
        m0 = schedule.get_by_t(t0, device=device, dtype=dtype)
        # m0 形状可能是 (1,1,H,W) 或 (1,H,W)，统一检查
        self.assertTrue(torch.all(m0 == 1.0), "M_0 should be all ones")

        # M_T
        tT = torch.tensor([self.T], device=device)
        mT = schedule.get_by_t(tT, device=device, dtype=dtype)
        # 计算中心区域宽度
        if mT.dim() == 4:  # (1,1,H,W)
            mT_squeezed = mT.squeeze()
        else:
            mT_squeezed = mT
        center_sum = mT_squeezed.sum().item()
        expected_sum = self.h * self.center_core_size
        self.assertEqual(center_sum, expected_sum,
                         f"M_T center sum should be {expected_sum}, got {center_sum}")

    def test_monotonic_decreasing(self):
        """测试掩码宽度（非零元素数量）随时间步增加而单调非增"""
        schedule = CenterRectangleSchedule(
            h=self.h,
            w=self.w,
            timesteps=self.T,
            center_core_size=self.center_core_size,
            schedule_type=self.schedule_type
        )
        device = torch.device('cpu')
        dtype = torch.float32

        prev_sum = None
        for t in range(self.T + 1):
            mt = schedule.get_by_t(torch.tensor([t], device=device), device=device, dtype=dtype)
            curr_sum = mt.sum().item()
            if prev_sum is not None:
                self.assertGreaterEqual(prev_sum, curr_sum,
                                        f"Width should be non-increasing at t={t}")
            prev_sum = curr_sum

    def test_dense_mode_mid_slope(self):
        """测试 Dense 模式下，中间时间步的宽度变化量应大于两端"""
        schedule = CenterRectangleSchedule(
            h=self.h,
            w=self.w,
            timesteps=self.T,
            center_core_size=self.center_core_size,
            schedule_type=self.schedule_type
        )
        device = torch.device('cpu')
        dtype = torch.float32

        sums = []
        for t in range(self.T + 1):
            mt = schedule.get_by_t(torch.tensor([t], device=device), device=device, dtype=dtype)
            sums.append(mt.sum().item())

        diffs = [sums[i] - sums[i+1] for i in range(self.T)]
        mid_idx = self.T // 2
        # 至少中间的变化量不应小于第一个变化量（宽松条件，因 Dense 是中间陡峭）
        self.assertGreaterEqual(diffs[mid_idx], diffs[0],
                                "Dense mode: mid change should be >= first change")


if __name__ == "__main__":
    unittest.main()