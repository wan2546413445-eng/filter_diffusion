import sys
import unittest
import torch

sys.path.append(".")
from diffusion.filter_schedule import CenterRectangleSchedule, _ratio_at_t


class TestCenterRectangleSchedule(unittest.TestCase):

    def setUp(self):
        self.h = 256
        self.w = 256
        self.timesteps = 20
        self.center_core_size = 32   # 当前设置，测试时建议同时验证 64 的情况
        self.schedule_type = "dense"
        self.schedule = CenterRectangleSchedule(
            h=self.h,
            w=self.w,
            timesteps=self.timesteps,
            center_core_size=self.center_core_size,
            schedule_type=self.schedule_type
        )

    def test_initialization(self):
        """测试 schedule 正常初始化"""
        self.assertEqual(self.schedule.h, self.h)
        self.assertEqual(self.schedule.w, self.w)
        self.assertEqual(self.schedule.timesteps, self.timesteps)
        self.assertEqual(self.schedule.masks.shape[0], self.timesteps + 1)

    def test_endpoint_masks(self):
        """测试 M_0 全1，M_T 中心区域正确"""
        device = torch.device('cpu')
        t0 = torch.tensor([0], device=device)
        m0 = self.schedule.get_by_t(t0, device=device)
        self.assertEqual(m0.shape, (1, 1, self.h, self.w, 1))
        self.assertTrue(torch.all(m0 == 1.0), "M_0 should be all ones")

        tT = torch.tensor([self.timesteps], device=device)
        mT = self.schedule.get_by_t(tT, device=device)

        # 找出非零列（相位编码方向）
        col_sum = mT[0, 0, :, :, 0].sum(dim=0)  # 对高度方向求和，形状 (W,)
        non_zero_cols = torch.where(col_sum > 0)[0]
        width = non_zero_cols.max().item() - non_zero_cols.min().item() + 1 if len(non_zero_cols) > 0 else 0
        expected_width = self.center_core_size
        self.assertEqual(width, expected_width,
                         f"MT center width should be {expected_width}, got {width}")

        # 验证在中心区域内，每一列的所有行都为 1
        for col in non_zero_cols:
            self.assertEqual(col_sum[col].item(), self.h,
                             f"Column {col} should have all {self.h} rows set to 1")

    def test_monotonic_decreasing(self):
        """测试掩码宽度随 t 增加而单调非增"""
        device = torch.device('cpu')
        prev_width = None
        for t in range(self.timesteps + 1):
            mt = self.schedule.get_by_t(torch.tensor([t], device=device), device=device)
            non_zero_cols = torch.where(mt[0,0,:,:,0].sum(dim=0) > 0)[0]
            width = non_zero_cols.max().item() - non_zero_cols.min().item() + 1 if len(non_zero_cols) > 0 else 0
            if prev_width is not None:
                self.assertGreaterEqual(prev_width, width,
                                        f"Width at t={t} should be <= width at t={t-1}")
            prev_width = width

    def test_dense_mode_mid_slope(self):
        """测试 Dense 模式下中间时间步的宽度变化量大于两端"""
        device = torch.device('cpu')
        widths = []
        for t in range(self.timesteps + 1):
            mt = self.schedule.get_by_t(torch.tensor([t], device=device), device=device)
            non_zero_cols = torch.where(mt[0,0,:,:,0].sum(dim=0) > 0)[0]
            width = non_zero_cols.max().item() - non_zero_cols.min().item() + 1 if len(non_zero_cols) > 0 else 0
            widths.append(width)

        diffs = [widths[i] - widths[i+1] for i in range(self.timesteps)]
        mid_idx = self.timesteps // 2
        # Dense 模式下中间变化量应大于两端（至少第一个变化量）
        self.assertGreaterEqual(diffs[mid_idx], diffs[0],
                                "Dense mode: mid change should be >= first change")

    def test_ratio_function(self):
        """测试 _ratio_at_t 函数在 dense 模式下的行为"""
        T = 20
        r_min = 32 / 256  # 0.125
        ratios = [_ratio_at_t(t, T, "dense", r_min) for t in range(T+1)]
        self.assertAlmostEqual(ratios[0], 1.0, places=5)
        self.assertAlmostEqual(ratios[-1], r_min, places=5)
        # 中间值应在两端之间，且 dense 模式下前期下降较慢，后期加快
        # 验证曲线单调递减
        self.assertTrue(all(ratios[i] >= ratios[i+1] for i in range(T)))

    def test_batch_selection(self):
        """测试 get_by_t 对 batch 索引的支持"""
        device = torch.device('cpu')
        t_batch = torch.tensor([0, 10, 20], device=device)
        masks = self.schedule.get_by_t(t_batch, device=device)
        self.assertEqual(masks.shape[0], 3)
        # 分别检查对应的宽度
        for i, t in enumerate(t_batch.tolist()):
            mt = masks[i:i+1]
            t_single = torch.tensor([t], device=device)
            mt_single = self.schedule.get_by_t(t_single, device=device)
            self.assertTrue(torch.equal(mt, mt_single))


class TestCenterRectangleScheduleWithCorrectCoreSize(unittest.TestCase):
    """验证当 center_core_size 设置为正确的 1/4 宽度时的行为"""
    def test_core_size_64_for_4x(self):
        h = w = 256
        acc = 4
        core_w = w // acc  # 64
        schedule = CenterRectangleSchedule(
            h=h, w=w, timesteps=20,
            center_core_size=core_w,
            schedule_type="dense"
        )
        mT = schedule.get_by_t(torch.tensor([20]), device='cpu')
        non_zero_cols = torch.where(mT[0,0,:,:,0].sum(dim=0) > 0)[0]
        width = non_zero_cols.max().item() - non_zero_cols.min().item() + 1
        self.assertEqual(width, core_w)


if __name__ == "__main__":
    unittest.main()