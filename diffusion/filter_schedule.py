from typing import Tuple, Union
import torch

import numpy as np
CoreSize = Union[int, Tuple[int, int]]


_CURVE_PROFILE = {
    "linear": [
        1.00000000, 0.95000000, 0.90000000, 0.85000000, 0.80000000,
        0.75000000, 0.70000000, 0.65000000, 0.60000000, 0.55000000,
        0.50000000, 0.45000000, 0.40000000, 0.35000000, 0.30000000,
        0.25000000, 0.20000000, 0.15000000, 0.10000000, 0.05000000,
    ],
    "sparse": [
        1.00000000, 0.98787021, 0.97059083, 0.95115667, 0.93037302,
        0.90817255, 0.88399033, 0.85705806, 0.82661769, 0.79205430,
        0.75294826, 0.70904674, 0.66015452, 0.60594406, 0.54568489,
        0.47789235, 0.39989555, 0.30732471, 0.19351777, 0.04884625,
    ],
    "dense": [
        1.00000000, 0.85648223, 0.74267529, 0.65010445, 0.57210765,
        0.50431511, 0.44405594, 0.38984548, 0.34095326, 0.29705174,
        0.25794570, 0.22338231, 0.19294194, 0.16600967, 0.14182745,
        0.11962698, 0.09884333, 0.07940917, 0.06212979, 0.04913874,
    ],
}

def _resolve_core_size(center_core_size: CoreSize, h: int, w: int) -> Tuple[int, int]:
    if isinstance(center_core_size, int):
        core_h, core_w = center_core_size, center_core_size
    else:
        core_h, core_w = center_core_size
    core_h = max(1, min(h, int(core_h)))
    core_w = max(1, min(w, int(core_w)))
    return core_h, core_w
def _ratio_at_t(t: int, timesteps: int, schedule_type: str, r_min: float) -> float:
    """
    精确按步查表，不插值。

    这里默认作者图对应 20 个离散状态：t = 0..19
    - 若 timesteps=19：严格对应图上的 20 个点
    - 若 timesteps=20：额外在最后补一个终点平台，t=20 继续保持最后一个值
    """
    if schedule_type not in _CURVE_PROFILE:
        raise ValueError(f"Unsupported schedule_type: {schedule_type}")

    base_curve = np.asarray(_CURVE_PROFILE[schedule_type], dtype=np.float32)

    if timesteps == 19:
        curve = base_curve
    elif timesteps == 20:
        curve = np.concatenate([base_curve, [base_curve[-1]]], axis=0)
    else:
        raise ValueError(
            f"Exact step-matched schedule only supports timesteps=19 or 20, got {timesteps}."
        )

    t = int(max(0, min(t, timesteps)))
    profile = float(curve[t])

    # 图上 0.05~1.00 映射到实际 r_min~1.00
    profile_norm = (profile - 0.05) / 0.95
    profile_norm = min(max(profile_norm, 0.0), 1.0)

    ratio = r_min + (1.0 - r_min) * profile_norm
    return max(r_min, min(1.0, ratio))

class CenterRectangleSchedule:
    """
    Build center-rectangle masks M_t, t=0..T.
    M_0: all ones; M_T: center core rectangle.
    Tensor shape: [T+1, 1, H, W, 1]
    """

    def __init__(
        self,
        h: int,
        w: int,
        timesteps: int,
        center_core_size: CoreSize,
        schedule_type: str = "dense",
    ):
        self.h = int(h)
        self.w = int(w)
        self.timesteps = int(timesteps)
        self.center_core_size = center_core_size
        self.schedule_type = schedule_type

        self.masks = self._build_masks()

    def _build_masks(self) -> torch.Tensor:
        core_h, core_w = _resolve_core_size(self.center_core_size, self.h, self.w)
        # Paper setting keeps all rows and shrinks only center columns (phase-encoding direction).
        core_h = self.h
        r_min = float(core_w) / float(self.w)
        masks = []

        for t in range(self.timesteps + 1):
            r_t = _ratio_at_t(t, self.timesteps, self.schedule_type, r_min)

            cur_h = core_h
            cur_w = int(round(r_t * self.w))
            cur_h = max(core_h, min(self.h, cur_h))
            cur_w = max(core_w, min(self.w, cur_w))

            top = (self.h - cur_h) // 2
            left = (self.w - cur_w) // 2

            m = torch.zeros(1, self.h, self.w, 1, dtype=torch.float32)
            m[:, top : top + cur_h, left : left + cur_w, :] = 1.0
            masks.append(m)

        return torch.stack(masks, dim=0)

    def get_by_t(self, t: torch.Tensor, device=None, dtype=torch.float32) -> torch.Tensor:
        if t.dtype != torch.long:
            t = t.long()
        # 优先使用传入的device，若为None则使用t的device
        target_device = device if device is not None else t.device
        m = self.masks.to(device=target_device, dtype=dtype)
        return m.index_select(0, t)