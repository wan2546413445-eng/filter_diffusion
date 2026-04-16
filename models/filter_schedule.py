import math
from typing import Tuple, Union

import torch


CoreSize = Union[int, Tuple[int, int]]


def _resolve_core_size(center_core_size: CoreSize, h: int, w: int) -> Tuple[int, int]:
    if isinstance(center_core_size, int):
        core_h, core_w = center_core_size, center_core_size
    else:
        core_h, core_w = center_core_size
    core_h = max(1, min(h, int(core_h)))
    core_w = max(1, min(w, int(core_w)))
    return core_h, core_w


def _schedule_progress(t: int, timesteps: int, schedule_type: str) -> float:
    if timesteps <= 0:
        return 1.0
    p = float(t) / float(timesteps)
    if schedule_type == "linear":
        return p
    if schedule_type == "dense":
        return math.sqrt(p)
    if schedule_type == "sparse":
        return p ** 2
    raise ValueError(f"Unsupported schedule_type: {schedule_type}")


class CenterRectangleSchedule:
    """
    Build center-rectangle masks M_t, t=0..T.
    M_0: full pass; M_T: center core mask.
    Tensor shape: [T+1, 1, H, W, 1]
    """

    def __init__(
        self,
        h: int,
        w: int,
        timesteps: int,
        center_core_size: CoreSize,
        schedule_type: str = "linear",
    ):
        self.h = int(h)
        self.w = int(w)
        self.timesteps = int(timesteps)
        self.center_core_size = center_core_size
        self.schedule_type = schedule_type

        self.masks = self._build_masks()

    def _build_masks(self) -> torch.Tensor:
        core_h, core_w = _resolve_core_size(self.center_core_size, self.h, self.w)
        masks = []

        for t in range(self.timesteps + 1):
            p = _schedule_progress(t, self.timesteps, self.schedule_type)

            cur_h = int(round(self.h - p * (self.h - core_h)))
            cur_w = int(round(self.w - p * (self.w - core_w)))
            cur_h = max(core_h, min(self.h, cur_h))
            cur_w = max(core_w, min(self.w, cur_w))

            top = (self.h - cur_h) // 2
            left = (self.w - cur_w) // 2

            m = torch.zeros(1, self.h, self.w, 1, dtype=torch.float32)
            m[:, top: top + cur_h, left: left + cur_w, :] = 1.0
            masks.append(m)

        return torch.stack(masks, dim=0)  # [T+1,1,H,W,1]

    def get_by_t(self, t: torch.Tensor, device=None, dtype=torch.float32) -> torch.Tensor:
        """t: [B] long -> [B,1,H,W,1]"""
        if t.dtype != torch.long:
            t = t.long()
        m = self.masks.to(device=device, dtype=dtype)
        return m.index_select(0, t)