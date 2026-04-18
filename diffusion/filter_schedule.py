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


def _ratio_at_t(t: int, timesteps: int, schedule_type: str, r_min: float) -> float:
    """
    r_t schedule from paper:
      linear: 1 - (1-r_min) * (t/T)
      sparse: r_min + (1-r_min) * (1-t/T)^3
      dense : r_min + (1-r_min) * (1-t/T)^(1/3)
    """
    p = float(t) / float(timesteps)
    if schedule_type == "linear":
        r_t = 1.0 - (1.0 - r_min) * p
    elif schedule_type == "sparse":
        r_t = r_min + (1.0 - r_min) * ((1.0 - p) ** 3)
    elif schedule_type == "dense":
        r_t = r_min + (1.0 - r_min) * ((1.0 - p) ** (1.0 / 3.0))
    else:
        raise ValueError(f"Unsupported schedule_type: {schedule_type}")
    return max(r_min, min(1.0, r_t))


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
        m = self.masks.to(device=device, dtype=dtype)
        return m.index_select(0, t)