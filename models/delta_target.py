import torch


def build_delta_target(k0: torch.Tensor, m_t: torch.Tensor, m_t_minus_1: torch.Tensor) -> torch.Tensor:
    """
    delta_gt = (M_{t-1} - M_t) * k0
    k0: [B,Nc,H,W,2]
    m_t/m_t_minus_1: [B,1,H,W,1]
    """
    return (m_t_minus_1 - m_t) * k0
