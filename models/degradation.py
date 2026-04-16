import torch


def apply_filter_degradation(k0: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
    """
    k0: [B,Nc,H,W,2]
    m_t: [B,1,H,W,1] or broadcast-compatible
    """
    return k0 * m_t