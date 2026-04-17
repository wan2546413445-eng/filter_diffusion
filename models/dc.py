import torch


def explicit_data_consistency(k_pred: torch.Tensor, k_c: torch.Tensor, acq_mask: torch.Tensor) -> torch.Tensor:
    """
    Replace observed k-space entries by conditional measurements k_c.

    k_pred:   [B,Nc,H,W,2]
    k_c:      [B,Nc,H,W,2]
    acq_mask: [B,1,H,W] or [B,1,H,W,1], with 1=observed
    """
    if acq_mask.dim() == 4:
        acq_mask = acq_mask.unsqueeze(-1)
    mask = acq_mask.to(dtype=k_pred.dtype, device=k_pred.device)
    return k_pred * (1.0 - mask) + k_c * mask
