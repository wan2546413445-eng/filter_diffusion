import torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr_tensor
from torchmetrics.functional import structural_similarity_index_measure as ssim_tensor


def _normalize_pair(
    gt: torch.Tensor,
    pred: torch.Tensor,
    normalize_mode: str = "own_max",
    eps: float = 1e-8,
):
    gt = gt.float()
    pred = pred.float()
    normalize_mode = normalize_mode.lower()

    if normalize_mode == "none":
        return gt, pred, None

    if normalize_mode == "own_max":
        gt_scale = gt.max()
        pred_scale = pred.max()
        gt = gt / (gt_scale + eps)
        pred = pred / (pred_scale + eps)
        return gt, pred, 1.0

    if normalize_mode == "gt_max":
        scale = gt.max()
        gt = gt / (scale + eps)
        pred = pred / (scale + eps)
        return gt, pred, 1.0

    raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")


def calc_nmse_tensor(gt, pred, normalize_mode: str = "own_max", normalize_by_max=None):
    if normalize_by_max is not None:
        normalize_mode = "own_max" if normalize_by_max else "none"

    gt, pred, _ = _normalize_pair(gt, pred, normalize_mode=normalize_mode)
    return torch.linalg.norm(gt - pred) ** 2 / (torch.linalg.norm(gt) ** 2 + 1e-8)


def calc_ssim_tensor(gt, pred, data_range=None, normalize_mode: str = "own_max", normalize_by_max=None):
    if normalize_by_max is not None:
        normalize_mode = "own_max" if normalize_by_max else "none"

    gt, pred, norm_range = _normalize_pair(gt, pred, normalize_mode=normalize_mode)
    if norm_range is not None:
        data_range = norm_range
    elif data_range is None:
        data_range = gt.max() - gt.min()

    if gt.dim() == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif gt.dim() == 3:
        gt = gt.unsqueeze(0)
        pred = pred.unsqueeze(0)
    elif gt.dim() == 4:
        pass
    else:
        raise ValueError(f"Input dimension {gt.dim()} not supported. Expected 2, 3 or 4.")

    return ssim_tensor(pred, gt, data_range=data_range)


def calc_psnr_tensor(gt, pred, data_range=None, normalize_mode: str = "own_max", normalize_by_max=None):
    if normalize_by_max is not None:
        normalize_mode = "own_max" if normalize_by_max else "none"

    gt, pred, norm_range = _normalize_pair(gt, pred, normalize_mode=normalize_mode)
    if norm_range is not None:
        data_range = norm_range
    elif data_range is None:
        data_range = gt.max() - gt.min()

    if gt.dim() == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif gt.dim() == 3:
        gt = gt.unsqueeze(0)
        pred = pred.unsqueeze(0)
    elif gt.dim() == 4:
        pass
    else:
        raise ValueError(f"Input dimension {gt.dim()} not supported.")

    return psnr_tensor(pred, gt, data_range=data_range)