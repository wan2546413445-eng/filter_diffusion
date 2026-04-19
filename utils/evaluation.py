import torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr_tensor
from torchmetrics.functional import structural_similarity_index_measure as ssim_tensor


def _normalize_by_own_max(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    按各自最大值归一化：
        x_norm = x / max(x)

    兼容:
        (H, W)
        (C, H, W)
        (B, C, H, W)

    注意：这里是“各自归一化”，与 MATLAB:
        img_1 / max(img_1(:))
        img_gt / max(img_gt(:))
    一致。
    """
    x = x.float()

    if x.dim() == 2:
        max_val = x.max()
        return x / (max_val + eps)

    elif x.dim() == 3:
        # (C, H, W) -> 每个 C 整体共用一个最大值
        max_val = x.max()
        return x / (max_val + eps)

    elif x.dim() == 4:
        # (B, C, H, W) -> 每个样本各自归一化
        max_val = x.amax(dim=(-3, -2, -1), keepdim=True)
        return x / (max_val + eps)

    else:
        raise ValueError(f"Input dimension {x.dim()} not supported.")


def calc_nmse_tensor(gt, pred, normalize_by_max: bool = True):
    """
    NMSE
    默认先按各自最大值归一化，再计算：
        ||gt_norm - pred_norm||^2 / ||gt_norm||^2
    """
    gt = gt.float()
    pred = pred.float()

    if normalize_by_max:
        gt = _normalize_by_own_max(gt)
        pred = _normalize_by_own_max(pred)

    return torch.linalg.norm(gt - pred) ** 2 / (torch.linalg.norm(gt) ** 2 + 1e-8)


def calc_ssim_tensor(gt, pred, data_range=None, normalize_by_max: bool = True):
    """
    SSIM
    默认先按各自最大值归一化，再用 data_range=1.0
    """
    gt = gt.float()
    pred = pred.float()

    if normalize_by_max:
        gt = _normalize_by_own_max(gt)
        pred = _normalize_by_own_max(pred)
        data_range = 1.0
    elif data_range is None:
        data_range = gt.max() - gt.min()

    if gt.dim() == 2:          # (H, W)
        gt = gt.unsqueeze(0).unsqueeze(0)      # (1,1,H,W)
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif gt.dim() == 3:        # (C, H, W)
        gt = gt.unsqueeze(0)                   # (1,C,H,W)
        pred = pred.unsqueeze(0)
    elif gt.dim() == 4:        # (B, C, H, W)
        pass
    else:
        raise ValueError(f"Input dimension {gt.dim()} not supported. Expected 2, 3 or 4.")

    return ssim_tensor(pred, gt, data_range=data_range)


def calc_psnr_tensor(gt, pred, data_range=None, normalize_by_max: bool = True):
    """
    PSNR
    默认先按各自最大值归一化，再用 data_range=1.0
    """
    gt = gt.float()
    pred = pred.float()

    if normalize_by_max:
        gt = _normalize_by_own_max(gt)
        pred = _normalize_by_own_max(pred)
        data_range = 1.0
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