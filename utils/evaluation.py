import torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr_tensor
from torchmetrics.functional import structural_similarity_index_measure as ssim_tensor

def calc_nmse_tensor(gt, pred):
    """Normalized Mean Squared Error"""
    return torch.linalg.norm(gt - pred) ** 2 / torch.linalg.norm(gt) ** 2



   
def calc_ssim_tensor(gt, pred, data_range=None):
    if data_range is None:
        data_range = gt.max() - gt.min()
    # 处理不同维度
    if gt.dim() == 2:          # (H, W)
        gt = gt.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif gt.dim() == 3:        # (C, H, W)
        gt = gt.unsqueeze(0)               # (1,C,H,W)
        pred = pred.unsqueeze(0)
    elif gt.dim() == 4:        # (B, C, H, W)
        pass
    else:
        raise ValueError(f"Input dimension {gt.dim()} not supported. Expected 2, 3 or 4.")
    return ssim_tensor(pred, gt, data_range=data_range)


def calc_psnr_tensor(gt, pred, data_range=None):
    if data_range is None:
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
