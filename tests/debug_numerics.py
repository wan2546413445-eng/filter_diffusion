import torch
import sys
sys.path.append(".")

from data.ixi_singlecoil_dataset import IXISinglecoilSliceDataset
from mask import mask_func_factory
from diffusion.kspace_diffusion import KspaceDiffusion
from diffusion.filter_schedule import CenterRectangleSchedule
import fastmri

# 配置
image_size = 256
acceleration = 4
center_core_size = image_size // acceleration
timesteps = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载一个真实样本
mask_func = mask_func_factory(acceleration=acceleration, image_size=image_size)
dataset = IXISinglecoilSliceDataset(
    root="/mnt/SSD/wsy/data/train",   # 请改成实际路径
    mask_func=mask_func,
    image_size=image_size
)
kspace_full, mask_full, _ = dataset[0]
kspace_full = kspace_full.unsqueeze(0).to(device)   # (1,1,H,W,2)
mask_full = mask_full.unsqueeze(0).to(device)       # (1,1,H,W)

print("=== Data Stats ===")
print(f"kspace_full - real: min={kspace_full[...,0].min().item():.4f}, max={kspace_full[...,0].max().item():.4f}")
print(f"mask_full - sum: {mask_full.sum().item()} (should be ~{image_size*image_size//acceleration + image_size*center_core_size})")

# 创建扩散模型（不训练，仅用于计算中间量）
class DummyNet(torch.nn.Module):
    def forward(self, x, t):
        return torch.randn(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device)

diffusion = KspaceDiffusion(
    denoise_fn=DummyNet().to(device),
    image_size=image_size,
    device_of_kernel=device,
    timesteps=timesteps,
    center_core_size=center_core_size,
)
diffusion.to(device)

# 模拟一个时间步
t = torch.tensor([10], device=device).long()
m_t = diffusion.schedule.get_by_t(t, device=device)
k_t = m_t * kspace_full
k_c = diffusion._build_conditional_kc(kspace_full, mask_full)
delta_m = diffusion.schedule.get_by_t(t-1, device=device) - m_t
delta_gt = delta_m * kspace_full

print("\n=== Intermediate Stats ===")
print(f"m_t sum: {m_t.sum().item()}")
print(f"k_t magnitude - min={k_t.abs().min().item():.4e}, max={k_t.abs().max().item():.4e}")
print(f"k_c magnitude - min={k_c.abs().min().item():.4e}, max={k_c.abs().max().item():.4e}")
print(f"delta_gt magnitude - min={delta_gt.abs().min().item():.4e}, max={delta_gt.abs().max().item():.4e}")

# 检查是否有 NaN 或 Inf
assert torch.isfinite(k_t).all(), "k_t contains NaN/Inf"
assert torch.isfinite(k_c).all(), "k_c contains NaN/Inf"
assert torch.isfinite(delta_gt).all(), "delta_gt contains NaN/Inf"

print("\n✅ All tensors are finite.")