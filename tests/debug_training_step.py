import torch
import sys
sys.path.append(".")

from diffusion.kspace_diffusion import KspaceDiffusion
from diffusion.filter_schedule import CenterRectangleSchedule

# 模拟网络
class DummyNet(torch.nn.Module):
    def forward(self, x, t):
        # x: (B,5,H,W)
        return torch.randn(x.shape[0], 2, x.shape[2], x.shape[3])

# 配置
image_size = 256
acceleration = 4
center_core_size = image_size // acceleration   # 64
timesteps = 20
batch_size = 2

# 创建扩散模型
diffusion = KspaceDiffusion(
    denoise_fn=DummyNet(),
    image_size=image_size,
    device_of_kernel='cpu',
    channels=2,
    timesteps=timesteps,
    schedule_type='dense',
    center_core_size=center_core_size,
)

# 模拟一个 batch 的数据
kspace_full = torch.randn(batch_size, 1, image_size, image_size, 2)  # (B,1,H,W,2)
mask_full = torch.ones(batch_size, 1, image_size, image_size)        # 假设全1简化测试

# 随机时间步
t = torch.randint(1, timesteps+1, (batch_size,))

# 执行一次 p_losses
loss = diffusion.p_losses(kspace_full, mask_full, t)

print(f"Loss value: {loss.item():.6f}")
print("All shapes and operations passed without error.")