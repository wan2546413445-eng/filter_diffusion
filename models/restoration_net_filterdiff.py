import math
import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
#  FilterDiff standalone restoration network
#  ------------------------------------------------------------
#  设计目标：
#  1) 与现有 models/unet_diffusion.py 完全隔离，不相互依赖
#  2) 接口与当前训练框架兼容：forward(x, time) -> [B, 2, H, W]
#  3) 满足 FilterDiff 论文里恢复网络 φ_theta 的角色：
#     输入 [k_t(2), k_c(2), M_t(1)] 共 5 通道，输出图像域 x0_pred 的 2 通道复数表示
#  4) 方便后续你直接替换 train.py 中的 denoise_fn
# ============================================================


def exists(x):
    return x is not None


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int = None, groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = None
        if exists(time_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_ch)
            )

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))

        if exists(self.time_mlp) and exists(t_emb):
            h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        q = q.transpose(-2, -1)                       # [B, heads, HW, dim]
        attn = torch.matmul(q, k) / math.sqrt(max(head_dim, 1))
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.view(b, c, h, w)
        out = self.proj(out)
        return out + x_in


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.block1 = ResidualBlock(in_ch, out_ch, time_dim=time_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim=time_dim)
        self.attn = SelfAttention2d(out_ch)
        self.down = Downsample(out_ch)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = Upsample(in_ch)
        self.block1 = ResidualBlock(in_ch + skip_ch, out_ch, time_dim=time_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim=time_dim)
        self.attn = SelfAttention2d(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        x = self.attn(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.block1 = ResidualBlock(channels, channels, time_dim=time_dim)
        self.attn = SelfAttention2d(channels)
        self.block2 = ResidualBlock(channels, channels, time_dim=time_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, t_emb)
        x = self.attn(x)
        x = self.block2(x, t_emb)
        return x


class FilterDiffRestorationNet(nn.Module):
    """
    Standalone restoration network for FilterDiff.

    输入:
        x    : [B, 5, H, W]
               其中 5 通道对应 [k_t(real, imag), k_c(real, imag), M_t]
        time : [B]

    输出:
        x0_pred : [B, 2, H, W]
                  图像域复数图像的双通道表示

    说明:
        论文主逻辑约束的是恢复网络的“角色”和输入输出契约，
        即 φ_theta(M_t, k_t, k_c, t) -> x0_pred。
        这里单独提供一个与现有 Unet 隔离的 U-Net 风格实现，方便你后续独立替换。
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        with_time_emb: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.with_time_emb = with_time_emb

        print("[FilterDiffRestorationNet] Time embedding:", with_time_emb)

        time_dim = base_channels if with_time_emb else None
        if with_time_emb:
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbedding(base_channels),
                nn.Linear(base_channels, base_channels * 4),
                nn.SiLU(),
                nn.Linear(base_channels * 4, base_channels),
            )
        else:
            self.time_mlp = None

        chs = [base_channels * m for m in channel_mults]

        self.stem = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)

        self.enc1 = EncoderBlock(chs[0], chs[0], time_dim)
        self.enc2 = EncoderBlock(chs[0], chs[1], time_dim)
        self.enc3 = EncoderBlock(chs[1], chs[2], time_dim)

        self.pre_bottleneck = ResidualBlock(chs[2], chs[3], time_dim=time_dim)
        self.bottleneck = Bottleneck(chs[3], time_dim)

        self.dec3 = DecoderBlock(chs[3], chs[2], chs[2], time_dim)
        self.dec2 = DecoderBlock(chs[2], chs[1], chs[1], time_dim)
        self.dec1 = DecoderBlock(chs[1], chs[0], chs[0], time_dim)

        self.head = nn.Sequential(
            ResidualBlock(chs[0], chs[0], time_dim=time_dim),
            nn.GroupNorm(8, chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(time) if exists(self.time_mlp) else None

        x = self.stem(x)
        x, skip1 = self.enc1(x, t_emb)
        x, skip2 = self.enc2(x, t_emb)
        x, skip3 = self.enc3(x, t_emb)

        x = self.pre_bottleneck(x, t_emb)
        x = self.bottleneck(x, t_emb)

        x = self.dec3(x, skip3, t_emb)
        x = self.dec2(x, skip2, t_emb)
        x = self.dec1(x, skip1, t_emb)
        x = self.head(x)
        return x


def build_filterdiff_restoration_net(
    dim: int = 64,
    in_channels: int = 5,
    out_channels: int = 2,
    dim_mults=(1, 2, 4, 8),
    with_time_emb: bool = True,
):
    """
    便捷构造函数，保持与你当前 train.py 的调用风格接近。
    例如后续可直接替换成：

        denoise_fn = build_filterdiff_restoration_net(
            dim=config.model.dim,
            in_channels=5,
            out_channels=2,
            dim_mults=tuple(config.model.dim_mults),
            with_time_emb=True,
        )
    """
    return FilterDiffRestorationNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=dim,
        channel_mults=dim_mults,
        with_time_emb=with_time_emb,
    )
