import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# FilterDiff restoration network: Swin-DiTs style
# ------------------------------------------------------------
# Paper-aligned architectural principles:
#   1) Transformer-style backbone based on DiT
#   2) Replace self-attention with Swin-attention
#   3) Remove position embedding
#   4) Serve as φ_theta(M_t, k_t, k_c, t) -> x0_pred
#
# Input:
#   x    : [B, 5, H, W] = [k_t(2), k_c(2), M_t(1)]
#   time : [B]
#
# Output:
#   x0_pred : [B, 2, H, W]
#
# NOTE:
#   The PDF describes the restoration backbone at a high level, but
#   does not disclose every exact hidden size / depth / head count.
#   This implementation is therefore architecture-faithful to the
#   paper description, while remaining directly pluggable into your code.
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
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    x: [B, H, W, C]
    return: [num_windows * B, window_size * window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    windows: [num_windows * B, window_size * window_size, C]
    return: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # relative position bias for Swin attention
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [nW*B, N, C]
        mask: [nW, N, N] or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out


class SwinDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        time_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        H, W = input_resolution
        if min(H, W) <= window_size:
            self.window_size = min(H, W)
            self.shift_size = 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_dim=int(dim * mlp_ratio))

        # DiT-style adaptive LayerNorm modulation with time embedding
        self.adaLN_modulation = None
        if exists(time_dim):
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, dim * 4),
            )

        self.register_buffer("attn_mask", self._build_mask(H, W), persistent=False)

    def _build_mask(self, H: int, W: int) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, H, W, C]
        """
        H, W = self.input_resolution
        B, Hx, Wx, C = x.shape
        assert (Hx, Wx) == (H, W), f"Input resolution mismatch: {(Hx, Wx)} vs {(H, W)}"

        if exists(self.adaLN_modulation) and exists(t_emb):
            shift1, scale1, shift2, scale2 = self.adaLN_modulation(t_emb).chunk(4, dim=-1)
        else:
            shift1 = scale1 = shift2 = scale2 = None

        shortcut = x
        x = self.norm1(x)
        if exists(shift1):
            x = self._modulate(x, shift1, scale1)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        if exists(shift2):
            x = self._modulate(x, shift2, scale2)
        x = self.mlp(x)
        x = shortcut + x
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 320, patch_size: int = 4, in_channels: int = 5, embed_dim: int = 384):
        super().__init__()
        assert img_size % patch_size == 0, f"img_size={img_size} must be divisible by patch_size={patch_size}"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)          # [B, C, H/P, W/P]
        x = x.permute(0, 2, 3, 1) # [B, H/P, W/P, C]
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, time_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_size * 2)
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: [B, Ht, Wt, C]
        return: [B, Ht, Wt, patch_size*patch_size*out_channels]
        """
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
        x = self.linear(x)
        return x


class FilterDiffSwinDiTs(nn.Module):
    """
    Swin-DiTs style restoration net for FilterDiff.
    No position embedding by design.
    """

    def __init__(
        self,
        img_size: int = 320,
        patch_size: int = 4,
        in_channels: int = 5,
        out_channels: int = 2,
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 8,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        with_time_emb: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.with_time_emb = with_time_emb

        print("[FilterDiffSwinDiTs] Swin-attention enabled, position embedding removed")

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )
        grid_h, grid_w = self.patch_embed.grid_size
        self.token_resolution = (grid_h, grid_w)

        if with_time_emb:
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbedding(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )
        else:
            self.time_mlp = None

        # No position embedding here (paper explicitly removes it)
        self.blocks = nn.ModuleList([
            SwinDiTBlock(
                dim=hidden_size,
                input_resolution=self.token_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                time_dim=hidden_size if with_time_emb else None,
            )
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
            time_dim=hidden_size,
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, Ht, Wt, P*P*C]
        return: [B, C, H, W]
        """
        B, Ht, Wt, D = x.shape
        p = self.patch_size
        c = self.out_channels
        assert D == p * p * c, f"Unexpected last dim: {D}, expected {p * p * c}"
        x = x.view(B, Ht, Wt, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, c, Ht * p, Wt * p)
        return x

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 5, H, W]
        time: [B]
        output: [B, 2, H, W]
        """
        x = self.patch_embed(x)  # [B, Ht, Wt, C]

        if exists(self.time_mlp):
            t_emb = self.time_mlp(time)
        else:
            t_emb = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)

        for blk in self.blocks:
            x = blk(x, t_emb)

        x = self.final_layer(x, t_emb)
        x = self.unpatchify(x)
        return x


def build_filterdiff_restoration_net(
    img_size: int = 320,
    patch_size: int = 4,
    in_channels: int = 5,
    out_channels: int = 2,
    hidden_size: int = 384,
    depth: int = 8,
    num_heads: int = 8,
    window_size: int = 8,
    mlp_ratio: float = 4.0,
    with_time_emb: bool = True,
):
    return FilterDiffSwinDiTs(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        with_time_emb=with_time_emb,
    )