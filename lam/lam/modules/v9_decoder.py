"""
V9 Reconstruction Decoder.

比 V8 probe decoder 更强:
  - 多尺度 FiLM (3 层, probe 只有 1 层 bottleneck FiLM)
  - U-Net skip connections (probe 无 skip)
  - 接收 z_actor + z_bg 双重条件

设计:
  Encoder: crop_t (3,32,32) → 3 层 stride-2 conv → (128,4,4)
  FiLM(z) at ALL 3 encoder levels (multi-scale conditioning)
  Decoder: 3 层 ConvTranspose + skip connections → (3,32,32)
  Output: sigmoid → crop_pred [0,1]
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: z -> (gamma, beta), feat = gamma * feat + beta."""

    def __init__(self, z_dim: int, feat_channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(z_dim, feat_channels * 2)
        self.feat_channels = feat_channels

    def forward(self, feat: Tensor, z: Tensor) -> Tensor:
        gb = self.proj(z)
        gamma, beta = gb[:, : self.feat_channels], gb[:, self.feat_channels:]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta


class ReconDecoder(nn.Module):
    """V9 重建 decoder: crop_t + z_actor + z_bg -> crop_{t+1}.

    Multi-scale FiLM + U-Net skip connections.
    比 probe decoder (单层 FiLM, 无 skip) 更强, 确保 z_actor 如果有信息则被使用。

    Args:
        z_actor_dim: z_actor 维度 (default 16)
        z_bg_dim: z_bg 维度 (default 16)
        crop_size: crop 大小 (default 32)
    Input:
        crop_t: (N, 3, crop_size, crop_size) — 当前帧 actor crop
        z_actor: (N, z_actor_dim) — per-actor action latent
        z_bg: (N, z_bg_dim) — global camera motion latent
    Output:
        crop_pred: (N, 3, crop_size, crop_size) — 预测下一帧 crop
    """

    def __init__(
        self,
        z_actor_dim: int = 16,
        z_bg_dim: int = 16,
        crop_size: int = 32,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        z_cond_dim = z_actor_dim + z_bg_dim

        ch1, ch2, ch3 = 32, 64, 128

        # Encoder
        self.enc1 = nn.Conv2d(3, ch1, 3, 2, 1)       # 32 -> 16
        self.enc2 = nn.Conv2d(ch1, ch2, 3, 2, 1)      # 16 -> 8
        self.enc3 = nn.Conv2d(ch2, ch3, 3, 2, 1)      # 8 -> 4

        # Multi-scale FiLM (3 levels vs probe's 1)
        self.film1 = FiLMLayer(z_cond_dim, ch1)
        self.film2 = FiLMLayer(z_cond_dim, ch2)
        self.film3 = FiLMLayer(z_cond_dim, ch3)

        # Decoder with U-Net skip connections
        self.dec3 = nn.ConvTranspose2d(ch3, ch2, 3, 2, 1, output_padding=1)  # 4 -> 8
        self.dec2 = nn.ConvTranspose2d(ch2 * 2, ch1, 3, 2, 1, output_padding=1)  # 8 -> 16 (skip)
        self.dec1 = nn.ConvTranspose2d(ch1 * 2, 3, 3, 2, 1, output_padding=1)     # 16 -> 32 (skip)

    def forward(self, crop_t: Tensor, z_actor: Tensor, z_bg: Tensor) -> Tensor:
        """
        Args:
            crop_t: (N, 3, crop_size, crop_size)
            z_actor: (N, z_actor_dim)
            z_bg: (N, z_bg_dim)
        Returns:
            crop_pred: (N, 3, crop_size, crop_size) in [0, 1]
        """
        z_cond = torch.cat([z_actor, z_bg], dim=-1)  # (N, z_actor_dim + z_bg_dim)

        # Encode + multi-scale FiLM
        f1 = F.gelu(self.enc1(crop_t))   # (N, 32, 16, 16)
        f1 = self.film1(f1, z_cond)
        f2 = F.gelu(self.enc2(f1))       # (N, 64, 8, 8)
        f2 = self.film2(f2, z_cond)
        f3 = F.gelu(self.enc3(f2))       # (N, 128, 4, 4)
        f3 = self.film3(f3, z_cond)

        # Decode with skip connections
        d3 = F.gelu(self.dec3(f3))                    # (N, 64, 8, 8)
        d2 = F.gelu(self.dec2(torch.cat([d3, f2], 1)))  # (N, 32, 16, 16)
        d1 = self.dec1(torch.cat([d2, f1], 1))          # (N, 3, 32, 32)

        d1 = F.interpolate(d1, size=(self.crop_size, self.crop_size),
                           mode="bilinear", align_corners=False)
        return torch.sigmoid(d1)


def compute_psnr(pred: Tensor, target: Tensor) -> float:
    """PSNR (dB) for [0,1] images."""
    mse = F.mse_loss(pred, target)
    if mse.item() < 1e-10:
        return 100.0
    return float(10 * torch.log10(1.0 / mse))


def compute_ssim_simple(pred: Tensor, target: Tensor) -> float:
    """Simplified SSIM (global, per-channel mean)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p = pred.mean(dim=[2, 3])
    mu_t = target.mean(dim=[2, 3])
    var_p = pred.var(dim=[2, 3])
    var_t = target.var(dim=[2, 3])
    cov_pt = ((pred - mu_p.unsqueeze(-1).unsqueeze(-1)) *
              (target - mu_t.unsqueeze(-1).unsqueeze(-1))).mean(dim=[2, 3])
    ssim = ((2 * mu_p * mu_t + C1) * (2 * cov_pt + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2))
    return float(ssim.mean())
