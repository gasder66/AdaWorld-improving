"""
V8 Probe Decoder: 验证 z_actor / z_bg 是否包含足够的重建信息.

设计原则:
  - 不修改 V8 主模型, z_actor/z_bg 冻结
  - z_actor 通过 FiLM 注入 decoder, 作为条件而非外观来源
  - 无 Spatial Transformer: decoder 自己从 z 学位移
  - 三重 baseline (copy / z=0 / z_shuffle) 验证 z 的真实贡献

ActorProbeDecoder:
  mode A: crop_t + z_actor → crop_pred
  mode B: crop_t + z_actor + dbbox_pred + actor_type → crop_pred

BackgroundProbeDecoder:
  bg_t + z_bg → bg_pred (条件触发, 暂缓实现)
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: z → (gamma, beta), feat = gamma * feat + beta."""

    def __init__(self, z_dim: int, feat_channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(z_dim, feat_channels * 2)
        self.feat_channels = feat_channels

    def forward(self, feat: Tensor, z: Tensor) -> Tensor:
        """
        Args:
            feat: (B, C, H, W)
            z: (B, z_dim)
        Returns:
            (B, C, H, W)
        """
        gb = self.proj(z)
        gamma, beta = gb[:, :self.feat_channels], gb[:, self.feat_channels:]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta


class ActorProbeDecoder(nn.Module):
    """z_actor 条件化 actor crop 重建.

    输入: crop_t (B, 3, 32, 32) + z_actor (B, z_dim)
    输出: crop_pred (B, 3, 32, 32) — 预测 crop(I_{t+1}, bbox_{t+1})

    mode A: crop_t + z_actor
    mode B: crop_t + z_actor + dbbox_pred (4,) + actor_type (scalar)
    """

    def __init__(
        self,
        z_dim: int = 16,
        crop_size: int = 32,
        use_dbbox: bool = False,
        num_actor_types: int = 0,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.crop_size = crop_size
        self.use_dbbox = use_dbbox
        self.num_actor_types = num_actor_types

        z_cond_dim = z_dim
        if use_dbbox:
            z_cond_dim += 4
        if num_actor_types > 0:
            z_cond_dim += num_actor_types

        ch1, ch2, ch3 = 32, 64, 128

        self.encoder = nn.Sequential(
            nn.Conv2d(3, ch1, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(ch1, ch2, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(ch2, ch3, 3, 2, 1),
            nn.GELU(),
        )

        self.film = FiLMLayer(z_cond_dim, ch3)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch3, ch2, 3, 2, 1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(ch2, ch1, 3, 2, 1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(ch1, 3, 3, 2, 1, output_padding=1),
        )

    def forward(
        self,
        crop_t: Tensor,
        z_actor: Tensor,
        dbbox_pred: Optional[Tensor] = None,
        actor_type: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            crop_t: (B, 3, crop_size, crop_size) float [0,1]
            z_actor: (B, z_dim)
            dbbox_pred: (B, 4) — V8 预测的 Δbbox (mode B)
            actor_type: (B,) long — actor type (mode B)
        Returns:
            crop_pred: (B, 3, crop_size, crop_size) float [0,1]
        """
        feat = self.encoder(crop_t)

        z_cond = z_actor
        if self.use_dbbox and dbbox_pred is not None:
            z_cond = torch.cat([z_cond, dbbox_pred], dim=-1)
        if self.num_actor_types > 0 and actor_type is not None:
            onehot = F.one_hot(actor_type, self.num_actor_types).float()
            z_cond = torch.cat([z_cond, onehot], dim=-1)

        feat = self.film(feat, z_cond)
        out = self.decoder(feat)
        out = F.interpolate(out, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
        return torch.sigmoid(out)


class BackgroundProbeDecoder(nn.Module):
    """z_bg 条件化背景重建. (条件触发, 暂缓实现)

    输入: bg_t (B, 3, 256, 256) + z_bg (B, z_bg_dim)
    输出: bg_pred (B, 3, 256, 256) — 预测 I_{t+1} 背景区域
    """

    def __init__(self, z_bg_dim: int = 16, img_size: int = 256) -> None:
        super().__init__()
        raise NotImplementedError("BackgroundProbeDecoder 待 Phase 1 actor probe 验证后实现")


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
    cov_pt = ((pred - mu_p.unsqueeze(-1).unsqueeze(-1)) * (target - mu_t.unsqueeze(-1).unsqueeze(-1))).mean(dim=[2, 3])
    ssim = ((2 * mu_p * mu_t + C1) * (2 * cov_pt + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2))
    return float(ssim.mean())
