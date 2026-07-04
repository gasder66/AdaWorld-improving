"""
V13 Atari Renderer: fast CNN-based rendering from structure maps.
Does NOT receive z_action directly.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _boxes_to_render_maps(
    bbox: Tensor, valid: Tensor, H: int, W: int,
) -> Tensor:
    """Convert (B,T,K,4) cxcywh to (B,T,2,H,W) maps."""
    B, T, K, _ = bbox.shape
    device = bbox.device

    # Channel 0: bbox occupancy grid (8x8 downsampled).
    g = 8
    occ_flat = torch.zeros(B * T, g * g, device=device)
    for k in range(K):
        v = valid[..., k].reshape(B * T)  # (B*T,)
        if not v.any():
            continue
        cx = bbox[..., k, 0].reshape(B * T)
        cy = bbox[..., k, 1].reshape(B * T)
        gx = (cx * g).long().clamp(0, g - 1)
        gy = (cy * g).long().clamp(0, g - 1)
        idx = (gy * g + gx)  # (B*T,)
        # scatter: index selects column, add value.
        occ_flat[torch.arange(B * T, device=device), idx] += v.float()
    occ_map = (occ_flat / max(K, 1)).reshape(B * T, 1, g, g)
    occ_map = F.interpolate(occ_map, size=(H, W), mode="bilinear", align_corners=False)
    occ_map = occ_map.reshape(B, T, 1, H, W)

    # Channel 1: valid slot count per frame.
    count = valid.float().sum(dim=-1, keepdim=True) / max(K, 1)  # (B,T,1)
    count_map = count.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, H, W)

    maps = torch.cat([occ_map, count_map], dim=2)
    return maps


class AtariBBoxRenderer(nn.Module):
    """CNN renderer: video_t + bbox_maps -> reconstructed next frame."""

    def __init__(self, image_size: int = 84, base_dim: int = 32):
        super().__init__()
        self.image_size = image_size
        in_c = 3 + 2

        self.enc_down = nn.Sequential(
            nn.Conv2d(in_c, base_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim * 2, base_dim * 2, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.dec_up = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(base_dim, base_dim // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(base_dim // 2, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, video_t: Tensor, pred_bbox: Tensor, slot_valid: Tensor,
    ) -> Tensor:
        B, T1, C, H, W = video_t.shape
        bbox_maps = _boxes_to_render_maps(pred_bbox, slot_valid, H, W)

        video_flat = video_t.reshape(B * T1, C, H, W)
        maps_flat = bbox_maps.reshape(B * T1, 2, H, W)
        x = torch.cat([video_flat, maps_flat], dim=1)

        h = self.enc_down(x)  # (B*T1, base_dim*2, H/8, W/8)
        out = self.dec_up(h)  # (B*T1, 3, H, W)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out.reshape(B, T1, C, H, W)
