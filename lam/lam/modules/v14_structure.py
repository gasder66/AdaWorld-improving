"""
V14 Structure Path: mask-level StructureExtractor + CausalStructureEncoder + StructureHead.

Hard constraints:
  1. NO velocity / delta / flow / future info.
  2. Causal temporal attention.
  3. Content/RGB never enters this path.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskStructureExtractor(nn.Module):
    """Build raw structure from boxes + masks. No velocity, no RGB."""

    def __init__(
        self,
        mask_grid: int = 16,
        mask_feat_dim: int = 32,
        use_moments: bool = True,
        use_mask_structure: bool = True,
    ):
        super().__init__()
        self.mask_grid = mask_grid
        self.mask_feat_dim = mask_feat_dim
        self.use_moments = use_moments
        self.use_mask_structure = use_mask_structure

        if use_mask_structure:
            self.mask_compress = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(16, mask_feat_dim, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            )
            self.mask_proj = nn.Linear(mask_feat_dim * 4, mask_feat_dim)
        else:
            self.mask_compress = None
            self.mask_proj = None

        # raw_dim = bbox(4) + geometry(2) + moments(6 if use_moments else 0) + visible(2) + mask_feat(32 if mask else 0)
        self.raw_dim = 4 + 2 + (6 if use_moments else 0) + 2 + (mask_feat_dim if use_mask_structure else 0)

    def _mask_moments(self, masks: Tensor) -> Tensor:
        """6 moments per mask: area, mean_xy, var_xy, cov_xy."""
        B, T, K, H, W = masks.shape
        device = masks.device
        ys = torch.arange(H, device=device, dtype=masks.dtype).view(1, 1, 1, H, 1)
        xs = torch.arange(W, device=device, dtype=masks.dtype).view(1, 1, 1, 1, W)
        area = masks.sum(dim=(-2, -1)).clamp(min=1e-6)
        mx = (masks * xs).sum(dim=(-2, -1)) / area / W
        my = (masks * ys).sum(dim=(-2, -1)) / area / H
        vx = (masks * (xs - mx.unsqueeze(-1).unsqueeze(-1) * W) ** 2).sum(dim=(-2, -1)) / area / (W * W)
        vy = (masks * (ys - my.unsqueeze(-1).unsqueeze(-1) * H) ** 2).sum(dim=(-2, -1)) / area / (H * H)
        cxy = (masks * (xs - mx.unsqueeze(-1).unsqueeze(-1) * W) *
               (ys - my.unsqueeze(-1).unsqueeze(-1) * H)).sum(dim=(-2, -1)) / area / (W * H)
        return torch.stack([area / (H * W), mx, my, vx, vy, cxy], dim=-1)

    def forward(
        self,
        boxes: Tensor,
        masks: Tensor,
        visible_masks: Tensor,
        valid: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, T, K, _, _ = masks.shape
        H, W = masks.shape[-2:]

        # Bbox features.
        bbox = boxes  # (B,T,K,4) cxcywh

        # Geometry: area, aspect_ratio.
        area = (bbox[..., 2] * bbox[..., 3]).unsqueeze(-1)
        ar = (bbox[..., 2] / (bbox[..., 3] + 1e-6)).clamp(0.1, 10.0).unsqueeze(-1)
        geo = torch.cat([area, ar], dim=-1)

        # Mask moments.
        if self.use_moments:
            moments = self._mask_moments(masks)
        else:
            moments = torch.zeros(B, T, K, 0, device=boxes.device)

        # Visible ratio + occlusion flag.
        full_area = masks.sum(dim=(-2, -1)).clamp(min=1e-6)
        visible_area = visible_masks.sum(dim=(-2, -1))
        vis_ratio = (visible_area / full_area).unsqueeze(-1)
        occ_flag = (vis_ratio < 0.95).float()
        visible = torch.cat([vis_ratio, occ_flag], dim=-1)

        # Low-res mask CNN.
        if self.use_mask_structure:
            masks_flat = masks.reshape(B * T * K, 1, H, W)
            mask_low = F.adaptive_avg_pool2d(masks_flat, (self.mask_grid, self.mask_grid))
            mask_feat = self.mask_compress(mask_low).flatten(1)
            mask_feat = self.mask_proj(mask_feat).reshape(B, T, K, self.mask_feat_dim)
        else:
            mask_feat = torch.zeros(B, T, K, 0, device=boxes.device)

        raw_struct = torch.cat([bbox, geo, moments, visible, mask_feat], dim=-1)

        # Targets for GT mask at grid resolution.
        if self.use_mask_structure:
            mask_low_gt = mask_low.reshape(B, T, K, 1, self.mask_grid, self.mask_grid)
        else:
            mask_low_gt = torch.zeros(B, T, K, 1, self.mask_grid, self.mask_grid, device=boxes.device)

        targets = {
            "bbox": bbox,
            "mask_low": mask_low_gt,
            "mask_full": masks,
            "visible_mask": visible_masks,
            "valid": valid,
        }
        return raw_struct, targets


class CausalMaskStructureEncoder(nn.Module):
    """Causal temporal transformer + slot attention on mask-level structure."""

    def __init__(
        self,
        raw_dim: int,
        struct_dim: int = 128,
        temporal_layers: int = 2,
        slot_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.struct_dim = struct_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, 256),
            nn.GELU(),
            nn.Linear(256, struct_dim),
        )
        temp_layer = nn.TransformerEncoderLayer(
            d_model=struct_dim, nhead=heads, dim_feedforward=struct_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(temp_layer, num_layers=temporal_layers)
        slot_layer = nn.TransformerEncoderLayer(
            d_model=struct_dim, nhead=heads, dim_feedforward=struct_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.slot = nn.TransformerEncoder(slot_layer, num_layers=slot_layers)

    def forward(self, raw_struct: Tensor, valid: Tensor) -> Tensor:
        B, T, K, _ = raw_struct.shape
        s = self.mlp(raw_struct)

        # Causal temporal: (B*K, T, D).
        s = s.permute(0, 2, 1, 3).reshape(B * K, T, self.struct_dim)
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=s.device, dtype=s.dtype), diagonal=1,
        )
        s = self.temporal(s, mask=causal_mask)
        s = s.reshape(B, K, T, self.struct_dim).permute(0, 2, 1, 3)

        # Slot attention: (B*T, K, D).
        s = s.reshape(B * T, K, self.struct_dim)
        pad = valid.reshape(B * T, K).logical_not()
        s = self.slot(s, src_key_padding_mask=pad)
        s = s.reshape(B, T, K, self.struct_dim)
        return s * valid.unsqueeze(-1).float()


class V14StructureHead(nn.Module):
    """Predict mask + bbox from s_hat."""

    def __init__(
        self,
        struct_dim: int = 128,
        mask_grid: int = 16,
    ):
        super().__init__()
        self.mask_grid = mask_grid
        self.bbox_head = nn.Sequential(
            nn.Linear(struct_dim, struct_dim), nn.GELU(),
            nn.Linear(struct_dim, 4),
        )
        self.mask_head = nn.Sequential(
            nn.Linear(struct_dim, struct_dim), nn.GELU(),
            nn.Linear(struct_dim, mask_grid * mask_grid),
        )
        self.exist_head = nn.Sequential(
            nn.Linear(struct_dim, struct_dim // 2), nn.GELU(),
            nn.Linear(struct_dim // 2, 1),
        )
        self.vis_head = nn.Linear(struct_dim, 1)

    def forward(self, s_hat: Tensor) -> Dict[str, Tensor]:
        B, T, K, _ = s_hat.shape
        return {
            "bbox": self.bbox_head(s_hat),
            "mask_low": self.mask_head(s_hat).reshape(B, T, K, 1, self.mask_grid, self.mask_grid),
            "exist_logit": self.exist_head(s_hat),
            "visible_ratio": torch.sigmoid(self.vis_head(s_hat)),
        }
