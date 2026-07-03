"""
V12 Structure Path: StructureExtractor + StructureEncoder.

StructureExtractor computes raw structure features from masks/boxes only
(NO RGB). StructureEncoder maps raw -> latent structure embeddings s_t.

Hard constraint: StructureExtractor must not use RGB pixels.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def boxes_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """(B, T, K, 4) pixel xyxy -> normalized cxcywh in [0,1]."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    H = boxes.new_tensor(float(boxes.shape[-3]))  # not used; H/W passed explicitly
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], dim=-1)


def _normalize_boxes(boxes: Tensor, H: int, W: int) -> Tensor:
    """pixel xyxy -> normalized cxcywh in [0,1]."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return torch.stack([cx, cy, w, h], dim=-1)


def _mask_moments(masks: Tensor) -> Tensor:
    """Compute 6 moments from binary masks.

    Args:
        masks: (..., K, H, W) float
    Returns:
        moments: (..., K, 6) = [area, mean_x, mean_y, var_x, var_y, cov_xy]
                 normalized to [0,1]-ish range.
    """
    *prefix, K, H, W = masks.shape
    device = masks.device
    ys = torch.arange(H, device=device, dtype=masks.dtype).view(*[1] * len(prefix), 1, H, 1)
    xs = torch.arange(W, device=device, dtype=masks.dtype).view(*[1] * len(prefix), 1, 1, W)

    area = masks.sum(dim=(-2, -1)).clamp(min=1e-6)               # (..., K)
    mean_x = (masks * xs).sum(dim=(-2, -1)) / area / W
    mean_y = (masks * ys).sum(dim=(-2, -1)) / area / H
    var_x = (masks * (xs - mean_x[..., None, None] * W) ** 2).sum(dim=(-2, -1)) / area / (W * W)
    var_y = (masks * (ys - mean_y[..., None, None] * H) ** 2).sum(dim=(-2, -1)) / area / (H * H)
    cov_xy = (masks * (xs - mean_x[..., None, None] * W) * (ys - mean_y[..., None, None] * H)).sum(dim=(-2, -1)) / area / (W * H)
    return torch.stack([area / (H * W), mean_x, mean_y, var_x, var_y, cov_xy], dim=-1)


class StructureExtractor(nn.Module):
    """Compute raw structure features from masks/boxes. No learnable params except mask compressor.

    No RGB pixels are used.
    """

    def __init__(self, mask_grid: int = 16, mask_feat_dim: int = 32, use_velocity: bool = True):
        super().__init__()
        self.mask_grid = mask_grid
        self.mask_feat_dim = mask_feat_dim
        self.use_velocity = use_velocity
        self.mask_compress = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, mask_feat_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.mask_proj = nn.Linear(mask_feat_dim * 4, mask_feat_dim)

    @property
    def raw_dim(self) -> int:
        d = 4 + 6 + self.mask_feat_dim  # bbox + moments + mask_feat
        if self.use_velocity:
            d += 4
        return d

    def forward(
        self,
        masks: Tensor,       # (B, T, K, H, W)
        boxes: Tensor,       # (B, T, K, 4) pixel xyxy
        valid: Tensor,       # (B, T, K) bool
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, T, K, H, W = masks.shape

        # Normalized cxcywh boxes.
        cxcywh = _normalize_boxes(boxes, H, W)                    # (B, T, K, 4)

        # Mask moments.
        moments = _mask_moments(masks)                             # (B, T, K, 6)

        # Low-res mask compressed.
        masks_flat = masks.reshape(B * T * K, 1, H, W).float()
        mask_low = F.adaptive_avg_pool2d(masks_flat, (self.mask_grid, self.mask_grid))
        mask_feat = self.mask_compress(mask_low)                  # (B*T*K, D, 2, 2)
        mask_feat = mask_feat.flatten(1)
        mask_feat = self.mask_proj(mask_feat)                     # (B*T*K, mask_feat_dim)
        mask_feat = mask_feat.reshape(B, T, K, self.mask_feat_dim)

        parts = [cxcywh, moments, mask_feat]
        if self.use_velocity:
            velocity = torch.zeros_like(cxcywh)
            velocity[:, :-1] = cxcywh[:, 1:] - cxcywh[:, :-1]
            velocity[:, -1:] = velocity[:, -2:-1]
            parts.insert(1, velocity)

        raw_struct = torch.cat(parts, dim=-1)  # (B, T, K, D_raw)

        # Targets for structure loss (absolute values at each frame).
        mask_low_gt = mask_low.reshape(B, T, K, 1, self.mask_grid, self.mask_grid)
        targets = {
            "bbox": cxcywh,                                        # (B, T, K, 4)
            "mask_lowres": mask_low_gt,                            # (B, T, K, 1, g, g)
            "moments": moments,                                    # (B, T, K, 6)
        }
        return raw_struct, targets


class StructureEncoder(nn.Module):
    """raw_struct (B,T,K,D_raw) -> s (B,T,K,D_s). No RGB.

    encoder_mode:
      "bidirectional" — full temporal transformer (default, s_t sees all frames)
      "causal"        — causal temporal transformer (s_t sees ≤ t)
      "per_frame"     — per-frame MLP only, no temporal transformer
    """

    def __init__(
        self,
        raw_dim: int,
        struct_dim: int = 128,
        temporal_layers: int = 2,
        slot_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.0,
        encoder_mode: str = "bidirectional",
    ):
        super().__init__()
        self.struct_dim = struct_dim
        self.encoder_mode = encoder_mode
        self.mlp = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, 256),
            nn.GELU(),
            nn.Linear(256, struct_dim),
        )
        if encoder_mode in ("bidirectional", "causal"):
            temp_layer = nn.TransformerEncoderLayer(
                d_model=struct_dim, nhead=heads, dim_feedforward=struct_dim * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(temp_layer, num_layers=temporal_layers)
        else:
            self.temporal = None
        slot_layer = nn.TransformerEncoderLayer(
            d_model=struct_dim, nhead=heads, dim_feedforward=struct_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.slot = nn.TransformerEncoder(slot_layer, num_layers=slot_layers)

    def forward(self, raw_struct: Tensor, valid: Tensor) -> Tensor:
        """raw_struct: (B,T,K,D_raw), valid: (B,T,K) -> s: (B,T,K,D_s)."""
        B, T, K, _ = raw_struct.shape
        s = self.mlp(raw_struct)                                  # (B,T,K,D_s)

        if self.temporal is not None:
            s = s.permute(0, 2, 1, 3).reshape(B * K, T, self.struct_dim)
            if self.encoder_mode == "causal":
                mask = torch.triu(
                    torch.full((T, T), float("-inf"), device=s.device, dtype=s.dtype),
                    diagonal=1,
                )
                s = self.temporal(s, mask=mask)
            else:
                s = self.temporal(s)
            s = s.reshape(B, K, T, self.struct_dim).permute(0, 2, 1, 3)

        s = s.reshape(B * T, K, self.struct_dim)
        pad = ~valid.reshape(B * T, K)
        s = self.slot(s, src_key_padding_mask=pad)
        s = s.reshape(B, T, K, self.struct_dim)
        return s
