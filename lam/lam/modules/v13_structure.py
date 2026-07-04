"""
V13 Atari Structure Path: StructureExtractor + CausalStructureEncoder.

Hard constraints:
  1. NO velocity, NO bbox delta, NO optical flow, NO frame difference.
  2. StructureEncoder is CAUSAL (s_t can only attend to <= t).
  3. Content/RGB never enters this path.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AtariStructureExtractor(nn.Module):
    """Build raw structure from SlotBuilder output. No velocity, no RGB."""

    def __init__(
        self,
        role_emb_dim: int = 32,
        num_roles: int = 8,
        group_feat_dim: int = 19,
        use_geo: bool = True,
    ):
        super().__init__()
        self.role_emb_dim = role_emb_dim
        self.group_feat_dim = group_feat_dim
        self.use_geo = use_geo
        self.role_embed = nn.Embedding(num_roles, role_emb_dim)

        geo_dim = 2 if use_geo else 0
        rel_dim = 2
        self.raw_dim = 4 + geo_dim + rel_dim + role_emb_dim + group_feat_dim

    def forward(
        self,
        slot_bbox: Tensor,        # (B, T, K, 4) cxcywh
        slot_role: Tensor,        # (B, T, K) long
        slot_valid: Tensor,       # (B, T, K) bool
        slot_group_feat: Tensor,  # (B, T, K, D_g)
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, T, K, _ = slot_bbox.shape

        # Bbox features.
        bbox = slot_bbox

        # Geometry: area, aspect_ratio (clamped).
        if self.use_geo:
            area = (bbox[..., 2] * bbox[..., 3]).unsqueeze(-1)  # (B, T, K, 1)
            ar = (bbox[..., 2] / (bbox[..., 3] + 1e-6)).clamp(0.1, 10.0).unsqueeze(-1)
            geo = torch.cat([area, ar], dim=-1)  # (B, T, K, 2)
        else:
            geo = torch.zeros(B, T, K, 0, device=bbox.device)

        # Relative to first valid agent slot (assumed slot 0).
        agent_cx = bbox[..., 0, 0:1]  # (B, T, 1)
        agent_cy = bbox[..., 0, 1:2]
        rel_cx = bbox[..., 0:1] - agent_cx.unsqueeze(2)  # (B, T, K, 1)
        rel_cy = bbox[..., 1:2] - agent_cy.unsqueeze(2)
        rel = torch.cat([rel_cx, rel_cy], dim=-1)  # (B, T, K, 2)

        # Role embedding.
        role_clamped = slot_role.clamp(min=0)
        role_emb = self.role_embed(role_clamped)  # (B, T, K, role_emb_dim)

        # Group features.
        gf = slot_group_feat

        raw_struct = torch.cat([bbox, geo, rel, role_emb, gf], dim=-1)  # (B, T, K, D_raw)

        targets = {
            "bbox": bbox,
            "valid": slot_valid,
        }
        return raw_struct, targets


class CausalStructureEncoder(nn.Module):
    """raw_struct (B,T,K,D_raw) -> s (B,T,K,D_s). Causal temporal attention."""

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

    def forward(self, raw_struct: Tensor, slot_valid: Tensor) -> Tensor:
        B, T, K, _ = raw_struct.shape
        s = self.mlp(raw_struct)  # (B, T, K, D_s)

        # Causal temporal attention per object: (B*K, T, D_s)
        s = s.permute(0, 2, 1, 3).reshape(B * K, T, self.struct_dim)
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=s.device, dtype=s.dtype), diagonal=1,
        )
        s = self.temporal(s, mask=causal_mask)
        s = s.reshape(B, K, T, self.struct_dim).permute(0, 2, 1, 3)

        # Slot attention per frame: (B*T, K, D_s)
        s = s.reshape(B * T, K, self.struct_dim)
        pad = slot_valid.reshape(B * T, K).logical_not()
        s = self.slot(s, src_key_padding_mask=pad)
        s = s.reshape(B, T, K, self.struct_dim)
        s = s * slot_valid.unsqueeze(-1).float()
        return s


class AtariStructureHead(nn.Module):
    """s_hat (B,T,K,D_s) -> predicted bbox + existence + group_feat."""

    def __init__(self, struct_dim: int = 128, group_feat_dim: int = 19):
        super().__init__()
        self.bbox_head = nn.Sequential(
            nn.Linear(struct_dim, struct_dim),
            nn.GELU(),
            nn.Linear(struct_dim, 4),
        )
        self.exist_head = nn.Sequential(
            nn.Linear(struct_dim, struct_dim // 2),
            nn.GELU(),
            nn.Linear(struct_dim // 2, 1),
        )
        self.group_head = nn.Linear(struct_dim, group_feat_dim)

    def forward(self, s_hat: Tensor) -> Dict[str, Tensor]:
        return {
            "bbox": self.bbox_head(s_hat),           # (..., 4) cxcywh
            "exist_logit": self.exist_head(s_hat),   # (..., 1)
            "group_feat": self.group_head(s_hat),    # (..., D_g)
        }
