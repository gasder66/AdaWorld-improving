"""
V12 Decoder: StructureHead + FusionRenderer.

StructureHead: s_hat -> interpretable structure (bbox, mask_lowres, moments).
FusionRenderer: content + predicted structure -> RGB reconstruction.

Hard constraints:
  - Decoder NEVER receives z_action.
  - Decoder only gets c_obj, c_bg, s_hat, pred_struct, valid_mask.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam.modules.blocks import (
    SpatioTransformer, CrossAttention, PositionalEncoding,
)


class StructureHead(nn.Module):
    """Decode s_hat back to interpretable structure targets."""

    def __init__(self, struct_dim: int = 128, mask_grid: int = 16):
        super().__init__()
        self.mask_grid = mask_grid
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(struct_dim),
            nn.Linear(struct_dim, struct_dim),
            nn.GELU(),
            nn.Linear(struct_dim, 4),
        )
        self.moments_head = nn.Sequential(
            nn.LayerNorm(struct_dim),
            nn.Linear(struct_dim, struct_dim),
            nn.GELU(),
            nn.Linear(struct_dim, 6),
        )
        # Mask decoder: struct -> (mask_grid, mask_grid)
        self.mask_head = nn.Sequential(
            nn.Linear(struct_dim, struct_dim * 2),
            nn.GELU(),
            nn.Linear(struct_dim * 2, mask_grid * mask_grid),
        )

    def forward(self, s_hat: Tensor) -> Dict[str, Tensor]:
        """s_hat: (B, T, K, D_s) -> dict of (B, T, K, ...)."""
        bbox = self.bbox_head(s_hat)
        moments = self.moments_head(s_hat)
        mask_flat = self.mask_head(s_hat)                          # (B, T, K, g*g)
        mask_low = mask_flat.reshape(*s_hat.shape[:3], 1, self.mask_grid, self.mask_grid)
        return {"bbox": bbox, "mask_lowres": mask_low, "moments": moments}


class FusionRenderer(nn.Module):
    """content + structure -> RGB. No z_action allowed."""

    def __init__(
        self,
        content_dim: int = 128,
        struct_dim: int = 128,
        dec_dim: int = 256,
        patch_size: int = 16,
        image_size: int = 256,
        num_heads: int = 8,
        dec_blocks: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dec_dim = dec_dim
        self.patch_size = patch_size
        self.image_size = image_size
        grid = image_size // patch_size
        n_patches = grid * grid
        self.n_patches = n_patches

        patch_token_dim = 3 * patch_size * patch_size
        self.patch_token_dim = patch_token_dim

        # Project content+structure -> decoder tokens.
        self.render_proj = nn.Sequential(
            nn.LayerNorm(content_dim + struct_dim),
            nn.Linear(content_dim + struct_dim, dec_dim),
            nn.GELU(),
        )
        self.bg_proj = nn.Sequential(
            nn.LayerNorm(content_dim),
            nn.Linear(content_dim, dec_dim),
            nn.GELU(),
        )

        # Learned patch queries + positional encoding.
        self.patch_queries = nn.Parameter(torch.randn(1, n_patches, dec_dim) * 0.02)
        self.pos_enc = PositionalEncoding(dec_dim)

        self.cross_attn = CrossAttention(dec_dim, num_heads, dropout=dropout)
        self.decoder = SpatioTransformer(
            in_dim=dec_dim, model_dim=dec_dim, out_dim=patch_token_dim,
            num_blocks=dec_blocks, num_heads=num_heads, dropout=dropout,
        )

    def forward(
        self,
        c_obj: Tensor,      # (B, K, D_c)
        c_bg: Tensor,       # (B, D_c)
        s_hat: Tensor,      # (B, T, K, D_s)
        valid: Tensor,      # (B, T, K) bool (validity at target frames)
    ) -> Tensor:
        B, T, K, _ = s_hat.shape
        D = self.dec_dim

        # Render tokens: content + structure per object.
        c_obj_exp = c_obj.unsqueeze(1).expand(B, T, K, -1)        # (B, T, K, D_c)
        render_in = torch.cat([c_obj_exp, s_hat], dim=-1)         # (B, T, K, D_c+D_s)
        render_tok = self.render_proj(render_in)                  # (B, T, K, D)
        render_tok = render_tok * valid.unsqueeze(-1).float()

        # Background token (shared across time).
        bg_tok = self.bg_proj(c_bg).unsqueeze(1).unsqueeze(2).expand(B, T, 1, D)  # (B, T, 1, D)
        tokens = torch.cat([bg_tok, render_tok], dim=2)           # (B, T, K+1, D)

        # Patch queries.
        queries = self.patch_queries.expand(B * T, -1, -1)        # (B*T, N, D)
        queries = self.pos_enc(queries.unsqueeze(1)).squeeze(1)   # add pos

        # Cross-attention: (B*T, N, D) x (B*T, K+1, D)
        tokens_flat = tokens.reshape(B * T, K + 1, D)
        fused = self.cross_attn(q=queries, kv=tokens_flat)        # (B*T, N, D)

        # Decode patches.
        fused = fused.reshape(B, T, self.n_patches, D)            # (B, T, N, D)
        decoded = self.decoder(fused)                             # (B, T, N, patch_token_dim)

        # Unpatchify -> (B, T, H, W, C)
        from lam.modules.blocks import unpatchify
        recon = unpatchify(decoded, self.patch_size, self.image_size, self.image_size)
        recon = torch.sigmoid(recon)
        return recon
