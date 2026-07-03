"""
V12 Dynamics: FactorizedIDM + FactorizedFDM.

IDM:  s_t, s_{t+1} -> z_action  (inverse dynamics, VAE bottleneck)
FDM:  s_t, z_action -> s_hat_{t+1}  (forward dynamics, residual)

Hard constraints:
  - Content tensors never enter IDM or FDM.
  - FDM predicts structure residual, not RGB residual.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SlotSelfAttention(nn.Module):
    """Self-attention across K slots with valid_mask padding."""

    def __init__(self, dim: int, heads: int = 4, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        """x: (B, K, D), valid: (B, K) bool -> (B, K, D)."""
        pad = valid.to(torch.bool).logical_not()
        out = self.transformer(x, src_key_padding_mask=pad)
        return out


class FactorizedIDM(nn.Module):
    """Inverse dynamics: s_t (all slots), s_{t+1} (per slot) -> z_action (per slot).

    z_t^k = IDM(s_t^{1:K}, s_{t+1}^k)
    Only structure enters IDM.
    """

    def __init__(
        self,
        struct_dim: int = 128,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.struct_dim = struct_dim
        self.latent_dim = latent_dim
        self.context = SlotSelfAttention(struct_dim, heads, layers, dropout)
        # Input: [context, s_t, s_{t+1}, delta] = 4 * struct_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(4 * struct_dim),
            nn.Linear(4 * struct_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, latent_dim * 2)
        self.var_min = -5.0
        self.var_max = 3.0

    def forward(
        self,
        s_t: Tensor,        # (B, K, D_s) or (B, T, K, D_s)
        s_tp1: Tensor,      # same as s_t
        valid: Tensor,      # (B, K) or (B, T, K) bool
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Flatten optional time dim into batch.
        time_dim = s_t.dim() == 4
        if time_dim:
            B, T, K, D = s_t.shape
            s_t = s_t.reshape(B * T, K, D)
            s_tp1 = s_tp1.reshape(B * T, K, D)
            valid = valid.reshape(B * T, K)

        ctx = self.context(s_t, valid)                            # (B, K, D_s)
        delta = s_tp1 - s_t
        x = torch.cat([ctx, s_t, s_tp1, delta], dim=-1)          # (B, K, 4*D_s)
        h = self.mlp(x)
        mu_logvar = self.head(h)                                  # (B, K, 2*D_z)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, self.var_min, self.var_max)

        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        # Zero out invalid slots.
        valid_f = valid.unsqueeze(-1).float()
        z = z * valid_f
        mu = mu * valid_f
        logvar = logvar * valid_f

        if time_dim:
            z = z.reshape(B, T, K, -1)
            mu = mu.reshape(B, T, K, -1)
            logvar = logvar.reshape(B, T, K, -1)
        return z, mu, logvar


class FactorizedFDM(nn.Module):
    """Forward dynamics: s_t, z_action -> s_hat_{t+1} (residual).

    s_hat = s_t + FDM(s_t, z). Predicts structure residual only.
    """

    def __init__(
        self,
        struct_dim: int = 128,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.struct_dim = struct_dim
        self.input_proj = nn.Linear(struct_dim + latent_dim, struct_dim)
        self.slot_attn = SlotSelfAttention(struct_dim, heads, layers, dropout)
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(struct_dim, struct_dim),
        )

    def forward(
        self,
        s_t: Tensor,       # (B, K, D_s) or (B, T, K, D_s)
        z: Tensor,         # (B, K, D_z) or (B, T, K, D_z)
        valid: Tensor,     # (B, K) or (B, T, K) bool
    ) -> Tensor:
        time_dim = s_t.dim() == 4
        if time_dim:
            B, T, K, D = s_t.shape
            s_t = s_t.reshape(B * T, K, D)
            z = z.reshape(B * T, K, -1)
            valid = valid.reshape(B * T, K)

        x = torch.cat([s_t, z], dim=-1)                           # (B, K, D_s+D_z)
        h = self.input_proj(x)
        h = self.slot_attn(h, valid)
        delta_s = self.out(h)                                     # (B, K, D_s)
        s_hat = s_t + delta_s
        s_hat = s_hat * valid.unsqueeze(-1).float()

        if time_dim:
            s_hat = s_hat.reshape(B, T, K, -1)
        return s_hat
