import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lam.modules.embeddings import RotaryEmbedding
from torch import Tensor


def patchify(videos: Tensor, size: int) -> Tensor:
    B, T, H, W, C = videos.shape
    videos = videos[:, :, :H - (H % size), :W - (W % size), :]
    x = rearrange(videos, "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)", hp=size, wp=size)
    return x


def unpatchify(patches: Tensor, size: int, h_out: int, w_out: int) -> Tensor:
    h_pad = -h_out % size
    hn = (h_out + h_pad) // size
    x = rearrange(patches, "b t (hn wn) (hp wp c) -> b t (hn hp) (wn wp) c", hp=size, wp=size, hn=hn)
    return x[:, :, :h_out, :w_out]


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_enc = pe

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[:x.shape[2]].to(x.device)


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0, rot_emb: bool = False) -> None:
        super(SelfAttention, self).__init__()
        inner_dim = model_dim // num_heads
        self.scale = inner_dim ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout)
        )

        self.rot_emb = rot_emb
        if rot_emb:
            self.rotary_embedding = RotaryEmbedding(dim=inner_dim)

    def scaled_dot_product_attention(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            is_causal: bool = False,
            key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query)
        if is_causal:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(attn_bias)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        if key_padding_mask is not None:
            # key_padding_mask: (B, S), True = mask out this key
            attn_bias = attn_bias.unsqueeze(0) + key_padding_mask.unsqueeze(1).unsqueeze(2).to(attn_bias.dtype) * float("-inf")

        attn_weight = query @ key.transpose(-2, -1) * self.scale
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def forward(self, x: Tensor, is_causal: bool = False, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        if self.rot_emb:
            q = self.rotary_embedding.rotate_queries_or_keys(q, self.rotary_embedding.freqs)
            k = self.rotary_embedding.rotate_queries_or_keys(k, self.rotary_embedding.freqs)
            q, k = map(lambda t: t.contiguous(), (q, k))
        out = self.scaled_dot_product_attention(q, k, v, is_causal=is_causal, key_padding_mask=key_padding_mask)
        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SpatioBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        t_len = x.shape[1]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        if key_padding_mask is not None:
            mask = rearrange(key_padding_mask, "b t s -> (b t) s")
            x_ = self.spatial_attn(x_, key_padding_mask=~mask)
        else:
            x_ = self.spatial_attn(x_)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Feedforward
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioTemporalBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout, rot_emb=True)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, causal_temporal: bool = False, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        if key_padding_mask is not None:
            mask_s = rearrange(key_padding_mask, "b t s -> (b t) s")
            x_ = self.spatial_attn(x_, key_padding_mask=~mask_s)
        else:
            x_ = self.spatial_attn(x_)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Temporal attention
        x = rearrange(x, "b t s e -> (b s) t e")
        x_ = self.norm2(x)
        if key_padding_mask is not None:
            mask_t = rearrange(key_padding_mask, "b t s -> (b s) t")
            x_ = self.temporal_attn(x_, is_causal=causal_temporal, key_padding_mask=~mask_t)
        else:
            x_ = self.temporal_attn(x_, is_causal=causal_temporal)
        x = x + x_
        x = rearrange(x, "(b s) t e -> b t s e", s=s_len)

        # Feedforward
        x_ = self.norm3(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(SpatioTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.out(x)
        return x  # (B, T, E)


class SpatioTemporalTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            causal_temporal: bool = False
    ) -> None:
        super(SpatioTemporalTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)
        self.causal_temporal = causal_temporal

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal, key_padding_mask=key_padding_mask)
        x = self.out(x)
        return x  # (B, T, E)


class CrossAttention(nn.Module):
    """Cross-attention module for slot-patch fusion in the decoder."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(CrossAttention, self).__init__()
        inner_dim = model_dim // num_heads
        self.scale = inner_dim ** -0.5
        self.heads = num_heads
        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Dropout(dropout))

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        B, N, D = q.shape
        K = kv.shape[1]
        q = self.to_q(q).reshape(B, N, self.heads, D // self.heads).permute(0, 2, 1, 3)
        k = self.to_k(kv).reshape(B, K, self.heads, D // self.heads).permute(0, 2, 1, 3)
        v = self.to_v(kv).reshape(B, K, self.heads, D // self.heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.to_out(out)


class VectorQuantizer(nn.Module):
    def __init__(self, num_latents: int, latent_dim: int, code_restart: bool = False) -> None:
        super(VectorQuantizer, self).__init__()
        self.codebook = nn.Embedding(num_latents, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_latents, 1.0 / num_latents)

        # Initialize a usage buffer
        self.register_buffer("usage", torch.zeros(num_latents), persistent=False)
        self.num_latents = num_latents

        self.code_restart = code_restart

    def update_usage(self, min_enc) -> None:
        for idx in min_enc:
            self.usage[idx] = self.usage[idx] + 1  # Add used code

    def random_restart(self) -> None:
        if self.code_restart:
            # Randomly restart all dead codes
            dead_codes = torch.nonzero(self.usage < 1).squeeze(1)
            rand_codes = torch.randperm(self.num_latents)[0:len(dead_codes)]
            print(f"Restarting {len(dead_codes)} codes")
            with torch.no_grad():
                self.codebook.weight[dead_codes] = self.codebook.weight[rand_codes]

            if hasattr(self, "inner_vq"):
                self.inner_vq.random_restart()

    def reset_usage(self) -> None:
        if self.code_restart:
            # Reset usage between epochs
            self.usage.zero_()

            if hasattr(self, "inner_vq"):
                self.inner_vq.reset_usage()

    def forward(self, x: Tensor, delta_psnr: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Compute distances
        distance = torch.cdist(x, self.codebook.weight)

        # Get indices and embeddings
        indices = torch.argmin(distance, dim=1)
        if delta_psnr:
            shape = indices.shape
            rand_indices = torch.randint(0, self.num_latents, shape).to(distance.device)
            while torch.any(rand_indices == indices):
                new_indices = torch.randint(0, self.num_latents, shape).to(distance.device)
                rand_indices = torch.where(rand_indices == indices, new_indices, rand_indices)
            z = self.codebook(rand_indices)
        else:
            z = self.codebook(indices)

        # Update code usage
        if not self.training or self.code_restart:
            self.update_usage(indices)

        # Straight through estimator
        z_q = x + (z - x).detach()
        return z_q, z, x, indices


class ResidualVectorQuantizer(VectorQuantizer):
    def __init__(self, num_latents: int, latent_dim: int) -> None:
        super(ResidualVectorQuantizer, self).__init__(num_latents, latent_dim)
        self.inner_vq = VectorQuantizer(num_latents, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Compute distances
        distance = torch.cdist(x, self.codebook.weight)

        # Get indices and embeddings
        indices = torch.argmin(distance, dim=1)
        z = self.codebook(indices)

        # Residual quantization
        residual = x - z.detach()
        inner_z_q, inner_z, inner_x, inner_indices = self.inner_vq(residual)

        # Update code usage
        if not self.training or self.code_restart:
            self.update_usage(indices)
            self.inner_vq.update_usage(inner_indices)

        # Straight through estimator
        z_q = x + (z - x).detach()
        return z_q + inner_z_q, z, x, indices, inner_z, inner_x, inner_indices


class ObjectReconHead(nn.Module):
    """
    对象级重建头。

    从隐动作预测 slot 特征的帧间差分，提供无监督对象级学习信号。
    Input:  z: (B, T-1, K, latent_dim)
    Output: delta_pred: (B, T-1, K, model_dim)
    """

    def __init__(self, latent_dim: int, model_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


def compute_subject_positions(masks: Tensor) -> Tensor:
    """
    从二值 mask 计算每个主体的质心位置。

    Args:
        masks: (B, T, A, H, W) float32 binary

    Returns:
        positions: (B, T, A, 2) float32, normalized [0,1]
    """
    B, T, A, H, W = masks.shape
    device = masks.device
    ys = torch.arange(H, device=device).view(1, 1, 1, H, 1)
    xs = torch.arange(W, device=device).view(1, 1, 1, 1, W)
    mask_sum = masks.sum(dim=(-2, -1)).clamp(min=1e-6)
    cy = (masks * ys.to(masks.dtype)).sum(dim=(-2, -1)) / mask_sum
    cx = (masks * xs.to(masks.dtype)).sum(dim=(-2, -1)) / mask_sum
    return torch.stack([cx / W, cy / H], dim=-1)


class MaskedPool(nn.Module):
    """
    Mask-Guided 特征池化。

    将 patch 级别的特征按 GT mask 区域加权平均池化，得到每个主体的特征向量。

    Input:
        patches: (B, T, N, D)  — patch 特征
        masks:   (B, T, A, H, W) — 逐主体二值 mask (A = max_actors)

    Output:
        obj_feats: (B, T, A, D) — 每个主体的聚合特征
        valid_mask: (B, T, A)   — 该主体是否存在的指示 (mask 非空)
    """

    def __init__(self, grid_h: int, grid_w: int):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w

    def forward(self, patches: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, N, D = patches.shape
        *_, A, H, W = masks.shape
        assert H % self.grid_h == 0 and W % self.grid_w == 0, \
            f"mask size ({H}x{W}) must be divisible by grid ({self.grid_h}x{self.grid_w})"

        # 1. 将 mask 下采样到 patch 级别
        masks_flat = masks.reshape(B * T * A, 1, H, W)
        masks_down = F.adaptive_avg_pool2d(masks_flat, (self.grid_h, self.grid_w))
        masks_down = masks_down.reshape(B, T, A, self.num_patches)

        # 2. 检测哪些主体存在 (mask 非空)
        valid_mask = masks_down.sum(dim=-1) > 0.5  # (B, T, A), bool

        # 3. 加权平均池化
        weights = masks_down.unsqueeze(-1)                # (B, T, A, N, 1)
        feat = patches.unsqueeze(2) * weights             # (B, T, A, N, D)
        mask_sum = weights.sum(dim=-2)                    # (B, T, A, 1)
        mask_sum = mask_sum + (mask_sum < 1e-6).float() * 1e-6
        obj_feats = feat.sum(dim=-2) / mask_sum           # (B, T, A, D)

        return obj_feats, valid_mask


class ObjectSpatioTemporalAttention(nn.Module):
    """
    对象级时空注意力模块。

    对 (K)×T 的二维特征矩阵做 Transformer Self-Attention，
    同时建模主体间交互（空间）和时序动态（时间）。

    Input:  obj_feats (B, T, K, D)
            valid_mask (B, T, K)  — True=有效, False=padding
    Output: obj_feats (B, T, K, D)
    """

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads,
            dim_feedforward=dim * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, obj_feats: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        B, T, A1, D = obj_feats.shape

        seq = obj_feats.reshape(B, T * A1, D)

        if valid_mask is not None:
            pad = valid_mask.reshape(B, T * A1)
            padding_mask = ~pad  # True = mask out
            padding_mask = padding_mask.to(torch.bool)
        else:
            padding_mask = None

        out = self.transformer(seq, src_key_padding_mask=padding_mask)
        out = out.reshape(B, T, A1, D)
        return out
