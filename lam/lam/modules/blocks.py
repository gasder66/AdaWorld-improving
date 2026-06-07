from typing import Dict, Optional, Tuple

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lam.modules.embeddings import RotaryEmbedding
from torch import Tensor
from torch.utils.checkpoint import checkpoint


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
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout)
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        """
        Args:
            q: Query tensor of shape (B, N, D) - typically video patches
            kv: Key/Value tensor of shape (B, K, D) - typically action slots
        Returns:
            Output tensor of shape (B, N, D)
        """
        query = self.to_q(q)
        key = self.to_k(kv)
        value = self.to_v(kv)

        query, key, value = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (query, key, value)
        )

        attn_weight = query @ key.transpose(-2, -1) * self.scale
        attn_weight = torch.softmax(attn_weight, dim=-1)
        out = attn_weight @ value

        del query, key, value
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


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
            is_causal: bool = False
    ) -> Tensor:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query)
        if is_causal:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(attn_bias)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        attn_weight = query @ key.transpose(-2, -1) * self.scale
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        if self.rot_emb:
            q = self.rotary_embedding.rotate_queries_or_keys(q, self.rotary_embedding.freqs)
            k = self.rotary_embedding.rotate_queries_or_keys(k, self.rotary_embedding.freqs)
            q, k = map(lambda t: t.contiguous(), (q, k))
        out = self.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SpatioBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0,
                 use_grad_checkpointing: bool = False) -> None:
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
        self.use_grad_checkpointing = use_grad_checkpointing

        # Stored slot→patch attention maps for spatial supervision loss
        self.last_slot_attn_maps = None

    def _get_slot_attention(self, x: Tensor, num_slots: int) -> Tensor:
        """计算 slot→patch 注意力权重（用于空间监督损失）。
        
        Args:
            x: (BT, S, D) - layer norm 后的 tokens
            num_slots: slot 数量 K
            
        Returns:
            attn: (BT, H, K, N_patches) - 可微分注意力权重
        """
        H = self.spatial_attn.heads
        q = self.spatial_attn.to_q(x)
        k = self.spatial_attn.to_k(x)
        q = rearrange(q, "b s (h d) -> b h s d", h=H)
        k = rearrange(k, "b s (h d) -> b h s d", h=H)

        q_slots = q[:, :, :num_slots, :]       # (BT, H, K, d)
        k_patches = k[:, :, num_slots:, :]     # (BT, H, N, d)

        scale = (q.shape[-1]) ** -0.5
        attn = torch.softmax(
            q_slots @ k_patches.transpose(-2, -1) * scale, dim=-1
        )  # (BT, H, K, N_patches)
        return attn  # 不 detach，保留梯度

    def _forward_block(self, x: Tensor, num_slots: int = None) -> Tensor:
        t_len = x.shape[1]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_)
        x = x + x_

        # 存储 slot→patch 注意力（用于空间监督损失）
        if num_slots is not None and num_slots > 1 and self.training:
            self.last_slot_attn_maps = self._get_slot_attention(self.norm1(x), num_slots)
        else:
            self.last_slot_attn_maps = None

        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Feedforward
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x

    def forward(self, x: Tensor, num_slots: int = None) -> Tensor:
        if self.use_grad_checkpointing and self.training:
            return checkpoint(self._forward_block, x, num_slots, use_reentrant=False)
        return self._forward_block(x, num_slots)


class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0,
                 use_slot_competition: bool = False,
                 use_grad_checkpointing: bool = False) -> None:
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
        self.use_slot_competition = use_slot_competition
        self.use_grad_checkpointing = use_grad_checkpointing

        # Store attention for spatial supervision loss
        self.last_slot_attn_map = None

    def _get_slot_attention(self, x: Tensor, num_slots: int) -> Tensor:
        """计算 slot→patch 注意力权重（用于空间监督损失）。
        
        Args:
            x: (BT, S, D) - layer norm 后的 tokens
            num_slots: slot 数量 K
            
        Returns:
            attn: (BT, H, K, N_patches) - 可微分注意力权重
        """
        H = self.spatial_attn.heads
        q = self.spatial_attn.to_q(x)
        k = self.spatial_attn.to_k(x)
        q = rearrange(q, "b s (h d) -> b h s d", h=H)
        k = rearrange(k, "b s (h d) -> b h s d", h=H)

        q_slots = q[:, :, :num_slots, :]       # (BT, H, K, d)
        k_patches = k[:, :, num_slots:, :]     # (BT, H, N, d)

        scale = (q.shape[-1]) ** -0.5
        attn = torch.softmax(
            q_slots @ k_patches.transpose(-2, -1) * scale, dim=-1
        )  # (BT, H, K, N_patches)
        return attn  # 不 detach，保留梯度

    def _forward_block(self, x: Tensor, causal_temporal: bool = False,
                       num_slots: int = None) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)

        if self.use_slot_competition and num_slots is not None:
            x_, attn_map = self._spatial_attn_with_competition(x_, num_slots)
            # 训练时保留梯度用于辅助损失，评估时 detach
            self.last_slot_attn_map = attn_map if self.training else attn_map.detach()
        else:
            x_ = self.spatial_attn(x_)
            # 非竞争模式也计算 slot→patch 注意力用于空间监督（保留梯度）
            if num_slots is not None and num_slots > 1:
                slot_attn = self._get_slot_attention(x_, num_slots)
                self.last_slot_attn_map = slot_attn if self.training else slot_attn.detach()
            else:
                self.last_slot_attn_map = None
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Temporal attention
        x = rearrange(x, "b t s e -> (b s) t e")
        x_ = self.norm2(x)
        if causal_temporal:
            x_ = self.temporal_attn(x_, is_causal=True)
        else:
            x_ = self.temporal_attn(x_)
        x = x + x_
        x = rearrange(x, "(b s) t e -> b t s e", s=s_len)

        # Feedforward
        x_ = self.norm3(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x

    def forward(self, x: Tensor, causal_temporal: bool = False,
                num_slots: int = None) -> Tensor:
        if self.use_grad_checkpointing and self.training:
            return checkpoint(
                self._forward_block, x, causal_temporal, num_slots,
                use_reentrant=False)
        return self._forward_block(x, causal_temporal, num_slots)

    def _spatial_attn_with_competition(self, x: Tensor, num_slots: int) -> Tuple[Tensor, Tensor]:
        """
        Slot Attention 竞争机制：slot 之间通过 softmax 竞争 patch 的关注权。
        
        基于 Slot Attention 论文，对 slot→patch 的注意力在 slot 维度做 softmax，
        使得每个 patch 的注意力在不同 slot 之间竞争（归一化后加起来为 1）。
        
        数值稳定性：
        - 对 attention logits 进行裁剪，避免极端值导致梯度爆炸
        - 使用 detached 注意力图存储
        """
        attn = self.spatial_attn
        q = attn.to_q(x)   # (BT, S, D)
        k = attn.to_k(x)
        v = attn.to_v(x)
        
        BT, S, D = q.shape
        H = attn.heads
        head_dim = D // H
        
        # Split into heads
        q, k, v = map(lambda t: t.reshape(BT, S, H, head_dim).transpose(1, 2), (q, k, v))
        scale = head_dim ** -0.5
        
        # Slot queries and patch keys/values
        q_slots = q[:, :, :num_slots, :]          # (BT, H, K, head_dim)
        q_patches = q[:, :, num_slots:, :]        # (BT, H, N, head_dim)
        k_slots = k[:, :, :num_slots, :]
        k_patches = k[:, :, num_slots:, :]
        v_slots = v[:, :, :num_slots, :]
        v_patches = v[:, :, num_slots:, :]
        
        # === Slot→Slot (standard attention: softmax over keys) ===
        slot_to_slot = torch.softmax(q_slots @ k_slots.transpose(-2, -1) * scale, dim=-1)
        slot_to_slot_out = slot_to_slot @ v_slots  # (BT, H, K, head_dim)
        
        # === Slot→Patch (Slot Attention: softmax over slots, not keys) ===
        # Each patch's attention weights sum to 1 across slots → slots compete for patches
        # 数值稳定性: 裁剪 logits 避免极端 softmax 值
        slot_patch_logits = q_slots @ k_patches.transpose(-2, -1) * scale
        slot_patch_logits = torch.clamp(slot_patch_logits, -5.0, 5.0)  # 防止梯度爆炸
        slot_to_patch = torch.softmax(slot_patch_logits, dim=2)  # softmax over K slots
        slot_to_patch_out = slot_to_patch @ v_patches  # (BT, H, K, head_dim)
        
        # === Patch→All (standard attention) ===
        patch_to_all = torch.softmax(q_patches @ k.transpose(-2, -1) * scale, dim=-1)
        patch_to_all_out = patch_to_all @ v
        
        # Combine outputs
        slot_out = slot_to_slot_out + slot_to_patch_out  # slot query outputs
        combined = torch.cat([slot_out, patch_to_all_out], dim=2)  # (BT, H, S, head_dim)
        combined = combined.transpose(1, 2).reshape(BT, S, D)
        
        out = attn.to_out(combined)
        return out, slot_to_patch  # 保留梯度用于空间监督损失


class SpatioTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            use_grad_checkpointing: bool = False,
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
                    dropout,
                    use_grad_checkpointing=use_grad_checkpointing,
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor, num_slots: int = None) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x, num_slots=num_slots)
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
            causal_temporal: bool = False,
            use_slot_competition: bool = False,
            use_grad_checkpointing: bool = False,
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
                    dropout,
                    use_slot_competition=use_slot_competition,
                    use_grad_checkpointing=use_grad_checkpointing,
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)
        self.causal_temporal = causal_temporal

    def forward(self, x: Tensor, num_slots: int = None) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal, num_slots=num_slots)
        x = self.out(x)
        return x  # (B, T, E)


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


class MaskedPool(nn.Module):
    """
    Mask-Guided 特征池化。

    将 patch 级别的特征按 GT mask 区域加权平均池化，得到每个主体的特征向量。

    Input:
        patches: (B, T, N, D)  — patch 特征（N = grid_h * grid_w）
        masks:   (B, T, A, H, W) — 逐主体二值 mask（A = max_actors）

    Output:
        obj_feats: (B, T, A, D) — 每个主体的聚合特征
        valid_mask: (B, T, A)   — 该主体是否存在的指示（mask 非空）
    """

    def __init__(self, patch_size: int, grid_h: int, grid_w: int):
        super().__init__()
        self.patch_size = patch_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w

    def forward(self, patches: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, N, D = patches.shape
        *_, A, H, W = masks.shape
        assert H % self.grid_h == 0 and W % self.grid_w == 0, \
            f"mask size ({H}x{W}) must be divisible by grid ({self.grid_h}x{self.grid_w})"

        # 1. 将 mask 下采样到 patch 级别
        # masks: (B, T, A, H, W) → pool → (B, T, A, grid_h, grid_w) → flatten
        masks_flat = masks.reshape(B * T * A, 1, H, W)  # (B*T*A, 1, H, W)
        masks_down = F.adaptive_avg_pool2d(masks_flat, (self.grid_h, self.grid_w))
        # (B*T*A, 1, grid_h, grid_w)
        masks_down = masks_down.reshape(B, T, A, self.num_patches)  # (B, T, A, N)

        # 2. 检测哪些主体存在（mask 非空）
        valid_mask = masks_down.sum(dim=-1) > 0.5  # (B, T, A), bool

        # 3. 加权平均池化
        # patches: (B, T, N, D) → (B, T, 1, N, D)
        # masks_down: (B, T, A, N) → (B, T, A, N, 1)
        weights = masks_down.unsqueeze(-1)           # (B, T, A, N, 1)
        feat = patches.unsqueeze(2) * weights        # (B, T, A, N, D)
        mask_sum = weights.sum(dim=-2)               # (B, T, A, 1), 安全 sum
        mask_sum = mask_sum + (mask_sum < 1e-6).float() * 1e-6  # 避免除零
        obj_feats = feat.sum(dim=-2) / mask_sum      # (B, T, A, D)

        return obj_feats, valid_mask


class ObjectInteractionModule(nn.Module):
    """
    主体间交互建模模块。

    使用 Self-Attention 让不同主体的隐动作互相交互，
    捕捉主体间的协同或竞争关系。

    Input:
        x: (B, A, D) — A 个主体的特征
        padding_mask: (B, A) — bool mask, True = 有效主体

    Output:
        (B, A, D) — 经过交互后的特征
    """

    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm 更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x: Tensor,
                padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (B, A, D) — A 个主体的特征
            padding_mask: (B, A), optional — True 表示有效主体
        Returns:
            (B, A, D) — 交互后的特征
        """
        return self.transformer(x, src_key_padding_mask=padding_mask)


class ObjectSpatioTemporalAttention(nn.Module):
    """
    对象级时空注意力模块。

    对 (A+1)×T 的二维特征矩阵做 Transformer Self-Attention，
    同时建模主体间交互（空间）和时序动态（时间）。

    输入: (B, T, A+1, D) — 含背景槽的对象级特征
    输出: (B, T, A+1, D) — 经过时空交互后的特征

    注意力模式:
        - 同一帧内跨对象槽：背景↔主体1↔主体2↔...（空间注意力）
        - 同一对象跨帧：obj@t0↔obj@t1↔...（时间注意力）
        - 全交互：任意 token 可以 attend 到任意其他 token
    """

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # 可学习的位置编码：对象类型 + 时序位置
        # 由调用方负责添加，这里只做 Transformer 编码

    def forward(self, obj_feats: Tensor, valid_mask: Tensor = None) -> Tensor:
        """
        Args:
            obj_feats: (B, T, A_plus1, D) — 含背景槽的对象级特征
            valid_mask: (B, T, A_plus1) — bool, 有效对象指示（背景槽始终为 True）

        Returns:
            (B, T, A_plus1, D) — 经过时空交互后的特征
        """
        B, T, A1, D = obj_feats.shape

        # 重排为 (B, T*(A+1), D) 的序列
        seq = obj_feats.reshape(B, T * A1, D)

        # 构建 padding_mask: (B, T*(A+1))
        # Transformer 的 key_padding_mask: True = mask out (ignore)
        if valid_mask is not None:
            pad = valid_mask.reshape(B, T * A1)
            padding_mask = ~pad  # True = 需要被 mask 掉
        else:
            padding_mask = None

        # Transformer Self-Attention
        out = self.transformer(seq, src_key_padding_mask=padding_mask)

        # 重排回 (B, T, A+1, D)
        out = out.reshape(B, T, A1, D)
        return out
