import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from external.lam.modules.embeddings import RotaryEmbedding
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

    def _forward_block(self, x: Tensor) -> Tensor:
        t_len = x.shape[1]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Feedforward
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.use_grad_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_block, x, use_reentrant=False)
        return self._forward_block(x)


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

        # Store attention for diversity loss (when competition is enabled)
        self.last_slot_attn_map = None

    def _forward_block(self, x: Tensor, causal_temporal: bool = False,
                       num_slots: int = None) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)

        if self.use_slot_competition and num_slots is not None:
            x_, attn_map = self._spatial_attn_with_competition(x_, num_slots)
            self.last_slot_attn_map = attn_map.detach()  # detach for stability
        else:
            x_ = self.spatial_attn(x_)
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
        
        使用 checkpoint 以避免反向传播时的内存问题。
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
        slot_patch_logits = q_slots @ k_patches.transpose(-2, -1) * scale
        slot_patch_logits = torch.clamp(slot_patch_logits, -5.0, 5.0)
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
        return out, slot_to_patch.detach()


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

    def forward(self, x: Tensor) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x)
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
