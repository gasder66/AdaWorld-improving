"""
Latent Action Model — 原始 AdaWorld LAM + 多槽 + P0 改进

架构: [learnable tokens + patches] → SpatioTemporalTransformer
      → tokens from frame 2 → VAE → z
      → Decoder: patches(frame 1) + z → reconstruct frame 2

P0 改进:
  - 多槽 (K slots) + 多槽 VAE
  - CrossAttn decoder (K>1 时)
  - valid_mask: 支持 padding slot 在注意力中被屏蔽
  - ObjectReconHead: 预测 slot 特征帧间差分，提供无监督学习信号
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lam.modules.blocks import (
    patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer,
    CrossAttention, ObjectReconHead,
)
from torch import Tensor


def build_key_padding_mask(valid_mask: Tensor, num_slots: int, num_patches: int, T: int) -> Tensor:
    """从 valid_mask 构建 key_padding_mask.

    Args:
        valid_mask: (B, K) bool, True=有效 slot, False=padding
        num_slots: K, num_patches: N, T: 时间步数

    Returns:
        key_padding_mask: (B, T, K+N) bool, True=有效(保留), False=padding(屏蔽)
    """
    B = valid_mask.shape[0]
    device = valid_mask.device
    slot_mask = valid_mask.unsqueeze(1).expand(B, T, num_slots)
    patch_mask = torch.ones(B, T, num_patches, dtype=torch.bool, device=device)
    return torch.cat([slot_mask, patch_mask], dim=2)


class LatentActionModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            num_slots: int = 1,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_slots = num_slots
        patch_token_dim = in_dim * patch_size ** 2
        grid_size = 256 // patch_size
        self.num_patches = grid_size * grid_size

        # 多槽 learnable tokens
        self.action_prompts = nn.Parameter(torch.empty(1, 1, num_slots, patch_token_dim))
        nn.init.uniform_(self.action_prompts, a=-1, b=1)

        # 编码器
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim, model_dim=model_dim, out_dim=model_dim,
            num_blocks=enc_blocks, num_heads=num_heads, dropout=dropout,
        )

        # 多槽 VAE
        self.fcs = nn.ModuleList([
            nn.Linear(model_dim, latent_dim * 2) for _ in range(num_slots)
        ])

        # 对象级重建头 (预测 slot 特征帧间差分)
        self.obj_recon_head = ObjectReconHead(latent_dim, model_dim)

        # Decoder
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        if num_slots > 1:
            self.cross_attn = CrossAttention(model_dim, num_heads, dropout=dropout)
        self.decoder = SpatioTransformer(
            in_dim=model_dim, model_dim=model_dim, out_dim=patch_token_dim,
            num_blocks=dec_blocks, num_heads=num_heads, dropout=dropout,
        )

        self.mu_record = None

    def encode(self, videos: Tensor, valid_mask: Optional[Tensor] = None) -> Dict:
        B, T = videos.shape[:2]
        K = self.num_slots
        N = self.num_patches
        patches = patchify(videos, self.patch_size)  # (B, T, N, D_patch)

        prompts = self.action_prompts.expand(B, T, -1, -1)
        model_input = torch.cat([prompts, patches], dim=2)

        # valid_mask → key_padding_mask (bool, True=保留, False=屏蔽)
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = build_key_padding_mask(valid_mask, K, N, T)

        encoded = self.encoder(model_input, key_padding_mask=key_padding_mask)

        # slot tokens (编码器输出的前 K 个)
        slot_feats = encoded[:, :, :K]  # (B, T, K, E)

        # 隐动作: 取第二帧及之后的 slot tokens
        z = slot_feats[:, 1:]  # (B, T-1, K, E)

        # 多槽 VAE
        B_T1 = B * (T - 1)
        z_mu_list, z_var_list, z_rep_list = [], [], []
        for k in range(K):
            z_k = z[:, :, k].reshape(B_T1, self.model_dim)
            moments = self.fcs[k](z_k)
            mu_k, var_k = torch.chunk(moments, 2, dim=1)
            var_k = torch.clamp(var_k, -5.0, 3.0)
            if self.training:
                rep_k = mu_k + torch.randn_like(var_k) * torch.exp(0.5 * var_k)
            else:
                rep_k = mu_k
            z_mu_list.append(mu_k.reshape(B, T - 1, 1, self.latent_dim))
            z_var_list.append(var_k.reshape(B, T - 1, 1, self.latent_dim))
            z_rep_list.append(rep_k.reshape(B, T - 1, 1, self.latent_dim))

        z_mu = torch.cat(z_mu_list, dim=2)
        z_var = torch.cat(z_var_list, dim=2)
        z_rep = torch.cat(z_rep_list, dim=2)

        if not self.training:
            if self.mu_record is None:
                self.mu_record = z_mu.detach().cpu()
            else:
                self.mu_record = torch.cat([self.mu_record, z_mu.detach().cpu()], dim=0)

        return {"patches": patches, "slot_feats": slot_feats,
                "z_rep": z_rep, "z_mu": z_mu, "z_var": z_var}

    def forward(self, batch: Dict) -> Dict:
        H, W = batch["videos"].shape[2:4]
        valid_mask = batch.get("valid_mask", None)
        outputs = self.encode(batch["videos"], valid_mask=valid_mask)

        K = self.num_slots
        z_rep = outputs["z_rep"]
        slot_feats = outputs["slot_feats"]

        # 对象级重建 (从隐动作预测下一帧 slot 特征, 辅助 VAE 防后验坍缩)
        z_actors = z_rep  # (B, T-1, K, latent_dim)
        delta_pred = self.obj_recon_head(z_actors)  # (B, T-1, K, model_dim)
        delta_target = slot_feats[:, 1:].detach()     # 预测下一帧 slot_feats (绝对特征)
        obj_recon_loss = F.mse_loss(delta_pred, delta_target)

        # Decoder
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_embed = self.action_up(z_rep)

        if K == 1:
            video_action_patches = video_patches + action_embed
        else:
            B1, T1, N, D = video_patches.shape
            v_p = video_patches.reshape(B1 * T1, N, D)
            a_e = action_embed.reshape(B1 * T1, K, D)
            fused = self.cross_attn(q=v_p, kv=a_e).reshape(B1, T1, N, D)
            video_action_patches = fused + video_patches

        recon = self.decoder(video_action_patches)
        recon = torch.sigmoid(recon)
        outputs["recon"] = unpatchify(recon, self.patch_size, H, W)
        outputs["obj_recon_loss"] = obj_recon_loss
        return outputs
