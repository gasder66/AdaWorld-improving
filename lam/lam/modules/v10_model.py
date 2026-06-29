"""
V10: V6c + Shared VAE.

V6c 有像素重建 (CrossAttn decoder, PSNR 27 dB) 但聚类失败 (per-slot VAE, NMI 0.05)。
V10 用单个 shared VAE 替代 per-slot VAE, 使所有 slot 的 z 在同一空间 → 可以跨 slot 聚类。

改动 (相对 V6c):
  - self.fcs (K 个独立 Linear) → self.fc (1 个 shared Linear)
  - encode() 中所有 slot 共享同一个 VAE head
  - 可选 FiLM: actor type → (gamma, beta) 条件化 z
  - decoder, CrossAttention, SpatioTransformer, MaskedPool 全部不变

损失 (同 V6c):
  L_recon (全帧 MSE) + β·L_KL (Free Bits) + λ·L_obj_recon + μ·L_delta + ν·L_contrast
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam.modules.blocks import (
    patchify, unpatchify,
    SpatioTemporalTransformer, SpatioTransformer,
    CrossAttention, ObjectReconHead,
    MaskedPool, ObjectSpatioTemporalAttention,
)


class LatentActionModelV10(nn.Module):
    """V6c + Shared VAE. 所有 slot 共享一个 VAE head."""

    def __init__(
        self,
        in_dim: int = 3,
        model_dim: int = 256,
        latent_dim: int = 32,
        patch_size: int = 16,
        enc_blocks: int = 4,
        dec_blocks: int = 4,
        num_heads: int = 8,
        max_actors: int = 4,
        dropout: float = 0.0,
        keep_background: bool = True,
        use_obj_st_attention: bool = True,
        free_bits_lambda: float = 0.1,
        num_actor_types: int = 0,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.max_actors = max_actors
        self.keep_background = keep_background
        self.use_obj_st_attention = use_obj_st_attention
        self.free_bits_lambda = free_bits_lambda
        self.num_actor_types = num_actor_types

        patch_token_dim = in_dim * patch_size ** 2
        grid_size = 256 // patch_size
        K = max_actors + 1 if keep_background else max_actors
        self.K = K

        # Encoder (same as V6c)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim, model_dim=model_dim, out_dim=model_dim,
            num_blocks=enc_blocks, num_heads=num_heads, dropout=dropout,
        )
        self.mask_pool = MaskedPool(grid_h=grid_size, grid_w=grid_size)

        if use_obj_st_attention:
            self.obj_st_attention = ObjectSpatioTemporalAttention(
                dim=model_dim, num_heads=num_heads, num_layers=2, dropout=dropout,
            )

        # === 核心改动: Shared VAE (替代 V6c 的 per-slot self.fcs) ===
        self.fc_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Linear(model_dim, latent_dim * 2)

        # 可选 FiLM: actor type → (gamma, beta)
        if num_actor_types > 0:
            self.film = nn.Sequential(
                nn.Embedding(num_actor_types + 1, model_dim * 2),
                nn.LayerNorm(model_dim * 2),
            )

        # ObjectReconHead (same as V6c, already shared)
        self.obj_recon_head = ObjectReconHead(latent_dim, model_dim)

        # Decoder (same as V6c, unchanged)
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.cross_attn = CrossAttention(model_dim, num_heads, dropout=dropout)
        self.decoder = SpatioTransformer(
            in_dim=model_dim, model_dim=model_dim, out_dim=patch_token_dim,
            num_blocks=dec_blocks, num_heads=num_heads, dropout=dropout,
        )

        self.var_min = -5.0
        self.var_max = 3.0
        self.mu_record = None

    def _build_masks_with_background(self, masks: Tensor) -> Tensor:
        bg_mask = 1.0 - masks.sum(dim=2, keepdim=True).clamp(0, 1)
        return torch.cat([bg_mask, masks], dim=2)

    def encode(self, videos: Tensor, masks: Tensor, actor_labels: Tensor = None) -> Dict:
        """编码 → z (shared VAE, 所有 slot 共享参数)."""
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)

        # 1. ST Encoder (same as V6c)
        encoded = self.encoder(patches)

        # 2. MaskedPool (same as V6c)
        all_masks = self._build_masks_with_background(masks)
        obj_feats, valid_mask = self.mask_pool(encoded, all_masks)
        valid_mask[:, :, 0] = True

        # 3. Object ST Attention (same as V6c)
        if self.use_obj_st_attention:
            obj_feats = self.obj_st_attention(obj_feats, valid_mask)

        # 4. Shared VAE (核心改动: 不再 per-slot 循环)
        obj_feats_in = obj_feats[:, :-1]  # (B, T-1, K, D)
        K = obj_feats_in.shape[2]
        B_T1_K = B * (T - 1) * K

        h = self.fc_norm(obj_feats_in.reshape(B_T1_K, self.model_dim))
        h = self.fc(h)  # (B*T-1*K, latent_dim*2)

        # 可选 FiLM 条件化
        if self.num_actor_types > 0 and actor_labels is not None:
            # actor_labels: (B, K) → expand to (B, T-1, K) → (B*T-1*K,)
            al = actor_labels.unsqueeze(1).expand(-1, T - 1, -1)  # (B, T-1, K)
            al_flat = al.reshape(B_T1_K).clamp(min=0).long()
            film_params = self.film(al_flat)  # (B*T-1*K, 2*D)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            # FiLM on the pre-split features — need to split first then modulate
            mu_raw, var_raw = torch.chunk(h, 2, dim=-1)
            mu_raw = mu_raw * (1 + gamma[:, :self.latent_dim]) + beta[:, :self.latent_dim]
            var_raw = var_raw * (1 + gamma[:, self.latent_dim:]) + beta[:, self.latent_dim:]
            mu, var = mu_raw, var_raw
        else:
            mu, var = torch.chunk(h, 2, dim=-1)

        var = torch.clamp(var, self.var_min, self.var_max)

        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * var)
        else:
            z = mu

        z_mu = mu.reshape(B, T - 1, K, self.latent_dim)
        z_var = var.reshape(B, T - 1, K, self.latent_dim)
        z_rep = z.reshape(B, T - 1, K, self.latent_dim)

        if not self.training:
            if self.mu_record is None:
                self.mu_record = z_mu.detach().cpu()
            else:
                self.mu_record = torch.cat([self.mu_record, z_mu.detach().cpu()], dim=0)

        return {
            "patches": patches,
            "obj_feats": obj_feats,
            "valid_mask": valid_mask,
            "z_rep": z_rep, "z_mu": z_mu, "z_var": z_var,
        }

    def _free_bits_kl(self, z_mu: Tensor, z_var: Tensor) -> Tensor:
        kl_dim = 0.5 * (z_mu ** 2 + z_var.exp() - z_var - 1)
        kl_dim = kl_dim.clamp(min=self.free_bits_lambda)
        return kl_dim.sum() / z_mu.reshape(-1).shape[0]

    def _delta_consistency_loss(self, z_mu: Tensor) -> Tensor:
        if z_mu.shape[1] < 3:
            return torch.tensor(0.0, device=z_mu.device)
        delta = z_mu[:, 1:] - z_mu[:, :-1]
        if delta.shape[1] < 2:
            return torch.tensor(0.0, device=z_mu.device)
        delta_prev = F.normalize(delta[:, :-1], dim=-1)
        delta_curr = F.normalize(delta[:, 1:], dim=-1)
        sim = (delta_prev * delta_curr).sum(dim=-1).mean()
        return 1.0 - sim

    def _contrastive_loss(self, z_mu: Tensor, temperature: float = 0.1) -> Tensor:
        B, T1, K, D = z_mu.shape
        if T1 <= 1 or K <= 1:
            return torch.tensor(0.0, device=z_mu.device)
        if not z_mu.requires_grad:
            return torch.tensor(0.0, device=z_mu.device)

        total_loss = 0.0
        n = 0
        for b in range(B):
            for t in range(T1 - 1):
                for k in range(K):
                    anchor = z_mu[b, t, k]
                    positive = z_mu[b, t + 1, k]
                    neg_idx = [j for j in range(K) if j != k]
                    negatives = z_mu[b, t, neg_idx]

                    anchor = F.normalize(anchor, dim=-1)
                    positive = F.normalize(positive, dim=-1)
                    negatives = F.normalize(negatives, dim=-1)

                    pos_sim = (anchor * positive).sum(dim=-1)
                    neg_sim = (anchor.unsqueeze(0) * negatives).sum(dim=-1)

                    pos_exp = torch.exp(pos_sim / temperature)
                    neg_exp_sum = torch.exp(neg_sim / temperature).sum()

                    total_loss += -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8))
                    n += 1
        return total_loss / max(n, 1)

    def forward(self, batch: Dict) -> Dict:
        videos = batch["videos"]
        masks = batch["masks"]
        actor_labels = batch.get("actor_labels", None)
        H, W = videos.shape[2:4]

        outputs = self.encode(videos, masks, actor_labels=actor_labels)

        z_rep = outputs["z_rep"]
        z_mu = outputs["z_mu"]
        z_var = outputs["z_var"]
        obj_feats = outputs["obj_feats"]

        # 1. obj_recon_loss (actor slots only)
        if self.keep_background:
            z_actors = z_rep[:, :, 1:]
            obj_target = obj_feats[:, 1:, 1:]
        else:
            z_actors = z_rep
            obj_target = obj_feats[:, 1:]
        delta_pred = self.obj_recon_head(z_actors)
        obj_recon_loss = F.mse_loss(delta_pred, obj_target.detach())

        # 2. Free Bits KL
        kl_loss = self._free_bits_kl(z_mu, z_var)

        # 3. Delta + Contrastive (actor slots only)
        z_actors_mu = z_mu[:, :, 1:] if self.keep_background else z_mu
        delta_loss = self._delta_consistency_loss(z_actors_mu)
        contrast_loss = self._contrastive_loss(z_actors_mu)

        # 4. Decoder (same as V6c, unchanged)
        K = z_rep.shape[2]
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_embed = self.action_up(z_rep)

        B1, T1, N, D = video_patches.shape
        v_p = video_patches.reshape(B1 * T1, N, D)
        a_e = action_embed.reshape(B1 * T1, K, D)
        fused = self.cross_attn(q=v_p, kv=a_e).reshape(B1, T1, N, D)
        video_action_patches = fused + video_patches

        recon = self.decoder(video_action_patches)
        recon = torch.sigmoid(recon)
        outputs["recon"] = unpatchify(recon, self.patch_size, H, W)

        outputs["obj_recon_loss"] = obj_recon_loss
        outputs["kl_loss"] = kl_loss
        outputs["delta_loss"] = delta_loss
        outputs["contrast_loss"] = contrast_loss
        return outputs
