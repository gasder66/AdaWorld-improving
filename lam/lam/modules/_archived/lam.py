"""
V6: ST Encoder + MaskedPool + Per-Subject VAE + 结构化隐空间约束

相比 V5 的改进:
  1. 背景槽始终保留 (keep_background=True) → 分离自我运动与独立运动
  2. Free Bits KL: 防止后验坍缩，允许部分维度编码更多信息
  3. 互信息最小化: 不同 slot 的 z 之间互信息最小化 → 强制 slot 编码不同主体
  4. 时序一致性: 相邻帧的 z 应平滑过渡 → 同类运动在 z 空间聚集

架构:
  videos → patchify → ST encoder → encoded patch features
  masks → MaskedPool(encoded) → per-subject obj_feats (B,T,K+1,D)  [含背景槽]
  → ObjectSpatioTemporalAttention
  → Per-Object VAE → z_k (B,T-1,K+1,32)  [z_0=背景, z_1..K=主体]
  → Decoder: CrossAttn(patches, z_rep) → recon

损失: L_recon + β·KL_freebits + λ·L_obj_recon + μ·L_mi + τ·L_temporal
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lam.modules.blocks import (
    patchify, unpatchify,
    SpatioTemporalTransformer, SpatioTransformer,
    CrossAttention, ObjectReconHead,
    MaskedPool, ObjectSpatioTemporalAttention,
)
from torch import Tensor


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
            max_actors: int = 4,
            dropout: float = 0.0,
            keep_background: bool = True,
            use_obj_st_attention: bool = True,
            free_bits_lambda: float = 0.5,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.max_actors = max_actors
        self.keep_background = keep_background
        self.use_obj_st_attention = use_obj_st_attention
        self.free_bits_lambda = free_bits_lambda
        patch_token_dim = in_dim * patch_size ** 2
        grid_size = 256 // patch_size

        # K = 主体数 + 背景槽 (始终保留)
        K = max_actors + 1 if keep_background else max_actors

        # 编码器
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim, model_dim=model_dim, out_dim=model_dim,
            num_blocks=enc_blocks, num_heads=num_heads, dropout=dropout,
        )

        # MaskedPool
        self.mask_pool = MaskedPool(grid_h=grid_size, grid_w=grid_size)

        # 对象级时空注意力
        if use_obj_st_attention:
            self.obj_st_attention = ObjectSpatioTemporalAttention(
                dim=model_dim, num_heads=num_heads, num_layers=2, dropout=dropout,
            )

        # Per-Object VAE (K 个独立线性层, 含背景槽)
        self.fcs = nn.ModuleList([
            nn.Linear(model_dim, latent_dim * 2) for _ in range(K)
        ])

        # 对象级重建头 (仅对主体槽, 不含背景)
        num_actor_slots = max_actors  # 不含背景
        self.obj_recon_head = ObjectReconHead(latent_dim, model_dim)

        # Decoder
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.cross_attn = CrossAttention(model_dim, num_heads, dropout=dropout)
        self.decoder = SpatioTransformer(
            in_dim=model_dim, model_dim=model_dim, out_dim=patch_token_dim,
            num_blocks=dec_blocks, num_heads=num_heads, dropout=dropout,
        )

        self.mu_record = None

    def _build_masks_with_background(self, masks: Tensor) -> Tensor:
        bg_mask = 1.0 - masks.sum(dim=2, keepdim=True).clamp(0, 1)
        return torch.cat([bg_mask, masks], dim=2)

    def encode(self, videos: Tensor, masks: Tensor) -> Dict:
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)

        # 1. ST Encoder
        encoded = self.encoder(patches)

        # 2. MaskedPool (含背景槽)
        all_masks = self._build_masks_with_background(masks)
        obj_feats, valid_mask = self.mask_pool(encoded, all_masks)
        valid_mask[:, :, 0] = True  # 背景槽始终有效

        # 3. 对象级时空注意力
        if self.use_obj_st_attention:
            obj_feats = self.obj_st_attention(obj_feats, valid_mask)

        # 4. Per-Object VAE
        obj_feats_in = obj_feats[:, :-1]
        K = obj_feats_in.shape[2]
        B_T1 = B * (T - 1)
        z_mu_list, z_var_list, z_rep_list = [], [], []
        for k in range(K):
            z_k = obj_feats_in[:, :, k].reshape(B_T1, self.model_dim)
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

        return {
            "patches": patches,
            "obj_feats": obj_feats,
            "valid_mask": valid_mask,
            "z_rep": z_rep, "z_mu": z_mu, "z_var": z_var,
        }

    def _free_bits_kl(self, z_mu: Tensor, z_var: Tensor) -> Tensor:
        """Free Bits KL: 每个 dim 的 KL 低于 λ 时不惩罚。

        KL_dim = 0.5 * (mu^2 + exp(var) - var - 1)
        KL_free = sum(max(KL_dim, λ))
        """
        kl_dim = 0.5 * (z_mu ** 2 + z_var.exp() - z_var - 1)  # (B, T-1, K, D)
        kl_dim = kl_dim.clamp(min=self.free_bits_lambda)  # free bits
        return kl_dim.sum() / z_mu.reshape(-1).shape[0]

    def _mutual_info_loss(self, z_mu: Tensor) -> Tensor:
        """互信息最小化: 不同 slot 的 z_mu 之间应正交。

        使用 cosine similarity 矩阵的 Frobenius 范数作为代理:
        MI ≈ -0.5 * log(1 - cos_sim^2) → 最小化 cos_sim^2
        """
        B, T1, K, D = z_mu.shape
        # 对每帧每 batch, 计算 K 个 slot 之间的 cosine similarity
        z_flat = z_mu.reshape(B * T1, K, D)  # (B*T1, K, D)
        z_norm = F.normalize(z_flat, dim=-1)  # 单位向量
        cos_sim = torch.bmm(z_norm, z_norm.transpose(1, 2))  # (B*T1, K, K)
        # 去掉对角线 (自相似)
        eye = torch.eye(K, device=z_mu.device).unsqueeze(0)
        off_diag = (cos_sim ** 2) * (1 - eye)
        # 平均 off-diagonal 元素
        num_pairs = K * (K - 1)
        return off_diag.sum() / (B * T1 * num_pairs + 1e-8)

    def _temporal_consistency_loss(self, z_mu: Tensor) -> Tensor:
        """时序一致性: 相邻帧的 z_mu 应平滑过渡。"""
        if z_mu.shape[1] <= 1:
            return torch.tensor(0.0, device=z_mu.device)
        return F.mse_loss(z_mu[:, 1:], z_mu[:, :-1].detach())

    def _delta_consistency_loss(self, z_mu: Tensor) -> Tensor:
        """动作级分离: 连续相同动作应产生一致的方向变化量 δz。

        约束不是 z 的绝对值, 而是 z 的帧间差分:
            L_delta = 1 - cosine_sim(δz_{t-1}, δz_t)

        当动作持续时 → δz 方向一致 → L_delta 小。
        当动作改变时 → δz 方向变化 → L_delta 可以大 → z 自动分离。
        """
        if z_mu.shape[1] < 3:  # 至少 3 帧 (2 个 δz)
            return torch.tensor(0.0, device=z_mu.device)
        delta = z_mu[:, 1:] - z_mu[:, :-1]  # (B, T1-1, K, D)
        if delta.shape[1] < 2:
            return torch.tensor(0.0, device=z_mu.device)
        delta_prev = F.normalize(delta[:, :-1], dim=-1)  # (B, T1-2, K, D)
        delta_curr = F.normalize(delta[:, 1:], dim=-1)
        sim = (delta_prev * delta_curr).sum(dim=-1).mean()
        return 1.0 - sim

    def _contrastive_loss(self, z_mu: Tensor, temperature: float = 0.1) -> Tensor:
        """对比学习损失: 同一主体跨帧的 z 在空间中应接近 (正例),
        不同主体在同一帧的 z 应远离 (负例).

        使用 InfoNCE 形式:
            L_contrast = -log( exp(sim(z_anchor, z_positive)/τ) /
                              Σ exp(sim(z_anchor, z_k)/τ) )
        其中 positive = 同一 slot 的相邻帧, 负例 = 其他所有 slot
        """
        B, T1, K, D = z_mu.shape
        if T1 <= 1 or K <= 1:
            return torch.tensor(0.0, device=z_mu.device)
        if not z_mu.requires_grad:
            return torch.tensor(0.0, device=z_mu.device)

        device = z_mu.device
        total_loss = 0.0
        n = 0

        for b in range(B):
            for t in range(T1 - 1):
                for k in range(K):
                    # anchor: z of slot k at time t
                    anchor = z_mu[b, t, k]  # (D,)
                    # positive: z of same slot k at time t+1
                    positive = z_mu[b, t + 1, k]  # (D,)

                    # negatives: z of ALL other slots at time t
                    # (exclude slot k itself, include other slots)
                    neg_idx = [j for j in range(K) if j != k]
                    negatives = z_mu[b, t, neg_idx]  # (K-1, D)

                    # cosine similarity
                    anchor = F.normalize(anchor, dim=-1)
                    positive = F.normalize(positive, dim=-1)
                    negatives = F.normalize(negatives, dim=-1)

                    pos_sim = (anchor * positive).sum(dim=-1)  # scalar
                    neg_sim = (anchor.unsqueeze(0) * negatives).sum(dim=-1)  # (K-1,)

                    pos_exp = torch.exp(pos_sim / temperature)
                    neg_exp_sum = torch.exp(neg_sim / temperature).sum()

                    loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8))
                    total_loss += loss
                    n += 1

        return total_loss / max(n, 1)

    def forward(self, batch: Dict) -> Dict:
        videos = batch["videos"]
        masks = batch["masks"]
        H, W = videos.shape[2:4]
        outputs = self.encode(videos, masks)

        z_rep = outputs["z_rep"]
        z_mu = outputs["z_mu"]
        z_var = outputs["z_var"]
        obj_feats = outputs["obj_feats"]

        # === 损失计算 ===

        # 1. 对象级重建 (仅主体槽, 跳过背景槽 index=0)
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

        # 3. 主体分离 (z_actors = 仅主体槽)
        if self.keep_background:
            z_actors_mu = z_mu[:, :, 1:]
        else:
            z_actors_mu = z_mu

        # 4. Delta 一致性 (动作级分离: 连续同动作时 δz 方向应一致)
        delta_loss = self._delta_consistency_loss(z_actors_mu)

        # 5. 对比学习损失 (保留作为辅助)
        contrast_loss = self._contrastive_loss(z_actors_mu)

        # === Decoder ===
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