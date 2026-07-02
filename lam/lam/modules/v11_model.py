"""
V11: V10 + Frame-Diff (iVideoGPT-style).

V10 重建下一帧 RGB → z 混合了外观和动作。
V11 重建下一帧 ΔI (帧差) → z 只编码"变化"=action, 与 appearance 解耦。

核心改动 (相对 V10):
  - Encoder 输入: [I_0, ΔI_1, ..., ΔI_{T-1}] (第一帧原始 RGB, 后续帧差)
  - Decoder: 去掉 sigmoid (帧差可负)
  - 重建目标: ΔI_{t+1} (非 RGB)
  - 去掉 L_contrast, L_obj_recon
  - 保留 L_delta (时序一致性), L_KL (Free Bits)

损失:
  L_recon (ΔI MSE) + β·L_KL (Free Bits) + μ·L_delta
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam.modules.blocks import (
    patchify, unpatchify,
    SpatioTemporalTransformer, SpatioTransformer,
    CrossAttention,
    MaskedPool, ObjectSpatioTemporalAttention,
)


class LatentActionModelV11(nn.Module):
    """V10 + Frame-Diff. 使用帧差重建迫使 z 只编码 action."""

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

        # Encoder (same as V10)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim, model_dim=model_dim, out_dim=model_dim,
            num_blocks=enc_blocks, num_heads=num_heads, dropout=dropout,
        )
        self.mask_pool = MaskedPool(grid_h=grid_size, grid_w=grid_size)

        if use_obj_st_attention:
            self.obj_st_attention = ObjectSpatioTemporalAttention(
                dim=model_dim, num_heads=num_heads, num_layers=2, dropout=dropout,
            )

        # Shared VAE (same as V10)
        self.fc_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Linear(model_dim, latent_dim * 2)

        # 可选 FiLM
        if num_actor_types > 0:
            self.film = nn.Sequential(
                nn.Embedding(num_actor_types + 1, model_dim * 2),
                nn.LayerNorm(model_dim * 2),
            )

        # Decoder (same as V10, but output goes through NO sigmoid)
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
        """编码 [I_0, ΔI_1, ...] → z (shared VAE)."""
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)

        encoded = self.encoder(patches)

        all_masks = self._build_masks_with_background(masks)
        obj_feats, valid_mask = self.mask_pool(encoded, all_masks)
        valid_mask[:, :, 0] = True

        if self.use_obj_st_attention:
            obj_feats = self.obj_st_attention(obj_feats, valid_mask)

        obj_feats_in = obj_feats[:, :-1]
        K = obj_feats_in.shape[2]
        B_T1_K = B * (T - 1) * K

        h = self.fc_norm(obj_feats_in.reshape(B_T1_K, self.model_dim))
        h = self.fc(h)

        if self.num_actor_types > 0 and actor_labels is not None:
            al = actor_labels.unsqueeze(1).expand(-1, T - 1, -1)
            al_flat = al.reshape(B_T1_K).clamp(min=0).long()
            film_params = self.film(al_flat)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
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

    def forward(self, batch: Dict) -> Dict:
        videos = batch["videos"]                          # (B, T, H, W, C) raw RGB
        masks = batch["masks"]                             # (B, T, A, H, W)
        actor_labels = batch.get("actor_labels", None)
        H, W = videos.shape[2:4]

        # === 核心改动: 构造 frame-diff 混合输入 ===
        # model_input = [I_0, ΔI_1, ..., ΔI_{T-1}]  where ΔI_t = I_t - I_{t-1}
        diff = videos[:, 1:] - videos[:, :-1]              # (B, T-1, H, W, C)
        model_input = torch.cat([videos[:, :1], diff], dim=1)  # (B, T, H, W, C)

        outputs = self.encode(model_input, masks, actor_labels=actor_labels)

        z_rep = outputs["z_rep"]
        z_mu = outputs["z_mu"]
        z_var = outputs["z_var"]

        # 1. Free Bits KL (same as V10)
        kl_loss = self._free_bits_kl(z_mu, z_var)

        # 2. Delta consistency (actor slots only, same as V10)
        z_actors_mu = z_mu[:, :, 1:] if self.keep_background else z_mu
        delta_loss = self._delta_consistency_loss(z_actors_mu)

        # 3. Decoder: 重建帧差 ΔI (去掉 sigmoid!)
        K = z_rep.shape[2]
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_embed = self.action_up(z_rep)

        B1, T1, N, D = video_patches.shape
        v_p = video_patches.reshape(B1 * T1, N, D)
        a_e = action_embed.reshape(B1 * T1, K, D)
        fused = self.cross_attn(q=v_p, kv=a_e).reshape(B1, T1, N, D)
        video_action_patches = fused + video_patches

        recon = self.decoder(video_action_patches)
        # No sigmoid — frame-diff can be negative

        outputs["recon"] = unpatchify(recon, self.patch_size, H, W)
        outputs["diff"] = diff                                  # (B, T-1, H, W, C) GT
        outputs["raw_videos"] = videos                          # save for RGB PSNR recovery

        outputs["kl_loss"] = kl_loss
        outputs["delta_loss"] = delta_loss
        return outputs
