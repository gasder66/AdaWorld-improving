"""
改进方向 LAM V3：多主体感知隐动作模型

支持两种输出模式：
- single_vector: Mean Pool + 单 VAE → 单向量 z̃ ∈ R³²（与原始 LAM 接口兼容）
- multi_vector:  Per-Object VAE → 多向量 (A+1) × z̃ ∈ R³²（每主体独立隐动作）

核心改进（两种模式共享）：
1. 逐帧检测模块（YOLO/LocateAnything）→ 自动生成 mask + 背景槽
2. Mask Pooling（含背景槽）→ (B, T, A+1, D)
3. 对象级时空注意力 → 在特征空间(256d)建模主体间交互和时序动态
4. Per-Object VAE / Mean Pool + VAE → 编码为隐动作

multi_vector 模式额外步骤：
5. 多 slot Decoder（含背景槽）
6. 对象级重建头：从隐动作用于预测下一帧特征，提供无监督学习信号
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lam.modules.blocks import (
    patchify, unpatchify,
    SpatioTemporalTransformer, SpatioTransformer, CrossAttention,
    MaskedPool, ObjectSpatioTemporalAttention, ObjectReconHead,
)
from torch import Tensor


class LatentActionModel(nn.Module):
    """
    多主体感知隐动作模型 (V3)。

    支持单向量/多向量输出模式。

    Input batch keys:
        "videos":  (B, T, H, W, C)  float32 [0,1]
        "masks":   (B, T, A, H, W)  float32 binary  (A = max_actors)

    Output (single_vector):
        "recon":          (B, T-1, H, W, C)
        "z_mu":           (B*(T-1), latent_dim)
        "z_var":          (B*(T-1), latent_dim)
        "z_rep":          (B, T-1, latent_dim)

    Output (multi_vector):
        "recon":          (B, T-1, H, W, C)
        "z_mu":           (B, T-1, A+1, latent_dim)
        "z_var":          (B, T-1, A+1, latent_dim)
        "z_rep":          (B, T-1, A+1, latent_dim)
        "valid_mask":     (B, T, A+1)
        "next_feat_pred": (B, T-1, A, model_dim)   (预测的下一帧每主体特征)
        "obj_recon_loss": scalar                     (对象级重建损失)
    """

    def __init__(
            self,
            in_dim: int = 3,
            model_dim: int = 256,
            latent_dim: int = 32,
            patch_size: int = 16,
            enc_blocks: int = 4,
            dec_blocks: int = 4,
            num_heads: int = 8,
            dropout: float = 0.0,
            max_actors: int = 4,
            num_actions: int = 5,
            img_size: int = 256,
            use_obj_st_attention: bool = True,
            obj_st_heads: int = 8,
            obj_st_layers: int = 2,
            multi_vector: bool = False,
            use_grad_checkpointing: bool = False,
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.max_actors = max_actors
        self.num_actions = num_actions
        self.multi_vector = multi_vector
        grid_size = img_size // patch_size

        patch_token_dim = in_dim * patch_size ** 2  # 3*16*16 = 768

        # === Encoder ===
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            use_grad_checkpointing=use_grad_checkpointing,
        )

        # === Mask Pooling ===
        self.mask_pool = MaskedPool(patch_size, grid_size, grid_size)

        # === 对象级时空注意力 ===
        if use_obj_st_attention:
            self.obj_st_attention = ObjectSpatioTemporalAttention(
                dim=model_dim,
                num_heads=obj_st_heads,
                num_layers=obj_st_layers,
                dropout=dropout,
            )
        else:
            self.obj_st_attention = None

        if multi_vector:
            # === 多向量模式：Per-Object VAE ===
            # A+1 个对象槽（含背景），每个独立 VAE
            self.vae_fc = nn.Linear(model_dim, latent_dim * 2)

            # === 对象级重建头（无监督） ===
            # 从每主体的隐动作重建其在特征空间的帧间差分
            self.obj_recon_head = ObjectReconHead(latent_dim, model_dim)

            # === 多 slot Decoder ===
            self.patch_up = nn.Linear(patch_token_dim, model_dim)
            self.action_up = nn.Linear(latent_dim, model_dim)
            self.cross_attn = CrossAttention(model_dim, num_heads, dropout=dropout)
            self.decoder = SpatioTransformer(
                in_dim=model_dim,
                model_dim=model_dim,
                out_dim=patch_token_dim,
                num_blocks=dec_blocks,
                num_heads=num_heads,
                dropout=dropout,
                use_grad_checkpointing=use_grad_checkpointing,
            )
        else:
            # === 单向量模式：Mean Pool + 单 VAE ===
            self.vae_fc = nn.Linear(model_dim, latent_dim * 2)

            # === 单 slot Decoder ===
            self.patch_up = nn.Linear(patch_token_dim, model_dim)
            self.action_up = nn.Linear(latent_dim, model_dim)
            self.cross_attn = CrossAttention(model_dim, num_heads, dropout=dropout)
            self.decoder = SpatioTransformer(
                in_dim=model_dim,
                model_dim=model_dim,
                out_dim=patch_token_dim,
                num_blocks=dec_blocks,
                num_heads=num_heads,
                dropout=dropout,
                use_grad_checkpointing=use_grad_checkpointing,
            )

        # === Analysis cache ===
        self.mu_record = None

    def _build_masks_with_background(self, masks: Tensor) -> Tensor:
        """从主体 mask 构建含背景槽的 mask。"""
        bg_mask = 1.0 - masks.sum(dim=2, keepdim=True).clamp(0, 1)
        all_masks = torch.cat([bg_mask, masks], dim=2)  # (B, T, A+1, H, W)
        return all_masks

    def encode(self, videos: Tensor, masks: Tensor) -> Dict:
        """
        Encode videos into latent codes.

        Args:
            videos: (B, T, H, W, C)  float32 [0,1]
            masks:  (B, T, A, H, W)  float32 binary (主体 mask，不含背景)

        Returns:
            dict with keys depending on multi_vector mode.
        """
        B, T = videos.shape[:2]
        A = self.max_actors

        # 1. Patchify & Encode
        patches = patchify(videos, self.patch_size)  # (B, T, N, D_patch)
        encoded = self.encoder(patches)               # (B, T, N, model_dim)

        # 2. 构建含背景槽的 mask
        all_masks = self._build_masks_with_background(masks)  # (B, T, A+1, H, W)

        # 3. Mask Pooling: (B, T, N, D) → (B, T, A+1, D)
        obj_feats, valid_mask = self.mask_pool(encoded, all_masks)

        # 4. 对象级时空注意力
        if self.obj_st_attention is not None:
            obj_feats = self.obj_st_attention(obj_feats, valid_mask)

        # 5. VAE 输入：取前 T-1 帧（与 Decoder 的 T-1 对齐）
        obj_feats_in = obj_feats[:, :-1]  # (B, T-1, A+1, model_dim)

        if self.multi_vector:
            # === 多向量模式 ===
            # Per-Object VAE：每个对象槽独立编码
            A1 = A + 1  # 含背景槽
            feat_flat = obj_feats_in.reshape(-1, self.model_dim)  # (B*(T-1)*(A+1), model_dim)
            moments = self.vae_fc(feat_flat)                       # (B*(T-1)*(A+1), latent_dim*2)
            z_mu, z_var = torch.chunk(moments, 2, dim=-1)         # 各 (B*(T-1)*(A+1), latent_dim)
            z_var = torch.clamp(z_var, -5.0, 3.0)

            if self.training:
                z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
            else:
                z_rep = z_mu

            z_mu = z_mu.reshape(B, T - 1, A1, self.latent_dim)
            z_var = z_var.reshape(B, T - 1, A1, self.latent_dim)
            z_rep = z_rep.reshape(B, T - 1, A1, self.latent_dim)

            # Cache for evaluation
            if not self.training:
                if self.mu_record is None:
                    self.mu_record = z_mu.detach().cpu()
                else:
                    self.mu_record = torch.cat(
                        [self.mu_record, z_mu.detach().cpu()], dim=0
                    )

            return {
                "z_mu": z_mu,
                "z_var": z_var,
                "z_rep": z_rep,
                "patches": patches,
                "obj_feats": obj_feats,       # (B, T, A+1, D) 完整特征（用于 obj_recon 目标）
                "obj_feats_in": obj_feats_in,  # (B, T-1, A+1, D) VAE 输入（前 T-1 帧）
                "valid_mask": valid_mask,
            }
        else:
            # === 单向量模式 ===
            feat_global = obj_feats_in.mean(dim=2)  # (B, T-1, model_dim)
            feat_flat = feat_global.reshape(-1, self.model_dim)
            moments = self.vae_fc(feat_flat)
            z_mu, z_var = torch.chunk(moments, 2, dim=-1)
            z_var = torch.clamp(z_var, -5.0, 3.0)

            if self.training:
                z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
            else:
                z_rep = z_mu

            z_rep = z_rep.reshape(B, T - 1, self.latent_dim)

            if not self.training:
                if self.mu_record is None:
                    self.mu_record = z_mu.detach().cpu()
                else:
                    self.mu_record = torch.cat(
                        [self.mu_record, z_mu.detach().cpu()], dim=0
                    )

            return {
                "z_mu": z_mu,
                "z_var": z_var,
                "z_rep": z_rep,
                "patches": patches,
                "obj_feats": obj_feats,
                "valid_mask": valid_mask,
            }

    def forward(self, batch: Dict) -> Dict:
        videos = batch["videos"]  # (B, T, H, W, C)
        masks = batch["masks"]    # (B, T, A, H, W)
        H, W = videos.shape[2:4]

        # === Encode ===
        enc_out = self.encode(videos, masks)
        patches = enc_out["patches"]   # (B, T, N, D_patch)
        valid_mask = enc_out["valid_mask"]  # (B, T, A+1)

        if self.multi_vector:
            return self._forward_multi_vector(enc_out, patches, valid_mask, H, W)
        else:
            return self._forward_single_vector(enc_out, patches, valid_mask, H, W)

    def _forward_single_vector(self, enc_out, patches, valid_mask, H, W):
        """单向量模式的前向传播（无 Action Head）。"""
        z_rep = enc_out["z_rep"]       # (B, T-1, latent_dim)
        z_mu = enc_out["z_mu"]         # (B*(T-1), latent_dim)
        z_var = enc_out["z_var"]       # (B*(T-1), latent_dim)

        # Reconstruction Decoder: 单 slot
        video_patches = self.patch_up(patches[:, :-1])  # (B, T-1, N, D)
        action_embed = self.action_up(z_rep)            # (B, T-1, D)

        B1, T1, N, D = video_patches.shape
        v_p = video_patches.reshape(B1 * T1, N, D)
        a_e = action_embed.reshape(B1 * T1, 1, D)
        fused = self.cross_attn(q=v_p, kv=a_e)
        fused = fused.reshape(B1, T1, N, D)

        video_action_patches = fused + video_patches
        recon_patches = self.decoder(video_action_patches)
        recon = unpatchify(recon_patches, self.patch_size, H, W)
        recon = F.sigmoid(recon)

        return {
            "recon": recon,
            "z_mu": z_mu,
            "z_var": z_var,
            "z_rep": z_rep,
            "valid_mask": valid_mask,
        }

    def _forward_multi_vector(self, enc_out, patches, valid_mask, H, W):
        """多向量模式的前向传播（无监督，无 Action Head）。"""
        z_rep = enc_out["z_rep"]       # (B, T-1, A+1, latent_dim)
        z_mu = enc_out["z_mu"]         # (B, T-1, A+1, latent_dim)
        z_var = enc_out["z_var"]       # (B, T-1, A+1, latent_dim)
        obj_feats = enc_out["obj_feats"]  # (B, T, A+1, model_dim)

        # === 对象级重建（无监督） ===
        # 从当前帧的隐动作预测下一帧的特征（仅对 A 个主体）
        # z_rep 编码了 obj_feats[:, :-1] 的信息
        # obj_recon: 预测 obj_feats[:, 1:, 1:, :]（下一帧的每主体特征）
        B, T1_m, A1, D_lat = z_rep.shape
        z_actors = z_rep[:, :, 1:, :]  # (B, T-1, A, latent_dim)
        next_feat_pred = self.obj_recon_head(z_actors)  # (B, T-1, A, model_dim)
        next_feat_target = obj_feats[:, 1:, 1:, :]       # (B, T-1, A, model_dim)
        obj_recon_loss = F.mse_loss(next_feat_pred, next_feat_target.detach())

        # === Reconstruction Decoder: 多 slot Cross-Attention ===
        video_patches = self.patch_up(patches[:, :-1])  # (B, T-1, N, D)
        action_embed = self.action_up(z_rep)            # (B, T-1, A+1, D)

        B1, T1, N, D = video_patches.shape
        v_p = video_patches.reshape(B1 * T1, N, D)
        a_e = action_embed.reshape(B1 * T1, A1, D)  # A+1 slots
        fused = self.cross_attn(q=v_p, kv=a_e)
        fused = fused.reshape(B1, T1, N, D)

        video_action_patches = fused + video_patches
        recon_patches = self.decoder(video_action_patches)
        recon = unpatchify(recon_patches, self.patch_size, H, W)
        recon = F.sigmoid(recon)

        return {
            "recon": recon,
            "z_mu": z_mu,
            "z_var": z_var,
            "z_rep": z_rep,
            "valid_mask": valid_mask,
            "next_feat_pred": next_feat_pred,
            "obj_recon_loss": obj_recon_loss,
        }
