"""
改进方向 LAM V3：多主体感知 + 单向量输出

核心改进：
1. 逐帧检测模块（YOLO/LocateAnything）→ 自动生成 mask + 背景槽
2. Mask Pooling（含背景槽）→ (B, T, A+1, D)
3. 对象级时空注意力 → 在特征空间(256d)做时空交互（先交互再差分）
4. Temporal Differencing → 帧间特征差分
5. Mean Pool + 单 VAE → 聚合回单向量 z̃ ∈ R³²（与原始 LAM 接口兼容）
6. Decoder → 单动作 embed Cross-Attention（与原始 LAM 一致）

与 V2（当前代码）的关键区别：
- V2: 差分后交互 + Per-Object VAE + 多向量输出 + 多 slot Decoder
- V3: 差分前交互 + 单 VAE + 单向量输出 + 单 slot Decoder
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lam.modules.blocks import (
    patchify, unpatchify,
    SpatioTemporalTransformer, SpatioTransformer, CrossAttention,
    MaskedPool, ObjectSpatioTemporalAttention,
)
from torch import Tensor


class LatentActionModel(nn.Module):
    """
    多主体感知 + 单向量输出的隐动作模型 (V3)。

    Architecture:
    1. Encoder: SpatioTemporalTransformer 编码 patch 特征（与原始相同）
    2. Detection: 逐帧检测生成 mask（训练时用 GT mask，推理时用 YOLO/LA）
    3. Mask Pooling: 含背景槽，按 mask 池化每主体特征 → (B, T, A+1, D)
    4. Object ST Attention: 对象级时空注意力（特征空间 256d，先交互再差分）
    5. Temporal Differencing: 帧间特征差分
    6. Mean Pool + VAE: 聚合回单向量 z̃ ∈ R³²
    7. Decoder: 单动作 embed Cross-Attention + SpatioTransformer 重建

    Input batch keys:
        "videos":  (B, T, H, W, C)  float32 [0,1]
        "masks":   (B, T, A, H, W)  float32 binary  (A = max_actors)

    Output:
        "recon":          (B, T-1, H, W, C)  重建帧
        "z_mu":           (B*(T-1), latent_dim)  全局 μ
        "z_var":          (B*(T-1), latent_dim)  全局 log-var
        "z_rep":          (B, T-1, latent_dim)   全局隐动作
        "action_logits":  (B, T-1, num_actions)  动作预测 logits
        "action_pred":    (B, T-1)               动作预测 argmax
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
            use_grad_checkpointing: bool = False,
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.max_actors = max_actors
        self.num_actions = num_actions
        grid_size = img_size // patch_size

        patch_token_dim = in_dim * patch_size ** 2  # 3*16*16 = 768

        # === Encoder ===
        # 与原始 AdaWorld LAM 相同，只编码 patch
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
        # 含背景槽：A 个主体 + 1 个背景 = A+1 个对象槽
        self.mask_pool = MaskedPool(patch_size, grid_size, grid_size)

        # === 对象级时空注意力 ===
        # 在特征空间 (model_dim=256) 做时空交互
        if use_obj_st_attention:
            self.obj_st_attention = ObjectSpatioTemporalAttention(
                dim=model_dim,
                num_heads=obj_st_heads,
                num_layers=obj_st_layers,
                dropout=dropout,
            )
        else:
            self.obj_st_attention = None

        # === 单 VAE ===
        # Mean Pool 后的单向量 VAE，与原始 LAM 一致
        self.vae_fc = nn.Linear(model_dim, latent_dim * 2)

        # === Action Prediction Head ===
        self.action_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_actions),
        )

        # === Decoder ===
        # 与原始 AdaWorld LAM 一致：单动作 embed Cross-Attention
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

    def _build_masks_with_background(self, masks: Tensor) -> tuple:
        """
        从主体 mask 构建含背景槽的 mask。

        Args:
            masks: (B, T, A, H, W) — 主体 mask

        Returns:
            all_masks: (B, T, A+1, H, W) — 含背景槽的 mask
            all_valid: (B, T, A+1) — 有效对象指示
        """
        # 背景mask = 1 - 所有主体mask的并集
        bg_mask = 1.0 - masks.sum(dim=2, keepdim=True).clamp(0, 1)
        # (B, T, 1, H, W)

        all_masks = torch.cat([bg_mask, masks], dim=2)  # (B, T, A+1, H, W)

        # valid_mask: 背景始终有效，主体按 mask 是否非空判断
        # MaskedPool 会自动计算 valid_mask，但我们需要在 A+1 维度上
        # 这里先返回 all_masks，让 MaskedPool 计算 valid_mask
        return all_masks

    def encode(self, videos: Tensor, masks: Tensor) -> Dict:
        """
        Encode videos into a single global latent code.

        Args:
            videos: (B, T, H, W, C)  float32 [0,1]
            masks:  (B, T, A, H, W)  float32 binary (主体 mask，不含背景)

        Returns:
            dict with:
                "z_mu":   (B*(T-1), latent_dim)  全局 μ
                "z_var":  (B*(T-1), latent_dim)  全局 log-var
                "z_rep":  (B, T-1, latent_dim)   全局隐动作
                "patches": (B, T, N, patch_token_dim)
                "obj_feats": (B, T, A+1, model_dim)  含背景的对象特征
                "valid_mask": (B, T, A+1)  有效对象指示
        """
        B, T = videos.shape[:2]

        # 1. Patchify & Encode（与原始 LAM 相同）
        patches = patchify(videos, self.patch_size)  # (B, T, N, D_patch)
        encoded = self.encoder(patches)               # (B, T, N, model_dim)

        # 2. 构建含背景槽的 mask
        all_masks = self._build_masks_with_background(masks)  # (B, T, A+1, H, W)

        # 3. Mask Pooling: (B, T, N, D) → (B, T, A+1, D)
        obj_feats, valid_mask = self.mask_pool(encoded, all_masks)
        # obj_feats: (B, T, A+1, model_dim)
        # valid_mask: (B, T, A+1), 背景槽(idx=0)始终 valid=True

        # 4. 对象级时空注意力（特征空间 256d，差分前交互）
        if self.obj_st_attention is not None:
            obj_feats = self.obj_st_attention(obj_feats, valid_mask)
            # (B, T, A+1, model_dim)

        # 5. Temporal Differencing: 帧间特征差分
        delta = obj_feats[:, 1:] - obj_feats[:, :-1]
        # delta: (B, T-1, A+1, model_dim)

        # 6. Mean Pool: 跨对象槽聚合 → 单向量
        delta_global = delta.mean(dim=2)  # (B, T-1, model_dim)

        # 7. 单 VAE 编码
        delta_flat = delta_global.reshape(-1, self.model_dim)  # (B*(T-1), model_dim)
        moments = self.vae_fc(delta_flat)                       # (B*(T-1), latent_dim*2)
        z_mu, z_var = torch.chunk(moments, 2, dim=-1)          # 各 (B*(T-1), latent_dim)
        z_var = torch.clamp(z_var, -5.0, 3.0)

        if self.training:
            z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
        else:
            z_rep = z_mu

        z_rep = z_rep.reshape(B, T - 1, self.latent_dim)  # (B, T-1, latent_dim)

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
            "obj_feats": obj_feats,
            "valid_mask": valid_mask,
        }

    def forward(self, batch: Dict) -> Dict:
        videos = batch["videos"]  # (B, T, H, W, C)
        masks = batch["masks"]    # (B, T, A, H, W)
        H, W = videos.shape[2:4]

        # === Encode ===
        enc_out = self.encode(videos, masks)
        z_rep = enc_out["z_rep"]       # (B, T-1, latent_dim)
        z_mu = enc_out["z_mu"]         # (B*(T-1), latent_dim)
        z_var = enc_out["z_var"]       # (B*(T-1), latent_dim)
        patches = enc_out["patches"]   # (B, T, N, D_patch)
        valid_mask = enc_out["valid_mask"]  # (B, T, A+1)

        # === Action Prediction ===
        action_logits = self.action_head(z_rep)  # (B, T-1, num_actions)

        # === Reconstruction Decoder ===
        # 与原始 LAM 一致：单动作 embed Cross-Attention
        video_patches = self.patch_up(patches[:, :-1])  # (B, T-1, N, D)
        action_embed = self.action_up(z_rep)            # (B, T-1, D)

        B1, T1, N, D = video_patches.shape
        v_p = video_patches.reshape(B1 * T1, N, D)
        a_e = action_embed.reshape(B1 * T1, 1, D)  # 单 slot
        fused = self.cross_attn(q=v_p, kv=a_e)
        fused = fused.reshape(B1, T1, N, D)

        video_action_patches = fused + video_patches  # 残差连接

        recon_patches = self.decoder(video_action_patches)  # (B, T-1, N, D_patch)
        recon = unpatchify(recon_patches, self.patch_size, H, W)
        recon = F.sigmoid(recon)  # (B, T-1, H, W, C)

        # === Output ===
        outputs = {
            "recon": recon,
            "z_mu": z_mu,
            "z_var": z_var,
            "z_rep": z_rep,
            "action_logits": action_logits,
            "action_pred": action_logits.argmax(dim=-1),
            "valid_mask": valid_mask,
        }
        return outputs
