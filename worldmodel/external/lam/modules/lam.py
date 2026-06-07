from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from external.lam.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, CrossAttention
from torch import Tensor


class LatentActionModel(nn.Module):
    """
    Latent action VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            num_slots: int = 1,
            use_slot_competition: bool = False,
            use_grad_checkpointing: bool = False,
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_slots = num_slots
        patch_token_dim = in_dim * patch_size ** 2

        # K 个可学习 object slot token（当 num_slots=1 时退化为原始 action_prompt）
        self.object_slots = nn.Parameter(torch.empty(1, 1, num_slots, patch_token_dim))
        nn.init.uniform_(self.object_slots, a=-1, b=1)

        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            use_slot_competition=use_slot_competition,
            use_grad_checkpointing=use_grad_checkpointing,
        )

        # Per-slot VAE: 每个 slot 有独立的 fc 层
        self.fc = nn.ModuleList(
            [nn.Linear(model_dim, latent_dim * 2) for _ in range(num_slots)]
        )

        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)

        # Cross-attention 用于解码器的一次前融合（当 num_slots > 1 时使用）
        self.cross_attn = CrossAttention(model_dim, num_heads, dropout=dropout) \
            if num_slots > 1 else None

        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
            use_grad_checkpointing=use_grad_checkpointing,
        )

    def encode(self, videos: Tensor) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        target_patches = None
        if T == 2:
            patches = patchify(videos, self.patch_size)
        elif T == 3:
            T -= 1
            patches = patchify(videos[:, :2], self.patch_size)
            target_patches = patchify(videos[:, 2:], self.patch_size)
        else:
            raise ValueError

        # 拼接 K 个 object slot token
        slot_pad = self.object_slots.expand(B, T, -1, -1)  # (B, T, K, D)
        padded_patches = torch.cat([slot_pad, patches], dim=2)  # (B, T, K+N, D)

        # Encode
        z = self.encoder(padded_patches, num_slots=self.num_slots)  # (B, T, K+N, E)
        # Get latent actions for all future frames: 取 K 个 slot 的输出
        z = z[:, 1:, :self.num_slots]  # (B, T-1, K, E)

        # Per-slot VAE
        z_mu_list, z_var_list, z_rep_list = [], [], []
        for k in range(self.num_slots):
            z_k = z[:, :, k, :].reshape(-1, self.model_dim)  # (B*(T-1), model_dim)
            moments = self.fc[k](z_k)
            z_mu_k, z_var_k = torch.chunk(moments, 2, dim=1)
            z_var_k = torch.clamp(z_var_k, -5.0, 3.0)
            # Reparameterization
            if not self.training:
                z_rep_k = z_mu_k
            else:
                z_rep_k = z_mu_k + torch.randn_like(z_var_k) * torch.exp(0.5 * z_var_k)
            z_mu_list.append(z_mu_k)
            z_var_list.append(z_var_k)
            z_rep_list.append(z_rep_k)

        z_mu = torch.stack(z_mu_list, dim=1)    # (B*(T-1), K, latent_dim)
        z_var = torch.stack(z_var_list, dim=1)  # (B*(T-1), K, latent_dim)
        z_rep = torch.stack(z_rep_list, dim=1)   # (B*(T-1), K, latent_dim)
        z_rep = z_rep.reshape(B, T - 1, self.num_slots, self.latent_dim)

        return {
            "patches": patches,
            "target_patches": target_patches,
            "z_rep": z_rep,
            "z_mu": z_mu,
            "z_var": z_var
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VAE
        H, W = batch["videos"].shape[2:4]
        outputs = self.encode(batch["videos"])
        if outputs["target_patches"] is None:
            video_patches = self.patch_up(outputs["patches"][:, :-1])
        else:
            video_patches = self.patch_up(outputs["target_patches"])
        action_slots = self.action_up(outputs["z_rep"])             # (B, T-1, K, model_dim)

        # 解码融合: Cross-Attention 一次前融合 or 简单相加
        if self.num_slots > 1 and self.cross_attn is not None:
            # (B, T-1, N, model_dim) + (B, T-1, K, model_dim) → cross-attn per time step
            B, T1, N, D = video_patches.shape
            K = action_slots.shape[2]
            # Flatten time dimension for cross-attention
            v_p = video_patches.reshape(B * T1, N, D)
            a_s = action_slots.reshape(B * T1, K, D)
            fused = self.cross_attn(q=v_p, kv=a_s)  # (B*T1, N, D)
            fused = fused.reshape(B, T1, N, D)
            video_action_patches = fused + video_patches  # 残差连接
        else:
            # num_slots=1 时退化为原始简单相加
            video_action_patches = video_patches + action_slots

        del outputs["patches"]
        del outputs["target_patches"]

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = F.sigmoid(video_recon)
        outputs.update(
            {
                "recon": unpatchify(video_recon, self.patch_size, H, W)
            }
        )
        return outputs
