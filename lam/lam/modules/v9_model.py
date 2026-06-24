"""
V9 Latent Action Model: V8 + 重建 decoder.

目标: 在保持 V8 高 NMI 聚类的同时, 恢复像素重建能力。

变体:
  V9-A: V8 (motion-only encoder) + recon decoder, recon loss 流入 z_actor
        → 测试: recon loss 能否让 z_actor 携带外观信息? (预期: 否, encoder 瓶颈)
  V9-B: V8 + RGB crops (所有 t) + recon decoder
        → 测试: 给 z_actor 外观信息后, NMI vs PSNR 权衡
  V9-C: 双路径 (z_actor motion + z_appearance appearance) + recon decoder
        → 测试: 分离路径能否同时达到高 NMI 和高 PSNR

架构 (V9-A):
  V8 encode: video → MotionTokenEncoder → z_actor (B,T-1,K,16), z_bg (B,T-1,16)
  V9 recon: crop(I_t, bbox_t) + z_actor + z_bg → ReconDecoder → crop(I_{t+1}, bbox_{t+1})

损失:
  L = L_motion + β·L_KL + γ·L_bg + δ·L_recon
  L_recon = L1(recon, target) + (1 - SSIM(recon, target))
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam.modules.slot_time_lam import LatentActionModelV8
from lam.modules.motion_token_encoder import MotionTokenEncoder, _crop_resize
from lam.modules.v9_decoder import ReconDecoder, compute_psnr, compute_ssim_simple


class RGBMotionTokenEncoder(nn.Module):
    """V9-B encoder: 所有 t 用 RGB crop (不用帧差), 让 z_actor 可访问外观.

    与 V8 MotionTokenEncoder 的区别:
      - t>0 用 RGB crop(I_t, bbox_t) 而非帧差 crop(ΔI_t, union_bbox)
      - 其余结构 (geom enc, bg enc, fuse) 与 V8 相同

    风险: RGB crop 含外观 → 可能导致 actor leakage (V6c 问题)
    """

    def __init__(self, model_dim: int = 256, crop_size: int = 32, img_size: int = 256) -> None:
        super().__init__()
        # 复用 V8 的子模块结构
        from lam.modules.motion_token_encoder import (
            ActorCropEncoder, BoxGeometryEncoder, BackgroundTokenEncoder,
        )
        self.actor_crop_enc = ActorCropEncoder(in_channels=3, model_dim=model_dim, crop_size=crop_size)
        self.geom_enc = BoxGeometryEncoder(model_dim=model_dim, img_size=img_size)
        self.bg_enc = BackgroundTokenEncoder(model_dim=model_dim, bg_size=crop_size)
        self.fuse_norm = nn.LayerNorm(model_dim * 2)
        self.fuse_proj = nn.Linear(model_dim * 2, model_dim)

    def forward(self, video: Tensor, boxes: Tensor, valid_mask: Tensor) -> Dict[str, Tensor]:
        """video: (B,T,H,W,3), boxes: (B,T,K,4), valid_mask: (B,T,K)."""
        B, T, K, _ = boxes.shape
        H, W = video.shape[2], video.shape[3]
        cs = self.actor_crop_enc.net[0].out_channels  # not used, just for shape

        # Actor crops: ALL timesteps use RGB (not frame-diff)
        crops = _crop_resize(video, boxes, crop_size=self.actor_crop_enc.crop_size if hasattr(self.actor_crop_enc, 'crop_size') else 32)
        # crops: (B, T, K, 3, cs, cs)
        crop_cs = crops.shape[-1]
        crop_tokens = self.actor_crop_enc(
            crops.reshape(B * T * K, 3, crop_cs, crop_cs)
        ).reshape(B, T, K, -1)

        # Geometry
        geom_tokens = self.geom_enc(boxes)  # (B, T, K, D)

        # Fuse
        actor_tokens = self.fuse_proj(
            self.fuse_norm(torch.cat([crop_tokens, geom_tokens], dim=-1))
        )

        # Background (same as V8: frame-diff for bg is OK, bg doesn't need appearance)
        bg_tokens = self.bg_enc(video)  # (B, T, D)

        motion_tokens = torch.cat([
            bg_tokens.unsqueeze(2),   # (B, T, 1, D)
            actor_tokens,             # (B, T, K, D)
        ], dim=2)

        return {"motion_tokens": motion_tokens, "actor_tokens": actor_tokens, "bg_tokens": bg_tokens}


class LatentActionModelV9(nn.Module):
    """V9: V8 + 重建 decoder.

    variant:
      'A': V8 encoder (motion-only) + recon decoder, z_actor gets recon gradient
      'B': RGB encoder (all-t RGB crops) + recon decoder
      'C': dual-pathway (z_actor motion + z_appearance appearance)
    """

    def __init__(
        self,
        model_dim: int = 256,
        z_dim: int = 16,
        z_bg_dim: int = 16,
        num_temporal_layers: int = 2,
        num_slot_layers: int = 1,
        num_heads: int = 4,
        max_actors: int = 4,
        crop_size: int = 32,
        img_size: int = 256,
        free_bits_lambda: float = 0.5,
        bbox_scale: float = 32.0,
        use_bg_slot: bool = True,
        bg_loss_weight: float = 1.0,
        camera_param_scale=(8.0, 8.0, 0.1, 0.05),
        num_actor_types: int = 0,
        dropout: float = 0.0,
        variant: str = "A",
        z_app_dim: int = 16,
        detach_z_actor_recon: bool = False,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.crop_size = crop_size
        self.z_dim = z_dim
        self.z_bg_dim = z_bg_dim
        self.z_app_dim = z_app_dim
        self.detach_z_actor_recon = detach_z_actor_recon

        # V8 backbone
        self.v8 = LatentActionModelV8(
            model_dim=model_dim, z_dim=z_dim, z_bg_dim=z_bg_dim,
            num_temporal_layers=num_temporal_layers, num_slot_layers=num_slot_layers,
            num_heads=num_heads, max_actors=max_actors, crop_size=crop_size,
            img_size=img_size, free_bits_lambda=free_bits_lambda,
            bbox_scale=bbox_scale, use_bg_slot=use_bg_slot,
            bg_loss_weight=bg_loss_weight, camera_param_scale=camera_param_scale,
            num_actor_types=num_actor_types, dropout=dropout,
        )

        # V9-B: replace motion encoder with RGB version
        if variant == "B":
            self.v8.motion_encoder = RGBMotionTokenEncoder(
                model_dim=model_dim, crop_size=crop_size, img_size=img_size,
            )

        # V9-C: dual-pathway — add appearance encoder + VAE
        if variant == "C":
            from lam.modules.motion_token_encoder import ActorCropEncoder
            self.app_encoder = ActorCropEncoder(
                in_channels=3, model_dim=model_dim, crop_size=crop_size,
            )
            self.app_vae = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, model_dim),
                nn.GELU(),
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, z_app_dim * 2),
            )
            self.var_min = -5.0
            self.var_max = 3.0
            # decoder uses z_actor(detached) + z_app + z_bg
            z_dec_actor = z_dim
            z_dec_bg = z_bg_dim if use_bg_slot else 0
            self.recon_decoder = ReconDecoder(
                z_actor_dim=z_dec_actor + z_app_dim,
                z_bg_dim=z_dec_bg,
                crop_size=crop_size,
            )
        else:
            # V9-A/B: decoder uses z_actor + z_bg
            z_dec_bg = z_bg_dim if use_bg_slot else 0
            self.recon_decoder = ReconDecoder(
                z_actor_dim=z_dim,
                z_bg_dim=z_dec_bg,
                crop_size=crop_size,
            )

    def _free_bits_kl_app(self, mu: Tensor, logvar: Tensor, lambda_: float) -> Tensor:
        kl_dim = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1)
        kl_dim = kl_dim.clamp(min=lambda_)
        return kl_dim.sum() / mu.reshape(-1).shape[0]

    def _encode_appearance(self, video: Tensor, boxes: Tensor, valid_mask: Tensor) -> Dict[str, Tensor]:
        """V9-C: encode z_appearance from RGB crops."""
        B, T, K, _ = boxes.shape
        # crop at t (for transition t→t+1, we use crop at t)
        crops_t = _crop_resize(video[:, :-1], boxes[:, :-1], crop_size=self.crop_size)
        # (B, T-1, K, 3, cs, cs)
        cs = crops_t.shape[-1]
        crops_flat = crops_t.reshape(B * (T - 1) * K, 3, cs, cs)
        feats = self.app_encoder(crops_flat)  # (B*(T-1)*K, D)
        moments = self.app_vae(feats)
        mu, logvar = torch.chunk(moments, 2, dim=-1)
        logvar = torch.clamp(logvar, self.var_min, self.var_max)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        D = mu.shape[-1]
        return {
            "z_app": z.reshape(B, T - 1, K, D),
            "mu_app": mu.reshape(B, T - 1, K, D),
            "logvar_app": logvar.reshape(B, T - 1, K, D),
        }

    def forward(self, batch: Dict) -> Dict[str, Tensor]:
        """前向传播: V8 losses + V9 recon loss."""
        video = batch["videos"]
        boxes = batch["boxes"]
        valid_mask = batch["valid_mask"]

        # V8 forward (motion + KL + bg losses)
        out = self.v8(batch)

        z_actor = out["z_actor"]      # (B, T-1, K, z_dim)
        z_bg = out.get("z_bg")        # (B, T-1, z_bg_dim) or None

        B, T1, K, D = z_actor.shape
        C = 3

        # Get crops: crop_t and crop_tp1 (target)
        crop_t = _crop_resize(video[:, :-1], boxes[:, :-1], crop_size=self.crop_size)
        # (B, T-1, K, 3, cs, cs)
        crop_tp1 = _crop_resize(video[:, 1:], boxes[:, 1:], crop_size=self.crop_size)

        # Flatten for decoder
        crop_t_flat = crop_t.reshape(B * T1 * K, C, self.crop_size, self.crop_size)
        crop_tp1_flat = crop_tp1.reshape(B * T1 * K, C, self.crop_size, self.crop_size)

        if self.variant == "C":
            # Dual-pathway: z_actor (detached) + z_app + z_bg
            app_out = self._encode_appearance(video, boxes, valid_mask)
            z_app = app_out["z_app"]  # (B, T-1, K, z_app_dim)
            out["z_app"] = z_app
            out["mu_app"] = app_out["mu_app"]
            out["logvar_app"] = app_out["logvar_app"]

            z_actor_dec = z_actor.detach() if self.detach_z_actor_recon else z_actor
            z_dec = torch.cat([z_actor_dec, z_app], dim=-1)  # (B, T-1, K, z_dim + z_app_dim)
            z_dec_flat = z_dec.reshape(B * T1 * K, -1)

            if z_bg is not None:
                z_bg_flat = z_bg.unsqueeze(2).expand(-1, -1, K, -1).reshape(B * T1 * K, -1)
            else:
                z_bg_flat = torch.zeros(B * T1 * K, 0, device=z_actor.device)

            recon_flat = self.recon_decoder(crop_t_flat, z_dec_flat, z_bg_flat)

            # KL on z_app
            kl_app = self._free_bits_kl_app(
                app_out["mu_app"], app_out["logvar_app"], self.v8.free_bits_lambda,
            )
            out["kl_app"] = kl_app
        else:
            # V9-A/B: z_actor + z_bg
            z_actor_flat = z_actor.reshape(B * T1 * K, -1)
            if z_bg is not None:
                z_bg_flat = z_bg.unsqueeze(2).expand(-1, -1, K, -1).reshape(B * T1 * K, -1)
            else:
                z_bg_flat = torch.zeros(B * T1 * K, 0, device=z_actor.device)

            recon_flat = self.recon_decoder(crop_t_flat, z_actor_flat, z_bg_flat)

        recon = recon_flat.reshape(B, T1, K, C, self.crop_size, self.crop_size)
        out["recon"] = recon
        out["crop_t"] = crop_t
        out["crop_tp1"] = crop_tp1

        # Recon loss: L1 + (1 - SSIM), masked by valid actors
        valid = valid_mask[:, 1:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        # (B, T1, K, 1, 1, 1) — broadcast to (B, T1, K, C, H, W)
        valid_exp = valid.expand_as(recon)
        l1 = (torch.abs(recon - crop_tp1) * valid_exp).sum() / (valid_exp.sum() + 1e-8)

        # SSIM per-sample then mask
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu_p = recon.mean(dim=[4, 5])  # (B, T1, K, C)
        mu_t = crop_tp1.mean(dim=[4, 5])
        var_p = recon.var(dim=[4, 5])
        var_t = crop_tp1.var(dim=[4, 5])
        r_flat = recon.reshape(B, T1, K, C, -1)
        t_flat = crop_tp1.reshape(B, T1, K, C, -1)
        cov = ((r_flat - mu_p.unsqueeze(-1)) * (t_flat - mu_t.unsqueeze(-1))).mean(dim=-1)
        ssim_map = ((2 * mu_p * mu_t + C1) * (2 * cov + C2)) / \
                   ((mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2))
        ssim_per = ssim_map.mean(dim=-1)  # (B, T1, K)
        ssim_valid = valid_mask[:, 1:].float()
        ssim_mean = (ssim_per * ssim_valid).sum() / (ssim_valid.sum() + 1e-8)
        ssim_loss = 1.0 - ssim_mean

        recon_loss = l1 + ssim_loss
        out["recon_loss"] = recon_loss
        out["recon_l1"] = l1
        out["recon_ssim_loss"] = ssim_loss
        out["recon_ssim"] = ssim_mean

        return out
