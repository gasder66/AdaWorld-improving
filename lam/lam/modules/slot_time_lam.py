"""
V8 Slot-Time Latent Action Model.

核心架构:
    motion_tokens (B,T,K+1,D)
        ↓ TransitionTokenBuilder: r_{t,k} = MLP([h_t, h_{t+1}, h_{t+1}-h_t])
        ↓ TemporalTransformer: 每个 slot 沿时间做 attention (2 layers)
        ↓ SlotTransformer: 同一时间步跨 slot attention (1 layer only)
        ↓ SharedActorActionHead: 所有 actor 共享 action latent space (无 actor 条件, Stage 1)
        ↓ BackgroundMotionHead: bg slot -> z_bg
    z_actor (B,T-1,K,d_z), z_bg (B,T-1,d_bg)

损失:
    L_motion: Δbbox_pred = CameraMotion(z_bg) + ActorMotion(z_actor) → MSE vs Δbbox_obs
    L_KL: Free Bits KL (z_actor + z_bg)

关键设计 (vs V6c/V7):
    - 不用 SharedEncoder (V7 失败原因): motion token 不含外观, 无 actor 高速通道
    - 所有 actor 共享 action head (vs V6c per-slot 独立 fc)
    - z_actor 从帧差+几何学习, 不从像素外观学习
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam.modules.blocks import SelfAttention, PositionalEncoding
from lam.modules.motion_token_encoder import MotionTokenEncoder


class TemporalTransformerBlock(nn.Module):
    """单层时序 Transformer: 对每个 slot 独立沿时间做 self-attention."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = SelfAttention(model_dim, num_heads, dropout=dropout, rot_emb=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, K+1, T-1, D) — 注意时序在 dim=2."""
        B, K, T, D = x.shape
        # 把 K 维并入 batch: (B*K, T, D)
        x_flat = x.reshape(B * K, T, D)
        x_ = self.norm1(x_flat)
        x_ = self.attn(x_, is_causal=False)
        x_flat = x_flat + x_
        x_ = self.norm2(x_flat)
        x_ = self.ffn(x_)
        x_flat = x_flat + x_
        return x_flat.reshape(B, K, T, D)


class SlotTransformerBlock(nn.Module):
    """单层 Slot Transformer: 同一时间步跨 slot 做 self-attention.

    注意: V6c blocks.py 的 SelfAttention.key_padding_mask 有 bug
    (False * -inf = NaN), 这里自行实现 masked attention。
    """

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (model_dim // num_heads) ** -0.5
        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """x: (B, T-1, K+1, D), key_padding_mask: (B, T-1, K+1) True=valid."""
        B, T, K, D = x.shape
        x_flat = x.reshape(B * T, K, D)
        x_ = self.norm1(x_flat)

        q = self.to_q(x_).reshape(B * T, K, self.num_heads, D // self.num_heads).transpose(1, 2)
        k = self.to_k(x_).reshape(B * T, K, self.num_heads, D // self.num_heads).transpose(1, 2)
        v = self.to_v(x_).reshape(B * T, K, self.num_heads, D // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*T, H, K, K)

        if key_padding_mask is not None:
            # key_padding_mask: (B, T, K) True=valid -> (B*T, 1, 1, K) True=mask out
            mask = (~key_padding_mask.reshape(B * T, K)).unsqueeze(1).unsqueeze(2)  # True=mask out
            attn = attn.masked_fill(mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * T, K, D)
        out = self.to_out(out)

        x_flat = x_flat + out
        x_ = self.norm2(x_flat)
        x_ = self.ffn(x_)
        x_flat = x_flat + x_
        return x_flat.reshape(B, T, K, D)


class TransitionTokenBuilder(nn.Module):
    """构造 transition token: r_{t,k} = MLP([h_{t,k}, h_{t+1,k}, h_{t+1,k}-h_{t,k}])."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim * 3),
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, motion_tokens: Tensor) -> Tensor:
        """motion_tokens: (B, T, K+1, D) -> transition_tokens: (B, T-1, K+1, D)."""
        h_t = motion_tokens[:, :-1]       # (B, T-1, K+1, D)
        h_t1 = motion_tokens[:, 1:]       # (B, T-1, K+1, D)
        delta = h_t1 - h_t                # (B, T-1, K+1, D)
        r = torch.cat([h_t, h_t1, delta], dim=-1)  # (B, T-1, K+1, 3D)
        return self.net(r)


class SharedActorActionHead(nn.Module):
    """共享 actor action head: 所有 actor slot 共享一个 action latent space.

    Stage 2C: 可选 actor label FiLM 条件化。
    actor label 是条件 (不是被编码进 z), 让 z_actor 编码
    "相对于该 actor 类型的残差 action"。
    """

    def __init__(
        self,
        model_dim: int,
        z_dim: int,
        num_actor_types: int = 0,
        var_min: float = -5.0,
        var_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.var_min = var_min
        self.var_max = var_max
        self.num_actor_types = num_actor_types
        self.use_conditioning = num_actor_types > 0

        self.norm1 = nn.LayerNorm(model_dim)
        self.fc1 = nn.Linear(model_dim, model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.fc2 = nn.Linear(model_dim, z_dim * 2)

        if self.use_conditioning:
            # FiLM: actor_label -> (gamma, beta) to modulate fc1 output
            self.film = nn.Sequential(
                nn.Embedding(num_actor_types + 1, model_dim * 2),  # +1 for unknown (-1→0)
                nn.LayerNorm(model_dim * 2),
            )

    def forward(self, x: Tensor, actor_labels: Tensor = None) -> Dict[str, Tensor]:
        """x: (..., D), actor_labels: (...,) long or None -> {mu, logvar, z}."""
        h = self.norm1(x)
        h = self.fc1(h)
        h = F.gelu(h)

        if self.use_conditioning and actor_labels is not None:
            # actor_labels: (...,) with -1 for invalid → clamp to 0
            labels = actor_labels.clamp(min=0).long()
            film_params = self.film(labels)  # (..., 2*D)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)  # each (..., D)
            h = h * (1 + gamma) + beta  # FiLM modulation

        h = self.norm2(h)
        moments = self.fc2(h)
        mu, logvar = torch.chunk(moments, 2, dim=-1)
        logvar = torch.clamp(logvar, self.var_min, self.var_max)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return {"mu": mu, "logvar": logvar, "z": z}


class BackgroundMotionHead(nn.Module):
    """背景运动 head: bg slot -> z_bg (mu, logvar)."""

    def __init__(self, model_dim: int, z_bg_dim: int, var_min: float = -5.0, var_max: float = 3.0) -> None:
        super().__init__()
        self.z_bg_dim = z_bg_dim
        self.var_min = var_min
        self.var_max = var_max
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, z_bg_dim * 2),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        moments = self.net(x)
        mu, logvar = torch.chunk(moments, 2, dim=-1)
        logvar = torch.clamp(logvar, self.var_min, self.var_max)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return {"mu": mu, "logvar": logvar, "z": z}


class CameraMotionPredictor(nn.Module):
    """从 z_bg + bbox 预测全局相机运动对每个 actor bbox 的影响 (Δbbox_bg).

    输入: z_bg (B, T-1, d_bg), bbox_t (B, T-1, K, 4)
    输出: Δbbox_bg (B, T-1, K, 4) — 每个 actor 受到的全局运动

    条件化 bbox 使模型能学习 zoom 效应 (边缘 bbox 移动更多)。
    """

    def __init__(self, z_bg_dim: int, hidden_dim: int = 64, img_size: int = 256) -> None:
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.LayerNorm(z_bg_dim + 4),
            nn.Linear(z_bg_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, z_bg: Tensor, bbox_t: Tensor) -> Tensor:
        """z_bg: (B, T-1, d_bg), bbox_t: (B, T-1, K, 4) -> Δbbox_bg: (B, T-1, K, 4)."""
        B, T1, d = z_bg.shape
        K = bbox_t.shape[2]
        z_bg_exp = z_bg.unsqueeze(2).expand(-1, -1, K, -1)  # (B, T-1, K, d)
        bbox_norm = bbox_t / self.img_size  # 归一化到 [0, 1]
        inp = torch.cat([z_bg_exp, bbox_norm], dim=-1)  # (B, T-1, K, d+4)
        return self.net(inp)


class ActorMotionPredictor(nn.Module):
    """从 z_actor 预测 actor 残差运动.

    输入: z_actor (B, T-1, K, d_z)
    输出: Δbbox_res (B, T-1, K, 4)
    """

    def __init__(self, z_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, z_actor: Tensor) -> Tensor:
        return self.net(z_actor)


class BgSupervisionHead(nn.Module):
    """从 z_bg 预测相机参数 (用于 L_bg 直接监督).

    输入: z_bg (B, T-1, d_bg)
    输出: camera_params_pred (B, T-1, 4) [dx, dy, dscale, dbright]
    """

    def __init__(self, z_bg_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_bg_dim),
            nn.Linear(z_bg_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, z_bg: Tensor) -> Tensor:
        return self.net(z_bg)


class LatentActionModelV8(nn.Module):
    """V8 MOT-Guided Slot-Time Latent Action Model."""

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
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.z_dim = z_dim
        self.z_bg_dim = z_bg_dim
        self.max_actors = max_actors
        self.free_bits_lambda = free_bits_lambda
        self.bbox_scale = bbox_scale
        self.use_bg_slot = use_bg_slot
        self.bg_loss_weight = bg_loss_weight
        self.num_actor_types = num_actor_types
        self.register_buffer("cam_scale", torch.tensor(camera_param_scale, dtype=torch.float32))

        K = max_actors + 1  # +1 for bg
        self.K = K

        # 1. Motion Token Encoder
        self.motion_encoder = MotionTokenEncoder(
            model_dim=model_dim, crop_size=crop_size, img_size=img_size,
        )

        # 2. Transition Token Builder
        self.transition_builder = TransitionTokenBuilder(model_dim)

        # 3. Temporal Transformer (2 layers)
        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(model_dim, num_heads, dropout)
            for _ in range(num_temporal_layers)
        ])

        # 4. Slot Transformer (1 layer only, per architecture.md)
        self.slot_blocks = nn.ModuleList([
            SlotTransformerBlock(model_dim, num_heads, dropout)
            for _ in range(num_slot_layers)
        ])

        # 5. Heads
        self.actor_action_head = SharedActorActionHead(model_dim, z_dim, num_actor_types=num_actor_types)
        if use_bg_slot:
            self.bg_motion_head = BackgroundMotionHead(model_dim, z_bg_dim)
            self.camera_motion_pred = CameraMotionPredictor(z_bg_dim, img_size=img_size)
            self.bg_supervision_head = BgSupervisionHead(z_bg_dim)

        # 6. Motion predictors (for L_motion)
        self.actor_motion_pred = ActorMotionPredictor(z_dim)

    def _free_bits_kl(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Free Bits KL: 每 dim 的 KL 低于 λ 时不惩罚."""
        kl_dim = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1)
        kl_dim = kl_dim.clamp(min=self.free_bits_lambda)
        return kl_dim.sum() / mu.reshape(-1).shape[0]

    def encode(self, video: Tensor, boxes: Tensor, valid_mask: Tensor, actor_labels: Tensor = None) -> Dict[str, Tensor]:
        """编码 -> z_actor, z_bg.

        Args:
            video: (B, T, H, W, 3)
            boxes: (B, T, K, 4)
            valid_mask: (B, T, K) bool
            actor_labels: (B, K) long, A2D actor type (optional, for conditioning)
        Returns:
            dict with z_actor, mu_actor, logvar_actor, z_bg, mu_bg, logvar_bg,
                     transition_tokens, motion_tokens
        """
        B, T, K, _ = boxes.shape
        # 1. Motion tokens
        enc = self.motion_encoder(video, boxes, valid_mask)
        motion_tokens = enc["motion_tokens"]  # (B, T, K+1, D)

        # 2. Transition tokens
        trans = self.transition_builder(motion_tokens)  # (B, T-1, K+1, D)

        # 3. Temporal attention (per slot along time)
        x = trans.permute(0, 2, 1, 3)  # (B, K+1, T-1, D)
        for block in self.temporal_blocks:
            x = block(x)
        x = x.permute(0, 2, 1, 3)  # (B, T-1, K+1, D)

        # 4. Slot attention (per timestep across slots)
        # valid mask for slots: bg always valid (if use_bg_slot), actors per valid_mask
        T1 = x.shape[1]
        slot_valid = torch.zeros(B, T1, self.K, dtype=torch.bool, device=x.device)
        if self.use_bg_slot:
            slot_valid[:, :, 0] = True  # bg always valid
        slot_valid[:, :, 1:] = valid_mask[:, 1:]  # actor valid from t=1 onward (transitions)

        for block in self.slot_blocks:
            x = block(x, key_padding_mask=slot_valid)

        # 5. Heads
        # actor slots (index 1..K)
        # actor_labels for FiLM: (B, K) → (B, T-1, K) by broadcasting
        if actor_labels is not None and self.num_actor_types > 0:
            al = actor_labels.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, T-1, K)
        else:
            al = None
        actor_out = self.actor_action_head(x[:, :, 1:], actor_labels=al)  # (B, T-1, K, d_z)

        result = {
            "z_actor": actor_out["z"], "mu_actor": actor_out["mu"], "logvar_actor": actor_out["logvar"],
            "transition_tokens": trans,
            "motion_tokens": motion_tokens,
        }

        if self.use_bg_slot:
            bg_out = self.bg_motion_head(x[:, :, 0])  # (B, T-1, d_bg)
            result["z_bg"] = bg_out["z"]
            result["mu_bg"] = bg_out["mu"]
            result["logvar_bg"] = bg_out["logvar"]

        return result

    def forward(self, batch: Dict) -> Dict[str, Tensor]:
        """前向传播 + 损失计算.

        batch keys: videos, boxes, valid_mask, [camera_params]
        Returns: losses + latents
        """
        video = batch["videos"]
        boxes = batch["boxes"]
        valid_mask = batch["valid_mask"]
        actor_labels = batch.get("actor_labels", None)
        camera_params = batch.get("camera_params", None)

        B, T, K, _ = boxes.shape
        out = self.encode(video, boxes, valid_mask, actor_labels=actor_labels)

        z_actor = out["z_actor"]      # (B, T-1, K, d_z)
        mu_actor = out["mu_actor"]
        logvar_actor = out["logvar_actor"]

        # === L_motion ===
        dbbox_obs = boxes[:, 1:] - boxes[:, :-1]  # (B, T-1, K, 4)
        dbbox_res = self.actor_motion_pred(z_actor)  # (B, T-1, K, 4)

        if self.use_bg_slot:
            z_bg = out["z_bg"]
            mu_bg = out["mu_bg"]
            logvar_bg = out["logvar_bg"]
            # CameraMotion(z_bg, bbox_t) → Δbbox_bg (条件化 bbox 位置)
            bbox_t = boxes[:, :-1]  # (B, T-1, K, 4) — bbox at time t
            dbbox_bg = self.camera_motion_pred(z_bg, bbox_t)  # (B, T-1, K, 4)
            dbbox_pred = dbbox_bg + dbbox_res
        else:
            dbbox_bg = torch.zeros_like(dbbox_res)
            dbbox_pred = dbbox_res

        actor_valid = valid_mask[:, 1:].unsqueeze(-1).float()  # (B, T-1, K, 1)
        dbbox_pred_n = dbbox_pred / self.bbox_scale
        dbbox_obs_n = dbbox_obs / self.bbox_scale
        motion_loss = ((dbbox_pred_n - dbbox_obs_n) ** 2 * actor_valid).sum() / (actor_valid.sum() * 4 + 1e-8)

        # === L_KL (Free Bits) ===
        kl_actor = self._free_bits_kl(mu_actor, logvar_actor)
        if self.use_bg_slot:
            kl_bg = self._free_bits_kl(mu_bg, logvar_bg)
            kl_loss = kl_actor + kl_bg
        else:
            kl_loss = kl_actor

        out["motion_loss"] = motion_loss
        out["kl_loss"] = kl_loss
        out["dbbox_pred"] = dbbox_pred
        out["dbbox_obs"] = dbbox_obs
        out["dbbox_bg"] = dbbox_bg
        out["dbbox_res"] = dbbox_res

        # === L_bg (相机参数直接监督) ===
        if self.use_bg_slot and camera_params is not None:
            cam_pred = self.bg_supervision_head(z_bg)  # (B, T-1, 4)
            cam_target = camera_params / self.cam_scale  # 归一化到 O(1)
            bg_loss = ((cam_pred - cam_target) ** 2).mean()
            out["bg_loss"] = bg_loss
        else:
            out["bg_loss"] = torch.tensor(0.0, device=video.device)

        return out
