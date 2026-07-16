"""V16 minimal Boxing object-wise IDM/FDM with a background context slot."""
from __future__ import annotations

from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialVisualEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, state_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, state_dim, 3, stride=2, padding=1), nn.GroupNorm(8, state_dim), nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PerObjectIDM(nn.Module):
    """Infer z_i only from object i's adjacent visual states."""

    def __init__(self, state_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.compare = nn.Sequential(
            nn.Conv2d(state_dim * 3, state_dim, 1), nn.GELU(),
            nn.Conv2d(state_dim, state_dim, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Linear(state_dim, latent_dim)

    def forward(self, state_t: Tensor, state_tp1: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        delta = state_tp1 - state_t
        hidden = self.compare(torch.cat([state_t, state_tp1, delta], dim=1)).mean(dim=(-2, -1))
        mu = self.head(hidden)
        logvar = torch.zeros_like(mu)
        z = mu
        return z, mu, logvar


class IndependentObjectFDM(nn.Module):
    """Stage-1 FDM; shared parameters, no cross-object information."""

    def __init__(self, state_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.state_norm = nn.GroupNorm(8, state_dim)
        self.z_up = nn.Linear(latent_dim, state_dim * 2, bias=False)
        self.transition = nn.Sequential(
            nn.Conv2d(state_dim, state_dim, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(state_dim, state_dim, 3, padding=1, bias=False),
        )

    def forward(self, state_t: Tensor, z: Tensor) -> Tensor:
        gamma, beta = self.z_up(z).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        modulated = self.state_norm(state_t) * gamma + beta
        return state_t + self.transition(modulated)


class SlotDecoder(nn.Module):
    def __init__(self, state_dim: int, out_channels: int) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(state_dim, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
        )
        self.out = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, state: Tensor, output_size: Tuple[int, int]) -> Tensor:
        hidden = self.pre(state)
        hidden = F.interpolate(hidden, size=output_size, mode="bilinear", align_corners=False)
        return self.out(hidden)


class BoxingObjectLAM(nn.Module):
    """Two dynamic fighter slots plus one action-free background slot."""

    def __init__(self, state_dim: int = 96, latent_dim: int = 16) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.object_encoder = SpatialVisualEncoder(4, state_dim)
        self.background_encoder = SpatialVisualEncoder(4, state_dim)
        self.idm = PerObjectIDM(state_dim, latent_dim)
        self.fdm = IndependentObjectFDM(state_dim, latent_dim)
        self.object_decoder = SlotDecoder(state_dim, 4)
        self.background_decoder = SlotDecoder(state_dim, 3)

    @staticmethod
    def _shuffle_per_object(z: Tensor) -> Tensor:
        # z: [B,T,K,D]. Preserve object identity while shuffling transitions.
        result = torch.empty_like(z)
        for k in range(z.shape[2]):
            flat = z[:, :, k].reshape(-1, z.shape[-1])
            result[:, :, k] = flat.roll(1, dims=0).reshape_as(z[:, :, k])
        return result

    def _encode_slots(self, videos: Tensor, masks: Tensor, background_masks: Tensor) -> Tuple[Tensor, Tensor]:
        batch, time, _, height, width = videos.shape
        objects = videos.unsqueeze(2) * masks.unsqueeze(3)
        object_inputs = torch.cat([objects, masks.unsqueeze(3)], dim=3)
        object_states = self.object_encoder(object_inputs.reshape(batch * time * 2, 4, height, width))
        sh, sw = object_states.shape[-2:]
        object_states = object_states.reshape(batch, time, 2, self.state_dim, sh, sw)
        background = videos * background_masks.unsqueeze(2)
        background_input = torch.cat([background, background_masks.unsqueeze(2)], dim=2)
        background_states = self.background_encoder(background_input.reshape(batch * time, 4, height, width))
        background_states = background_states.reshape(batch, time, self.state_dim, sh, sw)
        return object_states, background_states

    def forward(
        self,
        batch: Dict[str, Tensor],
        ablation: Literal["normal", "zero", "shuffle"] = "normal",
    ) -> Dict[str, Tensor]:
        videos = batch["videos"]
        masks = batch["masks"]
        background_masks = batch["background_masks"]
        batch_size, time, _, height, width = videos.shape
        object_states, background_states = self._encode_slots(videos, masks, background_masks)
        state_t = object_states[:, :-1]
        state_tp1 = object_states[:, 1:]
        flat_t = state_t.reshape(-1, self.state_dim, *state_t.shape[-2:])
        flat_tp1 = state_tp1.reshape_as(flat_t)
        z, mu, logvar = self.idm(flat_t, flat_tp1)
        z = z.reshape(batch_size, time - 1, 2, self.latent_dim)
        mu = mu.reshape_as(z)
        logvar = logvar.reshape_as(z)
        if ablation == "zero":
            z_used = torch.zeros_like(z)
        elif ablation == "shuffle":
            z_used = self._shuffle_per_object(z)
        else:
            z_used = z
        predicted = self.fdm(flat_t, z_used.reshape(-1, self.latent_dim))
        predicted = predicted.reshape_as(state_t)

        object_logits = self.object_decoder(
            predicted.reshape(-1, self.state_dim, *predicted.shape[-2:]), (height, width)
        ).reshape(batch_size, time - 1, 2, 4, height, width)
        object_rgb = torch.sigmoid(object_logits[:, :, :, :3])
        object_mask_logits = object_logits[:, :, :, 3]
        object_alpha = torch.sigmoid(object_mask_logits)
        background_rgb = torch.sigmoid(
            self.background_decoder(
                background_states[:, :-1].reshape(-1, self.state_dim, *background_states.shape[-2:]),
                (height, width),
            )
        ).reshape(batch_size, time - 1, 3, height, width)
        reconstruction = background_rgb
        for k in range(2):
            alpha = object_alpha[:, :, k].unsqueeze(2)
            reconstruction = alpha * object_rgb[:, :, k] + (1.0 - alpha) * reconstruction

        target_video = videos[:, 1:]
        target_masks = masks[:, 1:]
        target_background = background_masks[:, 1:]
        state_loss = F.mse_loss(predicted, state_tp1.detach())
        object_den = target_masks.sum(dim=(-2, -1)).clamp_min(1.0)
        object_l1 = (
            (object_rgb - target_video.unsqueeze(2)).abs() * target_masks.unsqueeze(3)
        ).sum(dim=(-3, -2, -1)) / (3.0 * object_den)
        object_rgb_loss = object_l1.mean()
        positive = target_masks.sum().clamp_min(1.0)
        negative = (1.0 - target_masks).sum().clamp_min(1.0)
        pos_weight = (negative / positive).detach().clamp(max=100.0)
        mask_bce = F.binary_cross_entropy_with_logits(object_mask_logits, target_masks, pos_weight=pos_weight)
        probs = object_alpha
        dice = 1.0 - (
            (2.0 * (probs * target_masks).sum(dim=(-2, -1)) + 1.0)
            / (probs.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1)) + 1.0)
        ).mean()
        bg_den = target_background.sum(dim=(-2, -1)).clamp_min(1.0)
        background_loss = (
            (background_rgb - target_video).abs() * target_background.unsqueeze(2)
        ).sum(dim=(-3, -2, -1)).div(3.0 * bg_den).mean()
        reconstruction_loss = (reconstruction - target_video).abs().mean()
        kl = 0.5 * (mu.square() + logvar.exp() - logvar - 1.0).mean()
        z_flat = mu.reshape(-1, self.latent_dim)
        z_std = z_flat.std(dim=0, unbiased=False)
        z_variance = z_std.square().mean()
        variance_floor_loss = F.relu(0.1 - z_std).mean()
        identity_state_loss = F.mse_loss(state_t, state_tp1.detach())
        total = (
            5.0 * state_loss
            + object_rgb_loss
            + 0.5 * mask_bce
            + 0.5 * dice
            + 0.25 * background_loss
            + 0.25 * reconstruction_loss
            + 0.1 * variance_floor_loss
        )
        return {
            "loss": total,
            "state_loss": state_loss,
            "object_rgb_loss": object_rgb_loss,
            "mask_bce": mask_bce,
            "mask_dice_loss": dice,
            "background_loss": background_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl,
            "variance_floor_loss": variance_floor_loss,
            "z_variance": z_variance,
            "identity_state_loss": identity_state_loss,
            "z": z,
            "z_mu": mu,
            "z_logvar": logvar,
            "object_states": object_states,
            "predicted_object_states": predicted,
            "background_states": background_states,
            "object_rgb": object_rgb,
            "object_mask_logits": object_mask_logits,
            "background_rgb": background_rgb,
            "reconstruction": reconstruction,
        }
