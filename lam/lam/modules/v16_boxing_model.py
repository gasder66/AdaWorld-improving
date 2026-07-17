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
            # Keep H/4 spatial states. Boxing sprites are only ~14 px wide;
            # H/8 reduced them to roughly two cells and destroyed contours.
            nn.Conv2d(64, state_dim, 3, stride=1, padding=1), nn.GroupNorm(8, state_dim), nn.GELU(),
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

    def forward(
        self, state_t: Tensor, z: Tensor, opponent_ablation: str = "normal", target_slot: int | None = None
    ) -> Tensor:
        del opponent_ablation, target_slot
        shape = state_t.shape
        flat_state = state_t.reshape(-1, shape[-3], shape[-2], shape[-1])
        flat_z = z.reshape(-1, z.shape[-1])
        gamma, beta = self.z_up(flat_z).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        modulated = self.state_norm(flat_state) * gamma + beta
        return (flat_state + self.transition(modulated)).reshape(shape)


class InteractionObjectFDM(nn.Module):
    """Independent dynamics plus a small two-token interaction residual."""

    def __init__(
        self, state_dim: int, latent_dim: int, layers: int = 2, heads: int = 4,
        spatial_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_tokens = spatial_tokens
        self.base = IndependentObjectFDM(state_dim, latent_dim)
        self.state_norm = nn.GroupNorm(8, state_dim)
        self.state_token = nn.Linear(state_dim, state_dim, bias=False)
        if spatial_tokens:
            self.spatial_score = nn.Conv2d(state_dim, 1, 1)
            self.position_token = nn.Linear(2, state_dim, bias=False)
        self.z_token = nn.Linear(latent_dim, state_dim, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=state_dim,
            nhead=heads,
            dim_feedforward=state_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.condition = nn.Linear(state_dim, state_dim * 2, bias=False)
        self.transition = nn.Sequential(
            nn.Conv2d(state_dim, state_dim, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(state_dim, state_dim, 3, padding=1, bias=False),
        )
        nn.init.normal_(self.transition[-1].weight, mean=0.0, std=1e-3)

    @staticmethod
    def _intervene(value: Tensor, opponent: int, mode: str) -> Tensor:
        result = value.clone()
        if mode.startswith("mask_"):
            result[:, opponent] = 0
        elif mode.startswith("shuffle_"):
            shift = max(1, value.shape[0] // 2)
            result[:, opponent] = value[:, opponent].roll(shift, dims=0)
        return result

    def _visual_token(self, state_t: Tensor) -> Tensor:
        batch, slots, channels, height, width = state_t.shape
        if not self.spatial_tokens:
            return self.state_token(state_t.mean(dim=(-2, -1)))
        flat = state_t.reshape(batch * slots, channels, height, width)
        weights = self.spatial_score(flat).flatten(2).softmax(dim=-1)
        features = flat.flatten(2)
        pooled = (features * weights).sum(dim=-1)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=state_t.device, dtype=state_t.dtype),
            torch.linspace(-1.0, 1.0, width, device=state_t.device, dtype=state_t.dtype),
            indexing="ij",
        )
        coordinates = torch.stack([xx, yy], dim=0).reshape(1, 2, height * width)
        position = (coordinates * weights).sum(dim=-1)
        token = self.state_token(pooled) + self.position_token(position)
        return token.reshape(batch, slots, channels)

    def forward(
        self, state_t: Tensor, z: Tensor, opponent_ablation: str = "normal", target_slot: int | None = None
    ) -> Tensor:
        # state_t [N,2,C,H,W], z [N,2,D]
        if state_t.ndim != 5 or state_t.shape[1] != 2:
            raise ValueError(f"interaction FDM expects [N,2,C,H,W], got {tuple(state_t.shape)}")
        base_prediction = self.base(state_t, z)
        visual_tokens = self._visual_token(state_t)
        z_used = z
        if opponent_ablation != "normal":
            if target_slot not in (0, 1):
                raise ValueError("target_slot must be 0 or 1 for opponent ablations")
            opponent = 1 - int(target_slot)
            if opponent_ablation in {"mask_state", "shuffle_state"}:
                visual_tokens = self._intervene(visual_tokens, opponent, opponent_ablation)
            elif opponent_ablation in {"mask_z", "shuffle_z"}:
                z_used = self._intervene(z, opponent, opponent_ablation)
            else:
                raise ValueError(f"unknown opponent ablation: {opponent_ablation}")
        tokens = visual_tokens + self.z_token(z_used)
        mixed = self.interaction(tokens)
        gamma, beta = self.condition(mixed).chunk(2, dim=-1)
        flat_state = state_t.reshape(-1, *state_t.shape[-3:])
        gamma = gamma.reshape(-1, gamma.shape[-1]).unsqueeze(-1).unsqueeze(-1)
        beta = beta.reshape(-1, beta.shape[-1]).unsqueeze(-1).unsqueeze(-1)
        modulated = self.state_norm(flat_state) * gamma + beta
        correction = self.transition(modulated).reshape_as(state_t)
        return base_prediction + correction


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

    OBJECT_INPUT_CHANNELS = {
        "masked_rgb_mask": 4,
        "mask_only": 1,
        "masked_rgb": 3,
        "mask_structure_content": 1,
    }

    def __init__(
        self,
        state_dim: int = 96,
        latent_dim: int = 16,
        fdm_type: str = "independent",
        object_input_mode: str = "masked_rgb_mask",
    ) -> None:
        super().__init__()
        if object_input_mode not in self.OBJECT_INPUT_CHANNELS:
            raise ValueError(f"unknown object_input_mode: {object_input_mode}")
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.fdm_type = fdm_type
        self.object_input_mode = object_input_mode
        self.object_encoder = SpatialVisualEncoder(self.OBJECT_INPUT_CHANNELS[object_input_mode], state_dim)
        self.background_encoder = SpatialVisualEncoder(4, state_dim)
        if object_input_mode == "mask_structure_content":
            # Content is pooled from the current RGB frame under each oracle
            # mask. Spatial layout remains exclusively in the mask structure
            # pathway used by IDM/FDM.
            self.content_encoder = SpatialVisualEncoder(3, state_dim)
            self.content_state_norm = nn.GroupNorm(8, state_dim)
            self.content_conditioner = nn.Linear(state_dim, state_dim * 2)
        else:
            self.content_encoder = None
            self.content_state_norm = None
            self.content_conditioner = None
        self.idm = PerObjectIDM(state_dim, latent_dim)
        if fdm_type == "independent":
            self.fdm = IndependentObjectFDM(state_dim, latent_dim)
        elif fdm_type == "interaction":
            self.fdm = InteractionObjectFDM(state_dim, latent_dim)
        elif fdm_type == "spatial_interaction":
            self.fdm = InteractionObjectFDM(state_dim, latent_dim, spatial_tokens=True)
        else:
            raise ValueError(f"unknown fdm_type: {fdm_type}")
        self.object_decoder = SlotDecoder(state_dim, 4)
        self.background_decoder = SlotDecoder(state_dim, 3)

    @staticmethod
    def _shuffle_per_object(z: Tensor) -> Tensor:
        # z: [B,T,K,D]. Preserve object identity while shuffling transitions.
        result = torch.empty_like(z)
        for k in range(z.shape[2]):
            flat = z[:, :, k].reshape(-1, z.shape[-1])
            shift = max(1, flat.shape[0] // 2)
            result[:, :, k] = flat.roll(shift, dims=0).reshape_as(z[:, :, k])
        return result

    def _encode_slots(
        self, videos: Tensor, masks: Tensor, background_masks: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor | None]:
        batch, time, _, height, width = videos.shape
        objects = videos.unsqueeze(2) * masks.unsqueeze(3)
        if self.object_input_mode == "masked_rgb_mask":
            object_inputs = torch.cat([objects, masks.unsqueeze(3)], dim=3)
        elif self.object_input_mode in {"mask_only", "mask_structure_content"}:
            object_inputs = masks.unsqueeze(3)
        else:
            object_inputs = objects
        channels = self.OBJECT_INPUT_CHANNELS[self.object_input_mode]
        object_states = self.object_encoder(object_inputs.reshape(batch * time * 2, channels, height, width))
        sh, sw = object_states.shape[-2:]
        object_states = object_states.reshape(batch, time, 2, self.state_dim, sh, sw)
        background = videos * background_masks.unsqueeze(2)
        background_input = torch.cat([background, background_masks.unsqueeze(2)], dim=2)
        background_states = self.background_encoder(background_input.reshape(batch * time, 4, height, width))
        background_states = background_states.reshape(batch, time, self.state_dim, sh, sw)
        content_states = None
        if self.content_encoder is not None:
            content_map = self.content_encoder(videos.reshape(batch * time, 3, height, width))
            flat_masks = masks.reshape(batch * time * 2, 1, height, width)
            downsampled_masks = F.interpolate(
                flat_masks, size=(sh, sw), mode="bilinear", align_corners=False
            ).reshape(batch * time, 2, 1, sh, sw)
            weighted = content_map.unsqueeze(1) * downsampled_masks
            denominator = downsampled_masks.sum(dim=(-2, -1)).clamp_min(1e-6)
            content_states = (
                weighted.sum(dim=(-2, -1)) / denominator
            ).reshape(batch, time, 2, self.state_dim)
        return object_states, background_states, content_states

    def forward(
        self,
        batch: Dict[str, Tensor],
        ablation: Literal["normal", "zero", "shuffle"] = "normal",
        opponent_ablation: Literal["normal", "mask_state", "shuffle_state", "mask_z", "shuffle_z"] = "normal",
        target_slot: int | None = None,
        content_ablation: Literal["normal", "zero", "shuffle", "swap_slots"] = "normal",
    ) -> Dict[str, Tensor]:
        videos = batch["videos"]
        masks = batch["masks"]
        background_masks = batch["background_masks"]
        batch_size, time, _, height, width = videos.shape
        object_states, background_states, content_states = self._encode_slots(
            videos, masks, background_masks
        )
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
        grouped_t = state_t.reshape(-1, 2, self.state_dim, *state_t.shape[-2:])
        grouped_z = z_used.reshape(-1, 2, self.latent_dim)
        predicted = self.fdm(grouped_t, grouped_z, opponent_ablation, target_slot).reshape_as(state_t)

        # During normal training, force transition-specific z to outperform
        # zero and same-object shuffled latents. This uses no action labels.
        normal_error = (predicted - state_tp1.detach()).square().mean(dim=(-3, -2, -1))
        if self.training and ablation == "normal":
            shuffled_z = self._shuffle_per_object(z.detach())
            shuffled_prediction = self.fdm(
                grouped_t, shuffled_z.reshape(-1, 2, self.latent_dim)
            ).reshape_as(state_t)
            shuffled_error = (shuffled_prediction - state_tp1.detach()).square().mean(dim=(-3, -2, -1))
            zero_error = (state_t - state_tp1.detach()).square().mean(dim=(-3, -2, -1))
            margin = 5e-4
            action_contrast_loss = (
                F.relu(normal_error + margin - shuffled_error).mean()
                + F.relu(normal_error + margin - zero_error).mean()
            )
        else:
            action_contrast_loss = torch.zeros((), device=videos.device)

        decoder_states = predicted
        content_used = None
        if content_states is not None:
            content_used = content_states[:, :-1]
            if content_ablation == "zero":
                content_used = torch.zeros_like(content_used)
            elif content_ablation == "shuffle":
                content_used = self._shuffle_per_object(content_used)
            elif content_ablation == "swap_slots":
                content_used = content_used.flip(dims=(2,))
            elif content_ablation != "normal":
                raise ValueError(f"unknown content_ablation: {content_ablation}")
            flat_predicted = predicted.reshape(-1, self.state_dim, *predicted.shape[-2:])
            flat_content = content_used.reshape(-1, self.state_dim)
            gamma, beta = self.content_conditioner(flat_content).chunk(2, dim=-1)
            gamma = 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            decoder_states = (
                self.content_state_norm(flat_predicted) * (1.0 + gamma) + beta
            ).reshape_as(predicted)
        object_logits = self.object_decoder(
            decoder_states.reshape(-1, self.state_dim, *decoder_states.shape[-2:]), (height, width)
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
        predicted_binary_masks = probs >= 0.5
        target_binary_masks = target_masks >= 0.5
        intersection = (predicted_binary_masks & target_binary_masks).sum(dim=(-2, -1)).float()
        union = (predicted_binary_masks | target_binary_masks).sum(dim=(-2, -1)).float().clamp_min(1.0)
        mask_iou = (intersection / union).mean()
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
        z_norm_loss = mu.square().mean()
        identity_state_loss = F.mse_loss(state_t, state_tp1.detach())
        structure_first = self.object_input_mode in {"mask_only", "mask_structure_content"}
        object_rgb_weight = 0.25 if self.object_input_mode == "mask_only" else 1.0
        mask_weight = 1.0 if structure_first else 0.5
        total = (
            5.0 * state_loss
            + object_rgb_weight * object_rgb_loss
            + mask_weight * mask_bce
            + mask_weight * dice
            + 0.25 * background_loss
            + 0.25 * reconstruction_loss
            + 0.1 * variance_floor_loss
            + 10.0 * action_contrast_loss
            + 1e-3 * z_norm_loss
        )
        return {
            "loss": total,
            "state_loss": state_loss,
            "object_rgb_loss": object_rgb_loss,
            "mask_bce": mask_bce,
            "mask_dice_loss": dice,
            "mask_iou": mask_iou,
            "background_loss": background_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl,
            "variance_floor_loss": variance_floor_loss,
            "z_norm_loss": z_norm_loss,
            "action_contrast_loss": action_contrast_loss,
            "z_variance": z_variance,
            "identity_state_loss": identity_state_loss,
            "z": z,
            "z_mu": mu,
            "z_logvar": logvar,
            "object_states": object_states,
            "predicted_object_states": predicted,
            "background_states": background_states,
            "content_states": content_states,
            "content_used": content_used,
            "object_rgb": object_rgb,
            "object_mask_logits": object_mask_logits,
            "background_rgb": background_rgb,
            "reconstruction": reconstruction,
        }
