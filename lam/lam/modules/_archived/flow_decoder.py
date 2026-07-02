"""V9-D: Flow-based decoder.

Replaces ReconDecoder with a FlowHead that predicts per-pixel displacement
from z_actor (+ optional z_bg), then warps crop_t to produce crop_pred.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def warp_with_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp img by predicted flow field.

    Args:
        img:  (N, C, H, W) source image (crop_t)
        flow: (N, 2, H, W) flow field in PIXEL units (dx, dy)
    Returns:
        warped: (N, C, H, W)
    """
    N, C, H, W = img.shape
    # Build identity grid: pixel (j,i) -> normalized [-1,1]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    # Normalize to [-1, 1]
    grid_x = grid_x / (W - 1) * 2 - 1
    grid_y = grid_y / (H - 1) * 2 - 1
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, H, W, 2)

    # Add flow (pixel units -> normalized)
    flow_t = flow.permute(0, 2, 3, 1)  # (N, H, W, 2)
    flow_norm = flow_t.clone()
    flow_norm[..., 0] = flow_norm[..., 0] / (W - 1) * 2  # dx -> normalized
    flow_norm[..., 1] = flow_norm[..., 1] / (H - 1) * 2  # dy -> normalized

    grid = base_grid + flow_norm
    warped = F.grid_sample(img, grid, align_corners=True, padding_mode="border")
    return warped


class FlowDecoder(nn.Module):
    """Predict flow from latent and warp crop_t.

    z_actor (+ z_bg) -> conv decoder -> flow (2, H, W)
    crop_t + flow -> warp -> crop_pred
    """

    def __init__(
        self,
        z_actor_dim: int = 16,
        z_bg_dim: int = 16,
        crop_size: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.z_total = z_actor_dim + z_bg_dim
        self.hidden_dim = hidden_dim

        # Bottleneck: latent -> feature map
        self.z_to_feat = nn.Sequential(
            nn.Linear(self.z_total, hidden_dim * 4 * 4),
            nn.GELU(),
        )

        # Conv decoder: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 2 channels (flow)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, 2, 1), nn.GELU(),
            nn.Conv2d(hidden_dim // 8, 2, 3, 1, 1),
        )

    def forward(
        self,
        crop_t: torch.Tensor,
        z_actor: torch.Tensor,
        z_bg: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            crop_t: (N, 3, H, W) frame-t crop for all actors
            z_actor: (N, dz) actor latents
            z_bg: (N, dz_bg) or None — bg latent (pooled across time)
        Returns:
            crop_pred: (N, 3, H, W) predicted frame-t+1 crop
        """
        if z_bg is not None and z_bg.shape[-1] > 0:
            z = torch.cat([z_actor, z_bg], dim=-1)
        else:
            z = z_actor

        # Latent -> 4x4 feature map
        feat = self.z_to_feat(z)  # (N, hidden_dim * 4 * 4)
        feat = feat.reshape(-1, self.hidden_dim, 4, 4)  # (N, hidden_dim, 4, 4)

        # Conv decoder -> flow
        flow = self.dec(feat)  # (N, 2, 32, 32)

        # Warp crop_t
        crop_pred = warp_with_flow(crop_t, flow)

        return crop_pred
