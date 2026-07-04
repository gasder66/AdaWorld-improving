"""
V14.1 MaskWarp: eval-time mask warping by bbox prediction.

warp_mask_by_bbox(mask_t, bbox_t, bbox_pred):
  Translates mask_t from bbox_t position to bbox_pred position.
  For rigid-translation data (Bridge-1), this should perfectly capture mask motion.

Mask warp is eval-only — not used during training.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def warp_mask_by_bbox(
    mask_t: Tensor,
    bbox_t: Tensor,
    bbox_pred: Tensor,
    out_size: int = -1,
) -> Tensor:
    """Warp mask_t from bbox_t region to bbox_pred region via affine_grid_sample.

    Args:
        mask_t:     (B, K, 1, H, W)  — binary mask at time t (full-frame or low-res)
        bbox_t:     (B, K, 4)        — cxcywh normalized [0,1] at time t
        bbox_pred:  (B, K, 4)        — cxcywh normalized [0,1] predicted at t+1
        out_size:   optional output H,W. If -1, keeps mask_t spatial size.

    Returns:
        warped: (B, K, 1, H, W) — mask warped to predicted position
    """
    B, K, _, H, W = mask_t.shape
    device = mask_t.device

    # Affine: output → source mapping. Positive bbox delta means object moved right,
    # so source sample position should be left (negative) relative to output.
    # Equivalently: affine tx = -dcx.
    dcx = (bbox_pred[..., 0] - bbox_t[..., 0]) * 2.0  # (B, K)
    dcy = (bbox_pred[..., 1] - bbox_t[..., 1]) * 2.0

    # Scale ratio — if object grew, source sample should be denser (1/sx).
    sx = bbox_t[..., 2] / bbox_pred[..., 2].clamp(min=1e-4)
    sy = bbox_t[..., 3] / bbox_pred[..., 3].clamp(min=1e-4)

    # Build affine matrix (B*K, 2, 3) — maps output_grid → source_sample.
    theta = torch.zeros(B * K, 2, 3, device=device)
    theta[:, 0, 0] = sx.reshape(-1).clamp(0.5, 2.0)
    theta[:, 0, 2] = -dcx.reshape(-1)  # object moved right → sample left
    theta[:, 1, 1] = sy.reshape(-1).clamp(0.5, 2.0)
    theta[:, 1, 2] = -dcy.reshape(-1)

    # Mask as (B*K, 1, H, W).
    mask_flat = mask_t.reshape(B * K, 1, H, W)

    oh, ow = (out_size, out_size) if out_size > 0 else (H, W)
    grid = F.affine_grid(theta, (B * K, 1, oh, ow), align_corners=False)
    warped = F.grid_sample(mask_flat, grid, mode="bilinear",
                           align_corners=False, padding_mode="zeros")
    return warped.reshape(B, K, 1, oh, ow)


def warp_mask_flow(
    mask_t: Tensor,
    flow: Tensor,
) -> Tensor:
    """Warp mask by optical flow field (alternative, not used in Bridge-1)."""
    B, K, _, H, W = mask_t.shape
    device = mask_t.device

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 2)

    flow_norm = flow.clone()
    flow_norm[..., 0] = flow_norm[..., 0] / (W / 2)
    flow_norm[..., 1] = flow_norm[..., 1] / (H / 2)
    grid = base_grid + flow_norm

    mask_flat = mask_t.reshape(B * K, 1, H, W)
    grid_flat = grid.reshape(B * K, H, W, 2)
    warped = F.grid_sample(mask_flat, grid_flat, mode="bilinear",
                           align_corners=False, padding_mode="zeros")
    return warped.reshape(B, K, 1, H, W)
