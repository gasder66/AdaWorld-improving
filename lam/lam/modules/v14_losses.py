"""
V14 losses: structure (bbox + mask) + KL + reconstruction.
"""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def _dice_loss(pred: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    smooth = 1.0
    v = valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
    intersection = (pred.sigmoid() * target * v).sum()
    denom = ((pred.sigmoid() + target) * v).sum()
    return 1.0 - (2.0 * intersection + smooth) / (denom + smooth)


def structure_loss_v14(
    pred: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    valid_tp1: Tensor,
    lambda_box: float = 10.0,
    lambda_mask: float = 1.0,
    lambda_iou: float = 1.0,
    lambda_exist: float = 1.0,
    lambda_vis: float = 0.5,
) -> Tensor:
    v = valid_tp1.unsqueeze(-1).float()

    # Bbox SmoothL1.
    box_loss = F.smooth_l1_loss(pred["bbox"], targets["bbox"], reduction="none")
    box_loss = (box_loss * v).sum() / v.sum().clamp(min=1.0)

    # IoU.
    pb = pred["bbox"]; gb = targets["bbox"]
    px1 = pb[..., 0] - pb[..., 2] / 2; py1 = pb[..., 1] - pb[..., 3] / 2
    px2 = pb[..., 0] + pb[..., 2] / 2; py2 = pb[..., 1] + pb[..., 3] / 2
    gx1 = gb[..., 0] - gb[..., 2] / 2; gy1 = gb[..., 1] - gb[..., 3] / 2
    gx2 = gb[..., 0] + gb[..., 2] / 2; gy2 = gb[..., 1] + gb[..., 3] / 2
    ix1 = torch.max(px1, gx1); iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2); iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    ap = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    ag = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    iou = inter / (ap + ag - inter + 1e-6)
    iou_loss = ((1.0 - iou) * valid_tp1.float()).sum() / valid_tp1.float().sum().clamp(min=1.0)

    # Mask: BCE + Dice on low-res grid.
    mask_bce = F.binary_cross_entropy_with_logits(pred["mask_low"], targets["mask_low"], reduction="none")
    v3d = v.unsqueeze(-1).unsqueeze(-1)
    mask_bce = (mask_bce * v3d).sum() / v3d.sum().clamp(min=1.0)
    mask_dice = _dice_loss(pred["mask_low"], targets["mask_low"], valid_tp1)

    # Existence BCE.
    exist_target = valid_tp1.unsqueeze(-1).float()
    exist_loss = F.binary_cross_entropy_with_logits(
        pred["exist_logit"].squeeze(-1), exist_target.squeeze(-1), reduction="none",
    )
    exist_loss = (exist_loss * valid_tp1.float()).sum() / valid_tp1.float().sum().clamp(min=1.0)

    # Visible ratio.
    vis_target = (targets["visible_mask"].sum(dim=(-2, -1)) /
                  (targets["mask_full"].sum(dim=(-2, -1)) + 1e-6)).unsqueeze(-1)
    vis_loss = F.mse_loss(pred["visible_ratio"], vis_target, reduction="none")
    vis_loss = (vis_loss * v).sum() / v.sum().clamp(min=1.0)

    return (lambda_box * box_loss + lambda_iou * iou_loss +
            lambda_mask * (mask_bce + mask_dice) +
            lambda_exist * exist_loss + lambda_vis * vis_loss)


def reconstruction_loss_v14(
    recon: Tensor, target: Tensor, pred_mask: Tensor,
    valid: Tensor, lambda_object: float = 2.0, lambda_full: float = 0.1,
) -> Tensor:
    l1_full = F.l1_loss(recon, target)

    # Object-masked L1: interpolate predicted low-res mask to full resolution.
    B, T1, K, _, g, _ = pred_mask.shape
    H, W = recon.shape[-2:]
    mask_prob = pred_mask.sigmoid()  # (B, T1, K, 1, g, g)
    mask_prob = mask_prob.permute(0, 1, 2, 3, 5, 4).reshape(B * T1 * K, 1, g, g)
    mask_full = F.interpolate(mask_prob, size=(H, W), mode="bilinear", align_corners=False)
    mask_full = mask_full.reshape(B, T1, K, 1, H, W)
    # Sum over objects, expand to channels.
    obj_mask = mask_full.sum(dim=2).expand(-1, -1, 3, -1, -1).clamp(0, 1)
    l1_obj = ((recon - target).abs() * obj_mask).sum() / obj_mask.sum().clamp(min=1.0)

    return lambda_full * l1_full + lambda_object * l1_obj


def free_bits_kl_v14(mu, logvar, valid, free_bits=0.05):
    kl_dim = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1)
    kl_dim = kl_dim.clamp(min=free_bits)
    v = valid.unsqueeze(-1).float()
    n_valid = v.sum().clamp(min=1.0) * mu.shape[-1]
    return (kl_dim * v).sum() / n_valid
