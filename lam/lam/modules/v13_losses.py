"""
V13 Atari losses: structure + KL + reconstruction.
"""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def _dice_loss(pred: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    smooth = 1.0
    intersection = (pred * target * valid).sum()
    denom = ((pred + target) * valid).sum()
    return 1.0 - (2.0 * intersection + smooth) / (denom + smooth)


def structure_loss_v13(
    pred: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    valid_tp1: Tensor,            # (B, T-1, K) bool
    slot_is_group: Tensor,        # (B, T-1, K) bool
    lambda_box: float = 10.0,
    lambda_exist: float = 1.0,
    lambda_group: float = 1.0,
    lambda_iou: float = 1.0,
) -> Tensor:
    v = valid_tp1.unsqueeze(-1).float()
    is_obj = slot_is_group.logical_not().unsqueeze(-1).float()

    # Bbox: SmoothL1 on object slots only.
    box_raw = F.smooth_l1_loss(pred["bbox"], targets["bbox"], reduction="none")
    box_loss = (box_raw * v * is_obj).sum() / (v * is_obj).sum().clamp(min=1.0)

    # IoU loss for object slots.
    pred_cxcywh = pred["bbox"]
    gt_cxcywh = targets["bbox"]
    px1 = pred_cxcywh[..., 0] - pred_cxcywh[..., 2] / 2
    py1 = pred_cxcywh[..., 1] - pred_cxcywh[..., 3] / 2
    px2 = pred_cxcywh[..., 0] + pred_cxcywh[..., 2] / 2
    py2 = pred_cxcywh[..., 1] + pred_cxcywh[..., 3] / 2
    gx1 = gt_cxcywh[..., 0] - gt_cxcywh[..., 2] / 2
    gy1 = gt_cxcywh[..., 1] - gt_cxcywh[..., 3] / 2
    gx2 = gt_cxcywh[..., 0] + gt_cxcywh[..., 2] / 2
    gy2 = gt_cxcywh[..., 1] + gt_cxcywh[..., 3] / 2
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = area_p + area_g - inter + 1e-6
    iou = inter / union
    iou_loss = ((1.0 - iou) * valid_tp1.float() * is_obj.squeeze(-1)).sum() / (valid_tp1.float() * is_obj.squeeze(-1)).sum().clamp(min=1.0)

    # Existence: BCE on object slots.
    exist_target = valid_tp1.unsqueeze(-1).float()
    exist_loss = F.binary_cross_entropy_with_logits(
        pred["exist_logit"].squeeze(-1), exist_target.squeeze(-1),
        reduction="none",
    )
    exist_loss = (exist_loss.unsqueeze(-1) * v * is_obj).sum() / (v * is_obj).sum().clamp(min=1.0)

    # Group feature: MSE on group slots.
    group_loss = torch.tensor(0.0, device=v.device)
    if "group_feat" in pred and "group_feat" in targets:
        is_grp = slot_is_group.unsqueeze(-1).float()
        gf_loss = F.mse_loss(pred["group_feat"], targets["group_feat"], reduction="none")
        group_loss = (gf_loss * v * is_grp).sum() / (v * is_grp).sum().clamp(min=1.0)

    return lambda_box * box_loss + lambda_iou * iou_loss + lambda_exist * exist_loss + lambda_group * group_loss


def reconstruction_loss_v13(
    recon: Tensor,              # (B, T-1, C, H, W)
    target: Tensor,             # (B, T-1, C, H, W)
    slot_bbox: Tensor,          # (B, T-1, K, 4) predicted bbox
    slot_valid: Tensor,         # (B, T-1, K) bool
    image_size: int = 84,
    lambda_object_masked: float = 2.0,
) -> Tensor:
    l1 = F.l1_loss(recon, target, reduction="mean")

    # Object-masked L1: compute mask from predicted bboxes.
    B, T1, C, H, W = recon.shape
    obj_mask = torch.zeros(B, T1, 1, H, W, device=recon.device)
    for b in range(B):
        for t in range(T1):
            for k in range(slot_valid.shape[2]):
                if slot_valid[b, t, k]:
                    cx, cy, w_, h_ = slot_bbox[b, t, k].tolist()
                    x1 = max(0, int((cx - w_ / 2) * W))
                    y1 = max(0, int((cy - h_ / 2) * H))
                    x2 = min(W, int((cx + w_ / 2) * W))
                    y2 = min(H, int((cy + h_ / 2) * H))
                    if x2 > x1 and y2 > y1:
                        obj_mask[b, t, 0, y1:y2, x1:x2] = 1.0

    obj_l1 = ((recon - target).abs() * obj_mask).sum() / obj_mask.sum().clamp(min=1.0) / C
    return l1 + lambda_object_masked * obj_l1


def free_bits_kl(mu: Tensor, logvar: Tensor, valid: Tensor, free_bits: float = 0.05) -> Tensor:
    kl_dim = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1)
    kl_dim = kl_dim.clamp(min=free_bits)
    v = valid.unsqueeze(-1).float()
    n_valid = v.sum().clamp(min=1.0) * mu.shape[-1]
    return (kl_dim * v).sum() / n_valid
