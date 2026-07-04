"""
V14.1 BridgeBench evaluation — mask warp, center-safe swap, counterfactual swap.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/eval_v14.py \
      --checkpoint result/v14/bridge1/phaseC/model.pt \
      --batch_size 32 --max_batches 15 --mask_eval_mode warp --gpu 4
"""
from __future__ import annotations

import argparse, json, math, os, sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.modules.v14_model import V14Model
from lam.modules.v14_mask_warp import warp_mask_by_bbox

ACTION_DELTA = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
ACTION_NAMES = ["stay", "up", "down", "left", "right"]
N_ACTIONS = 5
STEP_SIZE = 8
MARGIN = 16
IMAGE_SIZE = 128


def _ensure_dir(p): os.makedirs(p, exist_ok=True)
def _psnr(mse): return float(-10.0 * math.log10(max(mse, 1e-10)))


def _collate(batch):
    out = {}
    for k in batch[0]:
        vs = [b[k] for b in batch]
        if isinstance(vs[0], torch.Tensor): out[k] = torch.stack(vs, dim=0)
    return out


# ============================== Metrics ==============================

def _compute_ious(pred_bbox, gt_bbox, valid):
    B, T, K = pred_bbox.shape[:3]; ious = [[] for _ in range(K)]
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if not valid[b, t, k]: continue
                pcx, pcy, pw, ph = pred_bbox[b, t, k].tolist()
                gcx, gcy, gw, gh = gt_bbox[b, t, k].tolist()
                px1, py1 = pcx - pw / 2, pcy - ph / 2; px2, py2 = pcx + pw / 2, pcy + ph / 2
                gx1, gy1 = gcx - gw / 2, gcy - gh / 2; gx2, gy2 = gcx + gw / 2, gcy + gh / 2
                ix1 = max(px1, gx1); iy1 = max(py1, gy1)
                ix2 = min(px2, gx2); iy2 = min(py2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                ap = max(0, px2 - px1) * max(0, py2 - py1)
                ag = max(0, gx2 - gx1) * max(0, gy2 - gy1)
                ious[k].append(float(inter / (ap + ag - inter + 1e-6)))
    return ious


def _compute_mask_dices(pred_mask, gt_mask, valid):
    B, T, K = pred_mask.shape[:3]; dices = [[] for _ in range(K)]
    pred_bin = (pred_mask.sigmoid() > 0.5).float() if pred_mask.min() < 0 else pred_mask
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if not valid[b, t, k]: continue
                inter = (pred_bin[b, t, k] * gt_mask[b, t, k]).sum()
                denom = pred_bin[b, t, k].sum() + gt_mask[b, t, k].sum() + 1e-6
                dices[k].append(float(2 * inter / denom))
    return dices


def _compute_warp_dices(pred_mask_warp, gt_mask, valid):
    B, T, K = pred_mask_warp.shape[:3]; dices = [[] for _ in range(K)]
    pw = pred_mask_warp.clamp(0, 1)
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if not valid[b, t, k]: continue
                inter = (pw[b, t, k] * gt_mask[b, t, k]).sum()
                denom = pw[b, t, k].sum() + gt_mask[b, t, k].sum() + 1e-6
                dices[k].append(float(2 * inter / denom))
    return dices


def _compute_dice_hard(pred, gt):
    """Dice coefficient with hard threshold (>0.5)."""
    pb = (pred > 0.5).float(); gb = gt.float()
    inter = (pb * gb).sum(); denom = pb.sum() + gb.sum() + 1e-6
    return float(2 * inter / denom)


def _compute_dice_soft(pred, gt):
    """Soft Dice coefficient (bilinear values)."""
    p = pred.clamp(0, 1); g = gt.clamp(0, 1)
    inter = (p * g).sum(); denom = p.sum() + g.sum() + 1e-6
    return float(2 * inter / denom)


def _compute_iou_hard(pred, gt):
    """IoU with hard threshold."""
    pb = (pred > 0.5).float(); gb = gt.float()
    inter = (pb * gb).sum(); union = pb.sum() + gb.sum() - inter + 1e-6
    return float(inter / union)


def _per_slot_mask_dice_soft(pred, gt, valid, B, T, K):
    """Soft Dice per slot, returns list of K lists."""
    dices = [[] for _ in range(K)]
    p = pred.clamp(0, 1)
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if not valid[b, t, k]: continue
                inter = (p[b, t, k] * gt[b, t, k].clamp(0, 1)).sum()
                denom = p[b, t, k].sum() + gt[b, t, k].clamp(0, 1).sum() + 1e-6
                dices[k].append(float(2 * inter / denom))
    return dices


def _per_slot_mask_dice_hard(pred, gt, valid, B, T, K):
    """Hard Dice (>0.5) per slot."""
    dices = [[] for _ in range(K)]
    pb = (pred > 0.5).float(); gb = gt.float()
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if not valid[b, t, k]: continue
                inter = (pb[b, t, k] * gb[b, t, k]).sum()
                denom = pb[b, t, k].sum() + gb[b, t, k].sum() + 1e-6
                dices[k].append(float(2 * inter / denom))
    return dices


def _per_slot_dice_flat(pred, gt, valid, B, T, K, min_area=1.0):
    """Flat list of per-slot soft Dice values, filtering empty masks."""
    vals = []
    p = pred.clamp(0, 1); g = gt.clamp(0, 1)
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if not valid[b, t, k]: continue
                if g[b, t, k].sum() < min_area: continue  # skip empty masks
                inter = (p[b, t, k] * g[b, t, k]).sum()
                denom = p[b, t, k].sum() + g[b, t, k].sum() + 1e-6
                vals.append(float(2 * inter / denom))
    return vals


def _per_slot_dice_flat_hard(pred, gt, valid, B, T, K, min_area=1.0):
    """Flat list of hard Dice values, filtering empty masks."""
    vals = []
    pb = (pred > 0.5).float(); gb = gt.float()
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if not valid[b, t, k]: continue
                if gb[b, t, k].sum() < min_area: continue
                inter = (pb[b, t, k] * gb[b, t, k]).sum()
                denom = pb[b, t, k].sum() + gb[b, t, k].sum() + 1e-6
                vals.append(float(2 * inter / denom))
    return vals


def _per_slot_iou_hard(pred, gt, valid, B, T, K, min_area=1.0):
    """Flat list of hard IoU values, filtering empty masks."""
    vals = []
    pb = (pred > 0.5).float(); gb = gt.float()
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if not valid[b, t, k]: continue
                if gb[b, t, k].sum() < min_area: continue
                inter = (pb[b, t, k] * gb[b, t, k]).sum()
                union = pb[b, t, k].sum() + gb[b, t, k].sum() - inter + 1e-6
                vals.append(float(inter / union))
    return vals


def _classify_delta(dcx, dcy):
    ax, ay = abs(dcx), abs(dcy)
    if ax < 0.005 and ay < 0.005: return 0
    return 1 if (ay > ax and dcy < 0) else (2 if (ay > ax and dcy > 0) else (3 if dcx < 0 else 4))


def is_center_safe(bbox, action, margin=MARGIN, image_size=IMAGE_SIZE):
    cx = bbox[0] * image_size; cy = bbox[1] * image_size
    dx, dy = ACTION_DELTA.get(int(action), (0, 0))
    nx = cx + dx * STEP_SIZE; ny = cy + dy * STEP_SIZE
    return margin <= nx <= image_size - margin and margin <= ny <= image_size - margin


def apply_action_to_bbox(bbox, action, step=STEP_SIZE, image_size=IMAGE_SIZE, clip=True):
    dx, dy = ACTION_DELTA.get(int(action), (0, 0))
    new_cx = (bbox[0] * image_size + dx * step) / image_size
    new_cy = (bbox[1] * image_size + dy * step) / image_size
    if clip:
        hw = bbox[2] * image_size / 2; hh = bbox[3] * image_size / 2
        new_cx = max(hw / image_size, min(1 - hw / image_size, new_cx))
        new_cy = max(hh / image_size, min(1 - hh / image_size, new_cy))
    return torch.tensor([new_cx, new_cy, bbox[2], bbox[3]], device=bbox.device)


# ============================== Visualization ==============================

def _save_ablation_bar(results, out_path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    modes = ["normal", "z_zero", "z_shuffle", "z_shuffle_actor"]
    labels = ["Normal", "z=0", "z=shuffle", "z=shuffle(act)"]
    x = np.arange(len(modes)); w = 0.6

    configs = [
        ("bbox_iou", "Box IoU"),
        ("mask_dice_head", "Mask Dice (Head)"),
        ("mask_dice_warp", "Mask Dice (Warp)"),
        ("obj_psnr", "Obj PSNR"),
    ]
    for ax, (key, ylab) in zip(axes, configs):
        vals = [results.get(f"{m}_{key}", 0) for m in modes]
        colors = ["#4CAF50", "#FF9800", "#F44336", "#9C27B0"][:len(vals)]
        ax.bar(x, vals, w, color=colors)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=7)
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
        for i, v in enumerate(vals):
            ax.text(i, max(0, v) + 0.01, f"{v:.3f}", ha="center", fontsize=6)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _save_umap(z, labels, title, out_path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        import umap
        coords = umap.UMAP(n_components=2, random_state=42,
                           n_neighbors=min(30, max(2, len(z) // 10))).fit_transform(z)
    except:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(z)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=6, alpha=0.75,
               cmap="tab20", edgecolors="none")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _save_recon_panel(videos, recon_n, recon_z, out_path, n_samples=2):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    B = min(n_samples, videos.shape[0]); T1 = recon_n.shape[1]
    n_cols, rows = 5, B * T1
    fig, axes = plt.subplots(rows, n_cols, figsize=(n_cols * 2.5, rows * 2.5))
    if rows == 1: axes = axes[np.newaxis, :]
    for b in range(B):
        for t in range(T1):
            row = b * T1 + t
            it = videos[b, t].permute(1, 2, 0).cpu().clamp(0, 1)
            gt = videos[b, t + 1].permute(1, 2, 0).cpu().clamp(0, 1)
            rn = recon_n[b, t].permute(1, 2, 0).cpu().clamp(0, 1)
            rz = recon_z[b, t].permute(1, 2, 0).cpu().clamp(0, 1)
            er = (gt - rn).abs().mean(dim=-1, keepdim=True)
            er = (er / max(er.max().item(), 1e-6)).expand(-1, -1, 3)
            for c, (img, ti) in enumerate([(it, "I_t"), (gt, "GT"), (it, "Copy"),
                                           (rn, "Recon(n)"), (er, "Error")]):
                axes[row, c].imshow(img.numpy()); axes[row, c].set_title(ti, fontsize=7)
                axes[row, c].axis("off")
    fig.tight_layout(pad=0.3); fig.savefig(out_path, dpi=100); plt.close(fig)


def _save_mask_warp_panel(model, device, batch, out_path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    with torch.no_grad():
        video = batch["video"]
        masks = batch["masks"]
        boxes = batch["boxes"]
        valid = batch["valid"]
        vm = batch.get("visible_masks", masks)
        B, T, C, H, W = video.shape
        K = boxes.shape[2]

        out_n = model.forward_ablation(batch, z_mode="normal")
        out_z = model.forward_ablation(batch, z_mode="zero")
        out_s = model.forward_ablation(batch, z_mode="shuffle")

        mask_t = masks[:, :-1]  # (B, T-1, K, H, W)
        gt_mask_tp1 = masks[:, 1:]
        bbox_t = boxes[:, :-1]
        valid_tp1 = valid[:, 1:]
        gt_mask_lr = F.adaptive_avg_pool2d(
            gt_mask_tp1.reshape(B * (T - 1) * K, 1, H, W), (16, 16),
        ).reshape(B, T - 1, K, 1, 16, 16)
        mask_t_lr = F.adaptive_avg_pool2d(
            mask_t.reshape(B * (T - 1) * K, 1, H, W), (16, 16),
        ).reshape(B, T - 1, K, 1, 16, 16)

        def _warp(out, tb):
            B, T1, K, _, g, _ = mask_t_lr.shape
            return warp_mask_by_bbox(
                mask_t_lr.reshape(B * T1, K, 1, g, g),
                bbox_t.reshape(B * T1, K, 4),
                out["pred_struct"]["bbox"].reshape(B * T1, K, 4),
                out_size=16,
            ).reshape(B, T1, K, 1, g, g)

        warp_n = _warp(out_n, mask_t_lr); warp_z = _warp(out_z, mask_t_lr); warp_s = _warp(out_s, mask_t_lr)

    s_idx, t_idx = 0, 0
    n_cols, n_rows = 6, min(3, K)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1: axes = axes[np.newaxis, :]

    for k in range(n_rows):
        if not valid_tp1[s_idx, t_idx, k]: continue
        mt = mask_t_lr[s_idx, t_idx, k, 0].cpu().numpy()
        gt = gt_mask_lr[s_idx, t_idx, k, 0].cpu().numpy()
        ph = out_n["pred_struct"]["mask_low"][s_idx, t_idx, k, 0].sigmoid().cpu().numpy()
        wn = warp_n[s_idx, t_idx, k, 0].cpu().clamp(0, 1).numpy()
        wz = warp_z[s_idx, t_idx, k, 0].cpu().clamp(0, 1).numpy()
        ws = warp_s[s_idx, t_idx, k, 0].cpu().clamp(0, 1).numpy()

        for c, (img, ti) in enumerate([
            (mt, "mask_t"), (gt, "GT t+1"), (ph, "Head(n)"),
            (wn, "Warp(n)"), (wz, "Warp(z0)"), (ws, "Warp(sh)"),
        ]):
            axes[k, c].imshow(img, vmin=0, vmax=1, cmap="Blues")
            axes[k, c].set_title(ti, fontsize=6); axes[k, c].axis("off")

    fig.tight_layout(pad=0.3); fig.savefig(out_path, dpi=120); plt.close(fig);


def _save_swap_panel(model, batch_a, batch_b, device, out_path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    def _compute(batch):
        video = batch["video"]; masks = batch["masks"]; boxes = batch["boxes"]
        valid = batch["valid"]; vm = batch.get("visible_masks", masks)
        B, T, C, H, W = video.shape
        raw, _ = model.structure_extractor(boxes, masks, vm, valid)
        s = model.structure_encoder(raw, valid)
        z, _, _ = model.idm(s[:, :-1], s[:, 1:], valid[:, :-1])
        return s, z, valid, video[:, 0]

    with torch.no_grad():
        s_a, z_a, valid_a, v0_a = _compute(batch_a)
        s_b, z_b, valid_b, v0_b = _compute(batch_b)
        s_t_a, stp1_a = s_a[:, :-1], s_a[:, 1:]
        v_t_a, v_tp1_a = valid_a[:, :-1], valid_a[:, 1:]
        s_t_b, v_t_b = s_b[:, :-1], valid_b[:, :-1]

        def _recon(v0, st, z, vt, vtp1):
            sh = model.fdm(st, z, vt)
            pred = model.structure_head(sh)
            return model.renderer(v0, pred["bbox"], pred["mask_low"], vtp1), pred["bbox"]

        rn_a, _ = _recon(v0_a, s_t_a, z_a, v_t_a, v_tp1_a)
        rz_a, _ = _recon(v0_a, s_t_a, torch.zeros_like(z_a), v_t_a, v_tp1_a)
        rs_a, pb_a = _recon(v0_a, s_t_a, z_b, v_t_a, v_tp1_a)

        rn_b, _ = _recon(v0_b, s_t_b, z_b, v_t_b, valid_b[:, 1:])
        rz_b, _ = _recon(v0_b, s_t_b, torch.zeros_like(z_b), v_t_b, valid_b[:, 1:])
        rs_b, pb_b = _recon(v0_b, s_t_b, z_a, v_t_b, valid_b[:, 1:])

    B = min(2, rn_a.shape[0]); T1 = rn_a.shape[1]; n_cols = 6
    n_rows = B * T1 * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1: axes = axes[np.newaxis, :]

    for b_idx in range(B):
        for t in range(T1):
            row_a = (b_idx * T1 + t) * 2; row_b = row_a + 1
            for row, (name, v, rn, rz, rs, donor_acts), swap_in in [
                (row_a, ("A", batch_a["video"], rn_a, rz_a, rs_a, batch_b["actions"].cpu().numpy()), True),
                (row_b, ("B", batch_b["video"], rn_b, rz_b, rs_b, batch_a["actions"].cpu().numpy()), True),
            ]:
                t_safe = min(t, donor_acts.shape[1] - 1)
                donor_act = int(donor_acts[b_idx, t_safe, 0])
                it = v[b_idx, t].permute(1, 2, 0).cpu().clamp(0, 1)
                gt = v[b_idx, t + 1].permute(1, 2, 0).cpu().clamp(0, 1)
                src_boxes = batch_a["boxes"] if name == "A" else batch_b["boxes"]
                pred_dcx = (rs[b_idx, t, 0, 0] - src_boxes[b_idx, t, 0, 0]).item()
                pred_dcy = (rs[b_idx, t, 0, 1] - src_boxes[b_idx, t, 0, 1]).item()
                pred_act = _classify_delta(pred_dcx, pred_dcy)
                safe_str = "CS" if is_center_safe(src_boxes[b_idx, t, 0], donor_act) else "edge"

                panels = [
                    (it, f"{name}: I_t"),
                    (gt, "GT t+1"),
                    (rn[b_idx, t].permute(1, 2, 0).cpu().clamp(0, 1), "Normal"),
                    (rz[b_idx, t].permute(1, 2, 0).cpu().clamp(0, 1), "z=0"),
                    (rs[b_idx, t].permute(1, 2, 0).cpu().clamp(0, 1), f"swap={donor_act}|{safe_str}"),
                    (torch.zeros_like(it), f"pred={pred_act} donor={donor_act}"),
                ]
                for c, (img, ti) in enumerate(panels):
                    axes[row, c].imshow(img.numpy()); axes[row, c].set_title(ti, fontsize=5)
                    axes[row, c].axis("off")

    fig.tight_layout(pad=0.3); fig.savefig(out_path, dpi=100); plt.close(fig)


def _save_per_action_mask_dice(results, out_path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    per_action = results.get("oracle_per_action", {})
    if not per_action: return
    acts = sorted(per_action.keys(), key=lambda x: int(x))
    labels = [ACTION_NAMES[int(a)] for a in acts]
    x = np.arange(len(acts)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    vals_gt = [per_action[a].get("mask_gt_warp_dice_h", 0) for a in acts]
    vals_n = [per_action[a].get("mask_pred_n_dice_h", 0) for a in acts]
    vals_z = [per_action[a].get("mask_pred_z_dice_h", 0) for a in acts]
    ax.bar(x - w, vals_gt, w, label="GT warp", color="#4CAF50")
    ax.bar(x, vals_n, w, label="Pred normal", color="#2196F3")
    ax.bar(x + w, vals_z, w, label="Pred z=0", color="#FF9800")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mask Dice (hard)"); ax.set_title("Per-Action Mask Dice")
    ax.legend(fontsize=8)
    for i in range(len(acts)):
        ax.text(i - w, vals_gt[i] + 0.01, f"{vals_gt[i]:.2f}", ha="center", fontsize=6)
        ax.text(i, vals_n[i] + 0.01, f"{vals_n[i]:.2f}", ha="center", fontsize=6)
        ax.text(i + w, vals_z[i] + 0.01, f"{vals_z[i]:.2f}", ha="center", fontsize=6)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _save_mask_gap_breakdown(results, out_path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    filters = [("oracle_all", "All"), ("oracle_ns", "Non-stay"), ("oracle_cs", "Center-safe"),
               ("oracle_cs_ns", "CS+NS")]
    x = np.arange(len(filters)); w = 0.3
    vals_n, vals_z = [], []
    valid_filters = []
    for fk, fn in filters:
        if fk in results:
            vals_n.append(results[fk].get("mask_pred_n_dice_h", 0))
            vals_z.append(results[fk].get("mask_pred_z_dice_h", 0))
            valid_filters.append(fn)
    if not valid_filters: return
    x_valid = np.arange(len(valid_filters))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_valid - w/2, vals_n, w, label="Pred normal", color="#2196F3")
    ax.bar(x_valid + w/2, vals_z, w, label="Pred z=0", color="#FF9800")
    ax.set_xticks(x_valid); ax.set_xticklabels(valid_filters)
    ax.set_ylabel("Mask Dice (hard)"); ax.set_title("Mask Gap Breakdown")
    ax.legend(fontsize=8)
    for i in range(len(valid_filters)):
        ax.text(i - w/2, vals_n[i] + 0.01, f"{vals_n[i]:.2f}", ha="center", fontsize=6)
        ax.text(i + w/2, vals_z[i] + 0.01, f"{vals_z[i]:.2f}", ha="center", fontsize=6)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _save_mask_oracle_panel(batch, ab, device, model, out_path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from lam.modules.v14_mask_warp import warp_mask_by_bbox

    video = batch["video"]; masks = batch["masks"]; boxes = batch["boxes"]
    valid = batch["valid"]; B, T, C, H, W = video.shape; K = boxes.shape[2]

    mask_t = masks[:, :-1]; gt_mask_tp1 = masks[:, 1:]
    bbox_t = boxes[:, :-1]; gt_bbox_tp1 = boxes[:, 1:]
    valid_tp1 = valid[:, 1:]
    B_T1 = B * (T - 1)
    mask_t_flat = mask_t.reshape(B_T1, K, 1, H, W)
    bbox_t_flat = bbox_t.reshape(B_T1, K, 4)

    identity = warp_mask_by_bbox(mask_t_flat, bbox_t_flat, bbox_t_flat, out_size=H).reshape(B, T - 1, K, 1, H, W)
    gt_warp = warp_mask_by_bbox(mask_t_flat, bbox_t_flat, gt_bbox_tp1.reshape(B_T1, K, 4), out_size=H).reshape(B, T - 1, K, 1, H, W)
    pred_nm = warp_mask_by_bbox(mask_t_flat, bbox_t_flat, ab["n"]["pred_struct"]["bbox"].reshape(B_T1, K, 4), out_size=H).reshape(B, T - 1, K, 1, H, W)
    pred_zm = warp_mask_by_bbox(mask_t_flat, bbox_t_flat, ab["z"]["pred_struct"]["bbox"].reshape(B_T1, K, 4), out_size=H).reshape(B, T - 1, K, 1, H, W)
    pred_sm = warp_mask_by_bbox(mask_t_flat, bbox_t_flat, ab["s"]["pred_struct"]["bbox"].reshape(B_T1, K, 4), out_size=H).reshape(B, T - 1, K, 1, H, W)

    n_cols, n_rows = 8, min(3, K)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 2.3))
    if n_rows == 1: axes = axes[np.newaxis, :]
    s_idx, t_idx = 0, 0

    for k in range(n_rows):
        if not valid_tp1[s_idx, t_idx, k]: continue
        rgb = video[s_idx, t_idx].permute(1, 2, 0).cpu().clamp(0, 1)
        mt = mask_t[s_idx, t_idx, k].cpu().numpy()
        gm = gt_mask_tp1[s_idx, t_idx, k].cpu().numpy()
        mi = identity[s_idx, t_idx, k, 0].cpu().clamp(0, 1).numpy()
        gw = gt_warp[s_idx, t_idx, k, 0].cpu().clamp(0, 1).numpy()
        pn = pred_nm[s_idx, t_idx, k, 0].cpu().clamp(0, 1).numpy()
        pz = pred_zm[s_idx, t_idx, k, 0].cpu().clamp(0, 1).numpy()
        ps = pred_sm[s_idx, t_idx, k, 0].cpu().clamp(0, 1).numpy()

        for c, (img, ti) in enumerate([
            (_to_rgb(rgb), "RGB_t"), (mt, "mask_t"), (gm, "GT t+1"),
            (mi, "Identity"), (gw, "GT warp"), (pn, "Pred(n)"), (pz, "Pred(z)"), (ps, "Pred(s)"),
        ]):
            cmap = None if img.ndim == 3 else "Blues"; vrange = None if img.ndim == 3 else (0, 1)
            axes[k, c].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[k, c].set_title(ti, fontsize=5); axes[k, c].axis("off")

    fig.tight_layout(pad=0.2); fig.savefig(out_path, dpi=120); plt.close(fig)


def _to_rgb(x): return x.cpu().clamp(0, 1).numpy() if hasattr(x, 'numpy') else x


# ============================== Main ==============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=15)
    parser.add_argument("--max_latent", type=int, default=3000)
    parser.add_argument("--mask_eval_mode", default="warp", choices=["head", "warp"])
    parser.add_argument("--mask_oracle", action="store_true",
                        help="Enable identity/GT warp oracle diagnostics")
    parser.add_argument("--output_dir", default=None, help="Override eval output dir")
    parser.add_argument("--num_batches", type=int, default=None,
                        help="Override max_batches (alias)")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ROOT = os.path.join(os.path.dirname(__file__), "../../..")
    val_dir = os.path.join(ROOT, "data", "bridgebench", "bridge1", "val")
    # Output dir: --output_dir > --out_dir > default, with oracle suffix if applicable.
    out_dir = (args.output_dir or args.out_dir or
               os.path.join(ROOT, "result", "v14", "bridge1",
                            "eval_v14_2" if args.mask_oracle else "eval"))
    if args.num_batches is not None:
        args.max_batches = args.num_batches
    _ensure_dir(out_dir)

    print(f"\n{'='*60}\nV14.1 Eval  ckpt={args.checkpoint}  mask_mode={args.mask_eval_mode}\n{'='*60}")

    model = V14Model(image_size=128, max_actors=4).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    files = sorted([os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".pt")])

    # === Accumulators ===
    acc = {}
    for m in ["n", "z", "s", "sa"]:
        for t in ["iou", "dice_head", "dice_warp"]:
            acc[f"{m}_{t}"] = []
    for m in ["n", "z"]:
        for t in ["psnr", "obj_psnr"]:
            acc[f"{m}_{t}"] = []
    acc["copy_psnr"] = []
    # Oracle accumulators (V14.2).
    oracle = {} if args.mask_oracle else None
    if oracle is not None:
        for t in ["identity_dice_s", "identity_dice_h", "identity_iou_h",
                  "gt_warp_dice_s", "gt_warp_dice_h", "gt_warp_iou_h"]:
            oracle[t] = []
        oracle["pred_warp_normal_dice_s"] = []
        oracle["pred_warp_normal_dice_h"] = []
        oracle["pred_warp_zero_dice_s"] = []
        oracle["pred_warp_zero_dice_h"] = []
        oracle["pred_warp_shuffle_dice_s"] = []
        oracle["pred_warp_shuffle_dice_h"] = []
        # Per-action.
        oracle["per_action"] = {}
        for a in range(5):
            oracle["per_action"][a] = {
                "bbox_n": [], "bbox_z": [], "bbox_s": [],
                "mask_gt_warp_dice_s": [], "mask_pred_n_dice_s": [],
                "mask_pred_z_dice_s": [], "mask_pred_s_dice_s": [],
                "mask_gt_warp_dice_h": [], "mask_pred_n_dice_h": [],
                "mask_pred_z_dice_h": [], "mask_pred_s_dice_h": [],
                "count": 0,
            }
        # Center-safe accumulators.
        for filt in ["all", "cs", "ns", "cs_ns"]:
            oracle[filt] = {
                "mask_gt_warp_dice_s": [], "mask_gt_warp_dice_h": [],
                "mask_pred_n_dice_s": [], "mask_pred_n_dice_h": [],
                "mask_pred_z_dice_s": [], "mask_pred_z_dice_h": [],
                "bbox_n": [], "bbox_z": [],
            }
    all_z, all_action, all_actor, all_cat = [], [], [], []
    swap_vals = {k: [] for k in [
        "acc_all", "acc_cs", "acc_ns", "acc_cs_ns",
        "eff_acc", "eff_iou", "eff_l1", "eff_acc_cs",
    ]}
    swap_count = 0
    first_batch, first_n, first_z, first_ab = None, None, None, None

    n_batches = min(args.max_batches, len(files) // args.batch_size)

    for bi in range(n_batches):
        start = bi * args.batch_size
        batch_files = files[start:start + args.batch_size]
        if len(batch_files) < args.batch_size: break

        batch_list = [torch.load(f, map_location="cpu", weights_only=False) for f in batch_files]
        batch = _collate(batch_list)
        for k in list(batch.keys()):
            if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(device)
        video = batch["video"]; valid = batch["valid"]
        B, T, C, H, W = video.shape; K = batch["boxes"].shape[2]

        # --- Ablation passes ---
        ab = {}
        for mode, z_mode in [("n", "normal"), ("z", "zero"), ("s", "shuffle")]:
            ab[mode] = model.forward_ablation(batch, z_mode=z_mode)

        # Actor-shuffle.
        out_sa = model.forward_ablation(batch, z_mode="normal")
        z_sa = out_sa["z"].clone()
        for k in range(K):
            perm = torch.randperm(B, device=device)
            if B > 1: z_sa[:, :, k, :] = z_sa[perm][:, :, k, :]
        s_hat_sa = model.fdm(out_sa["s"][:, :-1], z_sa, valid[:, :-1])
        ab["sa"] = {"pred_struct": model.structure_head(s_hat_sa),
                     "recon": model.renderer(video[:, 0], model.structure_head(s_hat_sa)["bbox"],
                                             model.structure_head(s_hat_sa)["mask_low"], valid[:, 1:]),
                     "z": z_sa, "s_hat": s_hat_sa}

        gt_bbox = batch["boxes"][:, 1:]
        valid_tp1 = valid[:, 1:]
        mask_t = batch["masks"][:, :-1]
        gt_mask_tp1 = batch["masks"][:, 1:]

        # GT mask low-res (16x16).
        gt_mask_lr = F.adaptive_avg_pool2d(
            gt_mask_tp1.reshape(B * (T - 1) * K, 1, H, W), (16, 16),
        ).reshape(B, T - 1, K, 1, 16, 16)
        gt_mask_lr = (gt_mask_lr > 0.0).float()

        bbox_t = batch["boxes"][:, :-1]

        for mode in ["n", "z", "s", "sa"]:
            out = ab[mode]
            pred_struct = out["pred_struct"]
            # Bbox IoU.
            ious = _compute_ious(pred_struct["bbox"], gt_bbox, valid_tp1)
            acc[f"{mode}_iou"].extend(sum(ious, []))
            # Mask head Dice.
            dices_h = _compute_mask_dices(pred_struct["mask_low"], gt_mask_lr, valid_tp1)
            acc[f"{mode}_dice_head"].extend(sum(dices_h, []))
            # Mask warp Dice — full resolution (128x128).
            mask_warp_full = warp_mask_by_bbox(
                mask_t.reshape(B * (T - 1), K, 1, H, W),
                bbox_t.reshape(B * (T - 1), K, 4),
                pred_struct["bbox"].reshape(B * (T - 1), K, 4),
                out_size=H,
            ).reshape(B, T - 1, K, 1, H, W)
            # Full-res mask warp Dice.
            dices_wf = _compute_warp_dices(mask_warp_full, gt_mask_tp1.unsqueeze(-3), valid_tp1)
            acc[f"{mode}_dice_warp"].extend(sum(dices_wf, []))

        # --- Oracle diagnostics (V14.2) ---
        if oracle is not None:
            B_T1 = B * (T - 1)
            mask_t_flat = mask_t.reshape(B_T1, K, 1, H, W)
            bbox_t_flat = bbox_t.reshape(B_T1, K, 4)
            gt_mask_5d = gt_mask_tp1.unsqueeze(-3)  # (B, T-1, K, 1, H, W)

            # Identity warp: warp(mask_t, bbox_t, bbox_t) vs mask_t.
            mask_identity = warp_mask_by_bbox(mask_t_flat, bbox_t_flat, bbox_t_flat,
                                              out_size=H).reshape(B, T - 1, K, 1, H, W)
            oracle["identity_dice_s"].extend(_per_slot_dice_flat(mask_identity, mask_t.unsqueeze(-3), valid_tp1, B, T - 1, K))
            oracle["identity_dice_h"].extend(_per_slot_dice_flat_hard(mask_identity, mask_t.unsqueeze(-3), valid_tp1, B, T - 1, K))
            identity_iou = _per_slot_iou_hard(mask_identity, mask_t.unsqueeze(-3), valid_tp1, B, T - 1, K)

            # GT bbox warp oracle: warp(mask_t, bbox_t, gt_bbox_{t+1}) vs gt_mask_{t+1}.
            gt_warp = warp_mask_by_bbox(mask_t_flat, bbox_t_flat,
                                        gt_bbox.reshape(B_T1, K, 4),
                                        out_size=H).reshape(B, T - 1, K, 1, H, W)
            oracle["gt_warp_dice_s"].extend(_per_slot_dice_flat(gt_warp, gt_mask_5d, valid_tp1, B, T - 1, K))
            oracle["gt_warp_dice_h"].extend(_per_slot_dice_flat_hard(gt_warp, gt_mask_5d, valid_tp1, B, T - 1, K))
            oracle["gt_warp_iou_h"].extend(_per_slot_iou_hard(gt_warp, gt_mask_5d, valid_tp1, B, T - 1, K))

            # Pred warp normal/zero/shuffle.
            for mode_key, out in [("normal", ab["n"]), ("zero", ab["z"]), ("shuffle", ab["s"])]:
                pw = warp_mask_by_bbox(mask_t_flat, bbox_t_flat,
                                       out["pred_struct"]["bbox"].reshape(B_T1, K, 4),
                                       out_size=H).reshape(B, T - 1, K, 1, H, W)
                oracle[f"pred_warp_{mode_key}_dice_s"].extend(
                    _per_slot_dice_flat(pw, gt_mask_5d, valid_tp1, B, T - 1, K))
                oracle[f"pred_warp_{mode_key}_dice_h"].extend(
                    _per_slot_dice_flat_hard(pw, gt_mask_5d, valid_tp1, B, T - 1, K))

            # Per-action and center-safe breakdown.
            actions_batch = batch["actions"].cpu().numpy()
            for b in range(B):
                for t in range(T - 1):
                    for k in range(K):
                        if not valid_tp1[b, t, k]: continue
                        act = int(actions_batch[b, t, k])
                        if act < 0 or act >= 5: continue
                        pa = oracle["per_action"][act]
                        pa["count"] += 1
                        # Bbox normal/zero.
                        bbox_n = ab["n"]["pred_struct"]["bbox"][b, t, k]
                        bbox_z = ab["z"]["pred_struct"]["bbox"][b, t, k]
                        bbox_s = ab["s"]["pred_struct"]["bbox"][b, t, k]
                        gb = gt_bbox[b, t, k]
                        for key, pb in [("bbox_n", bbox_n), ("bbox_z", bbox_z), ("bbox_s", bbox_s)]:
                            pcx, pcy, pw_, ph_ = pb.tolist()
                            gcx, gcy, gw_, gh_ = gb.tolist()
                            px1 = pcx - pw_/2; py1 = pcy - ph_/2; px2 = pcx + pw_/2; py2 = pcy + ph_/2
                            gx1 = gcx - gw_/2; gy1 = gcy - gh_/2; gx2 = gcx + gw_/2; gy2 = gcy + gh_/2
                            ix1 = max(px1, gx1); iy1 = max(py1, gy1); ix2 = min(px2, gx2); iy2 = min(py2, gy2)
                            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            ap = max(0, px2 - px1) * max(0, py2 - py1)
                            ag = max(0, gx2 - gx1) * max(0, gy2 - gy1)
                            pa[key].append(float(inter / (ap + ag - inter + 1e-6)))
                        # Mask GT warp.
                        pa["mask_gt_warp_dice_s"].append(_compute_dice_soft(
                            gt_warp[b, t, k].unsqueeze(0), gt_mask_5d[b, t, k].unsqueeze(0)))
                        pa["mask_gt_warp_dice_h"].append(_compute_dice_hard(
                            gt_warp[b, t, k].unsqueeze(0), gt_mask_5d[b, t, k].unsqueeze(0)))
                        # Pred warp.
                        for suffix, mw in [("n", mask_warp_full), ("z", warp_mask_by_bbox(
                                mask_t_flat, bbox_t_flat, ab["z"]["pred_struct"]["bbox"].reshape(B_T1, K, 4),
                                out_size=H).reshape(B, T - 1, K, 1, H, W)),
                                           ("s", warp_mask_by_bbox(
                                mask_t_flat, bbox_t_flat, ab["s"]["pred_struct"]["bbox"].reshape(B_T1, K, 4),
                                out_size=H).reshape(B, T - 1, K, 1, H, W))]:
                            ms = mw[b, t, k].unsqueeze(0); gs = gt_mask_5d[b, t, k].unsqueeze(0)
                            pa[f"mask_pred_{suffix}_dice_s"].append(_compute_dice_soft(ms, gs))
                            pa[f"mask_pred_{suffix}_dice_h"].append(_compute_dice_hard(ms, gs))

                        # Center-safe filters.
                        is_cs = is_center_safe(bbox_t[b, t, k], act)
                        is_ns = act != 0
                        for filt, cond in [("all", True), ("cs", is_cs), ("ns", is_ns),
                                           ("cs_ns", is_cs and is_ns)]:
                            if not cond: continue
                            of = oracle[filt]
                            of["mask_gt_warp_dice_s"].append(_compute_dice_soft(
                                gt_warp[b, t, k].unsqueeze(0), gt_mask_5d[b, t, k].unsqueeze(0)))
                            of["mask_gt_warp_dice_h"].append(_compute_dice_hard(
                                gt_warp[b, t, k].unsqueeze(0), gt_mask_5d[b, t, k].unsqueeze(0)))
                            for suf, mw in [("n", mask_warp_full), ("z", warp_mask_by_bbox(
                                    mask_t_flat, bbox_t_flat,
                                    ab["z"]["pred_struct"]["bbox"].reshape(B_T1, K, 4),
                                    out_size=H).reshape(B, T - 1, K, 1, H, W))]:
                                ms = mw[b, t, k].unsqueeze(0)
                                gs = gt_mask_5d[b, t, k].unsqueeze(0)
                                of[f"mask_pred_{suf}_dice_s"].append(_compute_dice_soft(ms, gs))
                                of[f"mask_pred_{suf}_dice_h"].append(_compute_dice_hard(ms, gs))
                            of["bbox_n"].append(pa["bbox_n"][-1])
                            of["bbox_z"].append(pa["bbox_z"][-1])

        # PSNR.
        if len(acc["n_psnr"]) < 50:
            target = video[:, 1:]
            for mode, out in [("n", ab["n"]), ("z", ab["z"])]:
                mse = ((out["recon"] - target) ** 2).mean().item()
                acc[f"{mode}_psnr"].append(_psnr(mse))
                pm = F.interpolate(
                    out["pred_struct"]["mask_low"].sigmoid().reshape(B * (T - 1) * K, 1, 16, 16),
                    size=(H, W), mode="bilinear", align_corners=False,
                ).reshape(B, T - 1, K, 1, H, W)
                obj_mask = pm.sum(dim=2).expand(-1, -1, C, -1, -1).clamp(0, 1)
                mse_obj = (((out["recon"] - target) ** 2) * obj_mask).sum() / obj_mask.sum().clamp(min=1.0)
                acc[f"{mode}_obj_psnr"].append(_psnr(mse_obj.item()))
            copy_mse = ((video[:, :-1] - target) ** 2).mean().item()
            acc["copy_psnr"].append(_psnr(copy_mse))

        # Latents.
        mu = ab["n"]["mu"].detach().cpu().numpy()
        sv = valid[:, :-1].cpu().numpy()
        actions = batch["actions"].cpu().numpy()
        actors = batch["actor_id"].cpu().numpy()
        cats = batch["category"].cpu().numpy()
        for b in range(B):
            for t in range(T - 1):
                for k in range(K):
                    if sv[b, t, k] and actions[b, t, k] >= 0:
                        all_z.append(mu[b, t, k]); all_action.append(int(actions[b, t, k]))
                        all_actor.append(int(actors[b, k])); all_cat.append(int(cats[b, k]))

        # --- Center-safe + counterfactual swap ---
        if B >= 2:
            s_t = ab["n"]["s"][:, :-1]
            z_n = ab["n"]["z"]
            for i in range(min(B, 8)):
                for j in range(min(B, 8)):
                    if i == j: continue
                    for t in range(T - 1):
                        for k in range(K):
                            if not valid_tp1[i, t, k] or actions[j, t, k] < 0:
                                continue
                            donor_act = int(actions[j, t, k])
                            recv_box = bbox_t[i, t, k]
                            # Swap prediction.
                            z_swap_ij = z_n.clone(); z_swap_ij[i] = z_n[j]
                            s_hat_swap = model.fdm(s_t, z_swap_ij, valid_tp1)
                            pred_swap = model.structure_head(s_hat_swap)
                            pred_box = pred_swap["bbox"][i, t, k]
                            dcx = (pred_box[0] - recv_box[0]).item()
                            dcy = (pred_box[1] - recv_box[1]).item()
                            pred_act = _classify_delta(dcx, dcy)

                            # swap_action.
                            swap_vals["acc_all"].append(1 if pred_act == donor_act else 0)
                            cs = is_center_safe(recv_box, donor_act)
                            if cs: swap_vals["acc_cs"].append(1 if pred_act == donor_act else 0)
                            if donor_act != 0:
                                swap_vals["acc_ns"].append(1 if pred_act == donor_act else 0)
                                if cs: swap_vals["acc_cs_ns"].append(1 if pred_act == donor_act else 0)

                            # Counterfactual effect.
                            target_box = apply_action_to_bbox(recv_box, donor_act)
                            # IoU between pred_box and counterfactual target.
                            px1, py1 = pred_box[0] - pred_box[2] / 2, pred_box[1] - pred_box[3] / 2
                            px2, py2 = pred_box[0] + pred_box[2] / 2, pred_box[1] + pred_box[3] / 2
                            tx1, ty1 = target_box[0] - target_box[2] / 2, target_box[1] - target_box[3] / 2
                            tx2, ty2 = target_box[0] + target_box[2] / 2, target_box[1] + target_box[3] / 2
                            ix1, iy1 = max(px1, tx1), max(py1, ty1)
                            ix2, iy2 = min(px2, tx2), min(py2, ty2)
                            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                            ap = max(0.0, px2 - px1) * max(0.0, py2 - py1)
                            at = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
                            eff_iou = float(inter / (ap + at - inter + 1e-6))
                            pred_delta_act = _classify_delta(dcx, dcy)
                            eff_act_correct = 1 if pred_delta_act == donor_act else 0
                            swap_vals["eff_acc"].append(eff_act_correct)
                            swap_vals["eff_iou"].append(eff_iou)
                            swap_vals["eff_l1"].append(float((abs(dcx) + abs(dcy))))
                            if cs: swap_vals["eff_acc_cs"].append(eff_act_correct)
                            swap_count += 1

        if first_batch is None:
            first_batch = batch; first_n = ab["n"]["recon"]; first_z = ab["z"]["recon"]
            first_ab = ab

        if bi % 5 == 0:
            ni = np.mean(acc["n_iou"][-100:]) if acc["n_iou"] else 0
            print(f"  [{bi}/{n_batches}] n_iou={ni:.3f}  n_warp={np.mean(acc['n_dice_warp'][-100:]):.3f}")

    print(f"\n  Collected: {len(acc['n_iou'])} IoUs, {len(all_z)} latents, {swap_count} swaps")

    # === Results ===
    res = {"n_samples": n_batches * args.batch_size, "n_latents": len(all_z),
           "checkpoint": args.checkpoint, "mask_eval_mode": args.mask_eval_mode}

    for mode, label in [("n", "normal"), ("z", "z_zero"), ("s", "z_shuffle"), ("sa", "z_shuffle_actor")]:
        for t in ["iou", "dice_head", "dice_warp"]:
            if acc[f"{mode}_{t}"]:
                res[f"{label}_{t}"] = float(np.mean(acc[f"{mode}_{t}"]))
    for mode, label in [("n", "normal"), ("z", "z_zero")]:
        for t in ["psnr", "obj_psnr"]:
            if acc[f"{mode}_{t}"]:
                res[f"{label}_{t}"] = float(np.mean(acc[f"{mode}_{t}"]))
    if acc["copy_psnr"]: res["copy_psnr"] = float(np.mean(acc["copy_psnr"]))

    res["bbox_gap_zero"] = res.get("normal_iou", 0) - res.get("z_zero_iou", 0)
    res["bbox_gap_shuffle"] = res.get("normal_iou", 0) - res.get("z_shuffle_iou", 0)
    res["mask_gap_head"] = res.get("normal_dice_head", 0) - res.get("z_zero_dice_head", 0)
    res["mask_gap_warp"] = res.get("normal_dice_warp", 0) - res.get("z_zero_dice_warp", 0)

    # Clustering.
    if len(all_z) >= 10 and len(np.unique(all_action)) >= 2:
        z = np.asarray(all_z); action = np.asarray(all_action)
        actor = np.asarray(all_actor); cat = np.asarray(all_cat)
        res["z_var"] = float(z.var(axis=0).mean()); res["z_std"] = float(z.std())
        try:
            from sklearn.cluster import KMeans
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import normalized_mutual_info_score
            nc = min(N_ACTIONS, len(np.unique(action)))
            pred = KMeans(n_clusters=nc, random_state=args.seed, n_init=10).fit_predict(z)
            res["overall_nmi"] = float(normalized_mutual_info_score(action, pred))
            order = np.random.RandomState(args.seed).permutation(len(z))
            nt = max(1, int(0.8 * len(z)))
            if len(z) - nt >= 1:
                for name, y in [("action", action), ("actor", actor), ("category", cat)]:
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(z[order[:nt]], y[order[:nt]])
                    res[f"{name}_probe_acc"] = float(clf.score(z[order[nt:]], y[order[nt:]]))
            per_slot = []
            for slot in np.unique(actor):
                idx = actor == slot
                if idx.sum() >= max(10, nc):
                    ps = KMeans(n_clusters=nc, random_state=args.seed, n_init=10).fit_predict(z[idx])
                    per_slot.append(float(normalized_mutual_info_score(action[idx], ps)))
            res["per_slot_nmi"] = per_slot
            res["per_slot_nmi_avg"] = float(np.mean(per_slot)) if per_slot else None
            res["conditional_nmi"] = res["per_slot_nmi_avg"]
        except Exception as e:
            res["clustering_error"] = str(e)

    # Swap.
    if swap_count > 0:
        res["swap_pairs"] = swap_count
        for k, v in swap_vals.items():
            if v: res[f"swap_{k}"] = float(np.mean(v))

    # === GO ===
    # --- Oracle results (V14.2) ---
    if oracle is not None:
        for key in ["identity_dice_s", "identity_dice_h", "identity_iou_h",
                    "gt_warp_dice_s", "gt_warp_dice_h", "gt_warp_iou_h"]:
            if oracle[key]:
                res[f"oracle_{key}"] = float(np.mean(oracle[key]))
        for mode in ["normal", "zero", "shuffle"]:
            for t in ["dice_s", "dice_h"]:
                k = f"pred_warp_{mode}_{t}"
                if oracle.get(k):
                    res[f"oracle_{k}"] = float(np.mean(oracle[k]))
        res["oracle_pred_warp_gap_zero_dice_h"] = (
            res.get("oracle_pred_warp_normal_dice_h", 0) -
            res.get("oracle_pred_warp_zero_dice_h", 0))
        # Per-action.
        res["oracle_per_action"] = {}
        for a in range(5):
            pa = oracle["per_action"][a]
            if pa["count"] == 0: continue
            res["oracle_per_action"][str(a)] = {
                "count": pa["count"],
                "bbox_iou_n": float(np.mean(pa["bbox_n"])),
                "bbox_iou_z": float(np.mean(pa["bbox_z"])),
                "bbox_gap_zero": float(np.mean(pa["bbox_n"]) - np.mean(pa["bbox_z"])),
                "mask_gt_warp_dice_s": float(np.mean(pa["mask_gt_warp_dice_s"])),
                "mask_gt_warp_dice_h": float(np.mean(pa["mask_gt_warp_dice_h"])),
                "mask_pred_n_dice_s": float(np.mean(pa["mask_pred_n_dice_s"])),
                "mask_pred_z_dice_s": float(np.mean(pa["mask_pred_z_dice_s"])),
                "mask_pred_n_dice_h": float(np.mean(pa["mask_pred_n_dice_h"])),
                "mask_pred_z_dice_h": float(np.mean(pa["mask_pred_z_dice_h"])),
                "mask_gap_pred_zero_h": float(np.mean(pa["mask_pred_n_dice_h"]) - np.mean(pa["mask_pred_z_dice_h"])),
            }
        # Center-safe.
        for filt in ["all", "cs", "ns", "cs_ns"]:
            of = oracle[filt]
            if not of["bbox_n"]: continue
            key = f"oracle_{filt}"
            res[key] = {
                "n": len(of["bbox_n"]),
                "bbox_iou_n": float(np.mean(of["bbox_n"])),
                "bbox_iou_z": float(np.mean(of["bbox_z"])),
                "mask_gt_warp_dice_h": float(np.mean(of["mask_gt_warp_dice_h"])),
                "mask_pred_n_dice_h": float(np.mean(of["mask_pred_n_dice_h"])),
                "mask_pred_z_dice_h": float(np.mean(of["mask_pred_z_dice_h"])),
                "mask_gap_pred_zero_h": float(np.mean(of["mask_pred_n_dice_h"]) - np.mean(of["mask_pred_z_dice_h"])),
            }

        # Diagnosis.
        id_h = res.get("oracle_identity_dice_h", 0)
        gt_h = res.get("oracle_gt_warp_dice_h", 0)
        pred_n_h = res.get("oracle_pred_warp_normal_dice_h", 0)
        gap_h = res.get("oracle_pred_warp_gap_zero_dice_h", 0)
        if id_h < 0.95:
            res["diagnosis_case"] = "warp_bug"
        elif gt_h < 0.85:
            res["diagnosis_case"] = "data_alignment_bug"
        elif pred_n_h < 0.5 and gt_h > 0.85:
            res["diagnosis_case"] = "model_bbox_error"
        elif pred_n_h >= 0.5 and gap_h < 0.1:
            res["diagnosis_case"] = "mask_not_z_sensitive"
        else:
            res["diagnosis_case"] = "pass"
        res["go_mask_oracle"] = gt_h > 0.85
        res["go_mask_pred"] = pred_n_h > res.get("oracle_pred_warp_zero_dice_h", 0)
        res["bridge2_ready"] = id_h > 0.95 and gt_h > 0.85

    res["go_bbox_gap"] = res.get("bbox_gap_zero", 0) > 0.1
    res["go_mask_gap"] = res.get("mask_gap_warp", 0) > 0.1
    if oracle is not None:
        res["go_mask_pred_gap"] = res.get("oracle_pred_warp_gap_zero_dice_h", 0) > 0.1
    res["go_action_probe"] = res.get("action_probe_acc", 0) > 0.8
    best_swap = max(res.get("swap_acc_ns", 0), res.get("swap_acc_cs_ns", 0),
                    res.get("swap_eff_acc", 0))
    res["go_swap"] = best_swap > 0.7
    res["go_obj_psnr"] = res.get("normal_obj_psnr", 0) > res.get("z_zero_obj_psnr", 0)
    res["go_count"] = sum([res["go_bbox_gap"], res["go_mask_gap"], res["go_action_probe"],
                           res["go_swap"], res["go_obj_psnr"]])

    print(f"\n{'='*60}\n  RESULTS  ({res['go_count']}/5 GO)\n{'='*60}")
    for k, v in res.items():
        if isinstance(v, float): print(f"  {k}: {v:.4f}")
        elif isinstance(v, bool): print(f"  {k}: {'✓' if v else '✗'}")
        elif isinstance(v, list): print(f"  {k}: {[f'{x:.4f}' for x in v]}")
    # Oracle summary.
    if oracle is not None:
        print(f"\n  ORACLE DIAGNOSTICS")
        for k in ["oracle_identity_dice_h", "oracle_gt_warp_dice_h", "oracle_pred_warp_normal_dice_h",
                  "oracle_pred_warp_zero_dice_h", "oracle_pred_warp_gap_zero_dice_h",
                  "diagnosis_case", "bridge2_ready"]:
            v = res.get(k, "N/A")
            print(f"  {k}: {v}")
        if "oracle_per_action" in res:
            print(f"\n  PER-ACTION (bbox_iou_n / mask_gt_warp_dice_h / mask_pred_n_dice_h / gap):")
            for an, ad in sorted(res["oracle_per_action"].items()):
                print(f"    {ACTION_NAMES[int(an)]:6s}: count={ad['count']:5d}  "
                      f"bbox_n={ad['bbox_iou_n']:.3f}  gt_warp_h={ad['mask_gt_warp_dice_h']:.3f}  "
                      f"pred_n_h={ad['mask_pred_n_dice_h']:.3f}  gap_h={ad['mask_gap_pred_zero_h']:.3f}")
        print(f"\n  CENTER-SAFE BREAKDOWN:")
        for fk in ["oracle_all", "oracle_cs", "oracle_ns", "oracle_cs_ns"]:
            if fk in res:
                d = res[fk]
                print(f"    {fk.replace('oracle_',''):8s}: n={d['n']:5d}  bbox_n={d['bbox_iou_n']:.3f}  "
                      f"gt_warp_h={d['mask_gt_warp_dice_h']:.3f}  pred_n_h={d['mask_pred_n_dice_h']:.3f}  "
                      f"gap_h={d['mask_gap_pred_zero_h']:.3f}")

    print(f"\n  GO: {res['go_count']}/5")
    for t, k in [("bbox gap > 0.1", "go_bbox_gap"), ("mask gap > 0.1", "go_mask_gap"),
                 ("action probe > 0.8", "go_action_probe"), ("swap > 0.7", "go_swap"),
                 ("obj psnr normal > zero", "go_obj_psnr")]:
        print(f"    {'✓' if res[k] else '✗'} {t}")

    # === Visualizations ===
    _save_ablation_bar(res, os.path.join(out_dir, "z_ablation.png"))
    print("  z_ablation.png")

    if first_batch is not None:
        _save_recon_panel(first_batch["video"].cpu(), first_n.cpu(), first_z.cpu(),
                          os.path.join(out_dir, "reconstruction_panel.png"))
        _save_mask_warp_panel(model, device, first_batch,
                              os.path.join(out_dir, "mask_warp_panel.png"))
        print("  reconstruction_panel.png, mask_warp_panel.png")

    # Oracle visualizations.
    if oracle is not None and first_batch is not None:
        try:
            _save_mask_oracle_panel(first_batch, first_ab, device, model,
                                    os.path.join(out_dir, "mask_oracle_panel.png"))
            print("  mask_oracle_panel.png")
            _save_per_action_mask_dice(res, os.path.join(out_dir, "per_action_mask_dice.png"))
            print("  per_action_mask_dice.png")
            _save_mask_gap_breakdown(res, os.path.join(out_dir, "mask_gap_breakdown.png"))
            print("  mask_gap_breakdown.png")
        except Exception as e:
            print(f"  Warning: oracle viz failed: {e}")

    if len(all_z) >= 10:
        z_arr = np.asarray(all_z)
        _save_umap(z_arr, np.asarray(all_action), "z by action", os.path.join(out_dir, "latent_umap_by_action.png"))
        _save_umap(z_arr, np.asarray(all_actor), "z by actor", os.path.join(out_dir, "latent_umap_by_actor.png"))
        _save_umap(z_arr, np.asarray(all_cat), "z by category", os.path.join(out_dir, "latent_umap_by_category.png"))
        print("  UMAP x3")
        np.savez(os.path.join(out_dir, "latents.npz"), z=z_arr, action=np.asarray(all_action),
                 actor=np.asarray(all_actor), category=np.asarray(all_cat))

    # Swap panel.
    if args.batch_size >= 16:
        try:
            f0 = batch_files[:2]; f1 = batch_files[8:10]
            ba = _collate([torch.load(f, map_location="cpu", weights_only=False) for f in f0])
            bb = _collate([torch.load(f, map_location="cpu", weights_only=False) for f in f1])
            for k in list(ba.keys()):
                if isinstance(ba[k], torch.Tensor):
                    ba[k] = ba[k].to(device); bb[k] = bb[k].to(device)
            _save_swap_panel(model, ba, bb, device, os.path.join(out_dir, "action_swap.png"))
            print("  action_swap.png")
        except Exception as e:
            print(f"  swap panel: {e}")

    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n  All saved to {out_dir}")


if __name__ == "__main__":
    main()
