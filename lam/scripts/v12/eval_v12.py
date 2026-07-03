"""
V12 comprehensive evaluation — Object-Centric Structure-Action World Model.

Implements all metrics and visualizations per reports/V12/plan.md:

1. Clustering & latent space organization:
   - Overall NMI, Per-slot NMI, Conditional NMI, ARI
   - Action Probe (linear), Actor Leakage (linear)
   - UMAP/t-SNE/PCA by actor / by action / by actor×action

2. Reconstruction & per-slot contribution:
   - Full-frame PSNR, Actor-masked PSNR, Copy baseline
   - z ablation: normal vs z=0 vs z_shuffle (structure + RGB)
   - Full-frame reconstruction panel: I_t | GT | Copy | Recon | Error
   - Per-slot reconstruction: slot-only recon (z_k kept, z_j=0 for j≠k)
   - Box IoU vs inertia baseline

3. Tracking overlay (for GridWorld/Atari)

Usage:
  PYTHONPATH=lam python lam/scripts/v12/eval_v12.py \
      --checkpoint result/v12/synthetic_minimal_nooverlap/phaseC/model.pt \
      --dataset synthetic_minimal_nooverlap --gpu 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.modules.v12_model import LatentActionModelV12
from lam.v12_dataset import V12ObjectVideoDataset

VERSION = "v12"


# ============================== Helpers ==============================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _psnr_from_mse(mse: float) -> float:
    return float(-10.0 * math.log10(max(mse, 1e-10)))


def _to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """(..., H, W, C) float [0,1] -> (..., H, W, C) uint8 [0,255]."""
    return (img_t.detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """boxes: (..., 4) xyxy. Returns IoU (...,)."""
    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


def _collate(batch: List[Dict]) -> Dict:
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], dict):
            out[k] = vals[0]
        else:
            out[k] = vals
    return out


# ============================== Latent Space ==============================

def _latent_coords(z: np.ndarray, seed: int) -> np.ndarray:
    """UMAP > t-SNE > PCA, fallback chain."""
    n = len(z)
    if n < 4:
        return z[:, :2]
    try:
        import umap
        return umap.UMAP(
            n_components=2, random_state=seed,
            n_neighbors=min(30, max(2, n // 10)),
        ).fit_transform(z)
    except Exception:
        pass
    try:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed, perplexity=min(30, max(2, n // 4))).fit_transform(z)
    except Exception:
        pass
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(z)


def _plot_latent_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    cmap: str = "tab20",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=labels, s=8, alpha=0.75, cmap=cmap, edgecolors="none",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _compute_clustering_metrics(
    z: np.ndarray,
    actor: np.ndarray,
    action: np.ndarray,
    seed: int,
) -> Dict:
    """Compute all clustering metrics."""
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    results = {}
    n_clusters = len(np.unique(action))

    # Overall NMI + ARI
    pred = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(z)
    results["overall_nmi"] = float(normalized_mutual_info_score(action, pred))
    results["overall_ari"] = float(adjusted_rand_score(action, pred))

    # Per-slot NMI (within each actor, cluster z by action)
    per_slot = []
    for slot in np.unique(actor):
        idx = actor == slot
        if idx.sum() >= max(10, n_clusters):
            ps = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(z[idx])
            per_slot.append(float(normalized_mutual_info_score(action[idx], ps)))
    results["per_slot_nmi"] = per_slot
    results["per_slot_nmi_avg"] = float(np.mean(per_slot)) if per_slot else None

    # Conditional NMI: given actor, does z still predict action?
    # (Average of per-slot NMI — same as per_slot_nmi_avg but explicitly named)
    results["conditional_nmi"] = results["per_slot_nmi_avg"]

    # Action Probe (linear classifier)
    order = np.random.RandomState(seed).permutation(len(z))
    n_train = max(1, int(0.8 * len(z)))
    if len(z) - n_train >= 1:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(z[order[:n_train]], action[order[:n_train]])
        results["action_probe_acc"] = float(clf.score(z[order[n_train:]], action[order[n_train:]]))

        # Actor Leakage
        clf_a = LogisticRegression(max_iter=1000)
        clf_a.fit(z[order[:n_train]], actor[order[:n_train]])
        results["actor_leakage_acc"] = float(clf_a.score(z[order[n_train:]], actor[order[n_train:]]))

    # z statistics
    results["z_var"] = float(z.var(axis=0).mean())
    results["z_std"] = float(z.std())

    return results


# ============================== Reconstruction Viz ==============================

def _save_recon_panel(
    videos: torch.Tensor,       # (B, T, H, W, C)
    recon: torch.Tensor,        # (B, T-1, H, W, C)
    masks: torch.Tensor,        # (B, T, K, H, W)
    out_path: str,
    n_samples: int = 2,
) -> None:
    """Full-frame reconstruction: I_t | GT I_{t+1} | Copy | Recon | Error map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B = min(n_samples, videos.shape[0])
    T1 = recon.shape[1]
    n_cols = 5
    fig, axes = plt.subplots(B * T1, n_cols, figsize=(n_cols * 2.5, B * T1 * 2.5))
    if B * T1 == 1:
        axes = axes[np.newaxis, :]

    for b in range(B):
        for t in range(T1):
            row = b * T1 + t
            i_t = videos[b, t].cpu().clamp(0, 1)
            gt = videos[b, t + 1].cpu().clamp(0, 1)
            copy = i_t
            pred = recon[b, t].cpu().clamp(0, 1)
            err = (gt - pred).abs().mean(dim=-1, keepdim=True).expand(-1, -1, 3)
            err = err / max(err.max().item(), 1e-6)

            panels = [
                (i_t, f"I_t (b{b},t{t})"),
                (gt, "GT I_{t+1}"),
                (copy, "Copy"),
                (pred, "Recon"),
                (err, "Error"),
            ]
            for c, (img, title) in enumerate(panels):
                ax = axes[row, c] if n_cols > 1 else axes[c]
                ax.imshow(img.numpy())
                ax.set_title(title, fontsize=7)
                ax.axis("off")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_per_slot_recon(
    videos: torch.Tensor,       # (B, T, H, W, C)
    masks: torch.Tensor,        # (B, T, K, H, W)
    recon_per_slot: Dict[int, torch.Tensor],  # k -> (B, T-1, H, W, C)
    out_path: str,
    sample_idx: int = 0,
    t_idx: int = 0,
) -> None:
    """Per-slot reconstruction panel: one row per slot.

    Each row: I_t + slot mask | GT slot region | Copy slot | Slot-only recon | Error
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = masks.shape[2]
    n_cols = 5
    fig, axes = plt.subplots(K, n_cols, figsize=(n_cols * 2.5, K * 2.5))
    if K == 1:
        axes = axes[np.newaxis, :]

    i_t = videos[sample_idx, t_idx].cpu().clamp(0, 1)       # (H, W, C)
    gt = videos[sample_idx, t_idx + 1].cpu().clamp(0, 1)    # (H, W, C)
    H, W = i_t.shape[:2]

    for k in range(K):
        mask_k = masks[sample_idx, t_idx + 1, k].cpu().float()  # (H, W)
        mask_3c = mask_k.unsqueeze(-1).expand(-1, -1, 3)

        # I_t with slot mask overlay
        i_t_overlay = i_t.clone()
        i_t_overlay[mask_k > 0] = i_t_overlay[mask_k > 0] * 0.5 + torch.tensor([1.0, 0.0, 0.0]) * 0.5

        # GT slot region (masked)
        gt_slot = gt * mask_3c

        # Copy slot (I_t in slot region)
        copy_slot = i_t * mask_3c

        # Slot-only recon
        if k in recon_per_slot:
            pred = recon_per_slot[k][sample_idx, t_idx].cpu().clamp(0, 1)
        else:
            pred = torch.zeros_like(i_t)
        pred_slot = pred * mask_3c

        # Error in slot region
        err = (gt - pred).abs().mean(dim=-1, keepdim=True).expand(-1, -1, 3) * mask_3c
        err = err / max(err.max().item(), 1e-6)

        panels = [
            (i_t_overlay, f"Slot {k}: I_t+mask"),
            (gt_slot, "GT slot"),
            (copy_slot, "Copy slot"),
            (pred_slot, "Slot-only recon"),
            (err, "Error slot"),
        ]
        for c, (img, title) in enumerate(panels):
            axes[k, c].imshow(img.numpy())
            axes[k, c].set_title(title, fontsize=7)
            axes[k, c].axis("off")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_z_ablation_plot(ablation_results: Dict, out_path: str) -> None:
    """Bar chart comparing normal / z=0 / z_shuffle for structure and RGB."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ["normal", "z_zero", "z_shuffle"]
    labels = ["Normal z", "z=0", "z=shuffle"]
    x = np.arange(len(modes))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Structure IoU
    ious = [ablation_results.get(f"{m}_box_iou", 0) for m in modes]
    inertia = ablation_results.get("inertia_box_iou", 0)
    bars1 = ax1.bar(x, ious, width, color=["#4CAF50", "#FF9800", "#F44336"])
    ax1.axhline(y=inertia, color="gray", linestyle="--", label=f"Inertia={inertia:.3f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Box IoU")
    ax1.set_title("Structure Prediction (z ablation)")
    ax1.legend(fontsize=8)
    for bar, v in zip(bars1, ious):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}", ha="center", fontsize=7)

    # RGB PSNR
    psnrs = [ablation_results.get(f"{m}_rgb_psnr", 0) for m in modes]
    copy_psnr = ablation_results.get("copy_psnr", 0)
    bars2 = ax2.bar(x, psnrs, width, color=["#4CAF50", "#FF9800", "#F44336"])
    ax2.axhline(y=copy_psnr, color="gray", linestyle="--", label=f"Copy={copy_psnr:.1f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("RGB PSNR (dB)")
    ax2.set_title("RGB Reconstruction (z ablation)")
    ax2.legend(fontsize=8)
    for bar, v in zip(bars2, psnrs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{v:.1f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================== z-Action Swap / Replay ==============================

def _save_action_swap_panel(
    model: LatentActionModelV12,
    batch_a: Dict, batch_b: Dict,
    device: torch.device,
    out_path: str,
) -> None:
    """z-action causal swap test.

    content_A + z_B -> A's appearance, B's motion.
    Rows: sample A, sample B.
    Cols: I_t | GT I_{t+1} | Recon(normal z) | Recon(z=0) | Recon(swapped z).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        def _get_content_and_s(batch):
            v = batch["videos"].to(device)
            m = batch["masks"].to(device)
            b = batch["bboxes"].to(device)
            vi = batch["valid_mask"].to(device)
            c_obj, c_bg = model.content_encoder(v[:, 0], m[:, 0], b[:, 0], vi[:, 0])
            raw_s, _ = model.structure_extractor(m, b, vi)
            s = model.structure_encoder(raw_s, vi)
            return c_obj, c_bg, s, vi, v, m

        c_obj_a, c_bg_a, s_a, valid_a, v_a, m_a = _get_content_and_s(batch_a)
        c_obj_b, c_bg_b, s_b, valid_b, v_b, m_b = _get_content_and_s(batch_b)

        s_t_a, s_tp1_a = s_a[:, :-1], s_a[:, 1:]
        valid_t_a, valid_tp1_a = valid_a[:, :-1], valid_a[:, 1:]

        s_t_b, s_tp1_b = s_b[:, :-1], s_b[:, 1:]
        valid_t_b, valid_tp1_b = valid_b[:, :-1], valid_b[:, 1:]

        if model.use_z and model.idm is not None:
            z_a, _, _ = model.idm(s_t_a, s_tp1_a, valid_t_a)
            z_b, _, _ = model.idm(s_t_b, s_tp1_b, valid_t_b)
        else:
            Bz, Tz, Kz, Dz = s_t_a.shape
            z_a = torch.zeros(Bz, Tz, Kz, Dz, device=device)
            z_b = z_a

        z_zero = torch.zeros_like(z_a)

        def _decode(c_obj, c_bg, s_t, z, valid_t, valid_tp1):
            s_hat = model.fdm(s_t, z, valid_t)
            return model.decoder(c_obj, c_bg, s_hat, valid_tp1)

        recon_a_normal = _decode(c_obj_a, c_bg_a, s_t_a, z_a, valid_t_a, valid_tp1_a)
        recon_a_zero   = _decode(c_obj_a, c_bg_a, s_t_a, z_zero, valid_t_a, valid_tp1_a)
        recon_a_swap   = _decode(c_obj_a, c_bg_a, s_t_a, z_b, valid_t_a, valid_tp1_a)

        recon_b_normal = _decode(c_obj_b, c_bg_b, s_t_b, z_b, valid_t_b, valid_tp1_b)
        recon_b_zero   = _decode(c_obj_b, c_bg_b, s_t_b, z_zero, valid_t_b, valid_tp1_b)
        recon_b_swap   = _decode(c_obj_b, c_bg_b, s_t_b, z_a, valid_t_b, valid_tp1_b)

    B = min(2, v_a.shape[0])
    T1 = recon_a_normal.shape[1]
    n_cols = 5
    n_rows = B * T1 * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for b_idx in range(B):
        for t in range(T1):
            row_a = (b_idx * T1 + t) * 2
            row_b = row_a + 1

            for row, (name, v, recon_normal, recon_zero, recon_swap) in [
                (row_a, ("A", v_a, recon_a_normal, recon_a_zero, recon_a_swap)),
                (row_b, ("B", v_b, recon_b_normal, recon_b_zero, recon_b_swap)),
            ]:
                i_t   = v[b_idx, t].cpu().clamp(0, 1)
                gt    = v[b_idx, t + 1].cpu().clamp(0, 1)
                r_n   = recon_normal[b_idx, t].cpu().clamp(0, 1)
                r_z   = recon_zero[b_idx, t].cpu().clamp(0, 1)
                r_sw  = recon_swap[b_idx, t].cpu().clamp(0, 1)

                panels = [
                    (i_t, f"{name}: I_t"),
                    (gt, "GT I_{t+1}"),
                    (r_n, "Recon(normal)"),
                    (r_z, "Recon(z=0)"),
                    (r_sw, f"Recon(z from {'B' if name=='A' else 'A'})"),
                ]
                for c, (img, title) in enumerate(panels):
                    ax = axes[row, c]
                    ax.imshow(img.numpy())
                    ax.set_title(title, fontsize=6)
                    ax.axis("off")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved action swap panel")


# ============================== Tracking Overlay ==============================

def _draw_rect(img: np.ndarray, box: np.ndarray, color: Tuple[int, int, int], label: str = "") -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    img[y1:y1 + 2, x1:x2] = color
    img[max(y2 - 2, y1):y2, x1:x2] = color
    img[y1:y2, x1:x1 + 2] = color
    img[y1:y2, max(x2 - 2, x1):x2] = color


def _save_tracking_overlay(sample: Dict, out_path: str) -> None:
    import cv2
    colors = [(230, 50, 50), (50, 100, 230), (235, 180, 45), (150, 80, 180), (40, 170, 110)]
    frames = _to_uint8(sample["videos"])
    boxes = sample["bboxes"].numpy()
    valid = sample["valid_mask"].numpy()
    for t in range(frames.shape[0]):
        for k in range(valid.shape[1]):
            if valid[t, k]:
                _draw_rect(frames[t], boxes[t, k], colors[k % len(colors)])
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


# ============================== Main Eval ==============================

def _make_model(args, device: torch.device) -> LatentActionModelV12:
    model = LatentActionModelV12(
        image_size=args.image_size, max_actors=args.max_actors,
        crop_size=args.crop_size, content_dim=args.content_dim,
        mask_grid=args.mask_grid, mask_feat_dim=args.mask_feat_dim,
        struct_dim=args.struct_dim, latent_dim=args.latent_dim,
        dec_dim=args.dec_dim, patch_size=args.patch_size,
        dec_blocks=args.dec_blocks, free_bits=args.free_bits,
        use_z=not args.no_z,
        use_velocity=not args.no_velocity,
        encoder_mode=args.encoder_mode,
    ).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded checkpoint: {args.checkpoint}")
    model.eval()
    return model


def _bbox_to_cxcywh(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """xyxy pixel -> cxcywh normalized."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                         (x2 - x1) / W, (y2 - y1) / H], dim=-1)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="V12 comprehensive evaluation")
    parser.add_argument("--checkpoint", required=True, help="path to model.pt")
    parser.add_argument("--dataset", default="synthetic_minimal_nooverlap",
                        help="dataset name under data/v12/")
    parser.add_argument("--data_root", default=None, help="override dataset path")
    parser.add_argument("--split", default="val")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = all")
    parser.add_argument("--max_latent", type=int, default=2000, help="max latent samples for clustering")
    parser.add_argument("--n_vis_samples", type=int, default=3, help="samples for reconstruction viz")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    # Model dims (must match training)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_actors", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--content_dim", type=int, default=128)
    parser.add_argument("--struct_dim", type=int, default=128)
    parser.add_argument("--mask_grid", type=int, default=16)
    parser.add_argument("--mask_feat_dim", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--dec_dim", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--dec_blocks", type=int, default=4)
    parser.add_argument("--free_bits", type=float, default=0.05)
    # V12.1 diagnostic flags (must match training)
    parser.add_argument("--no_z", action="store_true")
    parser.add_argument("--no_velocity", action="store_true")
    parser.add_argument("--encoder_mode", type=str, default="bidirectional",
                        choices=["bidirectional", "causal", "per_frame"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    # Paths
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
    if args.data_root is None:
        data_root = os.path.join(ROOT, "data", "v12", args.dataset)
    else:
        data_root = args.data_root
    split_dir = os.path.join(data_root, args.split)
    if args.out_dir is None:
        out_dir = os.path.join(ROOT, "result", VERSION, args.dataset, "eval")
    else:
        out_dir = args.out_dir
    _ensure_dir(out_dir)

    print(f"\n{'='*60}")
    print(f"V12 Comprehensive Evaluation")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  dataset: {args.dataset} ({split_dir})")
    print(f"  output: {out_dir}")
    print(f"{'='*60}")

    dataset = V12ObjectVideoDataset(split_dir, output_format="t h w c")
    print(f"  Dataset: {len(dataset)} samples")

    model = _make_model(args, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=_collate,
    )

    # === Collect data ===
    all_z, all_actor, all_action, all_actor_action = [], [], [], []
    # z ablation metrics
    ablation_mse = {"normal": [], "z_zero": [], "z_shuffle": []}
    ablation_masked_mse = {"normal": [], "z_zero": [], "z_shuffle": []}
    ablation_copy_mse = []
    ablation_box_iou = {"normal": [], "z_zero": [], "z_shuffle": []}
    ablation_inertia_iou = []
    # Per-slot recon (first batch only for viz)
    first_batch = None
    second_batch = None
    recon_per_slot_viz: Optional[Dict[int, torch.Tensor]] = None
    recon_normal_viz: Optional[torch.Tensor] = None

    n_collected = 0
    n_latent = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["bboxes"].to(device)
            valid = batch["valid_mask"].to(device)
            actions = batch["actions"].cpu()
            obj_types = batch.get("object_types", None)
            obj_np = obj_types.numpy() if obj_types is not None else None

            B, T, H, W, C = videos.shape
            batch_input = {"videos": videos, "masks": masks, "bboxes": boxes, "valid_mask": valid}

            # --- z ablation: normal, z=0, z_shuffle ---
            ablation_recons = {}
            for mode_key, z_mode in [("normal", "normal"), ("z_zero", "zero"), ("z_shuffle", "shuffle")]:
                out = model.forward_ablation(batch_input, z_mode=z_mode)
                recon = out["recon"]
                pred_struct = out["pred_struct"]
                ablation_recons[mode_key] = out

                target = videos[:, 1:]
                copy = videos[:, :-1]

                # RGB MSE
                mse = ((recon - target) ** 2).mean(dim=[2, 3, 4])
                ablation_mse[mode_key].extend(mse.cpu().reshape(-1).tolist())

                # Actor-masked MSE
                actor_mask = masks[:, 1:].sum(dim=2).clamp(0, 1).unsqueeze(-1)
                masked_err = ((recon - target) ** 2) * actor_mask.float()
                denom = actor_mask.float().sum(dim=[2, 3, 4]).clamp(min=1.0) * C
                ablation_masked_mse[mode_key].extend((masked_err.sum(dim=[2, 3, 4]) / denom).cpu().reshape(-1).tolist())

                # Box IoU
                pred_bbox = pred_struct["bbox"].cpu()  # (B, T-1, K, 4) cxcywh normalized
                gt_bbox_cxcywh = _bbox_to_cxcywh(boxes[:, 1:].cpu(), H, W)
                pred_xyxy = _cxcywh_to_xyxy(pred_bbox)
                gt_xyxy = _cxcywh_to_xyxy(gt_bbox_cxcywh)
                iou = _box_iou(pred_xyxy.numpy(), gt_xyxy.numpy())
                ablation_box_iou[mode_key].extend(iou.reshape(-1).tolist())

                if mode_key == "normal":
                    ablation_copy_mse.extend(((copy - target) ** 2).mean(dim=[2, 3, 4]).cpu().reshape(-1).tolist())
                    # Inertia IoU: use t bbox as prediction for t+1
                    inertia_cxcywh = _bbox_to_cxcywh(boxes[:, :-1].cpu(), H, W)
                    inertia_xyxy = _cxcywh_to_xyxy(inertia_cxcywh)
                    inertia_iou = _box_iou(inertia_xyxy.numpy(), gt_xyxy.numpy())
                    ablation_inertia_iou.extend(inertia_iou.reshape(-1).tolist())

            # Copy MSE only once
            if batch_idx == 0:
                ablation_copy_mse = []
                target = videos[:, 1:]
                copy = videos[:, :-1]
                ablation_copy_mse.extend(((copy - target) ** 2).mean(dim=[2, 3, 4]).cpu().reshape(-1).tolist())
                # Inertia
                gt_bbox_cxcywh = _bbox_to_cxcywh(boxes[:, 1:].cpu(), H, W)
                gt_xyxy = _cxcywh_to_xyxy(gt_bbox_cxcywh)
                inertia_cxcywh = _bbox_to_cxcywh(boxes[:, :-1].cpu(), H, W)
                inertia_xyxy = _cxcywh_to_xyxy(inertia_cxcywh)
                inertia_iou = _box_iou(inertia_xyxy.numpy(), gt_xyxy.numpy())
                ablation_inertia_iou = inertia_iou.reshape(-1).tolist()

            # --- Collect latents ---
            out_normal = ablation_recons["normal"]
            mu = out_normal["mu"].detach().cpu().numpy()  # (B, T-1, K, D_z)
            act_np = actions.numpy()
            valid_np = valid[:, :-1].cpu().numpy()
            B_, T1, K, Dz = mu.shape
            for b in range(B_):
                for t in range(T1):
                    for k in range(K):
                        if valid_np[b, t, k] and act_np[b, t, k] >= 0 and n_latent < args.max_latent:
                            all_z.append(mu[b, t, k])
                            actor_id = int(obj_np[b, k]) if obj_np is not None else k
                            all_actor.append(actor_id)
                            all_action.append(int(act_np[b, t, k]))
                            all_actor_action.append(actor_id * 100 + int(act_np[b, t, k]))
                            n_latent += 1

            # --- Save first batch for viz ---
            if first_batch is None and args.n_vis_samples > 0:
                first_batch = {k: v[:args.n_vis_samples].clone() for k, v in batch_input.items()}
                recon_normal_viz = out_normal["recon"][:args.n_vis_samples].cpu()

                # Per-slot recon: keep only slot k, zero others
                recon_per_slot_viz = {}
                for k in range(K):
                    out_slot = model.forward_ablation(first_batch, z_mode="normal", slot_keep=k)
                    recon_per_slot_viz[k] = out_slot["recon"][:args.n_vis_samples].cpu()

            # Save second batch for action-swap viz
            if second_batch is None and batch_idx == 1:
                second_batch = {k: v[:args.n_vis_samples].clone() for k, v in batch_input.items()}

            n_collected += B
            if args.max_batches > 0 and batch_idx + 1 >= args.max_batches:
                break
            if n_latent >= args.max_latent and first_batch is not None:
                # Still continue for ablation metrics unless we have enough
                if batch_idx >= 20:
                    break

    print(f"\n  Collected: {n_collected} samples, {n_latent} latents")

    # === Compute results ===
    results: Dict = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "n_samples": n_collected,
        "n_latents": n_latent,
        "total_params": total_params,
    }

    # --- z ablation results ---
    copy_psnr = _psnr_from_mse(float(np.mean(ablation_copy_mse))) if ablation_copy_mse else None
    inertia_iou = float(np.mean(ablation_inertia_iou)) if ablation_inertia_iou else None

    for mode in ["normal", "z_zero", "z_shuffle"]:
        if ablation_mse[mode]:
            results[f"{mode}_rgb_psnr"] = _psnr_from_mse(float(np.mean(ablation_mse[mode])))
            results[f"{mode}_masked_psnr"] = _psnr_from_mse(float(np.mean(ablation_masked_mse[mode])))
            results[f"{mode}_box_iou"] = float(np.mean(ablation_box_iou[mode]))

    results["copy_psnr"] = copy_psnr
    results["inertia_box_iou"] = inertia_iou

    # --- Clustering metrics ---
    if len(all_z) >= 10 and len(np.unique(all_action)) >= 2:
        z = np.asarray(all_z)
        actor = np.asarray(all_actor)
        action = np.asarray(all_action)
        actor_action = np.asarray(all_actor_action)

        cluster_results = _compute_clustering_metrics(z, actor, action, args.seed)
        results.update(cluster_results)

        # --- Latent visualizations ---
        print("  Computing latent coordinates (UMAP/t-SNE/PCA)...")
        coords = _latent_coords(z, args.seed)
        _plot_latent_scatter(coords, actor, "Latent by Actor",
                             os.path.join(out_dir, "latent_umap_by_actor.png"))
        _plot_latent_scatter(coords, action, "Latent by Action",
                             os.path.join(out_dir, "latent_umap_by_action.png"))
        _plot_latent_scatter(coords, actor_action, "Latent by Actor×Action",
                             os.path.join(out_dir, "latent_umap_by_actor_action.png"))
        print(f"  Saved 3 latent scatter plots")

        # Save latents
        np.savez(
            os.path.join(out_dir, "latents.npz"),
            z=z, actor=actor, action=action, actor_action=actor_action,
        )
    else:
        results["latent_status"] = "skipped_insufficient_labeled_latents"

    # --- z ablation bar chart ---
    _save_z_ablation_plot(results, os.path.join(out_dir, "z_ablation.png"))
    print(f"  Saved z ablation plot")

    # --- Reconstruction visualizations ---
    if first_batch is not None and recon_normal_viz is not None:
        viz_batch = first_batch
        _save_recon_panel(
            viz_batch["videos"].cpu(), recon_normal_viz,
            viz_batch["masks"].cpu(),
            os.path.join(out_dir, "reconstruction_panel.png"),
            n_samples=min(args.n_vis_samples, viz_batch["videos"].shape[0]),
        )
        print(f"  Saved reconstruction panel")

        if recon_per_slot_viz is not None:
            _save_per_slot_recon(
                viz_batch["videos"].cpu(), viz_batch["masks"].cpu(),
                recon_per_slot_viz,
                os.path.join(out_dir, "per_slot_reconstruction.png"),
                sample_idx=0, t_idx=0,
            )
            print(f"  Saved per-slot reconstruction panel")

    # --- Action swap / z replay ---
    if first_batch is not None and second_batch is not None and model.use_z:
        try:
            _save_action_swap_panel(
                model, first_batch, second_batch, device,
                os.path.join(out_dir, "action_swap.png"),
            )
        except Exception as e:
            print(f"  Warning: action swap failed: {e}")

    # --- Tracking overlay ---
    sample0 = dataset[0]
    if sample0["valid_mask"].numel() and sample0["valid_mask"].any():
        try:
            _save_tracking_overlay(sample0, os.path.join(out_dir, "tracking_overlay.mp4"))
            print(f"  Saved tracking overlay")
        except Exception as e:
            print(f"  Warning: tracking overlay failed: {e}")

    # === Save results ===
    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(results, f, indent=2)

    # === Print summary ===
    print(f"\n{'='*60}")
    print(f"  EVAL RESULTS")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, list):
            v = [f"{x:.4f}" if isinstance(x, float) else x for x in v]
        elif isinstance(v, float):
            v = f"{v:.4f}"
        print(f"  {k}: {v}")
    print(f"\n  All outputs saved to: {out_dir}")
    print(f"  Files: eval.json, latents.npz, latent_umap_by_*.png,")
    print(f"         reconstruction_panel.png, per_slot_reconstruction.png,")
    print(f"         z_ablation.png, tracking_overlay.mp4")


if __name__ == "__main__":
    main()
