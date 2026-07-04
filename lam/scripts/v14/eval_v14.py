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


# ============================== Main ==============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=15)
    parser.add_argument("--max_latent", type=int, default=3000)
    parser.add_argument("--mask_eval_mode", default="warp", choices=["head", "warp"])
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ROOT = os.path.join(os.path.dirname(__file__), "../../..")
    val_dir = os.path.join(ROOT, "data", "bridgebench", "bridge1", "val")
    out_dir = args.out_dir or os.path.join(ROOT, "result", "v14", "bridge1", "eval")
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
    all_z, all_action, all_actor, all_cat = [], [], [], []
    swap_vals = {k: [] for k in [
        "acc_all", "acc_cs", "acc_ns", "acc_cs_ns",
        "eff_acc", "eff_iou", "eff_l1", "eff_acc_cs",
    ]}
    swap_count = 0
    first_batch, first_n, first_z = None, None, None

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
    res["go_bbox_gap"] = res.get("bbox_gap_zero", 0) > 0.1
    res["go_mask_gap"] = res.get("mask_gap_warp", 0) > 0.1
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
