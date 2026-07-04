"""
V13 Atari BBox evaluation — z-ablation, clustering, reconstruction, swap, overlay.

Usage:
  PYTHONPATH=lam python lam/scripts/v13/eval_atari.py \
      --game freeway --checkpoint result/v13/freeway/phaseC/model.pt --gpu 4
"""
from __future__ import annotations

import argparse, json, math, os, sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.atari_bbox_dataset import AtariBBoxDataset, GAME_CONFIGS
from lam.modules.v13_slot_builder import build_slot_builder
from lam.modules.v13_atari_model import V13AtariBBoxModel

ROLE_NAMES = {0: "agent", 1: "lane_group", 2: "nearby_car"}
ROLE_COLORS = {0: (255, 50, 50), 1: (50, 150, 255), 2: (50, 255, 50)}


def _ensure_dir(p): os.makedirs(p, exist_ok=True)
def _psnr(mse): return float(-10.0 * math.log10(max(mse, 1e-10)))


def _compute_ious(pred_bbox, gt_bbox, valid):
    """(B,T,K,4) cxcywh -> per-slot IoU list (K lists)."""
    B, T, K, _ = pred_bbox.shape
    ious = [[] for _ in range(K)]
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if not valid[b, t, k]:
                    continue
                pcx, pcy, pw, ph = pred_bbox[b, t, k].tolist()
                gcx, gcy, gw, gh = gt_bbox[b, t, k].tolist()
                px1, py1 = pcx - pw / 2, pcy - ph / 2
                px2, py2 = pcx + pw / 2, pcy + ph / 2
                gx1, gy1 = gcx - gw / 2, gcy - gh / 2
                gx2, gy2 = gcx + gw / 2, gcy + gh / 2
                ix1, iy1 = max(px1, gx1), max(py1, gy1)
                ix2, iy2 = min(px2, gx2), min(py2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_p = max(0, px2 - px1) * max(0, py2 - py1)
                area_g = max(0, gx2 - gx1) * max(0, gy2 - gy1)
                ious[k].append(float(inter / (area_p + area_g - inter + 1e-6)))
    return ious


def _per_role_ious(pred_bbox, gt_bbox, valid, slot_is_group):
    """Return IoUs grouped by role: agent, lane_group, nearby_car."""
    B, T, K, _ = pred_bbox.shape
    agent, lane, car = [], [], []
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if not valid[b, t, k]:
                    continue
                pcx, pcy, pw, ph = pred_bbox[b, t, k].tolist()
                gcx, gcy, gw, gh = gt_bbox[b, t, k].tolist()
                px1, py1 = pcx - pw / 2, pcy - ph / 2
                px2, py2 = pcx + pw / 2, pcy + ph / 2
                gx1, gy1 = gcx - gw / 2, gcy - gh / 2
                gx2, gy2 = gcx + gw / 2, gcy + gh / 2
                ix1, iy1 = max(px1, gx1), max(py1, gy1)
                ix2, iy2 = min(px2, gx2), min(py2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_p = max(0, px2 - px1) * max(0, py2 - py1)
                area_g = max(0, gx2 - gx1) * max(0, gy2 - gy1)
                iou = inter / (area_p + area_g - inter + 1e-6)
                if k == 0:
                    agent.append(float(iou))
                elif slot_is_group[b, t, k]:
                    lane.append(float(iou))
                else:
                    car.append(float(iou))
    return agent, lane, car


def _collate(batch):
    out = {}
    for k in batch[0]:
        vs = [b[k] for b in batch]
        if isinstance(vs[0], torch.Tensor):
            out[k] = torch.stack(vs, dim=0)
        elif isinstance(vs[0], str):
            out[k] = vs[0]
    return out


def _save_ablation_bar(results, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    modes = ["normal", "z_zero", "z_shuffle"]
    labels = ["Normal z", "z=0", "z=shuffle"]
    x = np.arange(3); w = 0.35
    ious = [results.get(f"agent_{m}_iou", 0) for m in modes]
    ax1.bar(x, ious, w, color=["#4CAF50", "#FF9800", "#F44336"])
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Agent Box IoU"); ax1.set_title("Agent z-ablation")
    for i, v in enumerate(ious): ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    psnrs = [results.get(f"{m}_psnr", 0) for m in modes]
    ax2.bar(x, psnrs, w, color=["#4CAF50", "#FF9800", "#F44336"])
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("RGB PSNR"); ax2.set_title("RGB z-ablation")
    copy = results.get("copy_psnr", 0)
    ax2.axhline(copy, color="gray", linestyle="--", label=f"copy={copy:.1f}")
    ax2.legend(fontsize=8)
    for i, v in enumerate(psnrs): ax2.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _save_recon_panel(videos, recon_n, recon_z, masks, out_path, n_samples=2):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B = min(n_samples, videos.shape[0])
    T1 = recon_n.shape[1]
    n_cols = 5
    rows = B * T1
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
            for c, (img, ti) in enumerate([
                (it, "I_t"), (gt, "GT t+1"), (it, "Copy"),
                (rn, "Recon(normal)"), (er, "Error"),
            ]):
                axes[row, c].imshow(img.numpy()); axes[row, c].set_title(ti, fontsize=7)
                axes[row, c].axis("off")
    fig.tight_layout(pad=0.3); fig.savefig(out_path, dpi=100); plt.close(fig)


def _save_umap(z, labels, label_name, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import umap
        coords = umap.UMAP(n_components=2, random_state=42,
                           n_neighbors=min(30, max(2, len(z) // 10))).fit_transform(z)
    except Exception:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(z)
    fig, ax = plt.subplots(figsize=(6, 5))
    s = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=6, alpha=0.75,
                   cmap="tab20", edgecolors="none")
    ax.set_title(f"z latent by {label_name}"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(s, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_latent", type=int, default=2000)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    cfg = GAME_CONFIGS[args.game]
    ROOT = os.path.join(os.path.dirname(__file__), "../../..")
    data_root = os.path.join(ROOT, "data", "v12_ocatari", f"ocatari_{args.game}")
    out_dir = args.out_dir or os.path.join(ROOT, "result", "v13", args.game, "eval")
    _ensure_dir(out_dir)

    print(f"\n{'='*60}\nV13 Atari Evaluation\n  game={args.game}\n  ckpt={args.checkpoint}\n  out={out_dir}\n{'='*60}")

    ds = AtariBBoxDataset(os.path.join(data_root, "val"), game=args.game, image_size=cfg["image_size"])
    print(f"  Val samples: {len(ds)}")

    sb = build_slot_builder(args.game)
    model = V13AtariBBoxModel(slot_builder=sb, image_size=cfg["image_size"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=0, collate_fn=_collate)

    # Accumulators
    normal_ious, zero_ious, shuffle_ious = [], [], []
    agent_normal, agent_zero, agent_shuffle = [], [], []
    lane_normal, car_normal = [], []
    normal_psnr, zero_psnr, copy_psnr = [], [], []
    all_z, all_role, all_slot = [], [], []
    first_batch = None

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            out_n = model.forward_ablation(batch, z_mode="normal")
            out_z = model.forward_ablation(batch, z_mode="zero")
            out_s = model.forward_ablation(batch, z_mode="shuffle")

            # IoU
            gt_bbox_tp1 = out_n["slot_bbox"][:, 1:]
            valid_tp1 = out_n["slot_valid"][:, 1:]
            sg = out_n["slot_is_group"][:, 1:]

            ious_n = _compute_ious(out_n["pred_struct"]["bbox"], gt_bbox_tp1, valid_tp1)
            ious_z = _compute_ious(out_z["pred_struct"]["bbox"], gt_bbox_tp1, valid_tp1)
            ious_s = _compute_ious(out_s["pred_struct"]["bbox"], gt_bbox_tp1, valid_tp1)

            normal_ious.extend(sum(ious_n, []))
            zero_ious.extend(sum(ious_z, []))
            shuffle_ious.extend(sum(ious_s, []))

            if len(ious_n) > 0 and len(ious_n[0]) > 0:
                agent_normal.extend(ious_n[0])
                agent_zero.extend(ious_z[0])
                agent_shuffle.extend(ious_s[0])

            ag, la, ca = _per_role_ious(out_n["pred_struct"]["bbox"], gt_bbox_tp1, valid_tp1, sg)
            agent_normal.extend(ag); lane_normal.extend(la); car_normal.extend(ca)

            # PSNR
            if len(normal_psnr) < 100:
                target = batch["video"][:, 1:]
                copy = batch["video"][:, :-1]
                for mode, out in [("n", out_n), ("z", out_z)]:
                    mse = ((out["recon"] - target) ** 2).mean().item()
                    if mode == "n": normal_psnr.append(_psnr(mse))
                    else: zero_psnr.append(_psnr(mse))
                copy_psnr.append(_psnr(((copy - target) ** 2).mean().item()))

            # Latents
            mu = out_n["mu"].detach().cpu().numpy()
            sv = out_n["slot_valid"][:, :-1].cpu().numpy()
            sg_np = out_n["slot_is_group"][:, :-1].cpu().numpy()
            for b in range(mu.shape[0]):
                for t in range(mu.shape[1]):
                    for k in range(mu.shape[2]):
                        if sv[b, t, k] and len(all_z) < args.max_latent:
                            all_z.append(mu[b, t, k])
                            all_role.append(0 if k == 0 else (1 if sg_np[b, t, k] else 2))
                            all_slot.append(k)

            if first_batch is None:
                first_batch = {k: v[:2].clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                first_n = out_n["recon"][:2].clone()
                first_z = out_z["recon"][:2].clone()

            if args.max_batches and bi + 1 >= args.max_batches:
                break

    n = len(agent_normal)
    print(f"\n  Collected: {n} agent IoUs, {len(all_z)} latents")

    # Results
    res: Dict = {"game": args.game, "n_agent_iou": n, "n_latents": len(all_z)}
    res["agent_normal_iou"] = float(np.mean(agent_normal))
    res["agent_z_zero_iou"] = float(np.mean(agent_zero))
    res["agent_z_shuffle_iou"] = float(np.mean(agent_shuffle))
    res["agent_ablation_gap"] = res["agent_normal_iou"] - res["agent_z_zero_iou"]
    res["overall_normal_iou"] = float(np.mean(normal_ious))
    res["overall_zero_iou"] = float(np.mean(zero_ious))
    res["overall_shuffle_iou"] = float(np.mean(shuffle_ious))
    if lane_normal: res["lane_group_iou"] = float(np.mean(lane_normal))
    if car_normal: res["nearby_car_iou"] = float(np.mean(car_normal))
    if normal_psnr:
        res["normal_psnr"] = float(np.mean(normal_psnr))
        res["z_zero_psnr"] = float(np.mean(zero_psnr))
        res["copy_psnr"] = float(np.mean(copy_psnr))

    # Clustering
    if len(all_z) >= 10:
        z = np.asarray(all_z)
        role = np.asarray(all_role)
        z_var = float(z.var(axis=0).mean())
        res["z_var"] = z_var; res["z_std"] = float(z.std())
        try:
            from sklearn.cluster import KMeans
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import normalized_mutual_info_score
            n_clusters = min(5, len(np.unique(role)))
            pred = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10).fit_predict(z)
            res["overall_nmi"] = float(normalized_mutual_info_score(role, pred))
            order = np.random.RandomState(args.seed).permutation(len(z))
            n_train = max(1, int(0.8 * len(z)))
            if len(z) - n_train >= 1:
                clf = LogisticRegression(max_iter=1000)
                clf.fit(z[order[:n_train]], role[order[:n_train]])
                res["role_probe_acc"] = float(clf.score(z[order[n_train:]], role[order[n_train:]]))
            # Per-role NMI
            per_role = {}
            for ri, rn in ROLE_NAMES.items():
                idx = role == ri
                if idx.sum() >= 10 and n_clusters > 1:
                    pr = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10).fit_predict(z[idx])
                    per_role[rn] = float(normalized_mutual_info_score(
                        np.ones(idx.sum()) if idx.sum() < n_clusters else role[idx], pr))
            res["per_role_nmi"] = per_role
        except Exception as e:
            res["clustering_error"] = str(e)

    print(f"\n{'='*60}\n  EVAL RESULTS\n{'='*60}")
    for k, v in res.items():
        if isinstance(v, float): print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict): print(f"  {k}: {v}")
        else: print(f"  {k}: {v}")

    # Visualizations
    _save_ablation_bar(res, os.path.join(out_dir, "z_ablation.png"))
    print("  Saved z_ablation.png")

    if first_batch is not None:
        _save_recon_panel(first_batch["video"].cpu(), first_n.cpu(), first_z.cpu(),
                          torch.zeros(1), os.path.join(out_dir, "reconstruction_panel.png"))
        print("  Saved reconstruction_panel.png")

    if len(all_z) >= 10:
        z_arr = np.asarray(all_z)
        role_arr = np.asarray(all_role)
        slot_arr = np.asarray(all_slot)
        _save_umap(z_arr, role_arr, "role", os.path.join(out_dir, "latent_umap_by_role.png"))
        _save_umap(z_arr, slot_arr, "slot", os.path.join(out_dir, "latent_umap_by_slot.png"))
        print("  Saved UMAP plots")
        np.savez(os.path.join(out_dir, "latents.npz"), z=z_arr, role=role_arr, slot=slot_arr)

    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n  All results saved to {out_dir}")


if __name__ == "__main__":
    main()
