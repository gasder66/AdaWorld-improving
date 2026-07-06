"""
V14.6 Bridge-2 detailed reconstruction visualizations.
Shows: I_t | GT t+1 | Recon(normal) | Recon(z=0) | Recon(shuffle) | Error
With bbox overlays, mask predictions, and per-object breakdown.

Usage:
  OMP_NUM_THREADS=1 PYTHONPATH=lam python lam/scripts/v14/vis_recon_detail.py \
      --checkpoint result/v14/bridge2_occlusion_clean/mask_structure_seed0/phaseC/model.pt \
      --val_dir data/bridgebench/bridge2_occlusion_clean/val \
      --output result/v14/bridge2_occlusion_clean/mask_structure_seed0/eval/recon_detail \
      --n_samples 8 --gpu 4
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.modules.v14_model import V14Model


COLORS = [(255,50,50), (50,150,255), (50,255,50), (255,200,50)]


def _draw_rect(img, cx, cy, w, h, color, thickness=2):
    H, W = img.shape[:2]
    x1 = max(0, int((cx - w/2) * W)); y1 = max(0, int((cy - h/2) * H))
    x2 = min(W-1, int((cx + w/2) * W)); y2 = min(H-1, int((cy + h/2) * H))
    if x2 <= x1 or y2 <= y1: return
    img[y1:y1+thickness, x1:x2] = color
    img[y2-thickness:y2, x1:x2] = color
    img[y1:y2, x1:x1+thickness] = color
    img[y1:y2, x2-thickness:x2] = color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model = V14Model(image_size=128, max_actors=4).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    files = sorted([os.path.join(args.val_dir, f) for f in os.listdir(args.val_dir) if f.endswith(".pt")])
    rng = np.random.RandomState(args.seed + 1)
    selected = rng.choice(files, size=min(args.n_samples, len(files)), replace=False)
    print(f"  Val: {len(files)} files, showing {len(selected)}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    for si, fp in enumerate(selected):
        s = torch.load(fp, map_location="cpu", weights_only=False)
        batch = {k: v.cuda().unsqueeze(0) for k, v in s.items() if isinstance(v, torch.Tensor)}
        video = batch["video"]; B, T, C, H, W = video.shape; K = batch["boxes"].shape[2]

        with torch.no_grad():
            out_n = model.forward_ablation(batch, z_mode="normal")
            out_z = model.forward_ablation(batch, z_mode="zero")
            out_s = model.forward_ablation(batch, z_mode="shuffle")

        # Check validity
        valid = batch["valid"][0].cpu().numpy()
        has_occ = batch.get("is_occluded", torch.zeros(1,T-1,K)).cpu().numpy()
        occ_ratio = batch.get("occlusion_ratio", torch.zeros(1,T,K)).cpu().numpy()

        n_cols = 6
        T1 = out_n["recon"].shape[1]  # T-1
        n_rows = T1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_rows == 1: axes = axes[np.newaxis, :]

        for t in range(T1):
            it = video[0, t].permute(1,2,0).cpu().clamp(0,1).numpy()
            gt = video[0, t+1].permute(1,2,0).cpu().clamp(0,1).numpy()
            rn = out_n["recon"][0, t].permute(1,2,0).cpu().clamp(0,1).numpy()
            rz = out_z["recon"][0, t].permute(1,2,0).cpu().clamp(0,1).numpy()
            rs = out_s["recon"][0, t].permute(1,2,0).cpu().clamp(0,1).numpy()
            err = np.abs(gt - rn).mean(axis=-1)
            err = err / max(err.max(), 1e-6)

            # Draw GT bboxes on I_t and GT
            for k in range(K):
                if valid[t, k]:
                    cx, cy, w_, h_ = s["boxes"][t, k].tolist()
                    _draw_rect(it, cx, cy, w_, h_, COLORS[k % 4])
                    _draw_rect(gt, cx, cy, w_, h_, COLORS[k % 4])
                    if t < T1:
                        ocx, ocy, ow_, oh_ = s["boxes"][t+1, k].tolist()
                        _draw_rect(gt, ocx, ocy, ow_, oh_, (255,255,255))
                # Draw pred bboxes
                if t < T1 and valid[t, k]:
                    pb = out_n["pred_struct"]["bbox"][0, t, k]
                    pz = out_z["pred_struct"]["bbox"][0, t, k]
                    ps = out_s["pred_struct"]["bbox"][0, t, k]
                    _draw_rect(rn, pb[0].item(), pb[1].item(), pb[2].item(), pb[3].item(), (0,255,0))
                    _draw_rect(rz, pz[0].item(), pz[1].item(), pz[2].item(), pz[3].item(), (255,128,0))
                    _draw_rect(rs, ps[0].item(), ps[1].item(), ps[2].item(), ps[3].item(), (255,0,255))

            # Annotations
            occ_info = ""
            for k in range(K):
                if t < T1 and valid[t, k]:
                    occ_info += f"k{k}:occ={occ_ratio[0,t,k]:.2f} "
            occ_info = occ_info.strip()

            panels = [
                (it, f"I_t t={t}\n{occ_info}", None),
                (gt, f"GT t+1 (t={t+1})", None),
                (rn, "Recon(normal z)", None),
                (rz, "Recon(z=0)", None),
                (rs, "Recon(z_shuffle)", None),
                (err, f"Error |n-gt|\nmax={err.max():.3f}", 'hot'),
            ]
            for c, (img, title, cmap) in enumerate(panels):
                ax = axes[t, c]
                ax.imshow(img, cmap=cmap)
                ax.set_title(title, fontsize=6)
                ax.axis("off")

        fig.tight_layout(pad=0.3)
        out_path = os.path.join(args.output, f"recon_sample_{si:02d}.png")
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"  [{si+1}/{len(selected)}] {out_path}")

    # Also create a summary panel: side-by-side normal/z=0/shuffle for first 2 samples
    print(f"\n  Creating mask prediction detail panels...")
    for si in range(min(3, len(selected))):
        fp = selected[si]
        s = torch.load(fp, map_location="cpu", weights_only=False)
        batch = {k: v.cuda().unsqueeze(0) for k, v in s.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            out_n = model.forward_ablation(batch, z_mode="normal")
            out_z = model.forward_ablation(batch, z_mode="zero")
            out_s = model.forward_ablation(batch, z_mode="shuffle")

        video = batch["video"]; B, T, C, H, W = video.shape; K = batch["boxes"].shape[2]
        masks = batch["masks"]
        visible = batch.get("visible_masks", masks)

        t = 0  # Show first transition
        n_cols = 6; n_rows = min(4, K)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        if n_rows == 1: axes = axes[np.newaxis, :]

        for k in range(n_rows):
            # RGB at t
            rgb = video[0, t].permute(1,2,0).cpu().clamp(0,1).numpy()
            _draw_rect(rgb, s["boxes"][t,k,0].item(), s["boxes"][t,k,1].item(),
                       s["boxes"][t,k,2].item(), s["boxes"][t,k,3].item(), COLORS[k])

            # full_mask at t+1
            fm = masks[0, t+1, k].cpu().numpy()
            # visible at t+1
            vm = visible[0, t+1, k].cpu().numpy()
            # pred mask
            pm_n = out_n["pred_struct"]["mask_low"][0, t, k, 0].sigmoid().cpu().numpy()
            pm_z = out_z["pred_struct"]["mask_low"][0, t, k, 0].sigmoid().cpu().numpy()
            pm_s = out_s["pred_struct"]["mask_low"][0, t, k, 0].sigmoid().cpu().numpy()

            panels = [
                (rgb, f"RGB t, slot{k}"),
                (fm, f"full_mask t+1"),
                (vm, f"visible_mask t+1"),
                (pm_n, f"pred_mask(n)"),
                (pm_z, f"pred_mask(z=0)"),
                (pm_s, f"pred_mask(sh)"),
            ]
            for c, (img, title) in enumerate(panels):
                axes[k, c].imshow(img, cmap='Blues' if img.ndim==2 else None, vmin=0, vmax=1)
                axes[k, c].set_title(title, fontsize=5)
                axes[k, c].axis("off")

        fig.tight_layout(pad=0.2)
        out_path = os.path.join(args.output, f"mask_detail_{si:02d}.png")
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"  [{si+1}] {out_path}")

    print(f"\n  All saved to {args.output}")


if __name__ == "__main__":
    main()
