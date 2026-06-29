"""
V10 A2D Visualization: UMAP + Reconstruction comparison.

用法:
  PYTHONPATH=lam python lam/scripts/vis_v10_a2d.py --name v10_a2d_gt --gpu 0
"""
import os, sys, argparse
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from lam.modules.v10_model import LatentActionModelV10
from lam.a2d_box_dataset import A2DBoxDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_vis", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "result", "v10")

    # Load latents
    latents_path = os.path.join(RESULTS_DIR, f"latents_{args.name}.npz")
    data = np.load(latents_path)
    z_actor = data["z_actor"]
    try:
        actions = data["actions"]
        slots = data["slots"]
    except KeyError:
        actions = data["actions"]
        slots = np.arange(len(z_actor))  # fallback

    print(f"  {len(z_actor)} samples, z_dim={z_actor.shape[1]}")

    # KMeans
    n_clusters = min(len(np.unique(actions)), 8)
    km = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    pred = km.fit_predict(z_actor)
    nmi = normalized_mutual_info_score(actions, pred)

    # UMAP
    import umap
    reducer = umap.UMAP(random_state=args.seed, n_neighbors=min(30, len(z_actor) // 10), min_dist=0.3)
    z_2d = reducer.fit_transform(z_actor)

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

    # 1. Action
    axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=actions, cmap="tab10", s=12, alpha=0.7)
    axes[0].set_title(f"Color=Action (NMI={nmi:.3f})")

    # 2. Actor
    unique_slots = np.unique(slots)
    colors_slot = plt.cm.Set1(np.linspace(0, 1, len(unique_slots))) if len(unique_slots) <= 9 else plt.cm.tab20(np.linspace(0, 1, len(unique_slots)))
    for i, s in enumerate(unique_slots):
        idx = slots == s
        axes[1].scatter(z_2d[idx, 0], z_2d[idx, 1], c=[colors_slot[i]], s=12, alpha=0.7, label=f"actor {s}")
    axes[1].set_title("Color=Actor")
    axes[1].legend(fontsize=7, ncol=2)

    # 3. KMeans
    axes[2].scatter(z_2d[:, 0], z_2d[:, 1], c=pred, cmap="tab10", s=12, alpha=0.7)
    axes[2].set_title(f"Color=KMeans ({n_clusters} clusters)")

    # 4. Per-action breakdown — color by action, mark centroids
    for a in np.unique(actions):
        idx = actions == a
        if idx.sum() > 0:
            centroids = z_2d[idx].mean(axis=0)
            axes[3].scatter(z_2d[idx, 0], z_2d[idx, 1], s=8, alpha=0.5)
            axes[3].scatter(centroids[0], centroids[1], s=80, marker="X", edgecolors="black", linewidths=1)
    axes[3].set_title("Action clusters (X=centroid)")

    plt.tight_layout()
    umap_path = os.path.join(RESULTS_DIR, f"umap_{args.name}.png")
    plt.savefig(umap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  UMAP saved: {umap_path}")

    # === Reconstruction visualization ===
    ckpt_path = os.path.join(RESULTS_DIR, f"model_{args.name}.pt")
    if not os.path.exists(ckpt_path):
        print("  No model checkpoint, skipping recon vis")
        return

    model = LatentActionModelV10(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8, max_actors=4,
        keep_background=True, use_obj_st_attention=True,
        free_bits_lambda=0.1,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    ds = A2DBoxDataset(data_root="data/a2d", release_root="Release",
                       split="test", num_frames=5, frame_stride=1,
                       max_actors=4, img_size=256)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    crops_t = []
    crops_gt = []
    crops_recon = []
    valid_masks_list = []

    with torch.no_grad():
        for batch in dl:
            v = batch["videos"].to(device)
            m = batch["masks"].to(device)
            out = model({"videos": v, "masks": m})
            recon = out["recon"].cpu()  # (B, T-1, H, W, C)
            gt = v[:, 1:].cpu()
            ct = v[:, :-1].cpu()
            vm = batch["valid_mask"][:, 1:].numpy()
            B, T1 = ct.shape[0], ct.shape[1]
            for b in range(B):
                for t in range(T1):
                    if vm[b, t].sum() > 0:
                        crops_t.append(ct[b, t].numpy())
                        crops_gt.append(gt[b, t].numpy())
                        crops_recon.append(recon[b, t].numpy())
                        valid_masks_list.append(vm[b, t])
            if len(crops_t) >= args.n_vis:
                break

    n_show = min(args.n_vis, len(crops_t))
    fig, axes = plt.subplots(n_show, 3, figsize=(15, 5 * n_show))
    for i in range(n_show):
        for j, (img, title) in enumerate(zip(
            [crops_t[i], crops_gt[i], crops_recon[i]],
            ["Frame t", "Frame t+1 (GT)", "Recon"])):
            img_clip = np.clip(img, 0, 1)
            axes[i, j].imshow(img_clip)
            axes[i, j].set_title(title)
            axes[i, j].axis("off")
    plt.tight_layout()
    recon_path = os.path.join(RESULTS_DIR, f"recon_{args.name}.png")
    plt.savefig(recon_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Recon saved: {recon_path}")


if __name__ == "__main__":
    main()
