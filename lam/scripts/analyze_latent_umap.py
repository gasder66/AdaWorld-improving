"""
UMAP 隐动作聚类分析。

说明: 对模型输出的 z_mu 做 UMAP 降维，按 GT 动作标签着色，
验证隐动作是否无监督地编码了真正的运动信息。

用法:
  CUDA_VISIBLE_DEVICES=2 python analyze_latent_umap.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset


ACTION_NAMES = {0: "stay", 1: "up", 2: "down", 3: "left", 4: "right"}
COLORS = {0: "gray", 1: "blue", 2: "red", 3: "orange", 4: "green"}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8, num_slots=4,
    ).to(device)

    ckpt = os.path.join(os.path.dirname(__file__), "..", "..", "result",
                        "adaworld_lam", "model_p0_objrecon_001.pt")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded: {ckpt}")
    
    dataset = DiskSyntheticDataset(
        os.path.join(os.path.dirname(__file__), "..", "..", "data",
                     "synthetic_multi_actor", "val"),
        num_frames=5, output_format="t h w c",
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)

    # Collect latents + ground truth labels
    all_z = []        # (N, latent_dim) 每行 = 一个 slot 在一帧的隐动作
    all_actions = []  # GT 动作标签
    all_actor_ids = []  # slot/actor 编号
    all_sample_ids = []  # 样本编号

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            videos = batch["videos"].to(device)
            actions = batch["actions"]      # (B, T-1, max_actors)  GT action labels
            positions = batch["positions"]  # (B, T, max_actors, 2) GT grid positions
            num_actors = batch["num_actors"]  # (B,) actual actor count

            out = model({"videos": videos})
            z_mu = out["z_mu"]  # (B, T-1, K, 32)
            B, T1, K, D = z_mu.shape

            for b in range(B):
                for t in range(T1):
                    for k in range(K):
                        all_z.append(z_mu[b, t, k].cpu().numpy())
                        all_actor_ids.append(k)
                        all_sample_ids.append(batch_idx * 64 + b)

                        # GT action for this actor (may be -1 if padding)
                        if k < num_actors[b]:
                            act = int(actions[b, t, k])
                            all_actions.append(act)
                        else:
                            all_actions.append(-1)

    all_z = np.array(all_z)
    all_actions = np.array(all_actions)
    all_actor_ids = np.array(all_actor_ids)
    print(f"\nCollected {len(all_z)} latent vectors (shape: {all_z.shape})")
    print(f"  z_mu mean: {all_z.mean():.4f}, std: {all_z.std():.4f}")

    # UMAP
    print("Running UMAP...")
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    z_2d = reducer.fit_transform(all_z)
    print(f"  UMAP done: {z_2d.shape}")

    # === Plot 1: Color by action type ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    ax = axes[0]
    mask_valid = all_actions >= 0
    for act in sorted(ACTION_NAMES.keys()):
        idx = (all_actions == act) & mask_valid
        ax.scatter(z_2d[idx, 0], z_2d[idx, 1],
                   c=COLORS[act], label=f"{ACTION_NAMES[act]}", s=5, alpha=0.6)
    ax.set_title("Latent Actions colored by GT Action Label", fontsize=13)
    ax.legend(markerscale=5, fontsize=10)
    ax.axis("off")

    # === Plot 2: Color by slot ID ===
    ax = axes[1]
    for k in range(4):
        idx = all_actor_ids == k
        ax.scatter(z_2d[idx, 0], z_2d[idx, 1],
                   c=[f"C{k}"], label=f"Slot {k}", s=3, alpha=0.3)
    ax.set_title("Latent Actions colored by Slot ID", fontsize=13)
    ax.legend(markerscale=8, fontsize=10)
    ax.axis("off")

    # === Plot 3: Color by actor (from ground truth) ===
    ax = axes[2]
    for a in range(4):
        idx = (all_actor_ids == a) & mask_valid
        ax.scatter(z_2d[idx, 0], z_2d[idx, 1],
                   label=f"Actor {a}", s=5, alpha=0.4)
    ax.set_title("Latent Actions colored by Actor Index", fontsize=13)
    ax.legend(markerscale=5, fontsize=10)
    ax.axis("off")

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "result",
                           "adaworld_lam", "umap")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "umap_latent_actions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_dir}/umap_latent_actions.png")

    # === Per-slot UMAP ===
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for k in range(4):
        ax = axes[k]
        idx_k = (all_actor_ids == k) & mask_valid
        z_k = all_z[idx_k]
        if len(z_k) == 0:
            continue
        z_k_2d = reducer.transform(z_k)
        act_k = all_actions[idx_k]
        for act in sorted(ACTION_NAMES.keys()):
            idx_a = act_k == act
            ax.scatter(z_k_2d[idx_a, 0], z_k_2d[idx_a, 1],
                       c=COLORS[act], label=ACTION_NAMES[act] if k == 0 else "",
                       s=10, alpha=0.6)
        ax.set_title(f"Slot {k} only ({len(z_k)} points)", fontsize=11)
        ax.axis("off")
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=COLORS[a], markersize=8, label=ACTION_NAMES[a])
               for a in sorted(ACTION_NAMES.keys())]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(out_dir, "umap_per_slot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/umap_per_slot.png")

    # Print stats
    print("\n=== Action distribution ===")
    for act in sorted(ACTION_NAMES.keys()):
        cnt = (all_actions == act).sum()
        print(f"  {ACTION_NAMES[act]:6s}: {cnt}")

    # === NMI / ARI metrics ===
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    mask = all_actions >= 0
    if mask.sum() > 0:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        pred_clusters = kmeans.fit_predict(all_z[mask])
        true_labels = all_actions[mask]
        nmi = normalized_mutual_info_score(true_labels, pred_clusters)
        ari = adjusted_rand_score(true_labels, pred_clusters)
        print(f"\n=== Clustering metrics (5 clusters vs GT actions) ===")
        print(f"  NMI: {nmi:.4f}")
        print(f"  ARI: {ari:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
