"""
UMAP 隐动作聚类分析 (V5: MaskedPool + per-subject VAE)。

检查:
1. 各 Slot 是否对应特定主体 (UMAP 按 Slot ID 着色应清晰分离)
2. 每个 Slot 的隐动作是否按动作类型聚类 (按 GT action 着色应分 5 簇)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset


ACTION_NAMES = {0: "stay", 1: "up", 2: "down", 3: "left", 4: "right"}
COLORS = {0: "gray", 1: "blue", 2: "red", 3: "orange", 4: "green"}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8, max_actors=4,
        keep_background=False, use_obj_st_attention=True,
    ).to(device)

    ckpt = os.path.join(os.path.dirname(__file__), "..", "..", "result",
                        "v5_maskedpool", "model_v5_maskedpool.pt")
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

    all_z = []         # (N, 32)
    all_actions = []   # GT action (0-4) or -1 for padding
    all_actor_ids = [] # slot index (0-3)
    all_num_actors = [] # actual actor count per sample

    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"]       # (B, T-1, K)
            num_actors = batch["num_actors"] # (B,)

            out = model({"videos": videos, "masks": masks})
            z_mu = out["z_mu"]  # (B, T-1, K, 32)
            B, T1, K, D = z_mu.shape

            for b in range(B):
                for t in range(T1):
                    for k in range(K):
                        all_z.append(z_mu[b, t, k].cpu().numpy())
                        all_actor_ids.append(k)
                        # GT action for this actor
                        if k < int(num_actors[b]):
                            act = int(actions[b, t, k])
                            all_actions.append(act)
                        else:
                            all_actions.append(-1)
                    all_num_actors.append(int(num_actors[b]))

    all_z = np.array(all_z)
    all_actions = np.array(all_actions)
    all_actor_ids = np.array(all_actor_ids)
    mask_valid = all_actions >= 0

    print(f"\nCollected {len(all_z)} latent vectors")
    print(f"  z_mu mean: {all_z.mean():.4f}, std: {all_z.std():.4f}")
    print(f"  Valid (with GT actions): {mask_valid.sum()} / {len(all_z)}")

    # UMAP
    print("Running UMAP (all data)...")
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    z_2d = reducer.fit_transform(all_z)
    print(f"  UMAP done: {z_2d.shape}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "result",
                           "v5_maskedpool", "umap")
    os.makedirs(out_dir, exist_ok=True)

    # === Plot 1: All slots, colored by slot ID ===
    fig, ax = plt.subplots(figsize=(10, 8))
    slot_colors = ['red', 'green', 'blue', 'orange']
    for k in range(4):
        idx = all_actor_ids == k
        ax.scatter(z_2d[idx, 0], z_2d[idx, 1],
                   c=slot_colors[k], label=f'Slot {k}', s=4, alpha=0.4)
    ax.set_title('V5: Latent Actions by Slot ID', fontsize=14)
    ax.legend(markerscale=8, fontsize=11)
    ax.axis('off')
    fig.savefig(f'{out_dir}/umap_by_slot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_dir}/umap_by_slot.png")

    # === Plot 2: All valid points, colored by action type ===
    fig, ax = plt.subplots(figsize=(10, 8))
    for act in sorted(ACTION_NAMES.keys()):
        idx = (all_actions == act) & mask_valid
        ax.scatter(z_2d[idx, 0], z_2d[idx, 1],
                   c=COLORS[act], label=ACTION_NAMES[act], s=4, alpha=0.5)
    ax.set_title('V5: Latent Actions by GT Action Label', fontsize=14)
    ax.legend(markerscale=8, fontsize=11)
    ax.axis('off')
    fig.savefig(f'{out_dir}/umap_by_action.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_dir}/umap_by_action.png")

    # === Plot 3: Per-slot UMAP (4 subplots, colored by action) ===
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for k in range(4):
        ax = axes[k]
        idx_k = (all_actor_ids == k) & mask_valid
        z_k = all_z[idx_k]
        if len(z_k) < 50:
            ax.set_title(f'Slot {k} (too few)', fontsize=12)
            ax.axis('off')
            continue
        z_k_2d = reducer.transform(z_k)
        act_k = all_actions[idx_k]
        for act in sorted(ACTION_NAMES.keys()):
            idx_a = act_k == act
            ax.scatter(z_k_2d[idx_a, 0], z_k_2d[idx_a, 1],
                       c=COLORS[act], label=ACTION_NAMES[act] if k == 0 else '',
                       s=10, alpha=0.6)
        ax.set_title(f'Slot {k} ({len(z_k)} points)', fontsize=12)
        ax.axis('off')
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=COLORS[a], markersize=8, label=ACTION_NAMES[a])
               for a in sorted(ACTION_NAMES.keys())]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f'{out_dir}/umap_per_slot_action.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_dir}/umap_per_slot_action.png")

    # === Plot 4: Subplots by action type (colored by slot) ===
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, act in enumerate(sorted(ACTION_NAMES.keys())):
        ax = axes[i]
        idx_a = (all_actions == act) & mask_valid
        for k in range(4):
            idx = idx_a & (all_actor_ids == k)
            ax.scatter(z_2d[idx, 0], z_2d[idx, 1],
                       c=slot_colors[k], label=f'Slot {k}', s=8, alpha=0.5)
        ax.set_title(f'Action: {ACTION_NAMES[act]}', fontsize=12)
        ax.axis('off')
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=slot_colors[k], markersize=8, label=f'Slot {k}')
               for k in range(4)]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f'{out_dir}/umap_by_action_slot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_dir}/umap_by_action_slot.png")

    # === Metrics ===
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.cluster import KMeans

    mask = mask_valid
    print("\n=== Clustering Metrics ===")

    # Overall (all slots together)
    kmeans_all = KMeans(n_clusters=5, random_state=42, n_init=10)
    pred_all = kmeans_all.fit_predict(all_z[mask])
    true_all = all_actions[mask]
    nmi_all = normalized_mutual_info_score(true_all, pred_all)
    ari_all = adjusted_rand_score(true_all, pred_all)
    print(f"  All slots (5 clusters vs GT actions):")
    print(f"    NMI = {nmi_all:.4f}, ARI = {ari_all:.4f}")

    # Per-slot
    nmi_per_slot, ari_per_slot = [], []
    for k in range(4):
        idx_k = (all_actor_ids == k) & mask
        if idx_k.sum() < 50:
            continue
        kmeans_k = KMeans(n_clusters=5, random_state=42, n_init=10)
        pred_k = kmeans_k.fit_predict(all_z[idx_k])
        true_k = all_actions[idx_k]
        nmi_k = normalized_mutual_info_score(true_k, pred_k)
        ari_k = adjusted_rand_score(true_k, pred_k)
        nmi_per_slot.append(nmi_k)
        ari_per_slot.append(ari_k)
        print(f"  Slot {k}: NMI = {nmi_k:.4f}, ARI = {ari_k:.4f}")

    if nmi_per_slot:
        print(f"  Per-slot avg: NMI = {np.mean(nmi_per_slot):.4f}, ARI = {np.mean(ari_per_slot):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
