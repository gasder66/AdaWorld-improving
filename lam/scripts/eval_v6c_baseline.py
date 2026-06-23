"""
V8 Phase 0: 评估 V6c 基线 (model_v6c_long.pt) 在合成数据上的 NMI / ARI / actor_leakage。
作为 V8 Stage 1 的对照基线。
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = "/home/xiaojy/projects/AdaWorld-improving/result/old_result/v6_structured/model_v6c_long.pt"

    model = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8, max_actors=4,
        keep_background=True, use_obj_st_attention=True, free_bits_lambda=0.1,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded V6c: {ckpt_path}")

    ds = DiskSyntheticDataset(
        "/home/xiaojy/projects/AdaWorld-improving/data/synthetic_multi_actor/val",
        num_frames=5, output_format="t h w c",
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0)

    all_z, all_actions, all_slots = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 15:
                break
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            out = model({"videos": videos, "masks": masks})
            z_mu = out["z_mu"]  # (B, T-1, K+1, 32), slot 0 = bg
            actions = batch["actions"]  # (B, T-1, K)
            num_actors = batch["num_actors"]
            B, T1, Kfull, D = z_mu.shape

            for b in range(B):
                for t in range(T1):
                    for k in range(1, Kfull):  # skip bg slot 0
                        all_z.append(z_mu[b, t, k].cpu().numpy())
                        all_slots.append(k - 1)  # actor slot 0..3
                        if k - 1 < int(num_actors[b]):
                            all_actions.append(int(actions[b, t, k - 1]))
                        else:
                            all_actions.append(-1)

    all_z = np.array(all_z)
    all_actions = np.array(all_actions)
    all_slots = np.array(all_slots)
    valid = all_actions >= 0

    print(f"\nCollected: z={all_z.shape}, valid={valid.sum()}/{len(all_z)}")

    # 1. Action Clustering: KMeans(5) on z_actor (slots 1..K), evaluate vs GT actions
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    pred = kmeans.fit_predict(all_z[valid])
    true = all_actions[valid]
    nmi = normalized_mutual_info_score(true, pred)
    ari = adjusted_rand_score(true, pred)
    print(f"\n[V6c Action Clustering]")
    print(f"  NMI = {nmi:.4f}")
    print(f"  ARI = {ari:.4f}")

    # 2. Actor Leakage: train z_actor -> slot_index classifier
    # chance = 1/4 = 0.25
    clf = LogisticRegression(max_iter=1000, C=1.0)
    n = len(all_z)
    idx = np.random.RandomState(42).permutation(n)
    n_train = int(0.8 * n)
    tr, te = idx[:n_train], idx[n_train:]
    clf.fit(all_z[tr], all_slots[tr])
    acc = clf.score(all_z[te], all_slots[te])
    print(f"\n[V6c Actor Leakage]")
    print(f"  z_actor -> slot acc = {acc:.4f} (chance = 0.2500)")

    # 3. Per-slot NMI (each slot's z should cluster by action within itself)
    print(f"\n[V6c Per-Slot NMI]")
    nmi_per_slot = []
    for k in range(4):
        idx_k = (all_slots == k) & valid
        if idx_k.sum() < 50:
            continue
        kmeans_k = KMeans(n_clusters=5, random_state=42, n_init=10)
        pred_k = kmeans_k.fit_predict(all_z[idx_k])
        true_k = all_actions[idx_k]
        nmi_k = normalized_mutual_info_score(true_k, pred_k)
        nmi_per_slot.append(nmi_k)
        print(f"  Slot {k}: NMI = {nmi_k:.4f}")
    if nmi_per_slot:
        print(f"  Avg: NMI = {np.mean(nmi_per_slot):.4f}")

    results = {
        "model": "V6c (model_v6c_long.pt)",
        "n_samples": int(len(all_z)),
        "n_valid": int(valid.sum()),
        "action_nmi": round(float(nmi), 4),
        "action_ari": round(float(ari), 4),
        "actor_leakage_acc": round(float(acc), 4),
        "actor_leakage_chance": 0.25,
        "per_slot_nmi": [round(float(x), 4) for x in nmi_per_slot],
        "per_slot_nmi_avg": round(float(np.mean(nmi_per_slot)), 4) if nmi_per_slot else None,
    }
    out_path = "/home/xiaojy/projects/AdaWorld-improving/reports/V8/v6c_baseline_metrics.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
