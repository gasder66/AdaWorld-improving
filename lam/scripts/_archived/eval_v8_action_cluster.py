"""
V8 Phase 5: 评估 V8 MOT-LAT 在合成数据上的 action clustering 质量。

指标:
  1. Overall NMI: KMeans(5) on ALL z_actor → vs GT actions (目标 ≥ 0.20)
  2. Per-Slot NMI: 每个 actor 内部 KMeans(5) → vs GT actions (目标 ≥ 0.30)
  3. Actor Leakage: z_actor → actor_id 分类器 acc (目标 ≤ 0.50, chance=0.25)
  4. ARI: Overall adjusted Rand index
  5. UMAP 可视化

对比基线: V6c (Overall NMI=0.0525, Per-Slot NMI=0.3885, Leakage=1.0000)

用法:
  PYTHONPATH=lam python lam/scripts/eval_v8_action_cluster.py --name v8_stage1
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="v8_stage1")
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "v8_mot_lam"
    )
    latents_path = os.path.join(RESULTS_DIR, f"latents_{args.name}.npz")
    data = np.load(latents_path)
    z_actor = data["z_actor"]      # (N, D)
    z_bg = data["z_bg"]            # (N, D_bg)
    actions = data["actions"]      # (N,)
    actor_ids = data["actor_ids"]  # (N,)
    dbbox_pred = data["dbbox_pred"]
    dbbox_obs = data["dbbox_obs"]

    print(f"\n{'='*60}")
    print(f"V8 Action Clustering Evaluation: {args.name}")
    print(f"  Samples: {len(z_actor)}")
    print(f"  z_actor: {z_actor.shape}, z_bg: {z_bg.shape}")
    print(f"  Actions: {np.unique(actions, return_counts=True)}")
    print(f"  Actors:  {np.unique(actor_ids, return_counts=True)}")
    print(f"{'='*60}")

    results = {
        "model": f"V8 ({args.name})",
        "n_samples": int(len(z_actor)),
        "n_actions": int(len(np.unique(actions))),
        "n_actors": int(len(np.unique(actor_ids))),
    }

    # === 1. Overall NMI / ARI ===
    # 所有 z_actor 混在一起聚类, 看 action 是否可恢复
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    pred_all = kmeans.fit_predict(z_actor)
    nmi_overall = normalized_mutual_info_score(actions, pred_all)
    ari_overall = adjusted_rand_score(actions, pred_all)
    print(f"\n[Overall Action Clustering]")
    print(f"  NMI = {nmi_overall:.4f}  (V6c: 0.0525, target ≥ 0.20)")
    print(f"  ARI = {ari_overall:.4f}")
    results["overall_nmi"] = round(float(nmi_overall), 4)
    results["overall_ari"] = round(float(ari_overall), 4)

    # === 2. Per-Slot (Per-Actor) NMI ===
    # 每个 actor 内部聚类, 看 action 是否可恢复
    print(f"\n[Per-Actor NMI]")
    nmi_per_actor = []
    for k in np.unique(actor_ids):
        idx_k = actor_ids == k
        if idx_k.sum() < 50:
            continue
        kmeans_k = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
        pred_k = kmeans_k.fit_predict(z_actor[idx_k])
        true_k = actions[idx_k]
        nmi_k = normalized_mutual_info_score(true_k, pred_k)
        nmi_per_actor.append(nmi_k)
        print(f"  Actor {k}: NMI = {nmi_k:.4f} (n={idx_k.sum()})")
    nmi_per_avg = float(np.mean(nmi_per_actor)) if nmi_per_actor else 0.0
    print(f"  Avg: NMI = {nmi_per_avg:.4f}  (V6c: 0.3885, target ≥ 0.30)")
    results["per_actor_nmi"] = [round(float(x), 4) for x in nmi_per_actor]
    results["per_actor_nmi_avg"] = round(nmi_per_avg, 4)

    # === 3. Actor Leakage ===
    # z_actor → actor_id 分类器, chance = 1/n_actors
    n_actors = len(np.unique(actor_ids))
    chance = 1.0 / n_actors
    clf = LogisticRegression(max_iter=1000, C=1.0)
    n = len(z_actor)
    idx = np.random.RandomState(args.seed).permutation(n)
    n_train = int(0.8 * n)
    tr, te = idx[:n_train], idx[n_train:]
    clf.fit(z_actor[tr], actor_ids[tr])
    leakage_acc = clf.score(z_actor[te], actor_ids[te])
    print(f"\n[Actor Leakage]")
    print(f"  z_actor → actor_id acc = {leakage_acc:.4f}  "
          f"(V6c: 1.0000, chance = {chance:.4f}, target ≤ 0.50)")
    results["actor_leakage_acc"] = round(float(leakage_acc), 4)
    results["actor_leakage_chance"] = round(float(chance), 4)

    # === 4. Action Classification (probe) ===
    # z_actor → action 分类器, 直接测 action 可解码性
    clf_act = LogisticRegression(max_iter=1000, C=1.0)
    clf_act.fit(z_actor[tr], actions[tr])
    action_acc = clf_act.score(z_actor[te], actions[te])
    action_chance = 1.0 / args.n_clusters
    print(f"\n[Action Probe]")
    print(f"  z_actor → action acc = {action_acc:.4f}  (chance = {action_chance:.4f})")
    results["action_probe_acc"] = round(float(action_acc), 4)
    results["action_probe_chance"] = round(float(action_chance), 4)

    # === 5. dbbox Prediction Quality ===
    dbbox_mse = float(((dbbox_pred - dbbox_obs) ** 2).mean())
    dbbox_rmse = float(np.sqrt(dbbox_mse))
    print(f"\n[Δbbox Prediction]")
    print(f"  MSE = {dbbox_mse:.2f} px², RMSE = {dbbox_rmse:.2f} px "
          f"(action step = 32 px)")
    results["dbbox_mse"] = round(dbbox_mse, 2)
    results["dbbox_rmse"] = round(dbbox_rmse, 2)

    # === 6. Latent Stats ===
    z_var = float(z_actor.var(axis=0).mean())
    z_active = int((z_actor.var(axis=0) > 0.01).sum())
    z_bg_var = float(z_bg.var(axis=0).mean())
    print(f"\n[Latent Stats]")
    print(f"  z_actor variance: {z_var:.4f}, active dims: {z_active}/{z_actor.shape[1]}")
    print(f"  z_bg variance:    {z_bg_var:.4f}")
    results["z_actor_variance"] = round(z_var, 4)
    results["z_actor_active_dims"] = z_active
    results["z_bg_variance"] = round(z_bg_var, 4)

    # === 7. UMAP Visualization ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap

        reducer = umap.UMAP(random_state=args.seed, n_neighbors=30, min_dist=0.3)
        z_2d = reducer.fit_transform(z_actor)

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        # Color by action
        scatter1 = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=actions, cmap="tab10",
                                    s=8, alpha=0.6)
        axes[0].set_title("z_actor UMAP (color=action)")
        axes[0].legend(*scatter1.legend_elements(), title="action", loc="best")

        # Color by actor_id
        scatter2 = axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=actor_ids, cmap="Set1",
                                    s=8, alpha=0.6)
        axes[1].set_title("z_actor UMAP (color=actor_id)")
        axes[1].legend(*scatter2.legend_elements(), title="actor", loc="best")

        # Color by KMeans cluster
        scatter3 = axes[2].scatter(z_2d[:, 0], z_2d[:, 1], c=pred_all, cmap="tab10",
                                    s=8, alpha=0.6)
        axes[2].set_title(f"z_actor UMAP (color=KMeans, NMI={nmi_overall:.3f})")
        axes[2].legend(*scatter3.legend_elements(), title="cluster", loc="best")

        plt.tight_layout()
        umap_path = os.path.join(RESULTS_DIR, f"umap_{args.name}.png")
        plt.savefig(umap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  UMAP saved: {umap_path}")
        results["umap_path"] = umap_path
    except ImportError:
        print("\n  (umap-learn not installed, skipping visualization)")

    # === Comparison Table ===
    print(f"\n{'='*60}")
    print(f"Comparison: V8 vs V6c vs V7")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'V6c':>10} {'V7v3':>10} {'V8':>10} {'Target':>10}")
    print(f"{'-'*65}")
    print(f"{'Overall NMI':<25} {'0.0525':>10} {'0.0047':>10} "
          f"{nmi_overall:>10.4f} {'≥0.20':>10}")
    print(f"{'Per-Slot NMI (avg)':<25} {'0.3885':>10} {'N/A':>10} "
          f"{nmi_per_avg:>10.4f} {'≥0.30':>10}")
    print(f"{'Actor Leakage':<25} {'1.0000':>10} {'0.8970':>10} "
          f"{leakage_acc:>10.4f} {'≤0.50':>10}")
    print(f"{'Action Probe Acc':<25} {'N/A':>10} {'N/A':>10} "
          f"{action_acc:>10.4f} {'-':>10}")
    print(f"{'dbbox RMSE (px)':<25} {'N/A':>10} {'N/A':>10} "
          f"{dbbox_rmse:>10.2f} {'-':>10}")
    print(f"{'='*65}")

    v6c = {
        "overall_nmi": 0.0525,
        "per_slot_nmi_avg": 0.3885,
        "actor_leakage_acc": 1.0000,
    }
    results["comparison"] = {
        "V6c": v6c,
        "V7v3": {"overall_nmi": 0.0047, "actor_leakage_acc": 0.8970},
        "V8": {
            "overall_nmi": round(float(nmi_overall), 4),
            "per_slot_nmi_avg": round(nmi_per_avg, 4),
            "actor_leakage_acc": round(float(leakage_acc), 4),
        },
        "targets": {
            "overall_nmi": 0.20,
            "per_slot_nmi_avg": 0.30,
            "actor_leakage_acc": 0.50,
        },
    }
    results["meets_targets"] = {
        "overall_nmi": bool(nmi_overall >= 0.20),
        "per_slot_nmi": bool(nmi_per_avg >= 0.30),
        "actor_leakage": bool(leakage_acc <= 0.50),
    }

    out_path = os.path.join(RESULTS_DIR, f"eval_{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
