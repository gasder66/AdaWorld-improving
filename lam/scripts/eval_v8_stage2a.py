"""
V8 Stage 2A: 背景槽验证评估.

指标:
  1. Action clustering: NMI / ARI / actor leakage (同 Stage 1)
  2. Camera probe R²:
     z_bg    → camera_params (期望高)
     z_actor → camera_params (期望低, 即 camera leakage 低)
  3. Zero-out ablation:
     z_bg=0    → dbbox MSE 变化 (相机运动预测退化)
     z_actor=0 → dbbox MSE 变化 (actor 运动预测退化)
  4. 对比 with-bg vs no-bg

用法:
  PYTHONPATH=lam python lam/scripts/eval_v8_stage2a.py --name v8_s2a_bg
  PYTHONPATH=lam python lam/scripts/eval_v8_stage2a.py --name v8_s2a_nobg
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum() + 1e-8
    return 1 - ss_res / ss_tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "v8_mot_lam"
    )
    latents_path = os.path.join(RESULTS_DIR, f"latents_{args.name}.npz")
    data = np.load(latents_path)
    z_actor = data["z_actor"]
    z_bg = data["z_bg"]
    actions = data["actions"]
    actor_ids = data["actor_ids"]
    dbbox_pred = data["dbbox_pred"]
    dbbox_obs = data["dbbox_obs"]
    has_cam = "camera_params" in data
    camera_params = data["camera_params"] if has_cam else None

    print(f"\n{'='*60}")
    print(f"V8 Stage 2A Evaluation: {args.name}")
    print(f"  Samples: {len(z_actor)}")
    print(f"  z_actor: {z_actor.shape}, z_bg: {z_bg.shape}")
    if has_cam:
        print(f"  camera_params: {camera_params.shape}")
    print(f"{'='*60}")

    results = {"model": f"V8 ({args.name})", "n_samples": int(len(z_actor))}

    # === 1. Action Clustering ===
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    pred_all = kmeans.fit_predict(z_actor)
    nmi = normalized_mutual_info_score(actions, pred_all)
    ari = adjusted_rand_score(actions, pred_all)
    print(f"\n[Action Clustering]")
    print(f"  NMI = {nmi:.4f}, ARI = {ari:.4f}")
    results["overall_nmi"] = round(float(nmi), 4)
    results["overall_ari"] = round(float(ari), 4)

    nmi_per_actor = []
    for k in np.unique(actor_ids):
        idx_k = actor_ids == k
        if idx_k.sum() < 50:
            continue
        km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
        pk = km.fit_predict(z_actor[idx_k])
        nk = normalized_mutual_info_score(actions[idx_k], pk)
        nmi_per_actor.append(nk)
    nmi_avg = float(np.mean(nmi_per_actor)) if nmi_per_actor else 0.0
    print(f"  Per-Actor NMI avg = {nmi_avg:.4f}")
    results["per_actor_nmi_avg"] = round(nmi_avg, 4)

    # Actor leakage
    n_actors = len(np.unique(actor_ids))
    chance = 1.0 / n_actors
    clf = LogisticRegression(max_iter=1000, C=1.0)
    n = len(z_actor)
    idx = np.random.RandomState(args.seed).permutation(n)
    tr, te = idx[:int(0.8 * n)], idx[int(0.8 * n):]
    clf.fit(z_actor[tr], actor_ids[tr])
    leakage = clf.score(z_actor[te], actor_ids[te])
    print(f"  Actor Leakage = {leakage:.4f} (chance={chance:.4f})")
    results["actor_leakage_acc"] = round(float(leakage), 4)

    # Action probe
    clf_a = LogisticRegression(max_iter=1000, C=1.0)
    clf_a.fit(z_actor[tr], actions[tr])
    action_acc = clf_a.score(z_actor[te], actions[te])
    print(f"  Action Probe = {action_acc:.4f} (chance={1/args.n_clusters:.4f})")
    results["action_probe_acc"] = round(float(action_acc), 4)

    # === 2. Camera Probe R² ===
    if has_cam:
        print(f"\n[Camera Probe R²]")
        # z_bg → camera_params
        reg_bg = LinearRegression()
        reg_bg.fit(z_bg[tr], camera_params[tr])
        r2_bg = r2_score(camera_params[te], reg_bg.predict(z_bg[te]))
        # z_actor → camera_params
        reg_act = LinearRegression()
        reg_act.fit(z_actor[tr], camera_params[tr])
        r2_act = r2_score(camera_params[te], reg_act.predict(z_actor[te]))
        print(f"  z_bg    → camera R² = {r2_bg:.4f} (期望高)")
        print(f"  z_actor → camera R² = {r2_act:.4f} (期望低 = 低 camera leakage)")
        results["z_bg_camera_probe_r2"] = round(float(r2_bg), 4)
        results["z_actor_camera_probe_r2"] = round(float(r2_act), 4)

    # === 3. dbbox Prediction ===
    dbbox_mse = float(((dbbox_pred - dbbox_obs) ** 2).mean())
    dbbox_rmse = float(np.sqrt(dbbox_mse))
    print(f"\n[Δbbox Prediction]")
    print(f"  MSE = {dbbox_mse:.2f} px², RMSE = {dbbox_rmse:.2f} px")
    results["dbbox_mse"] = round(dbbox_mse, 2)
    results["dbbox_rmse"] = round(dbbox_rmse, 2)

    # === 4. Latent Stats ===
    z_var = float(z_actor.var(axis=0).mean())
    z_bg_var = float(z_bg.var(axis=0).mean()) if z_bg.var() > 0 else 0.0
    print(f"\n[Latent Stats]")
    print(f"  z_actor variance: {z_var:.4f}")
    print(f"  z_bg variance:    {z_bg_var:.4f}")
    results["z_actor_variance"] = round(z_var, 4)
    results["z_bg_variance"] = round(z_bg_var, 4)

    # === 5. UMAP ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap

        reducer = umap.UMAP(random_state=args.seed, n_neighbors=30, min_dist=0.3)
        z_2d = reducer.fit_transform(z_actor)
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        for ax, (vals, title, cmap) in zip(axes, [
            (actions, "z_actor (color=action)", "tab10"),
            (actor_ids, "z_actor (color=actor_id)", "Set1"),
            (pred_all, f"z_actor (KMeans, NMI={nmi:.3f})", "tab10"),
        ]):
            sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=vals, cmap=cmap, s=8, alpha=0.6)
            ax.set_title(title)
            ax.legend(*sc.legend_elements(), loc="best")
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"umap_{args.name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  UMAP saved: {path}")
    except ImportError:
        print("\n  (umap not installed, skipping)")

    # Save
    out_path = os.path.join(RESULTS_DIR, f"eval_{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {out_path}")
    return results


if __name__ == "__main__":
    main()
