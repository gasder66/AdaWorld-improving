"""
V8 Stage 2B: A2D 评估.

指标:
  1. Action NMI (8 类 A2D action)
  2. Actor type leakage (z_actor → A2D actor type 1-7)
  3. Action probe accuracy
  4. z_bg camera probe (用背景帧差均值作为伪标签)
  5. UMAP

用法:
  PYTHONPATH=lam python lam/scripts/eval_v8_a2d.py --name v8_a2d
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "result", "v8_mot_lam")
    data = np.load(os.path.join(RESULTS_DIR, f"latents_{args.name}.npz"))
    z_actor = data["z_actor"]
    z_bg = data["z_bg"]
    actions = data["actions"]
    actor_types = data["actor_types"]

    print(f"\n{'='*60}")
    print(f"V8 Stage 2B A2D Evaluation: {args.name}")
    print(f"  Samples: {len(z_actor)}")
    print(f"  Actions: {np.unique(actions, return_counts=True)}")
    print(f"  Actor types: {np.unique(actor_types, return_counts=True)}")
    print(f"{'='*60}")

    results = {"model": f"V8 ({args.name})", "n_samples": int(len(z_actor))}

    # === 1. Action Clustering (8 classes) ===
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    pred = kmeans.fit_predict(z_actor)
    nmi = normalized_mutual_info_score(actions, pred)
    ari = adjusted_rand_score(actions, pred)
    print(f"\n[Action Clustering (8 A2D classes)]")
    print(f"  NMI = {nmi:.4f}")
    print(f"  ARI = {ari:.4f}")
    results["action_nmi"] = round(float(nmi), 4)
    results["action_ari"] = round(float(ari), 4)

    # === 2. Actor Type Leakage ===
    n_types = len(np.unique(actor_types))
    chance = 1.0 / n_types
    clf = LogisticRegression(max_iter=1000, C=1.0)
    n = len(z_actor)
    idx = np.random.RandomState(args.seed).permutation(n)
    tr, te = idx[:int(0.8 * n)], idx[int(0.8 * n):]
    clf.fit(z_actor[tr], actor_types[tr])
    leakage = clf.score(z_actor[te], actor_types[te])
    print(f"\n[Actor Type Leakage]")
    print(f"  z_actor → actor_type acc = {leakage:.4f} (chance = {chance:.4f})")
    results["actor_type_leakage"] = round(float(leakage), 4)

    # === 3. Action Probe ===
    clf_a = LogisticRegression(max_iter=1000, C=1.0)
    clf_a.fit(z_actor[tr], actions[tr])
    action_acc = clf_a.score(z_actor[te], actions[te])
    print(f"\n[Action Probe]")
    print(f"  z_actor → action acc = {action_acc:.4f} (chance = {1/args.n_clusters:.4f})")
    results["action_probe_acc"] = round(float(action_acc), 4)

    # === 4. Latent Stats ===
    z_var = float(z_actor.var(axis=0).mean())
    z_bg_var = float(z_bg.var(axis=0).mean())
    active = int((z_actor.var(axis=0) > 0.01).sum())
    print(f"\n[Latent Stats]")
    print(f"  z_actor variance: {z_var:.4f}, active dims: {active}/{z_actor.shape[1]}")
    print(f"  z_bg variance: {z_bg_var:.4f}")
    results["z_actor_variance"] = round(z_var, 4)
    results["z_bg_variance"] = round(z_bg_var, 4)
    results["z_actor_active_dims"] = active

    # === 5. UMAP ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap

        reducer = umap.UMAP(random_state=args.seed, n_neighbors=min(30, n - 1), min_dist=0.3)
        z_2d = reducer.fit_transform(z_actor)
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        ACTOR_NAMES = {1: "adult", 2: "baby", 3: "ball", 4: "bird", 5: "car", 6: "cat", 7: "dog"}
        ACTION_NAMES = {0: "climb", 1: "crawl", 2: "eat", 3: "fly", 4: "jump", 5: "roll", 6: "run", 7: "walk"}
        for ax, (vals, title, cmap) in zip(axes, [
            (actions, "z_actor (color=action)", "tab10"),
            (actor_types, "z_actor (color=actor_type)", "Set1"),
            (pred, f"KMeans (NMI={nmi:.3f})", "tab10"),
        ]):
            sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=vals, cmap=cmap, s=15, alpha=0.7)
            ax.set_title(title)
            ax.legend(*sc.legend_elements(), loc="best", fontsize=8)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"umap_{args.name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  UMAP saved: {path}")
    except ImportError:
        print("\n  (umap not installed, skipping)")

    out_path = os.path.join(RESULTS_DIR, f"eval_{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()
