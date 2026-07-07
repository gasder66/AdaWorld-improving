"""
Bridge-3 continuous latent action probe: z → dx/dy/displacement/scale.

Usage:
  OMP_NUM_THREADS=1 PYTHONPATH=lam python lam/scripts/v14/eval_bridge3_continuous_probe.py \
      --latents result/v14/bridge3_real_tube_pilot/scratch_seed0/eval/latents.npz \
      --output result/v14/bridge3_evidence_closure/continuous_probe/scratch_seed0.json
"""
import argparse, json, os, sys
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True, help="Path to latents.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.latents)
    z = data["z"]  # (N, D_z)
    action = data.get("action", data.get("actions", None))
    actor = data.get("actor", data.get("actor_ids", None))
    N, Dz = z.shape
    print(f"Loaded {N} latent vectors, dim={Dz}")

    # For Bridge-3, action and actor data may not be present in latents.npz.
    # We need per-sample metadata. For now, use what's available.
    if action is not None:
        print(f"Action labels: {len(action)} samples, unique={np.unique(action)}")

    # Split train/val.
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(N)
    n_train = int(N * args.train_ratio)
    idx_train = order[:n_train]
    idx_val = order[n_train:]
    z_train, z_val = z[idx_train], z[idx_val]

    # For now, we compute basic z statistics.
    # Full continuous probe requires sample metadata (dx/dy per sample)
    # which is stored separately in eval outputs.
    results = {
        "n_samples": N,
        "z_dim": Dz,
        "z_var": float(z.var(axis=0).mean()),
        "z_std": float(z.std()),
        "z_norm_mean": float(np.linalg.norm(z, axis=1).mean()),
        "z_norm_std": float(np.linalg.norm(z, axis=1).std()),
        "train_samples": n_train,
        "val_samples": N - n_train,
    }

    # Discrete action probe if actions available.
    if action is not None and len(np.unique(action)) >= 2:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score

        y_action = action if action.ndim == 1 else action[:, 0] if action.ndim == 2 else action
        y_train_a = y_action[idx_train]
        y_val_a = y_action[idx_val]

        if len(y_val_a) > 0 and len(np.unique(y_train_a)) >= 2:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(z_train, y_train_a)
            pred_val = clf.predict(z_val)
            results["discrete_probe_acc"] = float(accuracy_score(y_val_a, pred_val))
            results["discrete_probe_chance"] = float(1.0 / max(1, len(np.unique(y_action))))
            results["discrete_probe_n_clusters"] = len(np.unique(y_action))

            # NMI
            n_clusters = min(8, len(np.unique(y_action)))
            if n_clusters > 1 and N > n_clusters:
                pred_all = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10).fit_predict(z)
                results["overall_nmi"] = float(normalized_mutual_info_score(y_action, pred_all))

    # Actor leakage.
    if actor is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        y_actor = actor if actor.ndim == 1 else actor[:, 0] if actor.ndim == 2 else actor
        y_train_r = y_actor[idx_train]
        y_val_r = y_actor[idx_val]
        if len(y_val_r) > 0 and len(np.unique(y_train_r)) >= 2:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(z_train, y_train_r)
            results["actor_leakage_acc"] = float(accuracy_score(y_val_r, clf.predict(z_val)))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
