"""
V9 YOLO Evaluation with symmetric leakage:
  - Overall / Per-ActorType NMI (unsupervised action clustering)
  - Actor Type Leakage (z -> actor_type, linear probe)
  - Action Probe (z -> action, linear probe)
  - Action Probe given Actor Type (conditional)
  - Leakage given Action (conditional)
  - Symmetric summary

Reads latents .npz from run_v9_yolo.py output.
"""
import os, sys, json, argparse
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "result", "v9")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="latent file name (without .npz)")
    parser.add_argument("--latents_dir", type=str, default=None)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    latents_dir = args.latents_dir or RESULTS_DIR
    latents_path = os.path.join(latents_dir, f"latents_{args.name}.npz")
    if not os.path.exists(latents_path):
        raise FileNotFoundError(f"Latents not found: {latents_path}")

    data = np.load(latents_path, allow_pickle=True)
    z_actor = data["z_actor"]       # (N, z_dim)
    actions = data["actions"]       # (N,)
    actor_types = data["actor_types"]  # (N,)

    print(f"\n{'='*60}")
    print(f"V9 YOLO Evaluation (symmetric leakage): {args.name}")
    print(f"  Samples: {len(z_actor)}")
    print(f"  z_actor shape: {z_actor.shape}")
    print(f"  Actions:       {np.unique(actions, return_counts=True)}")
    print(f"  Actor Types:   {np.unique(actor_types, return_counts=True)}")
    print(f"{'='*60}")

    results = {"name": args.name, "n_samples": len(z_actor)}

    # === 1. Overall Action Clustering (NMI) ===
    print(f"\n[Overall Action Clustering]")
    km = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=args.seed)
    labels_km = km.fit_predict(z_actor)
    nmi = normalized_mutual_info_score(actions, labels_km, average_method="arithmetic")
    ari = adjusted_rand_score(actions, labels_km)
    print(f"  NMI = {nmi:.4f}")
    print(f"  ARI = {ari:.4f}")
    results["overall_nmi"] = round(float(nmi), 4)
    results["overall_ari"] = round(float(ari), 4)

    # === 2. Per-ActorType NMI ===
    print(f"\n[Per-ActorType NMI]")
    nmi_per_type = []
    for t in np.unique(actor_types):
        idx = actor_types == t
        if idx.sum() < 10:
            print(f"  ActorType {int(t)}: N/A (n={idx.sum()}, too few)")
            continue
        z_t = z_actor[idx]
        a_t = actions[idx]
        n_clusters_t = min(args.n_clusters, len(np.unique(a_t)))
        if n_clusters_t < 2:
            continue
        km_t = KMeans(n_clusters=n_clusters_t, n_init=10, random_state=args.seed)
        lab_t = km_t.fit_predict(z_t)
        nmi_t = normalized_mutual_info_score(a_t, lab_t, average_method="arithmetic")
        nmi_per_type.append(nmi_t)
        print(f"  ActorType {int(t)}: NMI = {nmi_t:.4f} (n={idx.sum()})")
    nmi_per_avg = float(np.mean(nmi_per_type)) if nmi_per_type else 0.0
    print(f"  Avg: NMI = {nmi_per_avg:.4f}")
    results["per_actor_type_nmi"] = [round(float(x), 4) for x in nmi_per_type]
    results["per_actor_type_nmi_avg"] = round(nmi_per_avg, 4)

    # === 3. Actor Type Leakage (z -> actor_type) ===
    print(f"\n[Actor Type Leakage]")
    n_types = len(np.unique(actor_types))
    chance_type = 1.0 / n_types
    n = len(z_actor)
    idx_perm = np.random.RandomState(args.seed).permutation(n)
    n_tr = int(0.8 * n)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(z_actor[idx_perm[:n_tr]], actor_types[idx_perm[:n_tr]])
    leakage = clf.score(z_actor[idx_perm[n_tr:]], actor_types[idx_perm[n_tr:]])
    print(f"  z -> actor_type acc = {leakage:.4f}  (chance = {chance_type:.4f})")
    results["actor_type_leakage_acc"] = round(float(leakage), 4)

    # === 4. Action Probe (z -> action) ===
    print(f"\n[Action Probe]")
    clf_act = LogisticRegression(max_iter=1000, C=1.0)
    clf_act.fit(z_actor[idx_perm[:n_tr]], actions[idx_perm[:n_tr]])
    act_acc = clf_act.score(z_actor[idx_perm[n_tr:]], actions[idx_perm[n_tr:]])
    act_chance = 1.0 / args.n_clusters
    print(f"  z -> action acc = {act_acc:.4f}  (chance = {act_chance:.4f})")
    results["action_probe_acc"] = round(float(act_acc), 4)

    # === 5. Action Probe given Actor Type (conditional) ===
    print(f"\n[Action Probe given Actor Type]")
    act_probe_per = []
    for t in np.unique(actor_types):
        idx_t = actor_types == t
        n_t = idx_t.sum()
        if n_t < 10:
            print(f"  ActorType {int(t)}: N/A (n={n_t}, too few)")
            continue
        a_t = actions[idx_t]
        if len(np.unique(a_t)) < 2:
            print(f"  ActorType {int(t)}: N/A (only 1 action class, n={n_t})")
            continue
        idx_perm_t = np.random.RandomState(args.seed).permutation(n_t)
        n_tr_t = max(int(0.8 * n_t), 2)
        z_t = z_actor[idx_t]
        clf_t = LogisticRegression(max_iter=1000, C=1.0)
        clf_t.fit(z_t[idx_perm_t[:n_tr_t]], a_t[idx_perm_t[:n_tr_t]])
        acc_t = clf_t.score(z_t[idx_perm_t[n_tr_t:]], a_t[idx_perm_t[n_tr_t:]])
        act_probe_per.append(acc_t)
        print(f"  ActorType {int(t)}: acc = {acc_t:.4f} (n={n_t}, tr={n_tr_t}, te={n_t - n_tr_t})")
    act_probe_cond_avg = float(np.mean(act_probe_per)) if act_probe_per else 0.0
    print(f"  Avg Action Probe (given actor_type) = {act_probe_cond_avg:.4f}")
    print(f"  Overall Action Probe                = {act_acc:.4f}")
    if act_probe_cond_avg > act_acc:
        print(f"  -> given actor_type > overall: z 在 actor_type 内有更强的 action 解码能力")
    else:
        print(f"  -> given actor_type <= overall: actor_type 信息对 action 解码帮助不大")
    results["action_probe_given_type_avg"] = round(act_probe_cond_avg, 4)
    results["action_probe_given_type"] = [round(float(x), 4) for x in act_probe_per]

    # === 6. Leakage given Action (conditional, symmetric) ===
    print(f"\n[Leakage given Action]")
    unique_actions = np.unique(actions)
    leak_per_action = []
    action_names = ["stay", "up", "down", "left", "right"]
    for a in unique_actions:
        idx_a = actions == a
        n_a = idx_a.sum()
        if n_a < 10:
            name = action_names[int(a)] if int(a) < len(action_names) else f"act{int(a)}"
            print(f"  Action {int(a)}({name}): N/A (n={n_a}, too few)")
            continue
        n_a = idx_a.sum()
        idx_perm_a = np.random.RandomState(args.seed).permutation(n_a)
        n_tr_a = int(0.8 * n_a)
        z_a = z_actor[idx_a]
        t_a = actor_types[idx_a]
        if len(np.unique(t_a)) < 2:
            name = action_names[int(a)] if int(a) < len(action_names) else f"act{int(a)}"
            print(f"  Action {int(a)}({name}): N/A (only 1 actor_type, n={n_a})")
            continue
        clf_a = LogisticRegression(max_iter=1000, C=1.0)
        clf_a.fit(z_a[idx_perm_a[:n_tr_a]], t_a[idx_perm_a[:n_tr_a]])
        acc_a = clf_a.score(z_a[idx_perm_a[n_tr_a:]], t_a[idx_perm_a[n_tr_a:]])
        leak_per_action.append(acc_a)
        name = action_names[int(a)] if int(a) < len(action_names) else f"act{int(a)}"
        print(f"  Action {int(a)}({name}): acc = {acc_a:.4f} (n={n_a})")
    leak_given_action_avg = float(np.mean(leak_per_action)) if leak_per_action else 0.0
    print(f"  Avg Leakage (given action) = {leak_given_action_avg:.4f}")
    print(f"  Overall Leakage            = {leakage:.4f}")
    if leak_given_action_avg > leakage:
        print(f"  -> given action > overall: 给定动作后, actor 身份更容易判断")
    else:
        print(f"  -> given action <= overall: actor 身份已充分编码在 z 中")
    results["leakage_given_action_avg"] = round(leak_given_action_avg, 4)
    results["leakage_given_action"] = [round(float(x), 4) for x in leak_per_action]

    # === 7. Symmetric Summary ===
    print(f"\n[Probe 对称性总结]")
    print(f"  {'':<30} {'z -> action':>15} {'z -> actor_type':>15}")
    print(f"  {'Overall (无条件)':<30} {act_acc:>15.4f} {leakage:>15.4f}")
    print(f"  {'Given ActorType (条件化)':<30} {act_probe_cond_avg:>15.4f} {'N/A (type=条件本身)':>15}")
    print(f"  {'Given Action (条件化)':<30} {'N/A (action=条件本身)':>15} {leak_given_action_avg:>15.4f}")

    # Save
    save_path = os.path.join(RESULTS_DIR, f"eval_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
