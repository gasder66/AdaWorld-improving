"""
V3 隐动作分析脚本 — UMAP 可视化 + 聚类质量评估。

支持单向量/多向量模式：
- 单向量: z_mu (B*(T-1), latent_dim)，用众数聚合 GT
- 多向量: z_mu (B*(T-1), A+1, latent_dim)，每主体独立 GT

分析内容：
1. UMAP 可视化：按动作着色 / 按主体着色
2. 聚类质量：ARI
3. 动作分类线性探针
4. 隐动作方差分析
5. 多向量模式：跨主体余弦相似度

用法:
  python analyze_latent_v3.py --name v3_baseline
  python analyze_latent_v3.py --name v3_multi --multi_vector
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset

ACTION_NAMES = ["stay", "up", "down", "left", "right"]
ACTION_COLORS = ["#999999", "#ff7f00", "#4daf4a", "#377eb8", "#e41a1c"]
ACTOR_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
NUM_ACTIONS = 5


def collect_latents_single(model, dataset, device, max_samples=2000):
    """收集单向量模式的隐动作。"""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)

    all_z_mu = []
    all_global_actions = []
    all_per_actor_actions = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(all_z_mu) * 64 >= max_samples:
                break

            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"]

            outputs = model({"videos": videos, "masks": masks})

            z_mu = outputs["z_mu"]
            B_actual = videos.shape[0]
            T1 = videos.shape[1] - 1
            z_mu = z_mu.reshape(B_actual, T1, -1)

            all_z_mu.append(z_mu.cpu())
            all_per_actor_actions.append(actions)

            valid_mask = actions >= 0
            global_actions = torch.full((B_actual, T1), -1, dtype=torch.long)
            for b in range(B_actual):
                for t in range(T1):
                    valid_acts = actions[b, t][valid_mask[b, t]]
                    if len(valid_acts) > 0:
                        counts = torch.bincount(valid_acts, minlength=NUM_ACTIONS)
                        global_actions[b, t] = counts.argmax().item()

            all_global_actions.append(global_actions)

    z_mu = torch.cat(all_z_mu, dim=0)
    global_actions = torch.cat(all_global_actions, dim=0)
    per_actor_actions = torch.cat(all_per_actor_actions, dim=0)

    return z_mu, global_actions, per_actor_actions


def collect_latents_multi(model, dataset, device, max_samples=2000):
    """收集多向量模式的隐动作。"""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)

    all_z_mu = []
    all_actions = []
    all_valid = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(all_z_mu) * 64 >= max_samples:
                break

            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"]  # (B, T-1, A)

            outputs = model({"videos": videos, "masks": masks})

            # z_mu: (B*(T-1), A+1, latent_dim) → (B, T-1, A+1, latent_dim)
            z_mu = outputs["z_mu"]
            B_actual = videos.shape[0]
            T1 = videos.shape[1] - 1
            A1 = z_mu.shape[-2]
            z_mu = z_mu.reshape(B_actual, T1, A1, -1)

            all_z_mu.append(z_mu.cpu())
            all_actions.append(actions)
            all_valid.append(actions >= 0)

    z_mu = torch.cat(all_z_mu, dim=0)          # (N, T-1, A+1, D)
    actions = torch.cat(all_actions, dim=0)     # (N, T-1, A)
    valid = torch.cat(all_valid, dim=0)         # (N, T-1, A)

    return z_mu, actions, valid


def plot_umap(z_flat, labels, label_names, title, save_path, colors=None):
    """绘制 UMAP 可视化。"""
    try:
        import umap
    except ImportError:
        print("  [WARN] umap-learn not installed, skipping UMAP")
        return

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(z_flat)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = labels == label
        color = colors[label] if colors and label < len(colors) else None
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color,
            label=label_names[label] if label < len(label_names) else str(label),
            alpha=0.5, s=8,
        )
    ax.legend(markerscale=3)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def linear_probe(z_flat, labels, n_classes):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    scores = cross_val_score(clf, z_flat, labels, cv=5, scoring="accuracy")
    return scores.mean(), scores.std()


def compute_ari(z_flat, labels, n_clusters):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(z_flat)
    return adjusted_rand_score(labels, pred)


def plot_latent_variance(z_mu_np, save_path, title):
    per_dim_var = z_mu_np.var(axis=0)
    dim_indices = np.arange(len(per_dim_var))

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    bars = ax.bar(dim_indices, per_dim_var, color="#6366f1", alpha=0.8)
    ax.axhline(y=0.01, color="#ef4444", linestyle="--", linewidth=1, label="Threshold (0.01)")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Variance")
    ax.set_title(title, fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    active = (per_dim_var > 0.01).sum()
    return per_dim_var, active


def plot_action_distance_matrix(z_mu_np, action_labels, save_path, title):
    means = []
    for act in range(NUM_ACTIONS):
        mask = action_labels == act
        if mask.sum() > 0:
            means.append(z_mu_np[mask].mean(axis=0))
        else:
            means.append(np.zeros(z_mu_np.shape[1]))

    means = np.array(means)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    means_norm = means / (norms + 1e-8)
    cos_sim = means_norm @ means_norm.T

    dist_matrix = np.zeros((NUM_ACTIONS, NUM_ACTIONS))
    for i in range(NUM_ACTIONS):
        for j in range(NUM_ACTIONS):
            dist_matrix[i, j] = np.linalg.norm(means[i] - means[j])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(cos_sim, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(NUM_ACTIONS))
    ax1.set_yticks(range(NUM_ACTIONS))
    ax1.set_xticklabels(ACTION_NAMES, fontsize=10)
    ax1.set_yticklabels(ACTION_NAMES, fontsize=10)
    ax1.set_title("Cosine Similarity", fontsize=12)
    for i in range(NUM_ACTIONS):
        for j in range(NUM_ACTIONS):
            ax1.text(j, i, f"{cos_sim[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(dist_matrix, cmap="YlOrRd")
    ax2.set_xticks(range(NUM_ACTIONS))
    ax2.set_yticks(range(NUM_ACTIONS))
    ax2.set_xticklabels(ACTION_NAMES, fontsize=10)
    ax2.set_yticklabels(ACTION_NAMES, fontsize=10)
    ax2.set_title("Euclidean Distance", fontsize=12)
    for i in range(NUM_ACTIONS):
        for j in range(NUM_ACTIONS):
            ax2.text(j, i, f"{dist_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im2, ax=ax2)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    return cos_sim, dist_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--multi_vector", action="store_true",
                        help="多向量模式分析")
    args = parser.parse_args()

    device = torch.device("cuda:0")

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "results", "latent_analysis_v3"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 加载模型
    ckpt_path = None
    results_json_path = None
    exp_dirs = ["v3_obj_st_attention", "slot_attention_exp_v2", "bbox_mask_exp"]

    for exp_dir in exp_dirs:
        path = os.path.join(
            os.path.dirname(__file__), "..", "results", exp_dir,
            f"model_{args.name}.pt"
        )
        if os.path.exists(path):
            ckpt_path = path
            results_json_path = os.path.join(
                os.path.dirname(__file__), "..", "results", exp_dir,
                f"results_{args.name}.json"
            )
            print(f"Found model in: {exp_dir}")
            break

    if ckpt_path is None:
        print(f"Checkpoint not found for name={args.name}")
        return

    use_obj_st_attention = True
    multi_vector = args.multi_vector
    if os.path.exists(results_json_path):
        with open(results_json_path) as f:
            prev_results = json.load(f)
        use_obj_st_attention = prev_results.get("use_obj_st_attention", True)
        multi_vector = prev_results.get("multi_vector", multi_vector)
        print(f"  Loaded config: use_obj_st_attention={use_obj_st_attention}, multi_vector={multi_vector}")

    model = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8,
        max_actors=4, num_actions=5, img_size=256,
        use_obj_st_attention=use_obj_st_attention,
        obj_st_heads=8, obj_st_layers=2,
        multi_vector=multi_vector,
        use_grad_checkpointing=False,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {ckpt_path}")

    # 加载验证集
    data_root = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
    )
    eval_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"),
        num_frames=5, output_format="t h w c",
    )

    if multi_vector:
        _analyze_multi_vector(model, eval_dataset, device, args, RESULTS_DIR)
    else:
        _analyze_single_vector(model, eval_dataset, device, args, RESULTS_DIR)


def _analyze_single_vector(model, dataset, device, args, results_dir):
    """单向量模式分析。"""
    print("\n=== 单向量模式分析 ===")
    print("Collecting latent codes...")
    z_mu, global_actions, per_actor_actions = collect_latents_single(
        model, dataset, device, args.max_samples
    )

    z_last = z_mu[:, -1, :]
    global_act_last = global_actions[:, -1]
    per_actor_last = per_actor_actions[:, -1, :]

    N, D = z_last.shape
    print(f"  Collected {N} samples, {D} latent dims")

    valid_mask = global_act_last >= 0
    z_valid = z_last[valid_mask].numpy()
    act_valid = global_act_last[valid_mask].numpy()
    M = z_valid.shape[0]

    # 1. UMAP
    print("\n--- UMAP Visualization ---")
    plot_umap(
        z_valid, act_valid, ACTION_NAMES,
        f"V3 Single-Vector (colored by ACTION) - {args.name}",
        os.path.join(results_dir, f"umap_action_{args.name}.png"),
        colors=ACTION_COLORS,
    )

    # 2. ARI
    print("\n--- Clustering Quality ---")
    ari_action = compute_ari(z_valid, act_valid, NUM_ACTIONS)
    print(f"  ARI (action): {ari_action:.4f}")

    # 3. 线性探针
    print("\n--- Linear Probe ---")
    acc_action, std_action = linear_probe(z_valid, act_valid, NUM_ACTIONS)
    print(f"  Action: {acc_action:.2%} ± {std_action:.2%}")

    # 4. 方差
    print("\n--- Latent Variance ---")
    per_dim_var, active_dims = plot_latent_variance(
        z_valid,
        os.path.join(results_dir, f"latent_variance_{args.name}.png"),
        f"V3 Single-Vector Variance - {args.name}",
    )
    print(f"  Active dims: {active_dims}/{D}")

    # 5. 动作距离矩阵
    print("\n--- Action Distance Matrix ---")
    cos_sim, dist_matrix = plot_action_distance_matrix(
        z_valid, act_valid,
        os.path.join(results_dir, f"action_distance_{args.name}.png"),
        f"V3 Single-Vector Action Distance - {args.name}",
    )

    # 保存
    results = {
        "model_name": args.name,
        "mode": "single_vector",
        "ari_action": round(float(ari_action), 4),
        "linear_probe_action": round(float(acc_action), 4),
        "active_dims": int(active_dims),
    }
    save_path = os.path.join(results_dir, f"analysis_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {save_path}")


def _analyze_multi_vector(model, dataset, device, args, results_dir):
    """多向量模式分析。"""
    print("\n=== 多向量模式分析 ===")
    print("Collecting latent codes...")
    z_mu, actions, valid = collect_latents_multi(
        model, dataset, device, args.max_samples
    )

    # z_mu: (N, T-1, A+1, D) → 取最后一帧
    z_last = z_mu[:, -1, :, :]       # (N, A+1, D)
    act_last = actions[:, -1, :]      # (N, A)
    valid_last = valid[:, -1, :]      # (N, A)

    N, A1, D = z_last.shape
    A = A1 - 1  # 主体数（不含背景）
    print(f"  Collected {N} samples, {A1} slots (1 bg + {A} actors), {D} latent dims")

    # 展平: 每个有效 (sample, actor) 对作为一个数据点
    # 只取主体（不含背景槽 idx=0）
    z_list, action_labels, actor_labels = [], [], []
    for a in range(A):
        mask_a = valid_last[:, a]
        z_list.append(z_last[mask_a, a + 1, :].numpy())  # idx 0 = 背景
        action_labels.append(act_last[mask_a, a].numpy())
        actor_labels.append(np.full(mask_a.sum().item(), a))

    z_flat = np.concatenate(z_list, axis=0)
    action_labels = np.concatenate(action_labels)
    actor_labels = np.concatenate(actor_labels)
    M = z_flat.shape[0]
    print(f"  Total valid (sample, actor) pairs: {M}")

    # ===== 分析 1: UMAP 可视化 =====
    print("\n--- UMAP Visualization ---")

    # 按动作着色
    plot_umap(
        z_flat, action_labels, ACTION_NAMES,
        f"V3 Multi-Vector (colored by ACTION) - {args.name}",
        os.path.join(results_dir, f"umap_action_{args.name}.png"),
        colors=ACTION_COLORS,
    )

    # 按主体着色
    plot_umap(
        z_flat, actor_labels, [f"Actor {i}" for i in range(A)],
        f"V3 Multi-Vector (colored by ACTOR) - {args.name}",
        os.path.join(results_dir, f"umap_actor_{args.name}.png"),
        colors=ACTOR_COLORS,
    )

    # ===== 分析 2: 聚类质量 =====
    print("\n--- Clustering Quality ---")

    ari_action = compute_ari(z_flat, action_labels, NUM_ACTIONS)
    ari_actor = compute_ari(z_flat, actor_labels, A)
    print(f"  ARI (action clustering): {ari_action:.4f}")
    print(f"  ARI (actor clustering):  {ari_actor:.4f}")

    # ===== 分析 3: 线性探针 =====
    print("\n--- Linear Probe ---")

    acc_action, std_action = linear_probe(z_flat, action_labels, NUM_ACTIONS)
    acc_actor, std_actor = linear_probe(z_flat, actor_labels, A)
    print(f"  Action classification: {acc_action:.2%} ± {std_action:.2%}")
    print(f"  Actor classification:   {acc_actor:.2%} ± {std_actor:.2%}")

    # ===== 分析 4: 方差 =====
    print("\n--- Latent Variance ---")

    per_dim_var, active_dims = plot_latent_variance(
        z_flat,
        os.path.join(results_dir, f"latent_variance_{args.name}.png"),
        f"V3 Multi-Vector Variance - {args.name}",
    )
    print(f"  Active dims: {active_dims}/{D}")
    print(f"  Mean variance: {per_dim_var.mean():.4f}")

    # ===== 分析 5: 动作距离矩阵 =====
    print("\n--- Action Distance Matrix ---")

    cos_sim, dist_matrix = plot_action_distance_matrix(
        z_flat, action_labels,
        os.path.join(results_dir, f"action_distance_{args.name}.png"),
        f"V3 Multi-Vector Action Distance - {args.name}",
    )

    # ===== 分析 6: 跨主体余弦相似度 =====
    print("\n--- Cross-Actor Cosine Similarity ---")

    for act in range(NUM_ACTIONS):
        means = []
        for a in range(A):
            mask = valid_last[:, a] & (act_last[:, a] == act)
            if mask.sum() > 0:
                mean_z = z_last[mask, a + 1, :].mean(0)
                means.append(mean_z)

        if len(means) >= 2:
            means = torch.stack(means)
            means_norm = F.normalize(means, dim=-1)
            cos_sim_actors = (means_norm @ means_norm.T).numpy()
            off_diag = cos_sim_actors[np.triu_indices(len(means), k=1)]
            print(f"  Action '{ACTION_NAMES[act]}': "
                  f"cross-actor cos_sim = {off_diag.mean():.3f} ± {off_diag.std():.3f}")

    # ===== 分析 7: 同动作 vs 不同动作距离 =====
    print("\n--- Same vs Different Action Distance ---")

    # 采样计算
    n_sample = min(200, M)
    idx = np.random.choice(M, n_sample, replace=False)
    z_sample = z_flat[idx]
    act_sample = action_labels[idx]

    same_dists = []
    diff_dists = []
    for i in range(min(100, n_sample)):
        for j in range(i + 1, min(100, n_sample)):
            d = np.linalg.norm(z_sample[i] - z_sample[j])
            if act_sample[i] == act_sample[j]:
                same_dists.append(d)
            else:
                diff_dists.append(d)

    same_mean = np.mean(same_dists) if same_dists else 0
    diff_mean = np.mean(diff_dists) if diff_dists else 0
    ratio = same_mean / (diff_mean + 1e-8)
    print(f"  Same-action dist: {same_mean:.3f}")
    print(f"  Diff-action dist: {diff_mean:.3f}")
    print(f"  Ratio (same/diff): {ratio:.3f} (< 1 means better clustering)")

    # ===== 保存 =====
    results = {
        "model_name": args.name,
        "mode": "multi_vector",
        "use_obj_st_attention": model.obj_st_attention is not None,
        "num_samples": N,
        "num_valid_pairs": M,
        "ari_action": round(float(ari_action), 4),
        "ari_actor": round(float(ari_actor), 4),
        "linear_probe_action": round(float(acc_action), 4),
        "linear_probe_actor": round(float(acc_actor), 4),
        "active_dims": int(active_dims),
        "total_dims": D,
        "mean_variance": round(float(per_dim_var.mean()), 4),
        "same_action_dist": round(float(same_mean), 4),
        "diff_action_dist": round(float(diff_mean), 4),
        "same_diff_ratio": round(float(ratio), 4),
    }

    save_path = os.path.join(results_dir, f"analysis_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {save_path}")


if __name__ == "__main__":
    main()
