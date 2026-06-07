"""
V3 隐动作分析脚本 — UMAP 可视化 + 聚类质量评估。

V3 输出单向量 z_mu: (B*(T-1), latent_dim)
需要与聚合后的全局 GT 动作对齐进行分析。

分析内容：
1. UMAP 可视化：按动作类别着色
2. 聚类质量：ARI（按动作聚类）
3. 动作分类线性探针
4. 隐动作各维度方差分析
5. 与 V2 对比：单向量 vs 多向量的隐动作空间结构

用法:
  python analyze_latent_v3.py --name v3_baseline
  python analyze_latent_v3.py --name v3_no_obj_st
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
NUM_ACTIONS = 5


def collect_latents(model, dataset, device, max_samples=2000):
    """
    收集 V3 模型的隐动作和对应标签。

    V3 输出:
        z_mu: (B*(T-1), latent_dim) — 全局单向量
        action_logits: (B, T-1, num_actions) — 全局动作预测

    需要将 GT 动作聚合为全局标签（众数）。
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)

    all_z_mu = []
    all_global_actions = []
    all_per_actor_actions = []  # 保留每主体动作用于分析

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(all_z_mu) * 64 >= max_samples:
                break

            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"]  # (B, T-1, A)

            outputs = model({"videos": videos, "masks": masks})

            # z_mu: (B*(T-1), latent_dim) → reshape to (B, T-1, latent_dim)
            z_mu = outputs["z_mu"]
            B_actual = videos.shape[0]
            T1 = videos.shape[1] - 1
            z_mu = z_mu.reshape(B_actual, T1, -1)

            all_z_mu.append(z_mu.cpu())
            all_per_actor_actions.append(actions)

            # 聚合 GT 动作：众数
            valid_mask = actions >= 0  # (B, T-1, A)
            global_actions = torch.full((B_actual, T1), -1, dtype=torch.long)
            for b in range(B_actual):
                for t in range(T1):
                    valid_acts = actions[b, t][valid_mask[b, t]]
                    if len(valid_acts) > 0:
                        counts = torch.bincount(valid_acts, minlength=NUM_ACTIONS)
                        global_actions[b, t] = counts.argmax().item()

            all_global_actions.append(global_actions)

    z_mu = torch.cat(all_z_mu, dim=0)              # (N, T-1, D)
    global_actions = torch.cat(all_global_actions, dim=0)  # (N, T-1)
    per_actor_actions = torch.cat(all_per_actor_actions, dim=0)  # (N, T-1, A)

    return z_mu, global_actions, per_actor_actions


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
    """线性探针：用线性分类器从 z_mu 预测标签。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    scores = cross_val_score(clf, z_flat, labels, cv=5, scoring="accuracy")
    return scores.mean(), scores.std()


def compute_ari(z_flat, labels, n_clusters):
    """ARI：聚类与真实标签的一致性。"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(z_flat)
    return adjusted_rand_score(labels, pred)


def plot_latent_variance(z_mu_np, save_path, title):
    """绘制隐动作各维度方差。"""
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
    """绘制各动作类别之间的距离矩阵。"""
    means = []
    for act in range(NUM_ACTIONS):
        mask = action_labels == act
        if mask.sum() > 0:
            means.append(z_mu_np[mask].mean(axis=0))
        else:
            means.append(np.zeros(z_mu_np.shape[1]))

    means = np.array(means)
    # 计算余弦相似度矩阵
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    means_norm = means / (norms + 1e-8)
    cos_sim = means_norm @ means_norm.T

    # 计算欧氏距离矩阵
    dist_matrix = np.zeros((NUM_ACTIONS, NUM_ACTIONS))
    for i in range(NUM_ACTIONS):
        for j in range(NUM_ACTIONS):
            dist_matrix[i, j] = np.linalg.norm(means[i] - means[j])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 余弦相似度
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

    # 欧氏距离
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


def analyze_action_consensus(per_actor_actions, save_path):
    """
    分析多主体场景中动作一致性。

    当多个主体执行不同动作时，众数聚合会丢失信息。
    统计各种情况的比例。
    """
    # per_actor_actions: (N, T-1, A)
    N, T1, A = per_actor_actions.shape
    valid = per_actor_actions >= 0

    all_consensus = []
    for n in range(N):
        for t in range(T1):
            valid_acts = per_actor_actions[n, t][valid[n, t]]
            if len(valid_acts) > 1:
                unique_acts = torch.unique(valid_acts)
                all_consensus.append(1 if len(unique_acts) == 1 else 0)

    consensus_rate = np.mean(all_consensus) if all_consensus else 0

    # 统计每种动作组合出现的频率
    from collections import Counter
    action_combos = Counter()
    for n in range(N):
        for t in range(T1):
            valid_acts = per_actor_actions[n, t][valid[n, t]]
            if len(valid_acts) > 0:
                combo = tuple(sorted(valid_acts.tolist()))
                action_combos[combo] += 1

    total = sum(action_combos.values())

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Top 15 最常见组合
    top_combos = action_combos.most_common(15)
    labels = []
    counts = []
    for combo, count in top_combos:
        label = "+".join([ACTION_NAMES[a] for a in combo])
        labels.append(label)
        counts.append(count)

    bars = ax.barh(range(len(labels)), [c / total for c in counts], color="#6366f1", alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Frequency")
    ax.set_title(f"Action Combination Distribution (Consensus Rate: {consensus_rate:.1%})", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    return consensus_rate, action_combos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=2000)
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

    # 读取配置
    use_obj_st_attention = True
    if os.path.exists(results_json_path):
        with open(results_json_path) as f:
            prev_results = json.load(f)
        use_obj_st_attention = prev_results.get("use_obj_st_attention", True)
        print(f"  Loaded config: use_obj_st_attention={use_obj_st_attention}")

    model = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8,
        max_actors=4, num_actions=5, img_size=256,
        use_obj_st_attention=use_obj_st_attention,
        obj_st_heads=8, obj_st_layers=2,
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

    # 收集隐动作
    print("Collecting latent codes...")
    z_mu, global_actions, per_actor_actions = collect_latents(
        model, eval_dataset, device, args.max_samples
    )

    # z_mu: (N, T-1, D) → 取最后一帧
    z_last = z_mu[:, -1, :]                    # (N, D)
    global_act_last = global_actions[:, -1]     # (N,)
    per_actor_last = per_actor_actions[:, -1, :]  # (N, A)

    N, D = z_last.shape
    print(f"  Collected {N} samples, {D} latent dims")

    # 过滤无效样本
    valid_mask = global_act_last >= 0
    z_valid = z_last[valid_mask].numpy()
    act_valid = global_act_last[valid_mask].numpy()
    per_actor_valid = per_actor_last[valid_mask]

    M = z_valid.shape[0]
    print(f"  Valid samples: {M}")

    # ===== 分析 0: 动作一致性分析 =====
    print("\n--- Action Consensus Analysis ---")
    consensus_rate, action_combos = analyze_action_consensus(
        per_actor_actions,
        os.path.join(RESULTS_DIR, f"action_consensus_{args.name}.png"),
    )
    print(f"  Consensus rate (all actors same action): {consensus_rate:.1%}")
    top5 = action_combos.most_common(5)
    for combo, count in top5:
        label = "+".join([ACTION_NAMES[a] for a in combo])
        print(f"    {label}: {count} ({count/sum(action_combos.values()):.1%})")

    # ===== 分析 1: UMAP 可视化 =====
    print("\n--- UMAP Visualization ---")

    # 按动作着色
    plot_umap(
        z_valid, act_valid, ACTION_NAMES,
        f"V3 Latent Actions (colored by ACTION) - {args.name}",
        os.path.join(RESULTS_DIR, f"umap_action_{args.name}.png"),
        colors=ACTION_COLORS,
    )

    # ===== 分析 2: 聚类质量 =====
    print("\n--- Clustering Quality ---")

    ari_action = compute_ari(z_valid, act_valid, NUM_ACTIONS)
    print(f"  ARI (action clustering): {ari_action:.4f}")

    # ===== 分析 3: 线性探针 =====
    print("\n--- Linear Probe ---")

    acc_action, std_action = linear_probe(z_valid, act_valid, NUM_ACTIONS)
    print(f"  Action classification: {acc_action:.2%} ± {std_action:.2%}")

    # ===== 分析 4: 隐动作方差分析 =====
    print("\n--- Latent Variance Analysis ---")

    per_dim_var, active_dims = plot_latent_variance(
        z_valid,
        os.path.join(RESULTS_DIR, f"latent_variance_{args.name}.png"),
        f"V3 Latent Variance per Dimension - {args.name}",
    )
    print(f"  Active dims (var > 0.01): {active_dims}/{D}")
    print(f"  Mean variance: {per_dim_var.mean():.4f}")
    print(f"  Max variance: {per_dim_var.max():.4f}")
    print(f"  Min variance: {per_dim_var.min():.6f}")

    # ===== 分析 5: 动作间距离矩阵 =====
    print("\n--- Action Distance Matrix ---")

    cos_sim, dist_matrix = plot_action_distance_matrix(
        z_valid, act_valid,
        os.path.join(RESULTS_DIR, f"action_distance_{args.name}.png"),
        f"V3 Action Distance Matrix - {args.name}",
    )

    # 打印关键距离
    for act_i in range(NUM_ACTIONS):
        for act_j in range(act_i + 1, NUM_ACTIONS):
            print(f"  {ACTION_NAMES[act_i]} vs {ACTION_NAMES[act_j]}: "
                  f"cos_sim={cos_sim[act_i, act_j]:.3f}, "
                  f"dist={dist_matrix[act_i, act_j]:.3f}")

    # ===== 分析 6: 一致 vs 不一致场景的隐动作质量 =====
    print("\n--- Consensus vs Disagreement Analysis ---")

    # 将样本分为"所有主体一致"和"主体间不一致"两组
    consensus_z = []
    disagreement_z = []
    consensus_acts = []
    disagreement_acts = []

    for n in range(N):
        if global_act_last[n] < 0:
            continue
        valid_acts = per_actor_last[n][per_actor_last[n] >= 0]
        if len(valid_acts) <= 1:
            continue
        unique_acts = torch.unique(valid_acts)
        if len(unique_acts) == 1:
            consensus_z.append(z_last[n].numpy())
            consensus_acts.append(global_act_last[n].item())
        else:
            disagreement_z.append(z_last[n].numpy())
            disagreement_acts.append(global_act_last[n].item())

    consensus_z = np.array(consensus_z)
    disagreement_z = np.array(disagreement_z)
    consensus_acts = np.array(consensus_acts)
    disagreement_acts = np.array(disagreement_acts)

    print(f"  Consensus samples: {len(consensus_z)}")
    print(f"  Disagreement samples: {len(disagreement_z)}")

    if len(consensus_z) > 10:
        acc_cons, _ = linear_probe(consensus_z, consensus_acts, NUM_ACTIONS)
        print(f"  Action accuracy (consensus): {acc_cons:.2%}")

    if len(disagreement_z) > 10:
        acc_dis, _ = linear_probe(disagreement_z, disagreement_acts, NUM_ACTIONS)
        print(f"  Action accuracy (disagreement): {acc_dis:.2%}")

    # UMAP: 分别可视化一致和不一致场景
    if len(consensus_z) > 50:
        plot_umap(
            consensus_z, consensus_acts, ACTION_NAMES,
            f"V3 Consensus Scenes (colored by ACTION) - {args.name}",
            os.path.join(RESULTS_DIR, f"umap_consensus_{args.name}.png"),
            colors=ACTION_COLORS,
        )

    if len(disagreement_z) > 50:
        plot_umap(
            disagreement_z, disagreement_acts, ACTION_NAMES,
            f"V3 Disagreement Scenes (colored by ACTION) - {args.name}",
            os.path.join(RESULTS_DIR, f"umap_disagreement_{args.name}.png"),
            colors=ACTION_COLORS,
        )

    # ===== 保存结果 =====
    results = {
        "model_name": args.name,
        "architecture": "v3_single_vector",
        "use_obj_st_attention": use_obj_st_attention,
        "num_samples": N,
        "num_valid_samples": M,
        "consensus_rate": round(float(consensus_rate), 4),
        "ari_action": round(float(ari_action), 4),
        "linear_probe_action": round(float(acc_action), 4),
        "active_dims": int(active_dims),
        "total_dims": D,
        "mean_variance": round(float(per_dim_var.mean()), 4),
    }

    if len(consensus_z) > 10:
        results["linear_probe_consensus"] = round(float(acc_cons), 4)
    if len(disagreement_z) > 10:
        results["linear_probe_disagreement"] = round(float(acc_dis), 4)

    save_path = os.path.join(RESULTS_DIR, f"analysis_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {save_path}")


if __name__ == "__main__":
    main()
