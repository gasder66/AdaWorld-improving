"""
隐动作分析脚本 — UMAP 可视化 + 聚类质量评估。

分析训练好的 Mask-Guided 模型的隐动作 z_mu：
1. UMAP 可视化：按动作类别着色 vs 按主体着色
2. 聚类质量：同动作 z_mu 是否聚集，不同主体是否分离
3. 动作分类线性探针：z_mu 线性可分性
4. 主体间余弦相似度矩阵

用法:
  python analyze_latent.py --name v2_baseline_1k
  python analyze_latent.py --name v2_no_interact_1k
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
NUM_ACTIONS = 5
ACTOR_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
ACTION_COLORS = ["#gray", "#ff7f00", "#4daf4a", "#377eb8", "#e41a1c"]


def collect_latents(model, dataset, device, max_samples=2000):
    """收集隐动作和对应标签。"""
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

            # z_mu: (B*(T-1), A, D) → reshape to (B, T-1, A, D)
            z_mu = outputs["z_mu"]  # (B*(T-1), A, D)
            B_actual = videos.shape[0]
            T1 = videos.shape[1] - 1
            z_mu = z_mu.reshape(B_actual, T1, -1, z_mu.shape[-1])

            all_z_mu.append(z_mu.cpu())
            all_actions.append(actions)
            all_valid.append(actions >= 0)

    z_mu = torch.cat(all_z_mu, dim=0)   # (N, T-1, A, D)
    actions = torch.cat(all_actions, dim=0)
    valid = torch.cat(all_valid, dim=0)

    return z_mu, actions, valid


def plot_umap(z_flat, labels, label_names, title, save_path, color_map="tab10"):
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
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[ACTOR_COLORS[label] if label < len(ACTOR_COLORS) else "gray"] if color_map == "actors" else None,
            label=label_names[label] if label < len(label_names) else str(label),
            alpha=0.5, s=8,
            cmap=color_map if color_map != "actors" else None,
        )
    ax.legend(markerscale=3)
    ax.set_title(title)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cuda:0")

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "results", "latent_analysis"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 加载模型 — 支持多个实验目录
    exp_dirs = [
        "yolo_closed_loop",  # YOLO+ByteTrack 闭环实验
        "slot_attention_exp_v2",  # V2 基线实验
        "bbox_mask_exp",  # bbox 填充实验
    ]
    
    ckpt_path = None
    results_json_path = None
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
        print("Please run training first.")
        return
    use_interaction = True  # default
    if os.path.exists(results_json_path):
        with open(results_json_path) as f:
            prev_results = json.load(f)
        use_interaction = prev_results.get("use_interaction", True)
        print(f"  Loaded config: use_interaction={use_interaction}")

    model = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8,
        max_actors=4, num_actions=5, img_size=256,
        use_interaction=use_interaction,
        interaction_heads=4, interaction_layers=2,
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
    z_mu, actions, valid = collect_latents(model, eval_dataset, device, args.max_samples)

    # z_mu: (N, T-1, A, D) → 取最后一帧
    z_last = z_mu[:, -1, :, :]     # (N, A, D)
    act_last = actions[:, -1, :]    # (N, A)
    valid_last = valid[:, -1, :]    # (N, A)

    N, A, D = z_last.shape
    print(f"  Collected {N} samples, {A} actors, {D} latent dims")

    # ===== 分析 1: UMAP 可视化 =====
    print("\n--- UMAP Visualization ---")

    # 展平: 每个有效 (sample, actor) 对作为一个数据点
    z_list, action_labels, actor_labels = [], [], []
    for a in range(A):
        mask_a = valid_last[:, a]  # (N,)
        z_list.append(z_last[mask_a, a, :].numpy())
        action_labels.append(act_last[mask_a, a].numpy())
        actor_labels.append(np.full(mask_a.sum().item(), a))

    z_flat = np.concatenate(z_list, axis=0)          # (M, D)
    action_labels = np.concatenate(action_labels)     # (M,)
    actor_labels = np.concatenate(actor_labels)       # (M,)
    M = z_flat.shape[0]

    print(f"  Total valid (sample, actor) pairs: {M}")

    # UMAP: 按动作着色
    plot_umap(
        z_flat, action_labels, ACTION_NAMES,
        f"Latent Actions (colored by ACTION) - {args.name}",
        os.path.join(RESULTS_DIR, f"umap_action_{args.name}.png"),
        color_map="Set1",
    )

    # UMAP: 按主体着色
    plot_umap(
        z_flat, actor_labels, [f"Actor {i}" for i in range(A)],
        f"Latent Actions (colored by ACTOR) - {args.name}",
        os.path.join(RESULTS_DIR, f"umap_actor_{args.name}.png"),
        color_map="actors",
    )

    # ===== 分析 2: 动作聚类质量 =====
    print("\n--- Clustering Quality ---")

    # ARI: 按动作聚类
    ari_action = compute_ari(z_flat, action_labels, NUM_ACTIONS)
    print(f"  ARI (action clustering): {ari_action:.4f}")

    # ARI: 按主体聚类
    ari_actor = compute_ari(z_flat, actor_labels, A)
    print(f"  ARI (actor clustering):  {ari_actor:.4f}")

    # ===== 分析 3: 线性探针 =====
    print("\n--- Linear Probe ---")

    acc_action, std_action = linear_probe(z_flat, action_labels, NUM_ACTIONS)
    print(f"  Action classification: {acc_action:.2%} ± {std_action:.2%}")

    acc_actor, std_actor = linear_probe(z_flat, actor_labels, A)
    print(f"  Actor classification:   {acc_actor:.2%} ± {std_actor:.2%}")

    # ===== 分析 4: 跨主体动作可分性 =====
    print("\n--- Cross-Actor Action Separability ---")

    # 对每对主体，计算同动作 z_mu 的距离 vs 不同动作 z_mu 的距离
    for a1 in range(A):
        for a2 in range(a1 + 1, A):
            # 同动作：主体 a1 和 a2 执行相同动作时的 z 距离
            z_a1 = z_last[valid_last[:, a1], a1, :]  # (N1, D)
            z_a2 = z_last[valid_last[:, a2], a2, :]  # (N2, D)
            act_a1 = act_last[valid_last[:, a1], a1]
            act_a2 = act_last[valid_last[:, a2], a2]

            # 找同动作对
            same_action_dists = []
            diff_action_dists = []
            n_pairs = min(200, len(z_a1), len(z_a2))
            for i in range(n_pairs):
                for j in range(n_pairs):
                    d = (z_a1[i] - z_a2[j]).norm().item()
                    if act_a1[i] == act_a2[j]:
                        same_action_dists.append(d)
                    else:
                        diff_action_dists.append(d)

            same_mean = np.mean(same_action_dists) if same_action_dists else 0
            diff_mean = np.mean(diff_action_dists) if diff_action_dists else 0
            ratio = same_mean / (diff_mean + 1e-8)
            print(f"  Actor {a1} vs {a2}: same_action_dist={same_mean:.3f}, "
                  f"diff_action_dist={diff_mean:.3f}, ratio={ratio:.3f}")

    # ===== 分析 5: 主体间余弦相似度 =====
    print("\n--- Per-Action Cosine Similarity ---")

    for act in range(NUM_ACTIONS):
        # 收集所有执行该动作的 z_mu，按主体分组
        means = []
        for a in range(A):
            mask = valid_last[:, a] & (act_last[:, a] == act)
            if mask.sum() > 0:
                mean_z = z_last[mask, a, :].mean(0)
                means.append(mean_z)

        if len(means) >= 2:
            means = torch.stack(means)
            means_norm = F.normalize(means, dim=-1)
            cos_sim = (means_norm @ means_norm.T).numpy()
            off_diag = cos_sim[np.triu_indices(len(means), k=1)]
            print(f"  Action '{ACTION_NAMES[act]}': "
                  f"cross-actor cos_sim = {off_diag.mean():.3f} ± {off_diag.std():.3f}")

    # ===== 保存结果 =====
    results = {
        "model_name": args.name,
        "num_samples": N,
        "num_valid_pairs": M,
        "ari_action": round(float(ari_action), 4),
        "ari_actor": round(float(ari_actor), 4),
        "linear_probe_action": round(float(acc_action), 4),
        "linear_probe_actor": round(float(acc_actor), 4),
    }

    save_path = os.path.join(RESULTS_DIR, f"analysis_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {save_path}")


if __name__ == "__main__":
    main()