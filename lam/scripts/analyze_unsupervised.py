"""
无监督模型隐动作聚类分析。

对无监督训练的模型进行：
1. UMAP 可视化（按 action 和 actor 着色）
2. 聚类质量评估（ARI, NMI）
3. 跨主体动作可分性分析
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset

ACTION_NAMES = ["stay", "up", "down", "left", "right"]
NUM_ACTIONS = 5
COLORS = plt.cm.tab10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="模型名")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    
    # 查找模型
    exp_dir = os.path.join(
        os.path.dirname(__file__), "..", "results", "unsupervised_lam"
    )
    ckpt_path = os.path.join(exp_dir, f"model_{args.name}.pt")
    results_json_path = os.path.join(exp_dir, f"results_{args.name}.json")
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    # 读取配置
    if os.path.exists(results_json_path):
        with open(results_json_path) as f:
            results_config = json.load(f)
        use_interaction = results_config.get("use_interaction", True)
        print(f"Config loaded: interaction={use_interaction}")
    else:
        use_interaction = True
        print(f"Config not found, default: interaction=True")

    # 数据集
    data_root = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
    )
    eval_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"),
        num_frames=5,
        output_format="t h w c",
    )
    print(f"Eval dataset: {len(eval_dataset)} samples")

    # 模型
    model = LatentActionModel(
        in_dim=3,
        model_dim=256,
        latent_dim=32,
        patch_size=16,
        enc_blocks=4,
        dec_blocks=4,
        num_heads=8,
        max_actors=4,
        num_actions=NUM_ACTIONS,
        use_interaction=use_interaction,
        interaction_heads=4,
        interaction_layers=2,
        use_grad_checkpointing=False,
    ).to(device)
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Model loaded: {ckpt_path}")
    
    # 结果保存
    out_dir = os.path.join(exp_dir, "latent_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # 收集隐动作和动作标签
    all_z_mu = []  # (B*(T-1), A, D)
    all_actions = []
    all_valid = []
    
    dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=64, num_workers=4
    )
    
    with torch.no_grad():
        collected = 0
        for batch in dataloader:
            if collected >= args.max_samples:
                break
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"].to(device)
            
            outputs = model({"videos": videos, "masks": masks})
            z_mu = outputs["z_mu"]  # (B*(T-1), A, D)
            
            B = videos.shape[0]
            T1 = videos.shape[1] - 1
            A = 4
            z_mu = z_mu.reshape(B, T1, A, -1)
            
            all_z_mu.append(z_mu.cpu())
            all_actions.append(actions.cpu())
            all_valid.append((actions >= 0).cpu())
            collected += B
    
    all_z_mu = torch.cat(all_z_mu, dim=0)  # (N, T-1, A, D)
    all_actions = torch.cat(all_actions, dim=0)  # (N, T-1, A)
    all_valid = torch.cat(all_valid, dim=0)
    
    N, T1, A, D = all_z_mu.shape
    print(f"\nCollected: {N} samples × {T1} timesteps × {A} actors × {D} dims")
    
    # ====== 1. 展平计算整体 ARI ======
    z_flat = all_z_mu.reshape(-1, D).numpy()
    actions_flat = all_actions.reshape(-1).numpy()
    valid_flat = all_valid.reshape(-1).numpy()
    
    valid_idx = valid_flat > 0
    z_valid = z_flat[valid_idx]
    actions_valid = actions_flat[valid_idx]
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.manifold import TSNE
    
    print(f"\n{'='*60}")
    print(f"聚类分析")
    print(f"  有效样本数: {len(actions_valid)}")
    
    # KMeans 聚类
    n_clusters = NUM_ACTIONS
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(z_valid)
    
    ari = adjusted_rand_score(actions_valid, cluster_labels)
    nmi = normalized_mutual_info_score(actions_valid, cluster_labels)
    
    print(f"  KMeans(k={n_clusters}) ARI: {ari:.4f}")
    print(f"  KMeans(k={n_clusters}) NMI: {nmi:.4f}")
    
    # ====== 2. 按主体分别聚类 ======
    print(f"\n  按主体分别聚类:")
    per_actor_ari = []
    for a in range(A):
        mask_a = all_valid[:, :, a].reshape(-1).numpy() > 0
        if mask_a.sum() > 100:
            z_a = all_z_mu[:, :, a, :].reshape(-1, D).numpy()[mask_a]
            act_a = all_actions[:, :, a].reshape(-1).numpy()[mask_a]
            kmeans_a = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels_a = kmeans_a.fit_predict(z_a)
            ari_a = adjusted_rand_score(act_a, labels_a)
            per_actor_ari.append(ari_a)
            print(f"    Actor {a}: ARI={ari_a:.4f} (n={mask_a.sum()})")
    
    # ====== 3. 跨主体混淆分析 ======
    print(f"\n  跨主体分析:")
    # 对每个动作，计算不同主体隐动作之间的余弦相似度
    for act_idx, act_name in enumerate(ACTION_NAMES):
        cos_sims = []
        for a1 in range(A):
            for a2 in range(a1+1, A):
                mask_a1 = (all_actions[:, :, a1] == act_idx) & all_valid[:, :, a1]
                mask_a2 = (all_actions[:, :, a2] == act_idx) & all_valid[:, :, a2]
                mask = mask_a1 & mask_a2
                if mask.sum() > 10:
                    z1 = all_z_mu[:, :, a1, :][mask].numpy()
                    z2 = all_z_mu[:, :, a2, :][mask].numpy()
                    # 对每个样本计算余弦相似度
                    z1_norm = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)
                    z2_norm = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)
                    sim = (z1_norm * z2_norm).sum(axis=1).mean()
                    cos_sims.append(sim)
        
        if cos_sims:
            print(f"    '{act_name}': cross-actor cos_sim = {np.mean(cos_sims):.3f} ± {np.std(cos_sims):.3f}")
    
    # ====== 4. 可视化 ======
    from sklearn.manifold import TSNE
    
    # 先构建 actor_labels（与 z_valid 对应）
    actor_labels = []
    for n in range(N):
        for t in range(T1):
            for a in range(A):
                if all_valid[n, t, a]:
                    actor_labels.append(a)
    actor_labels = np.array(actor_labels[:len(z_valid)])
    
    # 如果样本太多，采样
    max_vis = 3000
    if len(z_valid) > max_vis:
        # 从 z_valid 和 actor_labels 同时采样
        idx = np.random.choice(len(z_valid), max_vis, replace=False)
        z_vis = z_valid[idx]
        actions_vis = actions_valid[idx]
        actor_vis = actor_labels[idx]
        print(f"\n  Visualizing {max_vis} samples...")
    else:
        z_vis = z_valid
        actions_vis = actions_valid
        actor_vis = actor_labels
    
    # t-SNE 降维
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(z_vis[:min(len(z_vis), 5000)])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 按动作着色
    for act_idx in range(NUM_ACTIONS):
        mask_act = actions_vis[:len(z_2d)] == act_idx
        if mask_act.sum() > 0:
            axes[0].scatter(z_2d[mask_act, 0], z_2d[mask_act, 1], 
                           c=[COLORS(act_idx)], label=ACTION_NAMES[act_idx],
                           alpha=0.6, s=10)
    axes[0].set_title(f"t-SNE by Action (ARI={ari:.3f})")
    axes[0].legend(fontsize=8)
    
    # 按主体着色
    for a_idx in range(A):
        mask_a = actor_vis[:len(z_2d)] == a_idx
        if mask_a.sum() > 0:
            axes[1].scatter(z_2d[mask_a, 0], z_2d[mask_a, 1],
                           c=[COLORS(a_idx)], label=f"Actor {a_idx}",
                           alpha=0.6, s=10)
    axes[1].set_title("t-SNE by Actor")
    axes[1].legend(fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"tsne_{args.name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  t-SNE saved: {save_path}")
    
    # ====== 5. 保存分析结果 ======
    results = {
        "model": args.name,
        "use_interaction": use_interaction,
        "kmeans_ari": float(ari),
        "kmeans_nmi": float(nmi),
        "per_actor_ari": [float(a) for a in per_actor_ari],
        "n_samples": int(len(actions_valid)),
    }
    
    # 跨主体余弦相似度
    cross_actor_sims = {}
    for act_idx, act_name in enumerate(ACTION_NAMES):
        cos_sims = []
        for a1 in range(A):
            for a2 in range(a1+1, A):
                mask_a1 = (all_actions[:, :, a1] == act_idx) & all_valid[:, :, a1]
                mask_a2 = (all_actions[:, :, a2] == act_idx) & all_valid[:, :, a2]
                mask = mask_a1 & mask_a2
                if mask.sum() > 10:
                    z1 = all_z_mu[:, :, a1, :][mask].numpy()
                    z2 = all_z_mu[:, :, a2, :][mask].numpy()
                    z1_norm = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)
                    z2_norm = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)
                    sims = (z1_norm * z2_norm).sum(axis=1)
                    cos_sims.extend(sims.tolist())
        cross_actor_sims[act_name] = {
            "mean": float(np.mean(cos_sims)) if cos_sims else 0,
            "std": float(np.std(cos_sims)) if cos_sims else 0,
        }
    results["cross_actor_cosine_sim"] = cross_actor_sims
    
    with open(os.path.join(out_dir, f"analysis_{args.name}.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"分析结果摘要:")
    print(f"  KMeans ARI: {ari:.4f}")
    print(f"  KMeans NMI: {nmi:.4f}")
    print(f"  Per-actor ARI: {[f'{a:.4f}' for a in per_actor_ari]}")
    print(f"  跨主体余弦相似度:")
    for act_name, sim in cross_actor_sims.items():
        print(f"    '{act_name}': {sim['mean']:.3f} ± {sim['std']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()