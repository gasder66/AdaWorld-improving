"""
多 Slot 诊断脚本：正确评估 ARI + 注意力可视化

核心问题：原有评估 `gt = actions[:, 0]` 只用了第一个 actor 的动作来评估所有 slots。
这个脚本修复了评估方法，并可视化 slot 注意力与实际物体位置的关系。

修复的评估方法：
1. 匈牙利匹配：计算每个 slot 与每个 actor 的 ARI 矩阵，用匈牙利算法最优分配
2. 空间重叠：slot 注意力图与 ground-truth 物体位置的 IoU
3. 每个样本的 slot→agent 分配一致性
"""
import os, sys, argparse, json, time, math
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lam.modules import LatentActionModel
from lam.synthetic_dataset import SyntheticMultiActorDataset
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "slot_attention_exp_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_hungarian_ari(z_mu: np.ndarray, actions: np.ndarray, num_slots: int,
                         num_actors_max: int = 5) -> dict:
    """
    正确的多 slot ARI 评估：使用匈牙利匹配。

    z_mu: (N_samples, K, latent_dim) - 所有 slot 的隐码
    actions: (N_samples, num_actors) - 每个 actor 的动作 (-1 = padding)
    num_slots: K, slot 数量
    num_actors_max: 最大 actor 数

    返回:
        matched_aris: 匹配后的 ARI 列表（每个匹配对）
        slot_actor_map: 每个 slot 匹配到的 actor 索引
        ari_matrix: K × num_actors_max 的 ARI 矩阵
    """
    K = num_slots
    A = num_actors_max
    ari_matrix = np.zeros((K, A))

    for k in range(K):
        z_slot = z_mu[:, k, :]
        for a in range(A):
            gt = actions[:, a]
            mask = gt >= 0  # 只考虑有效 actor
            if mask.sum() > 1 and len(np.unique(gt[mask])) > 1:
                n_clusters = min(5, len(np.unique(gt[mask])))
                pred = KMeans(n_clusters=n_clusters, random_state=42,
                              n_init=10).fit_predict(z_slot[mask])
                ari_matrix[k, a] = adjusted_rand_score(gt[mask], pred)
            else:
                ari_matrix[k, a] = -1.0  # 无法计算

    # 匈牙利匹配：最大化 ARI 总和
    # 把 -1 变成 -inf 以避免匹配到无效 actor
    cost = -ari_matrix.copy()
    cost[cost == 1.0] = 1e6  # -(-1) = 1, 使无效匹配有高成本

    # 只对有效 actor（有数据）做匹配
    valid_actors = [a for a in range(A) if (actions[:, a] >= 0).sum() > 1]
    if len(valid_actors) == 0:
        return {"matched_aris": [], "slot_actor_map": {}, "ari_matrix": ari_matrix.tolist()}

    # 裁剪到有效 actor 子矩阵
    sub_cost = cost[:, valid_actors]
    sub_ari = ari_matrix[:, valid_actors]

    # 匈牙利匹配（最多 min(K, len(valid_actors)) 对）
    row_ind, col_ind = linear_sum_assignment(sub_cost)
    # 过滤负 ARI 匹配（即使匹配到负 ARI 也比不匹配好，但这里我们只取正 ARI 的匹配）
    matched_aris = []
    slot_actor_map = {}
    for r, c in zip(row_ind, col_ind):
        matched_ari = sub_ari[r, c]
        if matched_ari > 0:  # 只保留正 ARI 匹配
            matched_aris.append(float(matched_ari))
            slot_actor_map[r] = valid_actors[c]

    return {
        "matched_aris": matched_aris,
        "slot_actor_map": slot_actor_map,
        "ari_matrix": ari_matrix.tolist(),
        "mean_matched_ari": float(np.mean(matched_aris)) if matched_aris else 0.0,
    }


def compute_spatial_overlap(attn_maps: np.ndarray, positions: np.ndarray,
                            grid_size: int = 16, num_actors_max: int = 5) -> dict:
    """
    计算 slot 注意力与物体位置的空间重叠度。

    attn_maps: (B, K, N_patches) - 每个 slot 对每个 patch 的注意力权重
    positions: (B, T, max_actors, 2) - 每个 actor 的位置（左上角网格坐标）
    grid_size: 空间网格大小（16x16 for 256/16）
    num_actors_max: 最大 actor 数

    返回:
        overlap_matrix: K × max_actors 的 IoU 矩阵
    """
    B, K, N = attn_maps.shape
    H = W = grid_size
    assert N == H * W, f"Expected {H*W} patches, got {N}"

    # 注意力图 reshape 到 2D
    attn_2d = attn_maps.reshape(B, K, H, W)  # (B, K, H, W)

    overlap_matrix = np.zeros((K, num_actors_max))

    for b in range(B):
        for k in range(K):
            attn_map = attn_2d[b, k]  # (H, W)
            # 归一化到 [0, 1]
            attn_max, attn_min = attn_map.max(), attn_map.min()
            if attn_max > attn_min:
                attn_map = (attn_map - attn_min) / (attn_max - attn_min)
            # 二值化：top 20% 注意力视为"关注区域"
            threshold = np.percentile(attn_map, 80)
            attn_mask = attn_map >= threshold  # (H, W)

            for a in range(num_actors_max):
                pos_r, pos_c = positions[b, 0, a]  # 第一帧的位置
                # 2x2 agent 在 patch 网格中的范围
                # 位置是 grid 坐标（8x8 grid），而 patch 网格是 16x16
                # 每个 grid cell = 2 patches (因为 256/16=16 patches, 8 grid cells)
                p_r = pos_r * 2
                p_c = pos_c * 2
                if p_r < 0 or p_c < 0:
                    continue
                # agent 占 2x2 grid cells = 4x4 patches
                agent_mask = np.zeros((H, W), dtype=bool)
                r_start = max(0, p_r)
                r_end = min(H, p_r + 4)
                c_start = max(0, p_c)
                c_end = min(W, p_c + 4)
                agent_mask[r_start:r_end, c_start:c_end] = True

                intersection = (attn_mask & agent_mask).sum()
                union = (attn_mask | agent_mask).sum()
                if union > 0:
                    overlap_matrix[k, a] += intersection / union

    overlap_matrix /= B  # 平均

    return {"overlap_matrix": overlap_matrix.tolist()}


def visualize_slot_attention(model, dataset, device, num_samples=4, save_path=None):
    """可视化 slot 注意力图与 ground-truth agent 位置的关系。"""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)

    batch = next(iter(loader))
    videos = batch["videos"].to(device)
    positions = batch["actor_positions"].numpy()  # (B, T, max_actors, 2)
    num_actors = batch["num_actors"].numpy()

    # 获取模型的 encoder 中间输出（包括注意力图）
    # 我们利用 SpatioTemporalBlock.last_slot_attn_map
    # 先清空所有块的注意力
    for block in model.encoder.transformer_blocks:
        block.last_slot_attn_map = None

    # 前向传播
    with torch.no_grad():
        _ = model({"videos": videos})

    # 获取最后一个块的注意力图
    attn_map = model.encoder.transformer_blocks[-1].last_slot_attn_map
    if attn_map is None:
        attn_map = model.encoder.transformer_blocks[-2].last_slot_attn_map

    # attn_map shape: (BT, H, K, N_patches) or (BT, H, K, N)
    if attn_map is not None:
        attn_map = attn_map.cpu().numpy()
        # 取第一帧（t=0）
        B = videos.shape[0]
        T = videos.shape[1]
        attn_map = attn_map[:B, ...]  # (B, H, K, N)
        # 在 heads 维度平均
        attn_map = attn_map.mean(axis=1)  # (B, K, N)
        K = attn_map.shape[1]
        N = attn_map.shape[2]
        H = W = int(math.sqrt(N))
        attn_2d = attn_map.reshape(B, K, H, W)  # (B, K, H, W)
    else:
        attn_2d = None
        K = model.num_slots

    # 渲染
    fig, axes = plt.subplots(num_samples, K + 1, figsize=((K + 1) * 4, num_samples * 4))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for b in range(num_samples):
        # 第一列：输入图像 + agent 边界框
        img = videos[b, 0].cpu().numpy()  # (H, W, C)
        axes[b, 0].imshow(img)
        axes[b, 0].set_title(f"Input (B={b}, N={num_actors[b]} actors)")
        for a in range(int(num_actors[b])):
            pos = positions[b, 0, a]
            if pos[0] >= 0:
                r0 = pos[0] * (256 // 8)  # grid cell → pixels
                c0 = pos[1] * (256 // 8)
                size = (256 // 8) * 2  # 2x2 grid cells
                rect = Rectangle((c0, r0), size, size, fill=False,
                                 edgecolor=plt.cm.tab10(a), linewidth=2)
                axes[b, 0].add_patch(rect)
                axes[b, 0].text(c0, r0 - 5, f"A{a}", color=plt.cm.tab10(a),
                               fontsize=10, fontweight="bold")

        # 其余列：每个 slot 的注意力图
        for k in range(K):
            ax = axes[b, k + 1]
            if attn_2d is not None:
                attn_k = attn_2d[b, k]
                # 归一化
                attn_k = (attn_k - attn_k.min()) / (attn_k.max() - attn_k.min() + 1e-8)
                # 上采样到 256x256
                attn_k_up = np.kron(attn_k, np.ones((256 // H, 256 // W)))
                ax.imshow(img, alpha=0.5)
                ax.imshow(attn_k_up, alpha=0.5, cmap="hot")
            else:
                ax.imshow(img, alpha=0.5)
            ax.set_title(f"Slot {k}")
            # 画 agent 边界框
            img_for_boxes = np.zeros((256, 256))
            ax.imshow(img_for_boxes, alpha=0)
            # Re-draw boxes on this axis
            for a in range(int(num_actors[b])):
                pos = positions[b, 0, a]
                if pos[0] >= 0:
                    r0 = pos[0] * (256 // 8)
                    c0 = pos[1] * (256 // 8)
                    size = (256 // 8) * 2
                    rect = Rectangle((c0, r0), size, size, fill=False,
                                     edgecolor=plt.cm.tab10(a), linewidth=2)
                    ax.add_patch(rect)
                    ax.text(c0, r0 - 5, f"A{a}", color=plt.cm.tab10(a),
                           fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--competition", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--vis_only", action="store_true",
                       help="是否只做可视化（加载已有模型）")
    parser.add_argument("--load_path", type=str, default=None,
                       help="加载已有模型权重")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    exp_suffix = "comp" if args.competition else "no_comp"
    exp_name = f"v{args.num_slots}_slots_{exp_suffix}_diagnosis"

    print("=" * 70)
    print(f"  Multi-Slot Diagnosis: {exp_name}")
    print(f"  slots={args.num_slots}, competition={args.competition}, "
          f"batch={args.batch_size}, steps={args.steps}")
    print("=" * 70)

    # ===== 数据 =====
    train_dataset = SyntheticMultiActorDataset(
        resolution=256, num_frames=2, min_actors=2, max_actors=4,
        background_type="checkerboard",
        samples_per_epoch=args.steps * args.batch_size * 2, seed=42
    )
    eval_dataset = SyntheticMultiActorDataset(
        resolution=256, num_frames=2, min_actors=2, max_actors=4,
        background_type="checkerboard", samples_per_epoch=1000, seed=999
    )

    # ===== 模型 =====
    model = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8,
        num_slots=args.num_slots, max_actors=4,
        use_slot_competition=args.competition,
        use_grad_checkpointing=not args.competition,  # 竞争时不用 checkpoint 避免 NaN
        aux_loss_weight=0.0,  # 诊断时不使用辅助损失
    ).to(device)

    if args.load_path and os.path.exists(args.load_path):
        model.load_state_dict(torch.load(args.load_path, map_location=device))
        print(f"  Loaded model from {args.load_path}")
    elif not args.vis_only:
        # ===== 训练 =====
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-2)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

        losses = []
        step = 0
        t0 = time.time()
        while step < args.steps:
            for batch in dataloader:
                if step >= args.steps:
                    break
                batch["videos"] = batch["videos"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(batch)  # 传完整 batch，使用辅助损失
                    gt = batch["videos"][:, 1:]
                    mse_loss = ((gt - outputs["recon"]) ** 2).mean()
                    z_mu, z_var = outputs["z_mu"], outputs["z_var"]
                    kl_loss = -0.5 * torch.sum(1 + z_var - z_mu ** 2 - z_var.exp()) / z_mu.shape[0]
                    loss = mse_loss + 0.0002 * kl_loss
                    # 添加辅助损失
                    if args.num_slots > 1:
                        aux_total = outputs.get("loss_total", torch.tensor(0.0, device=device))
                        loss = loss + aux_total

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss))
                if step % 100 == 0:
                    aux_str = ""
                    if args.num_slots > 1:
                        aux_str = (f" assign={outputs.get('loss_assign',0):.4f}"
                                   f" div={outputs.get('loss_diversity',0):.4f}"
                                   f" ent={outputs.get('loss_spatial_entropy',0):.4f}")
                    print(f"  Step {step:3d}/{args.steps}: loss={loss:.4f}, "
                          f"mse={mse_loss:.4f}, kl={kl_loss:.2f}{aux_str}, "
                          f"{time.time()-t0:.0f}s")
                step += 1

        # 保存模型
        save_path = os.path.join(RESULTS_DIR, f"model_{exp_name}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"  Model saved to {save_path}")
    else:
        print("  vis_only=True but no model path provided. Training from scratch.")
        return

    # ===== 诊断评估 =====
    print("\n" + "=" * 70)
    print("  DIAGNOSIS EVALUATION")
    print("=" * 70)

    model.eval()
    all_z_mu, all_actions = [], []
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, num_workers=0)
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 20:  # 评估 1280 个样本
                break
            videos = batch["videos"].to(device)
            outputs = model({"videos": videos})
            all_z_mu.append(outputs["z_mu"].cpu())
            actions_np = batch["actor_actions"].numpy()
            B, T_act, A_act = actions_np.shape
            actions_flat = actions_np.reshape(B * T_act, A_act)
            all_actions.append(actions_flat)

    z_mu = torch.cat(all_z_mu, dim=0).numpy()
    actions = np.concatenate(all_actions, axis=0)
    num_samples = z_mu.shape[0]
    print(f"  Evaluating {num_samples} samples, K={args.num_slots} slots, "
          f"A={actions.shape[1]} actors max")

    # ===== 1. 错误的 ARI（原评估方式）=====
    if args.num_slots == 1:
        z_flat = z_mu[:, 0, :]
        gt = actions[:, 0]
        mask = gt >= 0
        if mask.sum() > 1 and len(np.unique(gt[mask])) > 1:
            n_clusters = min(5, len(np.unique(gt[mask])))
            pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(z_flat[mask])
            old_ari = adjusted_rand_score(gt[mask], pred)
        else:
            old_ari = 0.0
        print(f"\n  [旧方法] Single slot vs Actor 0: ARI = {old_ari:.4f}")
    else:
        # 旧的错误评估
        old_aris = []
        for k in range(args.num_slots):
            z_slot = z_mu[:, k, :]
            gt = actions[:, 0]
            mask = gt >= 0
            if mask.sum() > 1 and len(np.unique(gt[mask])) > 1:
                n_clusters = min(5, len(np.unique(gt[mask])))
                pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(z_slot[mask])
                old_aris.append(adjusted_rand_score(gt[mask], pred))
            else:
                old_aris.append(0.0)
        print(f"\n  [旧方法 - BUG!] 每个 slot vs Actor 0:")
        print(f"    Slot ARIs={[f'{a:.4f}' for a in old_aris]}")
        print(f"    Mean ARI={np.mean(old_aris):.4f}")

    # ===== 2. 正确的 ARI（匈牙利匹配） =====
    hungarian = compute_hungarian_ari(z_mu, actions, args.num_slots, num_actors_max=4)
    matched_aris = hungarian["matched_aris"]
    slot_map = hungarian["slot_actor_map"]

    print(f"\n  [新方法 - 匈牙利匹配]")
    print(f"    ARI Matrix (K x A):")
    ari_mat = np.array(hungarian["ari_matrix"])
    for k in range(args.num_slots):
        row_str = f"      Slot {k}: "
        for a in range(ari_mat.shape[1]):
            val = ari_mat[k, a]
            if val >= 0:
                row_str += f"A{a}={val:.4f}  "
            else:
                row_str += f"A{a}=N/A  "
        print(row_str)

    print(f"    Hungarian matched pairs:")
    for k, a in sorted(slot_map.items()):
        print(f"      Slot {k} → Actor {a} (ARI={ari_mat[k, a]:.4f})")
    print(f"    Mean matched ARI = {hungarian['mean_matched_ari']:.4f}")

    # ===== 3. 空间重叠分析 =====
    # 获取注意力图（需要再来一次 forward）
    for block in model.encoder.transformer_blocks:
        block.last_slot_attn_map = None

    # 收集注意力图
    all_attns = []
    all_positions = []
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 5:  # 5 batches = 320 samples for overlap
                break
            videos = batch["videos"].to(device)
            _ = model({"videos": videos})
            positions = batch["actor_positions"].numpy()

            # 最后一块的注意力图
            attn = model.encoder.transformer_blocks[-1].last_slot_attn_map
            if attn is None:
                attn = model.encoder.transformer_blocks[-2].last_slot_attn_map
            if attn is not None:
                attn_np = attn.cpu().numpy()
                B = videos.shape[0]
                # (BT, H, K, N) → (B, K, N) 取 t=0
                attn_np = attn_np[:B].mean(axis=1)  # (B, K, N) avg over heads
                all_attns.append(attn_np)
                all_positions.append(positions[:, 0])  # t=0 positions

    if all_attns:
        all_attns = np.concatenate(all_attns, axis=0)
        all_positions = np.concatenate(all_positions, axis=0)
        N_patches = all_attns.shape[2]
        H = W = int(math.sqrt(N_patches))
        overlap = compute_spatial_overlap(all_attns, all_positions.reshape(-1, 1, 4, 2),
                                          grid_size=H, num_actors_max=4)
        print(f"\n  [空间重叠分析] Slot vs Actor 的注意力重叠度 (IoU):")
        ov_mat = np.array(overlap["overlap_matrix"])
        for k in range(args.num_slots):
            row = f"      Slot {k}: "
            for a in range(ov_mat.shape[1]):
                row += f"A{a}={ov_mat[k, a]:.4f}  "
            print(row)

        # 找出每个 slot 关注最多的 actor
        print(f"\n    Slot → 最关注 Actor:")
        for k in range(args.num_slots):
            best_a = np.argmax(ov_mat[k])
            best_val = ov_mat[k, best_a]
            print(f"      Slot {k} → Actor {best_a} (IoU={best_val:.4f})")

    # ===== 4. 可视化 =====
    vis_path = os.path.join(RESULTS_DIR, f"attn_vis_{exp_name}.png")
    vis_dataset = SyntheticMultiActorDataset(
        resolution=256, num_frames=2, min_actors=2, max_actors=4,
        background_type="checkerboard", samples_per_epoch=100, seed=42
    )
    visualize_slot_attention(model, vis_dataset, device, num_samples=4, save_path=vis_path)

    # ===== 5. Slot 一致性分析 =====
    print(f"\n  [Slot 一致性分析]")
    # 看每个 actor 是否始终被同一个 slot 关注
    # 使用空间重叠矩阵做 per-sample 分配
    if isinstance(all_attns, np.ndarray) and all_attns.size > 0:
        per_sample_matches = []
        for b in range(min(all_attns.shape[0], 100)):
            attn_b = all_attns[b]  # (K, N)
            pos_b = all_positions[b]  # (max_actors, 2)
            H_local = W_local = int(math.sqrt(attn_b.shape[1]))
            attn_2d = attn_b.reshape(args.num_slots, H_local, W_local)

            for k in range(args.num_slots):
                attn_k = attn_2d[k]
                attn_k_norm = (attn_k - attn_k.min()) / (attn_k.max() - attn_k.min() + 1e-8)
                threshold = np.percentile(attn_k_norm, 80)
                attn_mask = attn_k_norm >= threshold

                best_iou = 0
                best_a = -1
                for a in range(4):
                    if pos_b[a, 0] < 0:
                        continue
                    p_r = int(pos_b[a, 0]) * 2
                    p_c = int(pos_b[a, 1]) * 2
                    agent_mask = np.zeros((H_local, W_local), dtype=bool)
                    agent_mask[max(0, p_r):min(H_local, p_r + 4),
                               max(0, p_c):min(W_local, p_c + 4)] = True
                    inter = (attn_mask & agent_mask).sum()
                    union = (attn_mask | agent_mask).sum()
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_a = a

                per_sample_matches.append((k, best_a, best_iou))

        # 统计一致性
        from collections import defaultdict
        slot_actor_counts = defaultdict(lambda: defaultdict(int))
        for k, a, iou in per_sample_matches:
            if a >= 0:
                slot_actor_counts[k][a] += 1

        print(f"    每个 slot 最常关注的 actor (统计 {len(per_sample_matches)} 个样本):")
        for k in range(args.num_slots):
            if k in slot_actor_counts:
                counts = slot_actor_counts[k]
                total = sum(counts.values())
                main_actor = max(counts, key=counts.get)
                main_pct = counts[main_actor] / total * 100
                print(f"      Slot {k}: 最关注 Actor {main_actor} ({main_pct:.0f}%), "
                      f"分布: {dict(counts)}")
            else:
                print(f"      Slot {k}: 无明确关注对象")

    print("\n" + "=" * 70)
    print("  DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()