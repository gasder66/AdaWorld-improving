"""
V10 评估: 聚类质量 + 重建质量 + Conditional NMI + Actor-masked PSNR.

指标:
  1. Overall NMI / ARI / Per-Slot NMI / Actor Leakage (同 V6c/V8 eval)
  2. Conditional NMI: 给定 actor 类型后的 action NMI
  3. Action Probe given Type: 每个 actor 类型内部 action 分类
  4. Actor-masked PSNR: 只在 actor mask 区域计算
  5. Copy baseline PSNR: PSNR(crop_t, crop_{t+1}) 作为参考
  6. UMAP 可视化 (action / actor_type / KMeans / conditional)

用法:
  PYTHONPATH=lam python lam/scripts/eval_v10.py --name v10_stage1 --gpu 0
"""
import os, sys, json, argparse
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from lam.modules.v10_model import LatentActionModelV10
from lam.disk_synthetic_dataset import DiskSyntheticDataset
from lam.modules.blocks import patchify, unpatchify


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse.item() < 1e-10:
        return 100.0
    return float(10 * torch.log10(1.0 / mse))


def compute_masked_psnr(pred, target, mask):
    """Actor-masked PSNR: 只在 mask 区域计算."""
    # pred, target: (B, T-1, H, W, C), mask: (B, T-1, K, H, W)
    # 合并所有 actor 的 mask
    mask_union = mask.sum(dim=2)  # (B, T-1, H, W)
    mask_union = mask_union.clamp(0, 1).unsqueeze(-1)  # (B, T-1, H, W, 1)
    mask_exp = mask_union.expand_as(pred)

    mse = ((pred - target) ** 2 * mask_exp).sum() / (mask_exp.sum() + 1e-8)
    if mse.item() < 1e-10:
        return 100.0
    return float(10 * torch.log10(1.0 / mse))


def compute_copy_psnr(videos):
    """Copy baseline: PSNR(crop_t, crop_{t+1}) = PSNR(videos[:,:-1], videos[:,1:])."""
    pred = videos[:, :-1]
    target = videos[:, 1:]
    return compute_psnr(pred, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--max_actors", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "v10"
    )

    # Load training results for model config
    results_path = os.path.join(RESULTS_DIR, f"results_{args.name}.json")
    with open(results_path) as f:
        train_results = json.load(f)

    # Load latents
    latents_path = os.path.join(RESULTS_DIR, f"latents_{args.name}.npz")
    data = np.load(latents_path)
    z_actor = data["z_actor"]       # (N, D) — actor slots only (bg excluded)
    slots = data["slots"]           # (N,) slot index 0..3
    actions = data["actions"]       # (N,) action 0..4

    print(f"\n{'='*60}")
    print(f"V10 Evaluation: {args.name}")
    print(f"  Samples: {len(z_actor)}")
    print(f"  z_actor: {z_actor.shape}")
    print(f"  Actions: {np.unique(actions, return_counts=True)}")
    print(f"  Slots:   {np.unique(slots, return_counts=True)}")
    print(f"{'='*60}")

    results = {
        "model": f"V10 ({args.name})",
        "n_samples": int(len(z_actor)),
    }

    # === 1. Overall NMI / ARI ===
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    pred_all = km.fit_predict(z_actor)
    nmi_overall = normalized_mutual_info_score(actions, pred_all)
    ari_overall = adjusted_rand_score(actions, pred_all)
    print(f"\n[Overall Clustering]")
    print(f"  NMI = {nmi_overall:.4f}  (V6c: 0.0525, V8: 0.7723)")
    print(f"  ARI = {ari_overall:.4f}")
    results["overall_nmi"] = round(float(nmi_overall), 4)
    results["overall_ari"] = round(float(ari_overall), 4)

    # === 2. Per-Slot NMI ===
    print(f"\n[Per-Slot NMI]")
    nmi_per = []
    for k in np.unique(slots):
        idx = slots == k
        if idx.sum() < 50:
            continue
        km_k = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
        pred_k = km_k.fit_predict(z_actor[idx])
        nmi_k = normalized_mutual_info_score(actions[idx], pred_k)
        nmi_per.append(nmi_k)
        print(f"  Slot {k}: NMI = {nmi_k:.4f} (n={idx.sum()})")
    nmi_per_avg = float(np.mean(nmi_per)) if nmi_per else 0.0
    print(f"  Avg: NMI = {nmi_per_avg:.4f}  (V6c: 0.3885)")
    results["per_slot_nmi"] = [round(float(x), 4) for x in nmi_per]
    results["per_slot_nmi_avg"] = round(nmi_per_avg, 4)

    # === 3. Actor Leakage ===
    n_slots = len(np.unique(slots))
    chance = 1.0 / n_slots
    clf = LogisticRegression(max_iter=1000, C=1.0)
    n = len(z_actor)
    idx_perm = np.random.RandomState(args.seed).permutation(n)
    n_tr = int(0.8 * n)
    clf.fit(z_actor[idx_perm[:n_tr]], slots[idx_perm[:n_tr]])
    leakage = clf.score(z_actor[idx_perm[n_tr:]], slots[idx_perm[n_tr:]])
    print(f"\n[Actor Leakage]")
    print(f"  z → slot_id acc = {leakage:.4f}  (chance = {chance:.4f})")
    results["actor_leakage_acc"] = round(float(leakage), 4)

    # === 4. Action Probe ===
    clf_act = LogisticRegression(max_iter=1000, C=1.0)
    clf_act.fit(z_actor[idx_perm[:n_tr]], actions[idx_perm[:n_tr]])
    act_acc = clf_act.score(z_actor[idx_perm[n_tr:]], actions[idx_perm[n_tr:]])
    act_chance = 1.0 / args.n_clusters
    print(f"\n[Action Probe]")
    print(f"  z → action acc = {act_acc:.4f}  (chance = {act_chance:.4f})")
    results["action_probe_acc"] = round(float(act_acc), 4)

    # === 5. Action Probe given Slot (有监督条件指标) ===
    # 与 Per-Slot NMI 的区别:
    #   Per-Slot NMI = 无监督 (KMeans 在每个 slot 内聚类, 再比 GT action)
    #   Action Probe given Slot = 有监督 (在每个 slot 内训练 action 分类器)
    # 有监督指标更直接衡量 "z 中可线性解码的 action 信息"
    print(f"\n[Action Probe given Slot]")
    act_probe_per = []
    for k in np.unique(slots):
        idx = slots == k
        if idx.sum() < 50:
            continue
        n_k = idx.sum()
        idx_perm_k = np.random.RandomState(args.seed).permutation(n_k)
        n_tr_k = int(0.8 * n_k)
        z_k = z_actor[idx]
        a_k = actions[idx]
        clf_k = LogisticRegression(max_iter=1000, C=1.0)
        clf_k.fit(z_k[idx_perm_k[:n_tr_k]], a_k[idx_perm_k[:n_tr_k]])
        acc_k = clf_k.score(z_k[idx_perm_k[n_tr_k:]], a_k[idx_perm_k[n_tr_k:]])
        act_probe_per.append(acc_k)
        print(f"  Slot {k}: acc = {acc_k:.4f} (n={n_k})")
    act_probe_cond_avg = float(np.mean(act_probe_per)) if act_probe_per else 0.0
    print(f"  Avg Action Probe (given slot) = {act_probe_cond_avg:.4f}")
    print(f"  Overall Action Probe          = {act_acc:.4f}")
    if act_probe_cond_avg > act_acc:
        print(f"  → given slot > overall: z 在 slot 内有更强的 action 解码能力 ✓")
    else:
        print(f"  → given slot ≈ overall: slot 信息对 action 解码帮助不大")
    results["action_probe_given_slot_avg"] = round(act_probe_cond_avg, 4)
    results["action_probe_given_slot"] = [round(float(x), 4) for x in act_probe_per]

    # === 5b. Leakage given Action (对称指标) ===
    # Action Probe 和 Actor Leakage 是同一方法论: Linear probe on z, 目标标签不同
    # Action Probe: z → action      (z 包含多少 action 信息?)
    # Leakage:      z → slot_id     (z 包含多少 actor 身份?)
    # Leakage given Action: 给定动作, z → slot_id  (外观信息是否在动作内也被保留?)
    print(f"\n[Leakage given Action]")
    unique_actions_arr = np.unique(actions)
    leak_per_action = []
    for a in unique_actions_arr:
        idx_a = actions == a
        if idx_a.sum() < 50:
            continue
        n_a = idx_a.sum()
        idx_perm_a = np.random.RandomState(args.seed).permutation(n_a)
        n_tr_a = int(0.8 * n_a)
        z_a = z_actor[idx_a]
        s_a = slots[idx_a]
        clf_a = LogisticRegression(max_iter=1000, C=1.0)
        clf_a.fit(z_a[idx_perm_a[:n_tr_a]], s_a[idx_perm_a[:n_tr_a]])
        acc_a = clf_a.score(z_a[idx_perm_a[n_tr_a:]], s_a[idx_perm_a[n_tr_a:]])
        leak_per_action.append(acc_a)
        action_names = ["stay", "up", "down", "left", "right"]
        name = action_names[int(a)] if int(a) < len(action_names) else f"act{int(a)}"
        print(f"  Action {int(a)}({name}): acc = {acc_a:.4f} (n={n_a})")
    leak_given_action_avg = float(np.mean(leak_per_action)) if leak_per_action else 0.0
    print(f"  Avg Leakage (given action) = {leak_given_action_avg:.4f}")
    print(f"  Overall Leakage            = {leakage:.4f}")
    if leak_given_action_avg > leakage:
        print(f"  → 给定动作后, actor 身份更容易判断 (动作信息减少了 actor 混淆)")
    else:
        print(f"  → 给定动作 ≈ 不给定: actor 身份已经充分编码在 z 中")
    results["leakage_given_action_avg"] = round(leak_given_action_avg, 4)
    results["leakage_given_action"] = [round(float(x), 4) for x in leak_per_action]

    # === 5c. 对称性总结 ===
    print(f"\n[Probe 对称性总结]")
    print(f"  {'':<30} {'z → action':>15} {'z → slot':>15}")
    print(f"  {'Overall (无条件)':<30} {act_acc:>15.4f} {leakage:>15.4f}")
    print(f"  {'Given Slot (条件化)':<30} {act_probe_cond_avg:>15.4f} {'N/A (slot=条件本身)':>15}")
    print(f"  {'Given Action (条件化)':<30} {'N/A (action=条件本身)':>15} {leak_given_action_avg:>15.4f}")
    print(f"  注: Action Probe 和 Leakage 是同一个方法论 (Linear probe), 目标标签不同")

    # === 6. UMAP Visualization ===
    # 合成数据 actor 类型 (slot → 形状+颜色):
    #   slot 0 = 红色方块, slot 1 = 绿色圆形, slot 2 = 蓝色三角, slot 3 = 黄色方块
    SHAPE_NAMES = ["Square", "Circle", "Triangle"]
    COLOR_NAMES = ["Red", "Green", "Blue", "Yellow", "Purple"]
    ACTOR_TYPES = []
    for s in slots:
        shape = SHAPE_NAMES[int(s) % 3]
        color = COLOR_NAMES[int(s) % 5]
        ACTOR_TYPES.append(f"{color} {shape}")
    actor_types = np.array(ACTOR_TYPES)

    # actor-action 复合标签 (e.g. "Red Square-Left")
    actor_action_labels = np.array([f"{t}-{a}" for t, a in zip(ACTOR_TYPES, actions)])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap

        reducer = umap.UMAP(random_state=args.seed, n_neighbors=30, min_dist=0.3)
        z_2d = reducer.fit_transform(z_actor)

        # 4 张图: action / slot / KMeans / actor-action
        # 注: 合成数据中 slot 与 actor_type 一一对应, 故不单独画 actor_type
        fig, axes = plt.subplots(1, 4, figsize=(28, 6))

        # 1. Color by action
        scatter1 = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=actions, cmap="tab10",
                                    s=8, alpha=0.6)
        axes[0].set_title(f"Color=Action (NMI={nmi_overall:.3f})")
        action_names = ["stay", "up", "down", "left", "right"]
        legend1 = axes[0].legend(*scatter1.legend_elements(),
                                 title="action", loc="best", fontsize=8)
        for t, name in zip(legend1.get_texts(), action_names[:len(legend1.get_texts())]):
            t.set_text(name)

        # 2. Color by slot
        scatter2 = axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=slots, cmap="Set1",
                                    s=8, alpha=0.6)
        axes[1].set_title(f"Color=Slot (Leakage={leakage:.3f})")
        legend2 = axes[1].legend(*scatter2.legend_elements(),
                                 title="slot", loc="best", fontsize=8)
        for t, name in zip(legend2.get_texts(), ACTOR_TYPES[:len(legend2.get_texts())]):
            t.set_text(name)

        # 3. Color by KMeans cluster
        scatter3 = axes[2].scatter(z_2d[:, 0], z_2d[:, 1], c=pred_all, cmap="tab10",
                                    s=8, alpha=0.6)
        axes[2].set_title(f"Color=KMeans (NMI={nmi_overall:.3f})")
        axes[2].legend(*scatter3.legend_elements(), title="cluster", loc="best", fontsize=8)

        # 4. Color by actor-action composite (4 actors × 5 actions = 20 classes)
        unique_aa = np.unique(actor_action_labels)
        aa_to_int = {t: i for i, t in enumerate(unique_aa)}
        aa_ints = np.array([aa_to_int[t] for t in actor_action_labels])
        scatter5 = axes[3].scatter(z_2d[:, 0], z_2d[:, 1], c=aa_ints, cmap="tab20",
                                    s=6, alpha=0.5)
        axes[3].set_title(f"Color=Actor+Action ({len(unique_aa)} classes)")
        axes[3].legend(*scatter5.legend_elements(),
                       title="actor-action", loc="best", fontsize=6,
                       ncol=2)

        plt.tight_layout()
        umap_path = os.path.join(RESULTS_DIR, f"umap_{args.name}.png")
        plt.savefig(umap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  UMAP saved: {umap_path}")
        results["umap_path"] = umap_path
        results["n_actor_action_classes"] = len(unique_aa)
    except ImportError:
        print("\n  (umap-learn not installed, skipping visualization)")

    # === 7. Reconstruction: Full-frame + Actor-masked + Copy baseline ===
    print(f"\n[Reconstruction]")

    # Load model
    model = LatentActionModelV10(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8, max_actors=args.max_actors,
        keep_background=True, use_obj_st_attention=True,
        free_bits_lambda=0.1,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(RESULTS_DIR, f"model_{args.name}.pt"), map_location=device
    ))
    model.eval()

    if args.data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    eval_ds = DiskSyntheticDataset(
        os.path.join(data_root, "val"),
        max_actors=args.max_actors, num_frames=args.num_frames,
        output_format="t h w c",
    )
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=8, shuffle=False, num_workers=0)

    all_recon_psnr = []
    all_masked_psnr = []
    all_copy_psnr = []
    n_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            out = model({"videos": videos, "masks": masks})
            recon = out["recon"]  # (B, T-1, H, W, C)
            gt = videos[:, 1:]     # (B, T-1, H, W, C)

            # Full-frame PSNR
            psnr_full = compute_psnr(recon, gt)

            # Actor-masked PSNR
            masks_t1 = masks[:, 1:]  # (B, T-1, K, H, W)
            psnr_masked = compute_masked_psnr(recon, gt, masks_t1)

            # Copy baseline
            psnr_copy = compute_psnr(videos[:, :-1], gt)

            all_recon_psnr.append(psnr_full)
            all_masked_psnr.append(psnr_masked)
            all_copy_psnr.append(psnr_copy)
            n_batches += 1
            if n_batches >= 15:
                break

    psnr_recon = float(np.mean(all_recon_psnr))
    psnr_masked = float(np.mean(all_masked_psnr))
    psnr_copy = float(np.mean(all_copy_psnr))

    print(f"  Full-frame PSNR:  {psnr_recon:.2f} dB  (V6c: 27.35)")
    print(f"  Actor-masked PSNR: {psnr_masked:.2f} dB  (只在 actor mask 区域)")
    print(f"  Copy baseline:    {psnr_copy:.2f} dB  (PSNR(crop_t, crop_{{t+1}}))")
    print(f"  ΔPSNR(recon-copy): {psnr_recon - psnr_copy:+.2f} dB")
    print(f"  ΔPSNR(masked-copy): {psnr_masked - psnr_copy:+.2f} dB")

    results["full_frame_psnr"] = round(psnr_recon, 2)
    results["actor_masked_psnr"] = round(psnr_masked, 2)
    results["copy_psnr"] = round(psnr_copy, 2)
    results["delta_psnr_recon_copy"] = round(psnr_recon - psnr_copy, 2)
    results["delta_psnr_masked_copy"] = round(psnr_masked - psnr_copy, 2)

    # === Comparison Table ===
    print(f"\n{'='*80}")
    print(f"Comparison: V6c vs V10 vs V8")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'V6c':>10} {'V10':>10} {'V8':>10}")
    print(f"{'-'*60}")
    print(f"{'Overall NMI':<30} {'0.0525':>10} {nmi_overall:>10.4f} {'0.7723':>10}")
    print(f"{'Per-Slot NMI (avg, unsupervised)':<30} {'0.3885':>10} {nmi_per_avg:>10.4f} {'0.7684':>10}")
    print(f"{'Actor Leakage':<30} {'1.0000':>10} {leakage:>10.4f} {'0.3350':>10}")
    print(f"{'Leakage given Action':<30} {'N/A':>10} {leak_given_action_avg:>10.4f} {'N/A':>10}")
    print(f"{'Action Probe (supervised)':<30} {'N/A':>10} {act_acc:>10.4f} {'0.8741':>10}")
    print(f"{'Action Probe given Slot':<30} {'N/A':>10} {act_probe_cond_avg:>10.4f} {'N/A':>10}")
    print(f"{'Full-frame PSNR (dB)':<30} {'27.35':>10} {psnr_recon:>10.2f} {'NO-GO':>10}")
    print(f"{'Actor-masked PSNR (dB)':<30} {'N/A':>10} {psnr_masked:>10.2f} {'N/A':>10}")
    print(f"{'Copy baseline PSNR (dB)':<30} {'N/A':>10} {psnr_copy:>10.2f} {'22.68':>10}")
    print(f"{'='*60}")

    out_path = os.path.join(RESULTS_DIR, f"eval_{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
