"""
V3 多主体感知 + 单向量输出 训练脚本。

核心改进（相比 V2）：
- 背景槽：mask 池化包含背景，A+1 个对象槽
- 对象级时空注意力：特征空间 256d，差分前交互
- Mean Pool + 单 VAE：聚合回单向量 z̃ ∈ R³²
- 单 slot Decoder：与原始 LAM 一致
- 单 KL 项

用法:
  python run_single.py --gpu 2 --name v3_baseline --batch_size 32 --steps 500
  python run_single.py --gpu 2 --name v3_no_obj_st --no_obj_st_attention --batch_size 32 --steps 500
"""
import os
import sys
import json
import time
import argparse
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset

ACTION_NAMES = ["stay", "up", "down", "left", "right"]
NUM_ACTIONS = 5


def compute_action_accuracy(action_logits, gt_actions, valid_mask=None):
    """
    计算动作预测准确率（V3: 全局单向量预测）。

    V3 输出 (B, T-1, num_actions)，需要与聚合后的 GT 动作对齐。
    策略：取所有有效主体中出现频率最高的动作作为全局 GT。

    Args:
        action_logits: (B, T-1, num_actions) 全局动作预测 logits
        gt_actions: (B, T-1, A) GT 动作标签 (-1 = padding)
        valid_mask: (B, T-1, A) bool, 有效主体指示 (可选)

    Returns:
        accuracy: float, 总体准确率
        per_action_acc: dict, 每动作准确率
    """
    B, T1, _ = action_logits.shape
    pred = action_logits.argmax(dim=-1)  # (B, T-1)

    if valid_mask is None:
        valid_mask = gt_actions >= 0

    # 聚合 GT 动作：取每个时间步中出现频率最高的有效动作
    # global_actions[b, t] = mode of valid gt_actions[b, t, :]
    global_actions = torch.full((B, T1), -1, dtype=torch.long, device=gt_actions.device)
    global_valid = torch.zeros((B, T1), dtype=torch.bool, device=gt_actions.device)

    for b in range(B):
        for t in range(T1):
            valid_acts = gt_actions[b, t][valid_mask[b, t]]
            if len(valid_acts) > 0:
                # 取众数
                counts = torch.bincount(valid_acts[valid_acts >= 0], minlength=NUM_ACTIONS)
                global_actions[b, t] = counts.argmax().item()
                global_valid[b, t] = True

    # 总体准确率
    correct = (pred == global_actions) & global_valid
    total_valid = global_valid.sum().float()
    accuracy = (correct.sum().float() / (total_valid + 1e-8)).item()

    # 每动作准确率
    per_action_acc = {}
    for act in range(NUM_ACTIONS):
        mask_act = (global_actions == act) & global_valid
        if mask_act.sum() > 0:
            acc_act = (correct & mask_act).sum().float() / mask_act.sum().float()
            per_action_acc[ACTION_NAMES[act]] = round(acc_act.item(), 4)
        else:
            per_action_acc[ACTION_NAMES[act]] = 0.0

    return accuracy, per_action_acc


def compute_recon_metrics(recon, gt_videos):
    """计算重建 PSNR 和 MSE。"""
    mse = ((recon - gt_videos) ** 2).mean().item()
    psnr = -10 * np.log10(mse + 1e-10)
    return mse, psnr


def main():
    parser = argparse.ArgumentParser(description="V3 多主体感知 + 单向量输出训练")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--no_obj_st_attention", action="store_true",
                        help="禁用对象级时空注意力模块")
    parser.add_argument("--action_weight", type=float, default=1.0,
                        help="动作预测损失权重")
    parser.add_argument("--recon_weight", type=float, default=1.0,
                        help="重建损失权重")
    parser.add_argument("--kl_beta", type=float, default=0.0002,
                        help="KL 损失权重")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="每 N 步保存一次 checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "results", "v3_obj_st_attention"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 数据集
    if args.data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    else:
        data_root = args.data_root

    print(f"\n{'='*60}")
    print(f"V3 多主体感知 + 单向量输出训练")
    print(f"  GPU={args.gpu}, name={args.name}, batch={args.batch_size}, steps={args.steps}")
    print(f"  obj_st_attention={not args.no_obj_st_attention}")
    print(f"  action_weight={args.action_weight}, recon_weight={args.recon_weight}, kl_beta={args.kl_beta}")
    print(f"  Data: {data_root}")
    print(f"{'='*60}")

    # 数据集（5 帧视频）
    train_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "train"),
        num_frames=5,
        output_format="t h w c",
    )
    eval_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"),
        num_frames=5,
        output_format="t h w c",
    )

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
        use_obj_st_attention=not args.no_obj_st_attention,
        obj_st_heads=8,
        obj_st_layers=2,
        use_grad_checkpointing=True,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # ====== 训练 ======
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    losses = {"total": [], "recon": [], "kl": [], "action": []}
    step = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            videos = batch["videos"].to(device, non_blocking=True)    # (B, T, H, W, C)
            masks = batch["masks"].to(device, non_blocking=True)      # (B, T, A, H, W)
            actions = batch["actions"].to(device, non_blocking=True)  # (B, T-1, A)

            with torch.cuda.amp.autocast():
                outputs = model({"videos": videos, "masks": masks})

                # 重建损失
                gt = videos[:, 1:]  # (B, T-1, H, W, C)
                recon_loss = ((gt - outputs["recon"]) ** 2).mean()

                # KL 损失（单 KL 项）
                z_mu = outputs["z_mu"]    # (B*(T-1), latent_dim)
                z_var = outputs["z_var"]  # (B*(T-1), latent_dim)
                kl_loss = -0.5 * torch.sum(
                    1 + z_var - z_mu ** 2 - z_var.exp()
                ) / z_mu.shape[0]

                # 动作预测损失
                # V3 输出 (B, T-1, num_actions)，需要与聚合后的 GT 对齐
                action_logits = outputs["action_logits"]  # (B, T-1, num_actions)
                B, T1, A = actions.shape

                # 聚合 GT 动作：取每个时间步中出现频率最高的有效动作
                valid_mask = actions >= 0  # (B, T-1, A)
                global_actions = torch.full((B, T1), -1, dtype=torch.long, device=device)
                global_valid = torch.zeros((B, T1), dtype=torch.bool, device=device)

                for b in range(B):
                    for t in range(T1):
                        valid_acts = actions[b, t][valid_mask[b, t]]
                        if len(valid_acts) > 0:
                            counts = torch.bincount(valid_acts, minlength=NUM_ACTIONS)
                            global_actions[b, t] = counts.argmax().item()
                            global_valid[b, t] = True

                action_loss = F.cross_entropy(
                    action_logits.reshape(-1, NUM_ACTIONS),
                    global_actions.reshape(-1),
                    ignore_index=-1,
                )

                # 总损失
                loss = (
                    args.recon_weight * recon_loss
                    + args.kl_beta * kl_loss
                    + args.action_weight * action_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses["total"].append(float(loss))
            losses["recon"].append(float(recon_loss))
            losses["kl"].append(float(kl_loss))
            losses["action"].append(float(action_loss))

            if step % 50 == 0:
                elapsed = time.time() - t0
                mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                print(
                    f"  Step {step:3d}/{args.steps}: "
                    f"loss={float(loss):.4f}, "
                    f"recon={float(recon_loss):.4f}, "
                    f"kl={float(kl_loss):.4f}, "
                    f"action={float(action_loss):.4f}, "
                    f"mem={mem_peak:.1f}GB, {elapsed:.0f}s"
                )

            step += 1

    training_time = time.time() - t0
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"\n  Training done. Peak mem: {mem_peak:.1f}GB, time: {training_time:.0f}s")

    # 保存损失曲线
    for key, vals in losses.items():
        np.savetxt(
            os.path.join(RESULTS_DIR, f"loss_{args.name}_{key}.txt"),
            np.array(vals),
        )

    # ====== 评估 ======
    model.eval()
    results = {
        "architecture": "v3_obj_st_attention",
        "use_obj_st_attention": not args.no_obj_st_attention,
        "batch_size": args.batch_size,
        "training_steps": args.steps,
        "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2),
        "action_weight": args.action_weight,
        "recon_weight": args.recon_weight,
        "kl_beta": args.kl_beta,
    }

    # ---- 评估 1: 动作预测准确率 ----
    all_preds = []
    all_gts = []
    all_valid = []

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=64, num_workers=4
    )

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 20:  # ~1280 样本
                break
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"].to(device)

            outputs = model({"videos": videos, "masks": masks})
            action_logits = outputs["action_logits"]

            all_preds.append(action_logits.cpu())
            all_gts.append(actions.cpu())
            valid = actions >= 0
            all_valid.append(valid.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    all_valid = torch.cat(all_valid, dim=0)

    acc, per_action_acc = compute_action_accuracy(
        all_preds, all_gts, all_valid
    )
    results["action_accuracy"] = round(acc, 4)
    results["per_action_accuracy"] = per_action_acc

    print(f"\n  Action Accuracy: {acc:.2%}")
    for act_name, acc_act in per_action_acc.items():
        print(f"    Action '{act_name}': {acc_act:.2%}")

    # 随机基线
    print(f"  Random baseline: {1/NUM_ACTIONS:.1%}")

    # ---- 评估 2: 重建质量 ----
    mse_vals = []
    psnr_vals = []
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 20:
                break
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            outputs = model({"videos": videos, "masks": masks})
            gt = videos[:, 1:]
            mse, psnr = compute_recon_metrics(outputs["recon"], gt)
            mse_vals.append(mse)
            psnr_vals.append(psnr)

    results["recon_mse"] = round(float(np.mean(mse_vals)), 6)
    results["psnr"] = round(float(np.mean(psnr_vals)), 2)
    print(f"  Recon MSE: {results['recon_mse']:.6f}")
    print(f"  PSNR: {results['psnr']:.1f} dB")

    # ---- 评估 3: 隐动作分析 ----
    z_mu = model.mu_record  # (N, latent_dim)
    if z_mu is not None:
        z_mu = z_mu.numpy()
        N, D = z_mu.shape

        # 隐动作方差
        latent_var = float(z_mu.var(axis=0).mean())
        results["latent_variance"] = round(latent_var, 4)
        print(f"  Latent variance: {latent_var:.4f}")

        # 隐动作各维度方差（判断是否有坍缩）
        per_dim_var = z_mu.var(axis=0)
        active_dims = (per_dim_var > 0.01).sum()
        results["active_dims"] = int(active_dims)
        results["per_dim_variance"] = [round(float(v), 4) for v in per_dim_var]
        print(f"  Active dims (var > 0.01): {active_dims}/{D}")

    # ---- 保存结果 ----
    save_path = os.path.join(RESULTS_DIR, f"results_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 保存模型 checkpoint
    ckpt_path = os.path.join(RESULTS_DIR, f"model_{args.name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Model saved: {ckpt_path}")
    print(f"  Results saved: {save_path}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    main()
