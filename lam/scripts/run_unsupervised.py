"""
无监督多主体隐动作训练 - Mask-Guided 架构。

核心修正：
- 使用合成数据的位置信息（mask/bbox）来池化主体特征
- **不使用动作标签训练**，让 LAM 无监督学习隐动作
- 训练完成后，用聚类分析等方法验证隐动作质量
- 动作标签只能用于**事后评估**（线性探针、聚类质量），不能用于训练

损失函数：
- 重建损失：MSE(recon, gt)
- KL 损失：KL(z_mu, z_var)
- **无动作监督**

用法:
  CUDA_VISIBLE_DEVICES=2 python run_unsupervised.py --name unsupervised_baseline --steps 2000
  CUDA_VISIBLE_DEVICES=2 python run_unsupervised.py --name unsupervised_no_interact --no_interaction --steps 2000
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


def compute_recon_metrics(recon, gt_videos):
    """计算重建 PSNR 和 MSE。"""
    mse = ((recon - gt_videos) ** 2).mean().item()
    psnr = -10 * np.log10(mse + 1e-10)
    return mse, psnr


def main():
    parser = argparse.ArgumentParser(description="无监督 Mask-Guided 多主体隐动作训练")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--no_interaction", action="store_true",
                        help="禁用主体间交互模块")
    parser.add_argument("--recon_weight", type=float, default=1.0,
                        help="重建损失权重")
    parser.add_argument("--kl_beta", type=float, default=0.0002,
                        help="KL 损失权重")
    parser.add_argument("--checkpoint_every", type=int, default=500)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "results", "unsupervised_lam"
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
    print(f"无监督 Mask-Guided 多主体隐动作训练")
    print(f"  GPU={args.gpu}, name={args.name}, batch={args.batch_size}, steps={args.steps}")
    print(f"  interaction={not args.no_interaction}")
    print(f"  recon_weight={args.recon_weight}, kl_beta={args.kl_beta}")
    print(f"  **无动作监督** - 纯无监督训练")
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
        num_actions=NUM_ACTIONS,  # 保留 action_head 用于事后评估
        use_interaction=not args.no_interaction,
        interaction_heads=4,
        interaction_layers=2,
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

    losses = {"total": [], "recon": [], "kl": []}
    step = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            videos = batch["videos"].to(device, non_blocking=True)    # (B, T, H, W, C)
            masks = batch["masks"].to(device, non_blocking=True)      # (B, T, A, H, W)
            # 注意：不使用 actions 标签进行训练！

            with torch.cuda.amp.autocast():
                outputs = model({"videos": videos, "masks": masks})

                # 重建损失
                gt = videos[:, 1:]  # (B, T-1, H, W, C)
                recon_loss = ((gt - outputs["recon"]) ** 2).mean()

                # KL 损失
                z_mu = outputs["z_mu"]  # (B*(T-1), A, latent_dim)
                z_var = outputs["z_var"]
                kl_loss = -0.5 * torch.sum(
                    1 + z_var - z_mu ** 2 - z_var.exp()
                ) / z_mu.shape[0]

                # 总损失（无动作监督）
                loss = args.recon_weight * recon_loss + args.kl_beta * kl_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses["total"].append(float(loss))
            losses["recon"].append(float(recon_loss))
            losses["kl"].append(float(kl_loss))

            if step % 100 == 0:
                elapsed = time.time() - t0
                mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                print(
                    f"  Step {step:4d}/{args.steps}: "
                    f"loss={float(loss):.4f}, "
                    f"recon={float(recon_loss):.4f}, "
                    f"kl={float(kl_loss):.4f}, "
                    f"mem={mem_peak:.1f}GB, {elapsed:.0f}s"
                )

            # 定期保存 checkpoint
            if step > 0 and step % args.checkpoint_every == 0:
                ckpt_path = os.path.join(RESULTS_DIR, f"model_{args.name}_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"    Saved checkpoint: {ckpt_path}")

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

    # ====== 事后评估 ======
    model.eval()
    results = {
        "architecture": "mask_guided_unsupervised",
        "use_interaction": not args.no_interaction,
        "batch_size": args.batch_size,
        "training_steps": args.steps,
        "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2),
        "recon_weight": args.recon_weight,
        "kl_beta": args.kl_beta,
        "supervised": False,  # 标记为无监督训练
    }

    # ---- 评估 1: 重建质量 ----
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=64, num_workers=4
    )

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

    # ---- 评估 2: 隐动作分析 ----
    z_mu = model.mu_record  # (N, A, latent_dim)
    if z_mu is not None:
        z_mu = z_mu.numpy()
        N, A, D = z_mu.shape

        # Slot variance: 每个主体隐动作的方差（越大表示越活跃）
        slot_var = [float(z_mu[:, a, :].var(axis=0).mean()) for a in range(A)]
        results["slot_variances"] = slot_var
        print(f"  Slot variances: {[f'{v:.4f}' for v in slot_var]}")

        # Slot 间余弦相似度（越低越独立）
        slot_means = np.array([z_mu[:, a, :].mean(0) for a in range(A)])
        norms = np.linalg.norm(slot_means, axis=1, keepdims=True)
        slot_means_norm = slot_means / (norms + 1e-8)
        cos_sim = slot_means_norm @ slot_means_norm.T
        results["slot_cosine_sim"] = cos_sim.tolist()
        print(f"  Slot cosine similarity:\n{np.array2string(cos_sim, precision=3)}")

    # ---- 评估 3: 线性探针（事后评估）----
    # 用动作标签训练一个简单的线性分类器，验证隐动作是否编码了动作信息
    print(f"\n  线性探针评估（事后评估，不参与训练）:")

    # 收集隐动作和动作标签
    all_z_mu = []
    all_actions = []
    all_valid = []

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 30:  # 更多样本用于线性探针
                break
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"].to(device)

            outputs = model({"videos": videos, "masks": masks})
            z_mu_batch = outputs["z_mu"]  # (B*(T-1), A, latent_dim)

            # 重塑为 (B, T-1, A, latent_dim)
            B = videos.shape[0]
            T1 = videos.shape[1] - 1
            A = 4
            z_mu_batch = z_mu_batch.reshape(B, T1, A, -1)

            all_z_mu.append(z_mu_batch.cpu())
            all_actions.append(actions.cpu())
            all_valid.append((actions >= 0).cpu())

    all_z_mu = torch.cat(all_z_mu, dim=0)  # (N, T-1, A, latent_dim)
    all_actions = torch.cat(all_actions, dim=0)  # (N, T-1, A)
    all_valid = torch.cat(all_valid, dim=0)  # (N, T-1, A)

    # 展平
    z_flat = all_z_mu.reshape(-1, 32)  # (N*(T-1)*A, 32)
    actions_flat = all_actions.reshape(-1)  # (N*(T-1)*A)
    valid_flat = all_valid.reshape(-1)  # (N*(T-1)*A)

    # 只用有效样本
    valid_idx = valid_flat.bool()
    z_valid = z_flat[valid_idx]  # (M, 32)
    actions_valid = actions_flat[valid_idx]  # (M)

    if len(actions_valid) > 100:
        # 简单线性分类器
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            z_valid.numpy(), actions_valid.numpy(), test_size=0.3, random_state=42
        )

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        acc_linear_probe = clf.score(X_test, y_test)

        results["linear_probe_accuracy"] = round(acc_linear_probe, 4)
        print(f"    Linear probe accuracy: {acc_linear_probe:.2%} (random: {1/NUM_ACTIONS:.1%})")
    else:
        print(f"    Not enough valid samples for linear probe")

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