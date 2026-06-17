"""
V3 多主体感知隐动作模型训练脚本（无监督v2）。

核心变化：
1. 仅多向量模式（已移除单向量模式）
2. 无动作监督（已移除 Action Head）
3. 新增对象级重建损失（Object-Level Reconstruction Loss）
4. 训练目标: L = L_recon + β·KL + λ·L_obj_recon

用法:
  python run_single.py --gpu 2 --name v3_unsup_v2 --batch_size 32 --steps 500
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


def compute_recon_metrics(recon, gt_videos):
    mse = ((recon - gt_videos) ** 2).mean().item()
    psnr = -10 * np.log10(mse + 1e-10)
    return mse, psnr


def main():
    parser = argparse.ArgumentParser(description="V3 无监督多主体隐动作模型训练")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--no_obj_st_attention", action="store_true",
                        help="禁用对象级时空注意力模块")
    parser.add_argument("--recon_weight", type=float, default=1.0,
                        help="帧重建损失权重")
    parser.add_argument("--kl_beta", type=float, default=0.0002,
                        help="KL 散度权重")
    parser.add_argument("--obj_recon_weight", type=float, default=0.1,
                        help="对象级重建损失权重")
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "v3_obj_st_attention"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    else:
        data_root = args.data_root

    print(f"\n{'='*60}")
    print(f"V3 无监督多主体隐动作模型训练")
    print(f"  GPU={args.gpu}, name={args.name}, batch={args.batch_size}, steps={args.steps}")
    print(f"  obj_st_attention={not args.no_obj_st_attention}")
    print(f"  obj_recon_weight={args.obj_recon_weight}, recon_weight={args.recon_weight}, kl_beta={args.kl_beta}")
    print(f"  训练目标: L = recon_weight * L_recon + kl_beta * KL + obj_recon_weight * L_obj_recon")
    print(f"  数据: {data_root}")
    print(f"{'='*60}")

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

    model = LatentActionModel(
        in_dim=3,
        model_dim=256,
        latent_dim=32,
        patch_size=16,
        enc_blocks=4,
        dec_blocks=4,
        num_heads=8,
        max_actors=4,
        num_actions=5,  # 保留但不再用于监督
        use_obj_st_attention=not args.no_obj_st_attention,
        obj_st_heads=8,
        obj_st_layers=2,
        multi_vector=True,  # 仅多向量模式
        use_grad_checkpointing=True,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  参数总计: {params:,}")

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

    losses = {"total": [], "recon": [], "kl": [], "obj_recon": []}
    step = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            videos = batch["videos"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model({"videos": videos, "masks": masks})

                # 1. 帧重建损失 (frame-level recon)
                gt = videos[:, 1:]
                recon_loss = ((gt - outputs["recon"]) ** 2).mean()

                # 2. KL 散度损失
                z_mu = outputs["z_mu"]
                z_var = outputs["z_var"]
                kl_loss = -0.5 * torch.sum(
                    1 + z_var - z_mu ** 2 - z_var.exp()
                ) / z_mu.reshape(-1).shape[0]

                # 3. 对象级重建损失 (per-object feature next-frame prediction)
                obj_recon_loss = outputs["obj_recon_loss"]  # 已由模型内部计算

                # 总损失
                loss = (
                    args.recon_weight * recon_loss
                    + args.kl_beta * kl_loss
                    + args.obj_recon_weight * obj_recon_loss
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
            losses["obj_recon"].append(float(obj_recon_loss))

            if step % 50 == 0:
                elapsed = time.time() - t0
                mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                print(
                    f"  Step {step:3d}/{args.steps}: "
                    f"loss={float(loss):.4f}, "
                    f"recon={float(recon_loss):.4f}, "
                    f"kl={float(kl_loss):.6f}, "
                    f"obj_recon={float(obj_recon_loss):.6f}, "
                    f"mem={mem_peak:.1f}GB, {elapsed:.0f}s"
                )

            step += 1

    training_time = time.time() - t0
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"\n  训练完成. 峰值内存: {mem_peak:.1f}GB, 耗时: {training_time:.0f}s")

    for key, vals in losses.items():
        np.savetxt(
            os.path.join(RESULTS_DIR, f"loss_{args.name}_{key}.txt"),
            np.array(vals),
        )

    # ====== 评估 ======
    model.eval()
    results = {
        "architecture": "v3_unsup_v2_multi_vector",
        "multi_vector": True,
        "use_obj_st_attention": not args.no_obj_st_attention,
        "batch_size": args.batch_size,
        "training_steps": args.steps,
        "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2),
        "recon_weight": args.recon_weight,
        "kl_beta": args.kl_beta,
        "obj_recon_weight": args.obj_recon_weight,
    }

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=64, num_workers=4
    )

    # ---- 评估 1: 重建质量 ----
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
    print(f"\n  重建 MSE: {results['recon_mse']:.6f}")
    print(f"  PSNR: {results['psnr']:.1f} dB")

    # ---- 评估 2: 隐动作分析 ----
    z_mu = model.mu_record
    if z_mu is not None:
        z_mu_np = z_mu.numpy()
        # 多向量模式: (N, A+1, D)
        z_mu_actors = z_mu_np[:, 1:, :]  # (N, A, D), 不含背景
        z_flat = z_mu_actors.reshape(-1, z_mu_actors.shape[-1])

        latent_var = float(z_flat.var(axis=0).mean())
        results["latent_variance"] = round(latent_var, 4)
        print(f"  隐方差: {latent_var:.4f}")

        per_dim_var = z_flat.var(axis=0)
        active_dims = int((per_dim_var > 0.01).sum())
        results["active_dims"] = active_dims
        results["total_dims"] = int(per_dim_var.shape[0])
        print(f"  活跃维度 (var > 0.01): {active_dims}/{per_dim_var.shape[0]}")

    # ---- 评估 3: 对象级重建质量 ----
    obj_recon_mse = []
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 20:
                break
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            outputs = model({"videos": videos, "masks": masks})
            obj_recon_mse.append(float(outputs["obj_recon_loss"]))

    results["obj_recon_mse"] = round(float(np.mean(obj_recon_mse)), 6)
    print(f"  对象级重建 MSE: {results['obj_recon_mse']:.6f}")

    # ---- 保存 ----
    save_path = os.path.join(RESULTS_DIR, f"results_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    ckpt_path = os.path.join(RESULTS_DIR, f"model_{args.name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  模型保存: {ckpt_path}")
    print(f"  结果保存: {save_path}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    main()