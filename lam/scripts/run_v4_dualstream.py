"""
原始 AdaWorld LAM 训练脚本。

架构: [learnable tokens + patches] → SpatioTemporalTransformer → VAE → z
Decoder: patches + z → reconstruction

多槽模式: num_slots=K, K 个 learnable token 独立学习不同运动模式。
K=1 时退化为原始 AdaWorld。

用法:
  CUDA_VISIBLE_DEVICES=2 python run_v4_dualstream.py --gpu 0 --name test --num_slots 4 --batch_size 32 --steps 2000
"""
import os, sys, json, time, argparse
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_slots", type=int, default=4,
                        help="learnable action tokens (K)")
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--enc_blocks", type=int, default=4)
    parser.add_argument("--dec_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--kl_beta", type=float, default=2e-4)
    parser.add_argument("--obj_recon_weight", type=float, default=0.0,
                        help="对象级重建损失权重 (0=禁用, 建议 1.0)")
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "adaworld_lam"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    else:
        data_root = args.data_root

    print(f"\n{'='*60}")
    print(f"AdaWorld Original LAM Training")
    print(f"  GPU={args.gpu}, name={args.name}, batch={args.batch_size}, steps={args.steps}")
    print(f"  num_slots={args.num_slots}, model_dim={args.model_dim}, latent_dim={args.latent_dim}")
    print(f"  enc_blocks={args.enc_blocks}, dec_blocks={args.dec_blocks}, kl_beta={args.kl_beta}")
    print(f"  obj_recon_weight={args.obj_recon_weight}")
    print(f"  Loss: L_recon + kl_beta * KL" + (" + obj_recon_weight * L_obj_recon" if args.obj_recon_weight > 0 else ""))
    print(f"  数据: {data_root}")
    print(f"{'='*60}")

    train_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "train"), num_frames=5, output_format="t h w c",
    )
    eval_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"), num_frames=5, output_format="t h w c",
    )

    model = LatentActionModel(
        in_dim=3, model_dim=args.model_dim, latent_dim=args.latent_dim,
        patch_size=16, enc_blocks=args.enc_blocks, dec_blocks=args.dec_blocks,
        num_heads=args.num_heads, num_slots=args.num_slots,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}, 可训练: {trainable_params:,}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    losses = {"total": [], "recon": [], "kl": []}
    step = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break
            videos = batch["videos"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model({"videos": videos})
                gt = videos[:, 1:]
                recon_loss = ((gt - outputs["recon"]) ** 2).mean()

                z_mu, z_var = outputs["z_mu"], outputs["z_var"]
                kl_loss = -0.5 * torch.sum(
                    1 + z_var - z_mu ** 2 - z_var.exp()
                ) / z_mu.reshape(-1).shape[0]

                loss = recon_loss + args.kl_beta * kl_loss
                if args.obj_recon_weight > 0:
                    obj_recon_loss = outputs["obj_recon_loss"]
                    loss = loss + args.obj_recon_weight * obj_recon_loss
                else:
                    obj_recon_loss = torch.tensor(0.0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses["total"].append(float(loss))
            losses["recon"].append(float(recon_loss))
            losses["kl"].append(float(kl_loss))
            if args.obj_recon_weight > 0:
                losses.setdefault("obj_recon", []).append(float(obj_recon_loss))

            if step % 50 == 0:
                elapsed = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                msg = (
                    f"  Step {step:3d}/{args.steps}: "
                    f"loss={float(loss):.4f}, recon={float(recon_loss):.4f}, "
                    f"kl={float(kl_loss):.6f}"
                )
                if args.obj_recon_weight > 0:
                    msg += f", obj_recon={float(obj_recon_loss):.6f}"
                msg += f", mem={mem:.1f}GB, {elapsed:.0f}s"
                print(msg)
            step += 1

    training_time = time.time() - t0
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"\n  训练完成. 峰值内存: {mem_peak:.1f}GB, 耗时: {training_time:.0f}s")

    for key, vals in losses.items():
        np.savetxt(os.path.join(RESULTS_DIR, f"loss_{args.name}_{key}.txt"), np.array(vals))

    # === 评估 ===
    model.eval()
    results = {
        "architecture": "adaworld_original_lam",
        "num_slots": args.num_slots, "model_dim": args.model_dim,
        "latent_dim": args.latent_dim,
        "enc_blocks": args.enc_blocks, "dec_blocks": args.dec_blocks,
        "training_steps": args.steps, "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2),
        "total_params": total_params, "kl_beta": args.kl_beta,
        "obj_recon_weight": args.obj_recon_weight,
    }

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, num_workers=4)
    mse_vals = []
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 20: break
            videos = batch["videos"].to(device)
            out = model({"videos": videos})
            mse = ((videos[:, 1:] - out["recon"]) ** 2).reshape(videos.shape[0], -1).mean(dim=1)
            mse_vals.extend(mse.cpu().tolist())

    mse_arr = np.array(mse_vals)
    psnr_arr = -10 * np.log10(mse_arr + 1e-10)
    results["recon_mse"] = float(mse_arr.mean())
    results["psnr"] = round(float(psnr_arr.mean()), 2)
    print(f"\n  重建 MSE: {results['recon_mse']:.6f}, PSNR: {results['psnr']:.1f} dB")

    # 隐动作分析
    z_mu = model.mu_record
    if z_mu is not None:
        z_np = z_mu.numpy()
        z_flat = z_np.reshape(-1, z_np.shape[-1])
        psnr = float(z_flat.var(axis=0).mean())
        results["latent_variance"] = round(psnr, 4)
        active = int((z_flat.var(axis=0) > 0.01).sum())
        results["active_dims"] = active
        print(f"  隐方差: {psnr:.4f}, 活跃维度: {active}/{z_flat.shape[-1]}")

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
