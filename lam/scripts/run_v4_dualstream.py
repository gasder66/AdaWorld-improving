"""
V6: ST Encoder + MaskedPool + Per-Subject VAE + 结构化隐空间约束

新增损失项:
  - Free Bits KL (防止后验坍缩)
  - 互信息最小化 (slot 间编码不同主体)
  - 时序一致性 (同类运动在 z 空间聚集)

用法:
  CUDA_VISIBLE_DEVICES=2 python run_v4_dualstream.py --gpu 0 \\
      --name v6 --batch_size 32 --steps 5000 \\
      --keep_background --mi_weight 0.1 --temporal_weight 0.1
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
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--max_actors", type=int, default=4)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--enc_blocks", type=int, default=4)
    parser.add_argument("--dec_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--kl_beta", type=float, default=2e-4)
    parser.add_argument("--obj_recon_weight", type=float, default=0.01)
    parser.add_argument("--mi_weight", type=float, default=0.0,
                        help="互信息损失权重 (已废弃, 默认0=禁用)")
    parser.add_argument("--temporal_weight", type=float, default=0.0,
                        help="时序一致性损失权重 (已废弃, 默认0=禁用)")
    parser.add_argument("--contrast_weight", type=float, default=0.0,
                        help="对比学习损失权重 (0=禁用)")
    parser.add_argument("--delta_weight", type=float, default=0.1,
                        help="Delta 一致性损失权重 (动作级分离)")
    parser.add_argument("--num_frames_a2d", type=int, default=3,
                        help="A2D 数据集帧数")
    parser.add_argument("--frame_stride_a2d", type=int, default=1,
                        help="A2D 帧间跳步 (2=跳过相邻标注, 时间窗口翻倍)")
    parser.add_argument("--free_bits_lambda", type=float, default=0.5,
                        help="Free Bits KL 阈值")
    parser.add_argument("--keep_background", action="store_true", default=True,
                        help="保留背景槽 (默认开启)")
    parser.add_argument("--no_obj_st_attention", action="store_true",
                        help="禁用对象级时空注意力")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "a2d", "mot_a2d", "arb_a2d"],
                        help="数据集类型: synthetic=合成, a2d=GT A2D, mot_a2d=MOT A2D, arb_a2d=任意帧 YOLO A2D")
    parser.add_argument("--a2d_release", type=str, default=None,
                        help="A2D Release 目录路径")
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--yolo_source", type=str, default="a2d",
                        choices=["a2d", "coco", "reid"],
                        help="MOT masks 来源 (mot_a2d 数据集时有效)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0=单进程, 用于 arb_a2d 避免 CUDA fork 冲突)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "result",
        f"v6_arb_{args.num_frames_a2d}_{args.frame_stride_a2d}" if args.dataset == "arb_a2d" else
        f"v6_a2d_{args.yolo_source}" if args.dataset == "mot_a2d" else
        "v6_a2d" if args.dataset == "a2d" else "v6_structured"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    else:
        data_root = args.data_root

    if args.a2d_release is None:
        args.a2d_release = os.path.join(
            os.path.dirname(__file__), "..", "..", "Release"
        )

    print(f"\n{'='*60}")
    print(f"V6: ST Encoder + MaskedPool + Structured Latent Space")
    print(f"  GPU={args.gpu}, name={args.name}, batch={args.batch_size}, steps={args.steps}")
    print(f"  max_actors={args.max_actors}, keep_background={args.keep_background}")
    print(f"  dataset={args.dataset}")
    print(f"  kl_beta={args.kl_beta}, free_bits_lambda={args.free_bits_lambda}")
    print(f"  obj_recon_weight={args.obj_recon_weight}")
    print(f"  mi_weight={args.mi_weight}, temporal_weight={args.temporal_weight}, contrast_weight={args.contrast_weight}, delta_weight={args.delta_weight}")
    print(f"  Loss: L_recon + kl_beta*KL_fb + obj_recon*L_obj + delta*L_delta + contrast*L_contrast")
    print(f"  数据: {data_root}")
    print(f"{'='*60}")

    if args.dataset == "a2d":
        from lam.a2d_dataset import A2DDataset
        ds_kwargs = dict(
            data_root=data_root, release_root=args.a2d_release,
            split="train", num_frames=args.num_frames_a2d, max_actors=args.max_actors,
            img_size=256, frame_stride=args.frame_stride_a2d,
        )
        train_dataset = A2DDataset(**ds_kwargs)
        ds_kwargs["split"] = "test"
        eval_dataset = A2DDataset(**ds_kwargs)
    elif args.dataset == "mot_a2d":
        from lam.mot_a2d_dataset import MOTA2DDataset
        ds_kwargs = dict(
            data_root=data_root, release_root=args.a2d_release,
            split="train", num_frames=args.num_frames_a2d, max_actors=args.max_actors,
            img_size=256, frame_stride=args.frame_stride_a2d,
        )
        train_dataset = MOTA2DDataset(yolo_source=args.yolo_source, **ds_kwargs)
        ds_kwargs["split"] = "test"
        eval_dataset = MOTA2DDataset(yolo_source=args.yolo_source, **ds_kwargs)
    elif args.dataset == "arb_a2d":
        from lam.arbitrary_a2d_dataset import ArbitraryA2DDataset
        num_frames = args.num_frames_a2d if args.num_frames_a2d else 2
        train_dataset = ArbitraryA2DDataset(
            video_dir=os.path.join(data_root, "train"),
            T=num_frames, stride=args.frame_stride_a2d,
            max_actors=args.max_actors,
            cache_dir="/home/xiaojy/projects/AdaWorld-improving/result/v6_a2d/arb_cache",
        )
        eval_dataset = ArbitraryA2DDataset(
            video_dir=os.path.join(data_root, "test"),
            T=num_frames, stride=args.frame_stride_a2d,
            max_actors=args.max_actors,
            cache_dir="/home/xiaojy/projects/AdaWorld-improving/result/v6_a2d/arb_cache",
            start_frame_offset=10,
        )
    else:
        train_dataset = DiskSyntheticDataset(
            os.path.join(data_root, "train"), num_frames=5, output_format="t h w c",
        )
        eval_dataset = DiskSyntheticDataset(
            os.path.join(data_root, "val"), num_frames=5, output_format="t h w c",
        )

    model = LatentActionModel(
        in_dim=3, model_dim=args.model_dim, latent_dim=args.latent_dim,
        patch_size=16, enc_blocks=args.enc_blocks, dec_blocks=args.dec_blocks,
        num_heads=args.num_heads, max_actors=args.max_actors,
        keep_background=args.keep_background,
        use_obj_st_attention=not args.no_obj_st_attention,
        free_bits_lambda=args.free_bits_lambda,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}, 可训练: {trainable_params:,}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.num_workers > 0),
    )

    losses = {"total": [], "recon": [], "kl": [], "obj_recon": [], "delta": [], "contrast": []}
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
                gt = videos[:, 1:]
                recon_loss = ((gt - outputs["recon"]) ** 2).mean()
                kl_loss = outputs["kl_loss"]
                obj_recon_loss = outputs["obj_recon_loss"]
                delta_loss = outputs["delta_loss"]
                contrast_loss = outputs["contrast_loss"]

                loss = (
                    recon_loss
                    + args.kl_beta * kl_loss
                    + args.obj_recon_weight * obj_recon_loss
                    + args.delta_weight * delta_loss
                    + args.contrast_weight * contrast_loss
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
            losses["delta"].append(float(delta_loss))
            losses["contrast"].append(float(contrast_loss))

            if step % 50 == 0:
                elapsed = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                print(
                    f"  Step {step:4d}/{args.steps}: "
                    f"loss={float(loss):.4f}, recon={float(recon_loss):.4f}, "
                    f"kl={float(kl_loss):.4f}, obj={float(obj_recon_loss):.4f}, "
                    f"delta={float(delta_loss):.4f}, cont={float(contrast_loss):.4f}, "
                    f"mem={mem:.1f}GB, {elapsed:.0f}s"
                )
            step += 1

    training_time = time.time() - t0
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"\n  训练完成. 峰值内存: {mem_peak:.1f}GB, 耗时: {training_time:.0f}s")

    for key, vals in losses.items():
        np.savetxt(os.path.join(RESULTS_DIR, f"loss_{args.name}_{key}.txt"), np.array(vals))

    # === 评估 ===
    model.eval()
    results = {
        "architecture": "v6_structured",
        "max_actors": args.max_actors, "model_dim": args.model_dim,
        "latent_dim": args.latent_dim,
        "enc_blocks": args.enc_blocks, "dec_blocks": args.dec_blocks,
        "keep_background": args.keep_background,
        "kl_beta": args.kl_beta, "free_bits_lambda": args.free_bits_lambda,
        "obj_recon_weight": args.obj_recon_weight,
        "mi_weight": args.mi_weight, "temporal_weight": args.temporal_weight,
        "training_steps": args.steps, "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2),
        "total_params": total_params,
    }

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32,
                                               num_workers=max(0, args.num_workers))
    mse_vals = []
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 20: break
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            out = model({"videos": videos, "masks": masks})
            mse = ((videos[:, 1:] - out["recon"]) ** 2).reshape(videos.shape[0], -1).mean(dim=1)
            mse_vals.extend(mse.cpu().tolist())

    mse_arr = np.array(mse_vals)
    psnr_arr = -10 * np.log10(mse_arr + 1e-10)
    results["recon_mse"] = float(mse_arr.mean())
    results["psnr"] = round(float(psnr_arr.mean()), 2)
    print(f"\n  重建 MSE: {results['recon_mse']:.6f}, PSNR: {results['psnr']:.1f} dB")

    z_mu = model.mu_record
    if z_mu is not None:
        z_np = z_mu.numpy()
        z_flat = z_np.reshape(-1, z_np.shape[-1])
        lvar = float(z_flat.var(axis=0).mean())
        results["latent_variance"] = round(lvar, 4)
        active = int((z_flat.var(axis=0) > 0.01).sum())
        results["active_dims"] = active
        print(f"  隐方差: {lvar:.4f}, 活跃维度: {active}/{z_flat.shape[-1]}")

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