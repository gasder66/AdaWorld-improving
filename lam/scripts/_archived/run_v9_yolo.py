"""
V9 YOLO: Train V9 on YOLO-detected A2D data (real video).

与合成数据 run_v9.py 的区别:
  - YOLOBoxDataset 替代 MOTSlotDataset
  - bbox_scale=48 (A2D bbox 更大)
  - num_actor_types=7 (可选 FiLM 条件化)
  - num_workers=4, persistent_workers=True (视频解码慢)
  - 大部分样本 action=-1 (仅用 motion_loss + recon_loss 训练)

用法:
  # V9-A on YOLO A2D
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python lam/scripts/run_v9_yolo.py \\
      --name v9a_yolo --variant A --batch_size 16 --steps 5000 --max_samples 2000 \\
      --recon_weight 0.1 --num_actor_types 7
"""
import os, sys, json, time, argparse
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules.v9_model import LatentActionModelV9
from lam.yolo_box_dataset import YOLOBoxDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--variant", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--video_dir", type=str, default="data/a2d/train")
    parser.add_argument("--eval_video_dir", type=str, default="data/a2d/test")
    parser.add_argument("--release_root", type=str, default="Release")
    parser.add_argument("--max_actors", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--z_bg_dim", type=int, default=16)
    parser.add_argument("--z_app_dim", type=int, default=16)
    parser.add_argument("--num_temporal_layers", type=int, default=2)
    parser.add_argument("--num_slot_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=32)
    parser.add_argument("--kl_beta", type=float, default=1.0)
    parser.add_argument("--free_bits_lambda", type=float, default=0.5)
    parser.add_argument("--recon_weight", type=float, default=0.1)
    parser.add_argument("--bbox_scale", type=float, default=48.0)
    parser.add_argument("--num_actor_types", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="result/v8_mot_lam/yolo_cache")
    parser.add_argument("--detach_z_actor", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "v9"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"V9 YOLO: V9 on YOLO-detected A2D (variant {args.variant})")
    print(f"  GPU={args.gpu}, name={args.name}")
    print(f"  batch={args.batch_size}, steps={args.steps}, lr={args.lr}")
    print(f"  stride={args.stride}, max_samples={args.max_samples}")
    print(f"  bbox_scale={args.bbox_scale}, num_actor_types={args.num_actor_types}")
    print(f"  recon_weight={args.recon_weight}")
    print(f"{'='*60}")

    train_dataset = YOLOBoxDataset(
        video_dir=args.video_dir, release_root=args.release_root,
        split="train", T=args.num_frames, stride=args.stride,
        max_actors=args.max_actors, img_size=256,
        cache_dir=args.cache_dir, max_samples=args.max_samples,
    )
    eval_dataset = YOLOBoxDataset(
        video_dir=args.eval_video_dir, release_root=args.release_root,
        split="test", T=args.num_frames, stride=args.stride,
        max_actors=args.max_actors, img_size=256,
        cache_dir=args.cache_dir, max_samples=min(200, args.max_samples or 200),
    )

    model = LatentActionModelV9(
        model_dim=args.model_dim, z_dim=args.z_dim, z_bg_dim=args.z_bg_dim,
        num_temporal_layers=args.num_temporal_layers, num_slot_layers=args.num_slot_layers,
        num_heads=args.num_heads, max_actors=args.max_actors, crop_size=args.crop_size,
        free_bits_lambda=args.free_bits_lambda, bbox_scale=args.bbox_scale,
        use_bg_slot=True, num_actor_types=args.num_actor_types,
        variant=args.variant, z_app_dim=args.z_app_dim,
        detach_z_actor_recon=args.detach_z_actor,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    v8_params = sum(p.numel() for p in model.v8.parameters())
    print(f"  总参数: {total_params:,} (V8: {v8_params:,}, decoder+: {total_params - v8_params:,})")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=(args.num_workers > 0),
    )

    losses = {"total": [], "motion": [], "kl": [], "recon": [], "recon_l1": [], "recon_ssim": []}
    step = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break
            batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(batch_gpu)
            motion_loss = outputs["motion_loss"]
            kl_loss = outputs["kl_loss"]
            recon_loss = outputs["recon_loss"]

            loss = (motion_loss + args.kl_beta * kl_loss
                    + args.recon_weight * recon_loss)
            if "kl_app" in outputs:
                loss = loss + args.kl_beta * outputs["kl_app"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses["total"].append(float(loss))
            losses["motion"].append(float(motion_loss))
            losses["kl"].append(float(kl_loss))
            losses["recon"].append(float(recon_loss))
            losses["recon_l1"].append(float(outputs["recon_l1"]))
            losses["recon_ssim"].append(float(outputs["recon_ssim"]))

            if step % 100 == 0:
                elapsed = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                z_var = float(outputs["mu_actor"].reshape(-1, args.z_dim).var(0).mean())
                z_bg_var = float(outputs["mu_bg"].reshape(-1, args.z_bg_dim).var(0).mean()) if "mu_bg" in outputs else 0.0
                print(
                    f"  Step {step:4d}/{args.steps}: "
                    f"loss={float(loss):.4f}, motion={float(motion_loss):.4f}, "
                    f"kl={float(kl_loss):.4f}, recon={float(recon_loss):.4f}, "
                    f"ssim={float(outputs['recon_ssim']):.3f}, "
                    f"z_var={z_var:.4f}, z_bg_var={z_bg_var:.4f}, "
                    f"mem={mem:.1f}GB, {elapsed:.0f}s"
                )

            if (step + 1) % args.checkpoint_every == 0:
                ckpt = os.path.join(RESULTS_DIR, f"model_{args.name}_step{step+1}.pt")
                torch.save(model.state_dict(), ckpt)
            step += 1

    training_time = time.time() - t0
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"\n  训练完成. 峰值内存: {mem_peak:.1f}GB, 耗时: {training_time:.0f}s")

    for key, vals in losses.items():
        np.savetxt(os.path.join(RESULTS_DIR, f"loss_{args.name}_{key}.txt"), np.array(vals))

    # === Eval: collect latents + recon (only samples with GT action labels) ===
    model.eval()
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=8, num_workers=args.num_workers, shuffle=False,
    )

    all_z_actor, all_z_bg, all_z_app = [], [], []
    all_actions, all_actor_types = [], []
    all_recon, all_crop_t, all_crop_tp1 = [], [], []
    all_dbbox_pred, all_dbbox_obs = [], []

    with torch.no_grad():
        for batch in eval_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(batch_gpu)
            z_a = out["mu_actor"].cpu().numpy()
            z_b = out["mu_bg"].cpu().numpy()
            z_app = out["mu_app"].cpu().numpy() if "mu_app" in out else None
            v_np = batch["valid_mask"][:, 1:].cpu().numpy()
            act_np = batch["actions"].cpu().numpy()
            actor_labels = batch["actor_labels"].cpu().numpy()
            recon_np = out["recon"].cpu().numpy()
            crop_t_np = out["crop_t"].cpu().numpy()
            crop_tp1_np = out["crop_tp1"].cpu().numpy()
            B, T1, K, D = z_a.shape
            for b in range(B):
                for t in range(T1):
                    for k in range(K):
                        if v_np[b, t, k] and act_np[b, t, k] >= 0:
                            all_z_actor.append(z_a[b, t, k])
                            all_z_bg.append(z_b[b, t])
                            if z_app is not None:
                                all_z_app.append(z_app[b, t, k])
                            all_actions.append(int(act_np[b, t, k]))
                            all_actor_types.append(int(actor_labels[b, k]))
                            all_recon.append(recon_np[b, t, k])
                            all_crop_t.append(crop_t_np[b, t, k])
                            all_crop_tp1.append(crop_tp1_np[b, t, k])
                            all_dbbox_pred.append(out["dbbox_pred"][b, t, k].cpu().numpy())
                            all_dbbox_obs.append(out["dbbox_obs"][b, t, k].cpu().numpy())

    z_actor_arr = np.array(all_z_actor) if all_z_actor else np.zeros((0, args.z_dim))
    z_bg_arr = np.array(all_z_bg) if all_z_bg else np.zeros((0, args.z_bg_dim))
    actions_arr = np.array(all_actions) if all_actions else np.zeros(0, dtype=int)
    actor_types_arr = np.array(all_actor_types) if all_actor_types else np.zeros(0, dtype=int)
    recon_arr = np.array(all_recon) if all_recon else np.zeros((0, 3, args.crop_size, args.crop_size))
    crop_t_arr = np.array(all_crop_t) if all_crop_t else np.zeros((0, 3, args.crop_size, args.crop_size))
    crop_tp1_arr = np.array(all_crop_tp1) if all_crop_tp1 else np.zeros((0, 3, args.crop_size, args.crop_size))
    dbbox_pred_arr = np.array(all_dbbox_pred) if all_dbbox_pred else np.zeros((0, 4))
    dbbox_obs_arr = np.array(all_dbbox_obs) if all_dbbox_obs else np.zeros((0, 4))
    z_app_arr = np.array(all_z_app) if all_z_app else None

    print(f"\n  Eval samples with GT action: {len(z_actor_arr)}")
    if len(z_actor_arr) > 0:
        z_var = float(z_actor_arr.var(axis=0).mean())
        z_bg_var = float(z_bg_arr.var(axis=0).mean())
        dbbox_mse = float(((dbbox_pred_arr - dbbox_obs_arr) ** 2).mean())
        print(f"  z_actor var: {z_var:.4f}, z_bg var: {z_bg_var:.4f}")
        print(f"  dbbox MSE: {dbbox_mse:.2f}")
        print(f"  Actions: {np.unique(actions_arr, return_counts=True)}")

        # Recon metrics
        from lam.modules.v9_decoder import compute_psnr, compute_ssim_simple
        recon_f = torch.from_numpy(recon_arr)
        target_f = torch.from_numpy(crop_tp1_arr)
        crop_t_f = torch.from_numpy(crop_t_arr)
        psnr_recon = compute_psnr(recon_f, target_f)
        psnr_copy = compute_psnr(crop_t_f, target_f)
        ssim_recon = compute_ssim_simple(recon_f, target_f)
        ssim_copy = compute_ssim_simple(crop_t_f, target_f)
        print(f"  Recon: PSNR={psnr_recon:.2f} dB, copy={psnr_copy:.2f} dB, Δ={psnr_recon - psnr_copy:+.2f}")
        print(f"  SSIM:  recon={ssim_recon:.4f}, copy={ssim_copy:.4f}")
    else:
        z_var = z_bg_var = dbbox_mse = 0.0
        psnr_recon = psnr_copy = ssim_recon = ssim_copy = 0.0

    results = {
        "architecture": "v9", "variant": args.variant, "stage": "yolo_a2d",
        "training_steps": args.steps, "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2), "total_params": total_params,
        "v8_params": v8_params, "decoder_params": total_params - v8_params,
        "z_actor_variance": round(z_var, 4), "z_bg_variance": round(z_bg_var, 4),
        "dbbox_mse": round(dbbox_mse, 2), "n_eval_samples": len(z_actor_arr),
        "n_train_samples": len(train_dataset),
        "bbox_scale": args.bbox_scale, "num_actor_types": args.num_actor_types,
        "recon_weight": args.recon_weight,
        "psnr_recon": round(psnr_recon, 2), "psnr_copy": round(psnr_copy, 2),
        "ssim_recon": round(ssim_recon, 4), "ssim_copy": round(ssim_copy, 4),
        "delta_psnr_recon_copy": round(psnr_recon - psnr_copy, 2),
    }

    # Save latents
    if len(z_actor_arr) > 0:
        save_dict = dict(
            z_actor=z_actor_arr, z_bg=z_bg_arr,
            actions=actions_arr, actor_types=actor_types_arr,
            recon=recon_arr, crop_t=crop_t_arr, crop_tp1=crop_tp1_arr,
            dbbox_pred=dbbox_pred_arr, dbbox_obs=dbbox_obs_arr,
        )
        if z_app_arr is not None:
            save_dict["z_app"] = z_app_arr
        np.savez(
            os.path.join(RESULTS_DIR, f"latents_{args.name}.npz"),
            **save_dict,
        )

    save_path = os.path.join(RESULTS_DIR, f"results_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    ckpt_path = os.path.join(RESULTS_DIR, f"model_{args.name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  模型保存: {ckpt_path}")
    print(f"  结果保存: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
