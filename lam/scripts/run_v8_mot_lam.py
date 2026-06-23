"""
V8: MOT-Guided Slot-Time Latent Action Model (Stage 1)

损失:
  L = L_motion + β · L_KL

  L_motion = MSE( (Δbbox_bg + Δbbox_res) / scale, Δbbox_obs / scale )
  L_KL     = FreeBits(z_actor) + FreeBits(z_bg)

Stage 1 限制:
  - 合成数据 + GT bbox
  - 无 actor conditioning
  - 无 camera perturbation (z_bg 预期学到 ~0)

用法:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python lam/scripts/run_v8_mot_lam.py \\
      --name v8_stage1 --batch_size 16 --steps 5000 --lr 1e-4 --kl_beta 1.0
"""
import os, sys, json, time, argparse
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules.slot_time_lam import LatentActionModelV8
from lam.mot_slot_dataset import MOTSlotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--max_actors", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--z_bg_dim", type=int, default=16)
    parser.add_argument("--num_temporal_layers", type=int, default=2)
    parser.add_argument("--num_slot_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=32)
    parser.add_argument("--kl_beta", type=float, default=1.0)
    parser.add_argument("--free_bits_lambda", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "v8_mot_lam"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    else:
        data_root = args.data_root

    print(f"\n{'='*60}")
    print(f"V8: MOT-Guided Slot-Time Latent Action Model (Stage 1)")
    print(f"  GPU={args.gpu}, name={args.name}")
    print(f"  batch={args.batch_size}, steps={args.steps}, lr={args.lr}")
    print(f"  model_dim={args.model_dim}, z_dim={args.z_dim}, z_bg_dim={args.z_bg_dim}")
    print(f"  temporal_layers={args.num_temporal_layers}, slot_layers={args.num_slot_layers}")
    print(f"  kl_beta={args.kl_beta}, free_bits_lambda={args.free_bits_lambda}")
    print(f"  Loss: L_motion + {args.kl_beta} * L_KL")
    print(f"  数据: {data_root}")
    print(f"{'='*60}")

    train_dataset = MOTSlotDataset(
        os.path.join(data_root, "train"),
        max_actors=args.max_actors, num_frames=args.num_frames,
    )
    eval_dataset = MOTSlotDataset(
        os.path.join(data_root, "val"),
        max_actors=args.max_actors, num_frames=args.num_frames,
    )

    model = LatentActionModelV8(
        model_dim=args.model_dim,
        z_dim=args.z_dim,
        z_bg_dim=args.z_bg_dim,
        num_temporal_layers=args.num_temporal_layers,
        num_slot_layers=args.num_slot_layers,
        num_heads=args.num_heads,
        max_actors=args.max_actors,
        crop_size=args.crop_size,
        free_bits_lambda=args.free_bits_lambda,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.num_workers > 0),
        drop_last=True,
    )

    losses = {"total": [], "motion": [], "kl": []}
    step = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break
            videos = batch["videos"].to(device, non_blocking=True)
            boxes = batch["boxes"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)

            outputs = model({"videos": videos, "boxes": boxes, "valid_mask": valid})
            motion_loss = outputs["motion_loss"]
            kl_loss = outputs["kl_loss"]
            loss = motion_loss + args.kl_beta * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses["total"].append(float(loss))
            losses["motion"].append(float(motion_loss))
            losses["kl"].append(float(kl_loss))

            if step % 50 == 0:
                elapsed = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                z_var = float(outputs["mu_actor"].reshape(-1, args.z_dim).var(0).mean())
                print(
                    f"  Step {step:4d}/{args.steps}: "
                    f"loss={float(loss):.4f}, motion={float(motion_loss):.4f}, "
                    f"kl={float(kl_loss):.4f}, z_var={z_var:.4f}, "
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
        np.savetxt(
            os.path.join(RESULTS_DIR, f"loss_{args.name}_{key}.txt"),
            np.array(vals),
        )

    # === 评估: 收集 z_actor 用于聚类分析 ===
    model.eval()
    results = {
        "architecture": "v8_mot_lam",
        "stage": "stage1_synthetic",
        "max_actors": args.max_actors,
        "model_dim": args.model_dim,
        "z_dim": args.z_dim,
        "z_bg_dim": args.z_bg_dim,
        "num_temporal_layers": args.num_temporal_layers,
        "num_slot_layers": args.num_slot_layers,
        "kl_beta": args.kl_beta,
        "free_bits_lambda": args.free_bits_lambda,
        "training_steps": args.steps,
        "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2),
        "total_params": total_params,
    }

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=32, num_workers=args.num_workers, shuffle=False
    )

    all_z_actor = []
    all_z_bg = []
    all_actions = []
    all_actor_ids = []
    all_dbbox_pred = []
    all_dbbox_obs = []
    n_collected = 0
    with torch.no_grad():
        for batch in eval_loader:
            videos = batch["videos"].to(device)
            boxes = batch["boxes"].to(device)
            valid = batch["valid_mask"].to(device)
            actions = batch["actions"]      # (B, T-1, K)
            track_ids = batch["track_ids"]  # (B, K)
            out = model({"videos": videos, "boxes": boxes, "valid_mask": valid})
            # z_actor: (B, T-1, K, D) — 收集所有 transition 的样本
            z_a = out["mu_actor"].cpu().numpy()       # (B, T-1, K, D)
            z_b = out["mu_bg"].cpu().numpy()           # (B, T-1, D)
            v_np = valid[:, 1:].cpu().numpy()          # (B, T-1, K)
            act_np = actions.cpu().numpy()             # (B, T-1, K)
            B, T1, K, D = z_a.shape
            for b in range(B):
                for t in range(T1):
                    for k in range(K):
                        if v_np[b, t, k] and act_np[b, t, k] >= 0:
                            all_z_actor.append(z_a[b, t, k])
                            all_z_bg.append(z_b[b, t])
                            all_actions.append(int(act_np[b, t, k]))
                            all_actor_ids.append(int(track_ids[b, k]))
                            all_dbbox_pred.append(out["dbbox_pred"][b, t, k].cpu().numpy())
                            all_dbbox_obs.append(out["dbbox_obs"][b, t, k].cpu().numpy())
            n_collected += 1
            if n_collected >= 30:
                break

    z_actor_arr = np.array(all_z_actor)
    z_bg_arr = np.array(all_z_bg)
    actions_arr = np.array(all_actions)
    actor_ids_arr = np.array(all_actor_ids)
    dbbox_pred_arr = np.array(all_dbbox_pred)
    dbbox_obs_arr = np.array(all_dbbox_obs)

    z_var = float(z_actor_arr.var(axis=0).mean())
    active = int((z_actor_arr.var(axis=0) > 0.01).sum())
    results["z_actor_variance"] = round(z_var, 4)
    results["z_actor_active_dims"] = active
    results["z_bg_variance"] = round(float(z_bg_arr.var(axis=0).mean()), 4)
    results["dbbox_mse"] = float(((dbbox_pred_arr - dbbox_obs_arr) ** 2).mean())

    print(f"\n  z_actor 方差: {z_var:.4f}, 活跃维度: {active}/{args.z_dim}")
    print(f"  z_bg 方差: {results['z_bg_variance']:.4f}")
    print(f"  dbbox MSE (pixel²): {results['dbbox_mse']:.2f}")

    # 保存隐变量供后续聚类评估
    np.savez(
        os.path.join(RESULTS_DIR, f"latents_{args.name}.npz"),
        z_actor=z_actor_arr,
        z_bg=z_bg_arr,
        actions=actions_arr,
        actor_ids=actor_ids_arr,
        dbbox_pred=dbbox_pred_arr,
        dbbox_obs=dbbox_obs_arr,
    )

    save_path = os.path.join(RESULTS_DIR, f"results_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    ckpt_path = os.path.join(RESULTS_DIR, f"model_{args.name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  模型保存: {ckpt_path}")
    print(f"  结果保存: {save_path}")
    print(f"  隐变量保存: {RESULTS_DIR}/latents_{args.name}.npz")
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    main()
