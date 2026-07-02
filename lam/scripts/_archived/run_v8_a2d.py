"""
V8 Stage 2B: Train V8 on real A2D data.

与 Stage 1/2A 的区别:
  - 数据: A2D 真实视频 + GT bbox (从 .mat 读取)
  - actor_labels: A2D actor type (1-7), 用于 Phase 2C actor conditioning
  - actions: A2D 8 类 (0-7)
  - bbox_scale: 48 (A2D bbox delta 更大)
  - 无 on-the-fly camera perturbation (真实视频自带相机运动)
  - z_bg 预期学到真实相机运动

用法:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python lam/scripts/run_v8_a2d.py \\
      --name v8_a2d --batch_size 8 --steps 3000 --lr 1e-4
"""
import os, sys, json, time, argparse
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules.slot_time_lam import LatentActionModelV8
from lam.a2d_box_dataset import A2DBoxDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_root", type=str, default="data/a2d")
    parser.add_argument("--release_root", type=str, default="Release")
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
    parser.add_argument("--bbox_scale", type=float, default=48.0)
    parser.add_argument("--num_actor_types", type=int, default=0,
                        help="A2D actor type count for FiLM conditioning (0=no conditioning, 7=A2D)")
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

    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "result", "v8_mot_lam")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"V8 Stage 2B: A2D Real Data")
    print(f"  GPU={args.gpu}, name={args.name}")
    print(f"  batch={args.batch_size}, steps={args.steps}, lr={args.lr}")
    print(f"  bbox_scale={args.bbox_scale}, kl_beta={args.kl_beta}")
    print(f"  num_actor_types={args.num_actor_types} ({'FiLM conditioning' if args.num_actor_types > 0 else 'no conditioning'})")
    print(f"{'='*60}")

    train_dataset = A2DBoxDataset(
        args.data_root, args.release_root, split="train",
        num_frames=args.num_frames, max_actors=args.max_actors, img_size=256,
    )
    eval_dataset = A2DBoxDataset(
        args.data_root, args.release_root, split="test",
        num_frames=args.num_frames, max_actors=args.max_actors, img_size=256,
    )

    model = LatentActionModelV8(
        model_dim=args.model_dim, z_dim=args.z_dim, z_bg_dim=args.z_bg_dim,
        num_temporal_layers=args.num_temporal_layers, num_slot_layers=args.num_slot_layers,
        num_heads=args.num_heads, max_actors=args.max_actors, crop_size=args.crop_size,
        free_bits_lambda=args.free_bits_lambda, bbox_scale=args.bbox_scale,
        use_bg_slot=True, num_actor_types=args.num_actor_types,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    losses = {"total": [], "motion": [], "kl": []}
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
                z_bg_var = float(outputs["mu_bg"].reshape(-1, args.z_bg_dim).var(0).mean()) if "mu_bg" in outputs else 0.0
                print(
                    f"  Step {step:4d}/{args.steps}: "
                    f"loss={float(loss):.4f}, motion={float(motion_loss):.4f}, "
                    f"kl={float(kl_loss):.4f}, z_var={z_var:.4f}, z_bg_var={z_bg_var:.4f}, "
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

    # === Eval: collect latents ===
    model.eval()
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=8, num_workers=args.num_workers, shuffle=False
    )

    all_z_actor, all_z_bg = [], []
    all_actions, all_actor_types, all_actor_ids = [], [], []
    all_dbbox_pred, all_dbbox_obs = [], []
    with torch.no_grad():
        for batch in eval_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(batch_gpu)
            z_a = out["mu_actor"].cpu().numpy()
            z_b = out["mu_bg"].cpu().numpy()
            v_np = batch["valid_mask"][:, 1:].cpu().numpy()
            act_np = batch["actions"].cpu().numpy()
            actor_labels = batch["actor_labels"].cpu().numpy()
            track_ids = batch["track_ids"].cpu().numpy()
            B, T1, K, D = z_a.shape
            for b in range(B):
                for t in range(T1):
                    for k in range(K):
                        if v_np[b, t, k] and act_np[b, t, k] >= 0:
                            all_z_actor.append(z_a[b, t, k])
                            all_z_bg.append(z_b[b, t])
                            all_actions.append(int(act_np[b, t, k]))
                            all_actor_types.append(int(actor_labels[b, k]))
                            all_actor_ids.append(int(track_ids[b, k]))
                            all_dbbox_pred.append(out["dbbox_pred"][b, t, k].cpu().numpy())
                            all_dbbox_obs.append(out["dbbox_obs"][b, t, k].cpu().numpy())

    z_actor_arr = np.array(all_z_actor)
    z_bg_arr = np.array(all_z_bg)
    actions_arr = np.array(all_actions)
    actor_types_arr = np.array(all_actor_types)
    actor_ids_arr = np.array(all_actor_ids)
    dbbox_pred_arr = np.array(all_dbbox_pred)
    dbbox_obs_arr = np.array(all_dbbox_obs)

    z_var = float(z_actor_arr.var(axis=0).mean())
    z_bg_var = float(z_bg_arr.var(axis=0).mean())
    dbbox_mse = float(((dbbox_pred_arr - dbbox_obs_arr) ** 2).mean())
    print(f"\n  z_actor var: {z_var:.4f}, z_bg var: {z_bg_var:.4f}")
    print(f"  dbbox MSE: {dbbox_mse:.2f} px²")
    print(f"  Samples: {len(z_actor_arr)}, actions: {np.unique(actions_arr, return_counts=True)}")

    np.savez(
        os.path.join(RESULTS_DIR, f"latents_{args.name}.npz"),
        z_actor=z_actor_arr, z_bg=z_bg_arr,
        actions=actions_arr, actor_types=actor_types_arr, actor_ids=actor_ids_arr,
        dbbox_pred=dbbox_pred_arr, dbbox_obs=dbbox_obs_arr,
    )

    results = {
        "architecture": "v8_mot_lam", "stage": "stage2b_a2d",
        "training_steps": args.steps, "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2), "total_params": total_params,
        "z_actor_variance": round(z_var, 4), "z_bg_variance": round(z_bg_var, 4),
        "dbbox_mse": round(dbbox_mse, 2), "n_eval_samples": len(z_actor_arr),
    }
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
