"""
V9: V8 + 重建 Decoder

变体:
  A: V8 motion-only encoder + recon decoder (recon loss 流入 z_actor)
  B: RGB encoder (all-t RGB crops) + recon decoder
  C: 双路径 (z_actor motion + z_appearance appearance) + recon decoder

Stage 1 (合成数据, 无相机扰动):
  L = L_motion + β·L_KL + δ·L_recon

用法:
  # V9-A (sanity check)
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python lam/scripts/run_v9.py \\
      --name v9a_stage1 --variant A --batch_size 16 --steps 5000 --recon_weight 0.1

  # V9-B (RGB crops)
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python lam/scripts/run_v9.py \\
      --name v9b_stage1 --variant B --batch_size 16 --steps 5000 --recon_weight 0.1

  # V9-C (dual-pathway, z_actor detached from recon)
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python lam/scripts/run_v9.py \\
      --name v9c_stage1 --variant C --batch_size 16 --steps 5000 --recon_weight 0.1
"""
import os, sys, json, time, argparse
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules.v9_model import LatentActionModelV9
from lam.mot_slot_dataset import MOTSlotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--variant", type=str, default="A", choices=["A", "B", "C", "D"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--max_actors", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=5)
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
    parser.add_argument("--recon_weight", type=float, default=0.1, help="L_recon 权重 (δ)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # Stage 2A: camera perturbation
    parser.add_argument("--camera_perturbation", action="store_true")
    parser.add_argument("--pan_range", type=float, default=8.0)
    parser.add_argument("--zoom_range", type=float, default=0.1)
    parser.add_argument("--brightness_range", type=float, default=0.05)
    parser.add_argument("--no_bg_slot", action="store_true")
    parser.add_argument("--bg_loss_weight", type=float, default=1.0)
    # V9-C: detach z_actor from recon gradient
    parser.add_argument("--detach_z_actor", action="store_true",
                        help="V9-C: detach z_actor from recon loss (default: keep gradient)")
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

    if args.data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    else:
        data_root = args.data_root

    use_bg = not args.no_bg_slot
    cam_str = (
        f"camera_perturbation ON (pan={args.pan_range}, zoom={args.zoom_range}, bright={args.brightness_range})"
        if args.camera_perturbation else "no camera perturbation"
    )
    print(f"\n{'='*60}")
    print(f"V9: V8 + Reconstruction Decoder (variant {args.variant})")
    print(f"  GPU={args.gpu}, name={args.name}")
    print(f"  batch={args.batch_size}, steps={args.steps}, lr={args.lr}")
    print(f"  model_dim={args.model_dim}, z_dim={args.z_dim}, z_bg_dim={args.z_bg_dim}")
    if args.variant == "C":
        print(f"  z_app_dim={args.z_app_dim}, detach_z_actor={args.detach_z_actor}")
    print(f"  temporal_layers={args.num_temporal_layers}, slot_layers={args.num_slot_layers}")
    print(f"  kl_beta={args.kl_beta}, recon_weight={args.recon_weight}")
    print(f"  use_bg_slot={use_bg}, bg_loss_weight={args.bg_loss_weight}")
    print(f"  {cam_str}")
    loss_str = f"L = L_motion + {args.kl_beta}*L_KL"
    if use_bg and args.camera_perturbation:
        loss_str += f" + {args.bg_loss_weight}*L_bg"
    loss_str += f" + {args.recon_weight}*L_recon"
    print(f"  Loss: {loss_str}")
    print(f"  数据: {data_root}")
    print(f"{'='*60}")

    cam_kwargs = dict(
        camera_perturbation=args.camera_perturbation,
        pan_range=args.pan_range,
        zoom_range=args.zoom_range,
        brightness_range=args.brightness_range,
    ) if args.camera_perturbation else {}

    train_dataset = MOTSlotDataset(
        os.path.join(data_root, "train"),
        max_actors=args.max_actors, num_frames=args.num_frames,
        **cam_kwargs,
    )
    eval_dataset = MOTSlotDataset(
        os.path.join(data_root, "val"),
        max_actors=args.max_actors, num_frames=args.num_frames,
        **cam_kwargs,
    )

    cam_scale = (args.pan_range, args.pan_range, args.zoom_range, args.brightness_range)
    model = LatentActionModelV9(
        model_dim=args.model_dim,
        z_dim=args.z_dim,
        z_bg_dim=args.z_bg_dim,
        num_temporal_layers=args.num_temporal_layers,
        num_slot_layers=args.num_slot_layers,
        num_heads=args.num_heads,
        max_actors=args.max_actors,
        crop_size=args.crop_size,
        free_bits_lambda=args.free_bits_lambda,
        use_bg_slot=use_bg,
        bg_loss_weight=args.bg_loss_weight,
        camera_param_scale=cam_scale,
        variant=args.variant,
        z_app_dim=args.z_app_dim,
        detach_z_actor_recon=args.detach_z_actor,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    v8_params = sum(p.numel() for p in model.v8.parameters())
    dec_params = total_params - v8_params
    print(f"  总参数: {total_params:,} (V8: {v8_params:,}, decoder+: {dec_params:,})")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.num_workers > 0),
        drop_last=True,
    )

    losses = {"total": [], "motion": [], "kl": [], "bg": [], "recon": [], "recon_l1": [], "recon_ssim": []}
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
            bg_loss = outputs["bg_loss"]
            recon_loss = outputs["recon_loss"]

            loss = (motion_loss + args.kl_beta * kl_loss
                    + args.bg_loss_weight * bg_loss
                    + args.recon_weight * recon_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses["total"].append(float(loss))
            losses["motion"].append(float(motion_loss))
            losses["kl"].append(float(kl_loss))
            losses["bg"].append(float(bg_loss))
            losses["recon"].append(float(recon_loss))
            losses["recon_l1"].append(float(outputs["recon_l1"]))
            losses["recon_ssim"].append(float(outputs["recon_ssim"]))

            if step % 50 == 0:
                elapsed = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                z_var = float(outputs["mu_actor"].reshape(-1, args.z_dim).var(0).mean())
                print(
                    f"  Step {step:4d}/{args.steps}: "
                    f"loss={float(loss):.4f}, motion={float(motion_loss):.4f}, "
                    f"kl={float(kl_loss):.4f}, recon={float(recon_loss):.4f}, "
                    f"ssim={float(outputs['recon_ssim']):.3f}, "
                    f"z_var={z_var:.4f}, mem={mem:.1f}GB, {elapsed:.0f}s"
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

    # === 评估: 收集 z_actor + recon 数据 ===
    model.eval()
    results = {
        "architecture": "v9",
        "variant": args.variant,
        "stage": "stage2a_camera" if args.camera_perturbation else "stage1_synthetic",
        "use_bg_slot": use_bg,
        "camera_perturbation": args.camera_perturbation,
        "max_actors": args.max_actors,
        "model_dim": args.model_dim,
        "z_dim": args.z_dim,
        "z_bg_dim": args.z_bg_dim,
        "z_app_dim": args.z_app_dim if args.variant == "C" else 0,
        "detach_z_actor": args.detach_z_actor,
        "recon_weight": args.recon_weight,
        "kl_beta": args.kl_beta,
        "training_steps": args.steps,
        "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2),
        "total_params": total_params,
        "v8_params": v8_params,
        "decoder_params": dec_params,
    }

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=32, num_workers=args.num_workers, shuffle=False
    )

    all_z_actor = []
    all_z_bg = []
    all_z_app = []
    all_actions = []
    all_actor_ids = []
    all_recon = []
    all_crop_t = []
    all_crop_tp1 = []
    all_dbbox_pred = []
    all_dbbox_obs = []
    n_collected = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            actions = batch["actions"]
            track_ids = batch["track_ids"]
            out = model(batch_gpu)
            z_a = out["mu_actor"].cpu().numpy()
            z_b = out["mu_bg"].cpu().numpy() if "mu_bg" in out else np.zeros((z_a.shape[0], z_a.shape[1], args.z_bg_dim))
            z_app = out["mu_app"].cpu().numpy() if "mu_app" in out else None
            v_np = batch["valid_mask"][:, 1:].cpu().numpy()
            act_np = actions.cpu().numpy()
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
                            all_actor_ids.append(int(track_ids[b, k]))
                            all_recon.append(recon_np[b, t, k])
                            all_crop_t.append(crop_t_np[b, t, k])
                            all_crop_tp1.append(crop_tp1_np[b, t, k])
                            all_dbbox_pred.append(out["dbbox_pred"][b, t, k].cpu().numpy())
                            all_dbbox_obs.append(out["dbbox_obs"][b, t, k].cpu().numpy())
            n_collected += 1
            if n_collected >= 30:
                break

    z_actor_arr = np.array(all_z_actor)
    z_bg_arr = np.array(all_z_bg)
    actions_arr = np.array(all_actions)
    actor_ids_arr = np.array(all_actor_ids)
    recon_arr = np.array(all_recon)
    crop_t_arr = np.array(all_crop_t)
    crop_tp1_arr = np.array(all_crop_tp1)
    dbbox_pred_arr = np.array(all_dbbox_pred)
    dbbox_obs_arr = np.array(all_dbbox_obs)
    z_app_arr = np.array(all_z_app) if all_z_app else None

    z_var = float(z_actor_arr.var(axis=0).mean())
    active = int((z_actor_arr.var(axis=0) > 0.01).sum())
    results["z_actor_variance"] = round(z_var, 4)
    results["z_actor_active_dims"] = active
    results["z_bg_variance"] = round(float(z_bg_arr.var(axis=0).mean()), 4)

    # Recon metrics
    recon_f = torch.from_numpy(recon_arr)
    target_f = torch.from_numpy(crop_tp1_arr)
    crop_t_f = torch.from_numpy(crop_t_arr)
    from lam.modules.v9_decoder import compute_psnr, compute_ssim_simple
    psnr_recon = compute_psnr(recon_f, target_f)
    psnr_copy = compute_psnr(crop_t_f, target_f)
    ssim_recon = compute_ssim_simple(recon_f, target_f)
    ssim_copy = compute_ssim_simple(crop_t_f, target_f)
    results["psnr_recon"] = round(psnr_recon, 2)
    results["psnr_copy"] = round(psnr_copy, 2)
    results["ssim_recon"] = round(ssim_recon, 4)
    results["ssim_copy"] = round(ssim_copy, 4)
    results["delta_psnr_recon_copy"] = round(psnr_recon - psnr_copy, 2)

    print(f"\n  z_actor 方差: {z_var:.4f}, 活跃维度: {active}/{args.z_dim}")
    print(f"  Recon: PSNR={psnr_recon:.2f} dB, copy={psnr_copy:.2f} dB, Δ={psnr_recon - psnr_copy:+.2f}")
    print(f"  SSIM:  recon={ssim_recon:.4f}, copy={ssim_copy:.4f}")

    # 保存隐变量
    save_dict = dict(
        z_actor=z_actor_arr,
        z_bg=z_bg_arr,
        actions=actions_arr,
        actor_ids=actor_ids_arr,
        recon=recon_arr,
        crop_t=crop_t_arr,
        crop_tp1=crop_tp1_arr,
        dbbox_pred=dbbox_pred_arr,
        dbbox_obs=dbbox_obs_arr,
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
    print(f"  隐变量保存: {RESULTS_DIR}/latents_{args.name}.npz")
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    main()
