"""
V10 A2D: Train V10 (V6c + Shared VAE) on A2D real video.

与合成 run_v10.py 的区别:
  - A2DBoxDataset (GT bbox) 或 YOLOBoxDataset (YOLO bbox)
  - 仅 recon_loss + kl_loss (obj_recon/delta/contrast 设为0, 避免爆炸)
  - bbox→mask 转换在数据集内完成 (masks 字段已添加)

用法:
  # GT bbox (A2DBoxDataset)
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python lam/scripts/run_v10_a2d.py \\
      --name v10_a2d_gt --dataset a2d --batch_size 8 --steps 3000

  # YOLO bbox (YOLOBoxDataset)
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python lam/scripts/run_v10_a2d.py \\
      --name v10_a2d_yolo --dataset yolo --batch_size 8 --steps 5000 --max_samples 2000
"""
import os, sys, json, time, argparse
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules.v10_model import LatentActionModelV10
from lam.a2d_box_dataset import A2DBoxDataset
from lam.yolo_box_dataset import YOLOBoxDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="a2d", choices=["a2d", "yolo"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_root", type=str, default="data/a2d")
    parser.add_argument("--release_root", type=str, default="Release")
    parser.add_argument("--max_actors", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--enc_blocks", type=int, default=4)
    parser.add_argument("--dec_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--free_bits_lambda", type=float, default=0.1)
    parser.add_argument("--num_actor_types", type=int, default=0)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--kl_beta", type=float, default=0.0002)
    parser.add_argument("--grad_clip", type=float, default=0.3)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="result/v8_mot_lam/yolo_cache")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "v10"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.dataset == "yolo":
        train_dataset = YOLOBoxDataset(
            video_dir=f"{args.data_root}/train", release_root=args.release_root,
            split="train", T=args.num_frames, stride=args.stride,
            max_actors=args.max_actors, img_size=256,
            cache_dir=args.cache_dir, max_samples=args.max_samples,
        )
        eval_dataset = YOLOBoxDataset(
            video_dir=f"{args.data_root}/test", release_root=args.release_root,
            split="test", T=args.num_frames, stride=args.stride,
            max_actors=args.max_actors, img_size=256,
            cache_dir=args.cache_dir,
            max_samples=min(200, args.max_samples or 200),
        )
    else:
        train_dataset = A2DBoxDataset(
            data_root=args.data_root, release_root=args.release_root,
            split="train", num_frames=args.num_frames, frame_stride=1,
            max_actors=args.max_actors, img_size=256,
        )
        eval_dataset = A2DBoxDataset(
            data_root=args.data_root, release_root=args.release_root,
            split="test", num_frames=args.num_frames, frame_stride=1,
            max_actors=args.max_actors, img_size=256,
        )

    print(f"\n{'='*60}")
    print(f"V10 A2D: V6c + Shared VAE on real video")
    print(f"  GPU={args.gpu}, name={args.name}, dataset={args.dataset}")
    print(f"  batch={args.batch_size}, steps={args.steps}, lr={args.lr}")
    print(f"  latent_dim={args.latent_dim}, free_bits_lambda={args.free_bits_lambda}")
    print(f"  recon_w={args.recon_weight}, kl_beta={args.kl_beta}")
    print(f"  Loss = recon + kl (obj_recon/delta/contrast disabled)")
    print(f"{'='*60}")

    model = LatentActionModelV10(
        in_dim=3, model_dim=args.model_dim, latent_dim=args.latent_dim,
        patch_size=args.patch_size, enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks, num_heads=args.num_heads,
        max_actors=args.max_actors, keep_background=True,
        use_obj_st_attention=True, free_bits_lambda=args.free_bits_lambda,
        num_actor_types=args.num_actor_types,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")
    if args.debug:
        return

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=(args.num_workers > 0),
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
            masks = batch["masks"].to(device, non_blocking=True)
            batch_input = {"videos": videos, "masks": masks}

            outputs = model(batch_input)
            recon = outputs["recon"]
            gt = videos[:, 1:]
            recon_loss = ((gt - recon) ** 2).mean()
            kl_loss = outputs["kl_loss"]

            loss = args.recon_weight * recon_loss + args.kl_beta * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses["total"].append(float(loss))
            losses["recon"].append(float(recon_loss))
            losses["kl"].append(float(kl_loss))

            if step % 50 == 0:
                elapsed = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                mse = float(recon_loss)
                psnr = -10 * np.log10(mse + 1e-10) if mse > 1e-10 else 100.0
                z_var = float(outputs["z_mu"].reshape(-1, args.latent_dim).var(0).mean())
                print(
                    f"  Step {step:4d}/{args.steps}: "
                    f"loss={float(loss):.4f}, recon={mse:.6f} (PSNR={psnr:.1f}), "
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
        np.savetxt(os.path.join(RESULTS_DIR, f"loss_{args.name}_{key}.txt"), np.array(vals))

    # === Eval ===
    model.eval()
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=8, num_workers=0, shuffle=False,
    )

    all_z, all_slots, all_acts, all_actor_types = [], [], [], []
    all_recon_mse = []
    n_collected = 0

    with torch.no_grad():
        for batch in eval_loader:
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"]
            out = model({"videos": videos, "masks": masks})
            z_mu = out["z_mu"].cpu().numpy()
            recon = out["recon"]
            gt = videos[:, 1:]
            recon_mse = ((recon - gt) ** 2).mean(dim=[2, 3, 4]).cpu().numpy()

            act_np = actions.cpu().numpy()
            v_mask = batch["valid_mask"]
            if v_mask is not None:
                v_np = v_mask[:, 1:].cpu().numpy()
            else:
                v_np = (act_np >= 0)
            actor_labels = batch["actor_labels"].cpu().numpy()

            B, T1, Kfull, D = z_mu.shape
            for b in range(B):
                for t in range(T1):
                    for k in range(1, Kfull):
                        k_idx = k - 1
                        if k_idx < v_np.shape[2] and v_np[b, t, k_idx] and act_np[b, t, k_idx] >= 0:
                            all_z.append(z_mu[b, t, k])
                            all_slots.append(k_idx)
                            all_acts.append(int(act_np[b, t, k_idx]))
                            all_actor_types.append(int(actor_labels[b, k_idx]))
                            all_recon_mse.append(recon_mse[b, t])
            n_collected += 1
            if n_collected >= 15:
                break

    z_arr = np.array(all_z) if all_z else np.zeros((0, args.latent_dim))
    psnr_full = -10 * np.log10(float(np.mean(all_recon_mse)) + 1e-10) if all_recon_mse else 0

    if len(z_arr) >= 50:
        from sklearn.metrics import normalized_mutual_info_score
        from sklearn.cluster import KMeans
        from sklearn.linear_model import LogisticRegression

        acts_arr = np.array(all_acts)
        slots_arr = np.array(all_slots)
        types_arr = np.array(all_actor_types)

        n_clusters = min(len(np.unique(acts_arr)), 8)
        if n_clusters >= 2:
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            pred = km.fit_predict(z_arr)
            nmi = normalized_mutual_info_score(acts_arr, pred)

            n = len(z_arr)
            idx_p = np.random.RandomState(42).permutation(n)
            n_tr = int(0.8 * n)
            clf_a = LogisticRegression(max_iter=1000, C=1.0)
            clf_a.fit(z_arr[idx_p[:n_tr]], acts_arr[idx_p[:n_tr]])
            act_probe = clf_a.score(z_arr[idx_p[n_tr:]], acts_arr[idx_p[n_tr:]])

            clf_s = LogisticRegression(max_iter=1000, C=1.0)
            clf_s.fit(z_arr[idx_p[:n_tr]], slots_arr[idx_p[:n_tr]])
            leak = clf_s.score(z_arr[idx_p[n_tr:]], slots_arr[idx_p[n_tr:]])
        else:
            nmi = act_probe = leak = 0.0

        print(f"\n  Eval: {len(z_arr)} labeled samples")
        print(f"  NMI: {nmi:.4f}, Action Probe: {act_probe:.4f}, Leakage: {leak:.4f}")
        print(f"  PSNR: {psnr_full:.2f} dB")
    else:
        nmi = act_probe = leak = 0.0
        print(f"\n  Too few labeled eval samples ({len(z_arr)}), skipping clustering")

    results = {
        "architecture": "v10_shared_vae", "name": args.name,
        "dataset": args.dataset, "total_params": total_params,
        "training_steps": args.steps, "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2),
        "n_eval_samples": len(z_arr),
        "nmi": round(float(nmi), 4),
        "action_probe": round(float(act_probe), 4),
        "leakage": round(float(leak), 4),
        "psnr": round(float(psnr_full), 2),
    }
    with open(os.path.join(RESULTS_DIR, f"results_{args.name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    if len(z_arr) > 0:
        np.savez(os.path.join(RESULTS_DIR, f"latents_{args.name}.npz"),
                 z_actor=z_arr, actions=np.array(all_acts),
                 slots=np.array(all_slots))
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"model_{args.name}.pt"))
    print(f"  模型: {RESULTS_DIR}/model_{args.name}.pt\n{'='*60}\n")


if __name__ == "__main__":
    main()
