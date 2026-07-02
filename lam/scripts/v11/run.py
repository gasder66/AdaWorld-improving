"""
V11: Frame-Diff LAM 训练脚本 (iVideoGPT-style).

用法:
  PYTHONPATH=lam python lam/scripts/run_v11.py \
      --config stage1 --dataset synthetic --batch_size 16 --steps 5000

输出路径: result/v11/{dataset}/{config}/
"""
import os, sys, json, time, argparse
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.modules.v11_model import LatentActionModelV11
from lam.disk_synthetic_dataset import DiskSyntheticDataset

VERSION = "v11"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True,
                        help="config name (e.g. stage1, stage1_k01)")
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--max_actors", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--enc_blocks", type=int, default=4)
    parser.add_argument("--dec_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--free_bits_lambda", type=float, default=0.1)
    parser.add_argument("--num_actor_types", type=int, default=0)
    # Loss weights
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--kl_beta", type=float, default=0.0002)
    parser.add_argument("--delta_weight", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.3)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    ROOT = os.path.join(os.path.dirname(__file__), "../../..")
    RESULTS_DIR = os.path.join(ROOT, "result", VERSION, args.dataset, args.config)
    CKT_DIR = os.path.join(RESULTS_DIR, "ckpts")
    LOSS_DIR = os.path.join(RESULTS_DIR, "losses")
    os.makedirs(CKT_DIR, exist_ok=True)
    os.makedirs(LOSS_DIR, exist_ok=True)

    if args.data_root is None:
        data_root = os.path.join(ROOT, "data", "synthetic_multi_actor")
    else:
        data_root = args.data_root

    print(f"\n{'='*60}")
    print(f"V11: Frame-Diff LAM (iVideoGPT-style)")
    print(f"  GPU={args.gpu}, version={VERSION}, dataset={args.dataset}, config={args.config}")
    print(f"  result => {RESULTS_DIR}")
    print(f"  batch={args.batch_size}, steps={args.steps}, lr={args.lr}")
    print(f"  model_dim={args.model_dim}, latent_dim={args.latent_dim}")
    print(f"  free_bits_lambda={args.free_bits_lambda}")
    print(f"  recon_w={args.recon_weight}, kl_beta={args.kl_beta}, "
          f"delta_w={args.delta_weight}")
    print( "  Encoder input: [I_0, dI_1, ..., dI_T-1]" )
    print( "  Reconstruction target: dI_t+1 (frame-diff, NO sigmoid)" )
    print(f"  数据: {data_root}")
    print(f"{'='*60}")

    train_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "train"),
        max_actors=args.max_actors, num_frames=args.num_frames,
        output_format="t h w c",
    )
    eval_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"),
        max_actors=args.max_actors, num_frames=args.num_frames,
        output_format="t h w c",
    )

    model = LatentActionModelV11(
        in_dim=3, model_dim=args.model_dim, latent_dim=args.latent_dim,
        patch_size=args.patch_size, enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks, num_heads=args.num_heads,
        max_actors=args.max_actors, keep_background=True,
        use_obj_st_attention=True, free_bits_lambda=args.free_bits_lambda,
        num_actor_types=args.num_actor_types,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=(args.num_workers > 0),
    )

    losses = {"total": [], "recon": [], "kl": [], "delta": []}
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
            recon = outputs["recon"]          # (B, T-1, H, W, C) predicted ΔI
            diff = outputs["diff"]            # (B, T-1, H, W, C) GT ΔI
            recon_loss = ((diff - recon) ** 2).mean()
            kl_loss = outputs["kl_loss"]
            delta_loss = outputs["delta_loss"]

            loss = (args.recon_weight * recon_loss
                    + args.kl_beta * kl_loss
                    + args.delta_weight * delta_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses["total"].append(float(loss))
            losses["recon"].append(float(recon_loss))
            losses["kl"].append(float(kl_loss))
            losses["delta"].append(float(delta_loss))

            if step % 50 == 0:
                elapsed = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                mse = float(recon_loss)
                psnr = -10 * np.log10(mse + 1e-10) if mse > 1e-10 else 100.0
                z_var = float(outputs["z_mu"].reshape(-1, args.latent_dim).var(0).mean())
                print(
                    f"  Step {step:4d}/{args.steps}: "
                    f"loss={float(loss):.4f}, recon_mse={mse:.6f} (diff_PSNR={psnr:.1f}), "
                    f"kl={float(kl_loss):.4f}, delta={float(delta_loss):.4f}, "
                    f"z_var={z_var:.4f}, mem={mem:.1f}GB, {elapsed:.0f}s"
                )

            if (step + 1) % args.checkpoint_every == 0:
                ckpt = os.path.join(CKT_DIR, f"step{step+1}.pt")
                torch.save(model.state_dict(), ckpt)

            step += 1

    training_time = time.time() - t0
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"\n  训练完成. 峰值内存: {mem_peak:.1f}GB, 耗时: {training_time:.0f}s")

    for key, vals in losses.items():
        np.savetxt(os.path.join(LOSS_DIR, f"{key}.txt"), np.array(vals))

    # === Eval: collect latents ===
    model.eval()
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=32, num_workers=0, shuffle=False
    )

    all_z, all_slots, all_actions = [], [], []
    all_recon_mse = []
    n_collected = 0
    with torch.no_grad():
        for batch in eval_loader:
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"]
            out = model({"videos": videos, "masks": masks})
            z_mu = out["z_mu"].cpu().numpy()  # (B, T-1, K+1, D)
            recon = out["recon"]               # (B, T-1, H, W, C) ΔI pred
            diff = out["diff"]                 # (B, T-1, H, W, C) ΔI GT
            recon_mse = ((recon - diff) ** 2).mean(dim=[2, 3, 4]).cpu().numpy()

            act_np = actions.cpu().numpy()
            B, T1, Kfull, D = z_mu.shape
            v_np = act_np >= 0

            for b in range(B):
                for t in range(T1):
                    for k in range(1, Kfull):
                        if v_np[b, t, k - 1] and act_np[b, t, k - 1] >= 0:
                            all_z.append(z_mu[b, t, k])
                            all_slots.append(k - 1)
                            all_actions.append(int(act_np[b, t, k - 1]))
                            all_recon_mse.append(recon_mse[b, t])
            n_collected += 1
            if n_collected >= 15:
                break

    z_arr = np.array(all_z)
    slots_arr = np.array(all_slots)
    actions_arr = np.array(all_actions)
    recon_mse_arr = np.array(all_recon_mse)

    diff_psnr = -10 * np.log10(float(recon_mse_arr.mean()) + 1e-10)

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression

    n_clusters = len(np.unique(actions_arr))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(z_arr)
    nmi = normalized_mutual_info_score(actions_arr, pred)
    ari = adjusted_rand_score(actions_arr, pred)

    nmi_per = []
    for k_slot in np.unique(slots_arr):
        idx = slots_arr == k_slot
        if idx.sum() < 50:
            continue
        km_k = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_k = km_k.fit_predict(z_arr[idx])
        nmi_k = normalized_mutual_info_score(actions_arr[idx], pred_k)
        nmi_per.append(nmi_k)

    n = len(z_arr)
    idx_perm = np.random.RandomState(42).permutation(n)
    n_tr = int(0.8 * n)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(z_arr[idx_perm[:n_tr]], slots_arr[idx_perm[:n_tr]])
    leak = clf.score(z_arr[idx_perm[n_tr:]], slots_arr[idx_perm[n_tr:]])

    clf2 = LogisticRegression(max_iter=1000, C=1.0)
    clf2.fit(z_arr[idx_perm[:n_tr]], actions_arr[idx_perm[:n_tr]])
    act_acc = clf2.score(z_arr[idx_perm[n_tr:]], actions_arr[idx_perm[n_tr:]])

    print(f"\n  Eval samples: {len(z_arr)}")
    print(f"  Overall NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")
    print(f"  Per-Slot NMI avg: {np.mean(nmi_per):.4f}")
    print(f"  Actor Leakage: {leak:.4f}  (chance: {1/len(np.unique(slots_arr)):.4f})")
    print(f"  Action Probe: {act_acc:.4f}  (chance: {1/n_clusters:.4f})")
    print(f"  Frame-diff PSNR: {diff_psnr:.2f} dB")

    results = {
        "architecture": "v11_frame_diff", "name": args.name,
        "total_params": total_params,
        "training_steps": args.steps, "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2),
        "n_eval_samples": len(z_arr),
        "overall_nmi": round(float(nmi), 4),
        "overall_ari": round(float(ari), 4),
        "per_slot_nmi_avg": round(float(np.mean(nmi_per)), 4),
        "per_slot_nmi": [round(float(x), 4) for x in nmi_per],
        "actor_leakage_acc": round(float(leak), 4),
        "action_probe_acc": round(float(act_acc), 4),
        "frame_diff_psnr": round(float(diff_psnr), 2),
        "z_actor_variance": round(float(z_arr.var(axis=0).mean()), 4),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savez(
        os.path.join(RESULTS_DIR, "latents.npz"),
        z_actor=z_arr, slots=slots_arr, actions=actions_arr,
    )
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))
    print(f"\n  Saved => {RESULTS_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
