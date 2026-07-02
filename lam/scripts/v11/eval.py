"""
V11 Frame-Diff LAM 评估脚本.

评估指标:
  1. 帧差 PSNR: PSNR(recon, ΔI_gt) — 直接衡量帧差重建质量
  2. RGB PSNR (推理恢复): I_pred = I_t + recon → PSNR(I_pred, I_{t+1}) — 与 V10 可比
  3. Copy baseline: PSNR(0, ΔI_gt) 和 PSNR(I_t, I_{t+1}) — 全零预测和 copy 参考
  4. Actor-masked PSNR: 只在 actor mask 区域计算 (帧差版 + RGB 版)
  5. NMI / Leakage / Action Probe: 与 V10 相同

 用法:
   PYTHONPATH=lam python lam/scripts/eval_v11.py \
       --dataset synthetic --config stage1

   # or specify checkpoint explicitly:
   PYTHONPATH=lam python lam/scripts/eval_v11.py \
       --checkpoint result/v11/synthetic/stage1/model.pt
"""
import os, sys, json, argparse
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.modules.v11_model import LatentActionModelV11
from lam.disk_synthetic_dataset import DiskSyntheticDataset

VERSION = "v11"


def compute_masked_psnr(pred, gt, mask):
    mask = np.expand_dims(mask, axis=-1)
    N = mask.sum() * pred.shape[-1]
    mse = ((pred - gt).cpu().numpy() ** 2 * mask).sum() / max(N, 1)
    return float(-10 * np.log10(mse + 1e-10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic",
                        help="dataset name (used if --checkpoint not given)")
    parser.add_argument("--config", type=str, default="stage1",
                        help="config name (used if --checkpoint not given)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="explicit checkpoint path (overrides --dataset/--config)")
    parser.add_argument("--output", type=str, default=None,
                        help="output JSON path (default: same dir as checkpoint)")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--max_actors", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ROOT = os.path.join(os.path.dirname(__file__), "../../..")
    if args.checkpoint is None:
        run_dir = os.path.join(ROOT, "result", VERSION, args.dataset, args.config)
        args.checkpoint = os.path.join(run_dir, "model.pt")
        args.output = args.output or os.path.join(run_dir, "eval.json")
    elif args.output is None:
        args.output = os.path.join(os.path.dirname(args.checkpoint), "eval.json")

    if args.data_root is None:
        data_root = os.path.join(ROOT, "data", "synthetic_multi_actor")
    else:
        data_root = args.data_root

    print(f"\n{'='*60}")
    print(f"V11 评估: {args.checkpoint}")
    print(f"  数据: {data_root}")
    print(f"{'='*60}")

    eval_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"),
        max_actors=args.max_actors, num_frames=args.num_frames,
        output_format="t h w c",
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False
    )

    model = LatentActionModelV11(
        in_dim=3, model_dim=args.model_dim, latent_dim=args.latent_dim,
        max_actors=args.max_actors, keep_background=True,
        use_obj_st_attention=True,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")

    all_z, all_slots, all_actions = [], [], []
    diff_mse_list, rgb_mse_list, rgb_copy_mse_list = [], [], []
    diff_zero_mse_list = []
    masked_diff_mse_list, masked_rgb_mse_list = [], []
    n_samples = 0
    b_count = 0

    with torch.no_grad():
        for batch in eval_loader:
            videos = batch["videos"].to(device)    # (B, T, H, W, C)
            masks = batch["masks"].to(device)      # (B, T, A, H, W)
            actions = batch["actions"]
            out = model({"videos": videos, "masks": masks})

            z_mu = out["z_mu"].cpu().numpy()
            recon = out["recon"]                   # (B, T-1, H, W, C) ΔI pred
            diff = out["diff"]                     # (B, T-1, H, W, C) ΔI GT

            # === 帧差 PSNR ===
            diff_mse = ((recon - diff) ** 2).mean(dim=[2, 3, 4]).cpu().numpy()
            diff_zero_mse = (diff ** 2).mean(dim=[2, 3, 4]).cpu().numpy()

            # === RGB PSNR: I_pred = I_t + recon ===
            I_t = videos[:, :-1]                   # (B, T-1, H, W, C)
            I_tp1 = videos[:, 1:]                  # (B, T-1, H, W, C)
            I_pred = I_t + recon
            rgb_mse = ((I_pred - I_tp1) ** 2).mean(dim=[2, 3, 4]).cpu().numpy()
            rgb_copy_mse = ((I_t - I_tp1) ** 2).mean(dim=[2, 3, 4]).cpu().numpy()

            # === Actor-masked PSNR ===
            masks_tp1 = masks[:, 1:]               # (B, T-1, A, H, W)
            actor_mask = (masks_tp1.sum(dim=2, keepdim=True) > 0.5).float()  # (B, T-1, 1, H, W)

            for b_idx in range(videos.shape[0]):
                for t_idx in range(videos.shape[1] - 1):
                    msk = actor_mask[b_idx, t_idx, 0].cpu().numpy()
                    if msk.sum() > 0:
                        mdt = compute_masked_psnr(
                            recon[b_idx, t_idx], diff[b_idx, t_idx],
                            msk
                        )
                        mrt = compute_masked_psnr(
                            I_pred[b_idx, t_idx], I_tp1[b_idx, t_idx],
                            msk
                        )
                        masked_diff_mse_list.append(mdt)
                        masked_rgb_mse_list.append(mrt)

            diff_mse_list.extend(diff_mse.ravel().tolist())
            diff_zero_mse_list.extend(diff_zero_mse.ravel().tolist())
            rgb_mse_list.extend(rgb_mse.ravel().tolist())
            rgb_copy_mse_list.extend(rgb_copy_mse.ravel().tolist())

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
                            n_samples += 1

            b_count += 1
            if args.max_batches > 0 and b_count >= args.max_batches:
                break

    z_arr = np.array(all_z)
    slots_arr = np.array(all_slots)
    actions_arr = np.array(all_actions)

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

    diff_psnr = -10 * np.log10(float(np.mean(diff_mse_list)) + 1e-10)
    diff_zero_psnr = -10 * np.log10(float(np.mean(diff_zero_mse_list)) + 1e-10)
    rgb_psnr = -10 * np.log10(float(np.mean(rgb_mse_list)) + 1e-10)
    rgb_copy_psnr = -10 * np.log10(float(np.mean(rgb_copy_mse_list)) + 1e-10)
    masked_diff_psnr = np.mean(masked_diff_mse_list) if masked_diff_mse_list else 0.0
    masked_rgb_psnr = np.mean(masked_rgb_mse_list) if masked_rgb_mse_list else 0.0

    print(f"\n{'='*60}")
    print(f"V11 评估结果 ({args.checkpoint})")
    print(f"  Eval samples: {n_samples}")
    print(f"\n  --- Clustering ---")
    print(f"  Overall NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")
    print(f"  Per-Slot NMI avg: {np.mean(nmi_per):.4f}  ({nmi_per})")
    print(f"  Actor Leakage: {leak:.4f}  (chance: {1/len(np.unique(slots_arr)):.4f})")
    print(f"  Action Probe: {act_acc:.4f}  (chance: {1/n_clusters:.4f})")
    print(f"\n  --- Reconstruction ---")
    print(f"  Frame-diff PSNR: {diff_psnr:.2f} dB  (zero={diff_zero_psnr:.2f}, Δ={diff_psnr-diff_zero_psnr:.2f})")
    print(f"  RGB PSNR (I_t+recon): {rgb_psnr:.2f} dB  (copy={rgb_copy_psnr:.2f}, Δ={rgb_psnr-rgb_copy_psnr:.2f})")
    print(f"  Actor-masked diff PSNR: {masked_diff_psnr:.2f} dB")
    print(f"  Actor-masked RGB PSNR: {masked_rgb_psnr:.2f} dB")
    print(f"{'='*60}\n")

    results = {
        "architecture": "v11_frame_diff",
        "checkpoint": args.checkpoint,
        "total_params": total_params,
        "n_eval_samples": n_samples,
        "overall_nmi": round(float(nmi), 4),
        "overall_ari": round(float(ari), 4),
        "per_slot_nmi_avg": round(float(np.mean(nmi_per)), 4),
        "per_slot_nmi": [round(float(x), 4) for x in nmi_per],
        "actor_leakage_acc": round(float(leak), 4),
        "action_probe_acc": round(float(act_acc), 4),
        "z_actor_variance": round(float(z_arr.var(axis=0).mean()), 4),
        "frame_diff_psnr": round(float(diff_psnr), 2),
        "frame_diff_zero_psnr": round(float(diff_zero_psnr), 2),
        "frame_diff_delta": round(float(diff_psnr - diff_zero_psnr), 2),
        "rgb_psnr": round(float(rgb_psnr), 2),
        "rgb_copy_psnr": round(float(rgb_copy_psnr), 2),
        "rgb_delta": round(float(rgb_psnr - rgb_copy_psnr), 2),
        "actor_masked_diff_psnr": round(float(masked_diff_psnr), 2),
        "actor_masked_rgb_psnr": round(float(masked_rgb_psnr), 2),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  结果保存: {args.output}")


if __name__ == "__main__":
    main()
