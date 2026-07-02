"""
V8 Probe Decoder 评估: 四重 PSNR + 可视化.

指标:
  1. PSNR(probe)       — probe 预测 vs GT
  2. PSNR(copy)        — crop_t 直接作为预测 (baseline)
  3. PSNR(z=0)         — z_actor 置零时的 probe 输出
  4. PSNR(z_shuffle)   — z_actor batch 内打乱 (保留分布, 破坏对应)
  5. ΔPSNR: probe - copy, probe - z=0, probe - z_shuffle

  同样对 crop_tp1_predbox (V8 预测 bbox 的 crop) 补充评估.

  可视化: [crop_t] [GT] [probe] [copy] [z=0] [z_shuffle] [diff]

用法:
  PYTHONPATH=lam python lam/scripts/eval_v8_reconstruction.py --name v8_stage1
  PYTHONPATH=lam python lam/scripts/eval_v8_reconstruction.py --name v8_stage1 --use_dbbox
"""
import os, sys, json, argparse

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules.probe_decoder import ActorProbeDecoder, compute_psnr, compute_ssim_simple


def video_grouped_split(video_ids, train_ratio=0.8, seed=42):
    """按 video_id 分组 split, 同一视频的所有样本只在 train 或 test."""
    unique_videos = np.unique(video_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_videos)
    n_train = int(len(unique_videos) * train_ratio)
    train_videos = set(unique_videos[:n_train])
    train_idx = np.array([i for i, v in enumerate(video_ids) if v in train_videos])
    test_idx = np.array([i for i, v in enumerate(video_ids) if v not in train_videos])
    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=32)
    parser.add_argument("--use_dbbox", action="store_true")
    parser.add_argument("--num_actor_types", type=int, default=0)
    parser.add_argument("--n_vis", type=int, default=20, help="可视化样本数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "v8_mot_lam"
    )
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    suffix = "_dbbox" if args.use_dbbox else ""
    probe_name = f"probe_{args.name}{suffix}"

    print(f"\n{'='*60}")
    print(f"V8 Reconstruction Evaluation: {probe_name}")
    print(f"{'='*60}")

    data = np.load(os.path.join(results_dir, f"latent_pairs_{args.name}.npz"))
    z_actor = torch.from_numpy(data["z_actor"]).float().to(device)
    crop_t = torch.from_numpy(data["crop_t"]).float().to(device)
    crop_gt = torch.from_numpy(data["crop_tp1_gtbox"]).float().to(device)
    crop_pred_box = torch.from_numpy(data["crop_tp1_predbox"]).float().to(device)
    dbbox_pred = torch.from_numpy(data["dbbox_pred"]).float().to(device)
    actor_types = torch.from_numpy(data["actor_types"]).long().to(device)
    video_ids = data["video_id"]

    _, test_idx = video_grouped_split(video_ids, seed=args.seed)
    print(f"  Test samples: {len(test_idx)}")

    z_te = z_actor[test_idx]
    ct_te = crop_t[test_idx]
    cg_te = crop_gt[test_idx]
    cpb_te = crop_pred_box[test_idx]
    db_te = dbbox_pred[test_idx]
    at_te = actor_types[test_idx]

    model = ActorProbeDecoder(
        z_dim=args.z_dim, crop_size=args.crop_size,
        use_dbbox=args.use_dbbox, num_actor_types=args.num_actor_types,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(results_dir, f"model_{probe_name}.pt"), map_location=device))
    model.eval()

    z_zero = torch.zeros_like(z_te)
    perm = torch.randperm(len(z_te), device=device)
    z_shuffle = z_te[perm]

    with torch.no_grad():
        pred_probe = model(ct_te, z_te, dbbox_pred=db_te, actor_type=at_te)
        pred_z0 = model(ct_te, z_zero, dbbox_pred=db_te, actor_type=at_te)
        pred_zsh = model(ct_te, z_shuffle, dbbox_pred=db_te, actor_type=at_te)

    psnr_probe = compute_psnr(pred_probe, cg_te)
    psnr_copy = compute_psnr(ct_te, cg_te)
    psnr_z0 = compute_psnr(pred_z0, cg_te)
    psnr_zsh = compute_psnr(pred_zsh, cg_te)

    ssim_probe = compute_ssim_simple(pred_probe, cg_te)
    ssim_copy = compute_ssim_simple(ct_te, cg_te)

    psnr_probe_predbox = compute_psnr(pred_probe, cpb_te)
    psnr_copy_predbox = compute_psnr(ct_te, cpb_te)

    d_psnr_copy = psnr_probe - psnr_copy
    d_psnr_z0 = psnr_probe - psnr_z0
    d_psnr_zsh = psnr_probe - psnr_zsh

    print(f"\n[Actor Crop Reconstruction — GT bbox target]")
    print(f"  PSNR(probe):     {psnr_probe:.2f} dB")
    print(f"  PSNR(copy):      {psnr_copy:.2f} dB")
    print(f"  PSNR(z=0):       {psnr_z0:.2f} dB")
    print(f"  PSNR(z_shuffle): {psnr_zsh:.2f} dB")
    print(f"  SSIM(probe):     {ssim_probe:.4f}")
    print(f"  SSIM(copy):      {ssim_copy:.4f}")
    print(f"  ΔPSNR(probe-copy):     {d_psnr_copy:+.2f} dB")
    print(f"  ΔPSNR(probe-z=0):      {d_psnr_z0:+.2f} dB")
    print(f"  ΔPSNR(probe-z_shuffle):{d_psnr_zsh:+.2f} dB")

    print(f"\n[Actor Crop Reconstruction — pred bbox target]")
    print(f"  PSNR(probe):     {psnr_probe_predbox:.2f} dB")
    print(f"  PSNR(copy):      {psnr_copy_predbox:.2f} dB")

    go = (d_psnr_z0 >= 1.0 and d_psnr_zsh >= 0.5 and d_psnr_copy >= 0.5)
    print(f"\n[GO/NO-GO]")
    print(f"  z=0 ≥ 1.0:       {'✓' if d_psnr_z0 >= 1.0 else '✗'} ({d_psnr_z0:+.2f})")
    print(f"  z_shuffle ≥ 0.5: {'✓' if d_psnr_zsh >= 0.5 else '✗'} ({d_psnr_zsh:+.2f})")
    print(f"  copy ≥ 0.5:      {'✓' if d_psnr_copy >= 0.5 else '✗'} ({d_psnr_copy:+.2f})")
    print(f"  → {'GO (Phase 2)' if go else 'NO-GO'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_vis = min(args.n_vis, len(test_idx))
        fig, axes = plt.subplots(n_vis, 7, figsize=(21, 3 * n_vis))
        if n_vis == 1:
            axes = axes.unsqueeze(0)

        labels = ["crop_t", "GT", "probe", "copy", "z=0", "z_shuffle", "diff"]
        for row in range(n_vis):
            imgs = [
                ct_te[row].cpu(),
                cg_te[row].cpu(),
                pred_probe[row].cpu(),
                ct_te[row].cpu(),
                pred_z0[row].cpu(),
                pred_zsh[row].cpu(),
                ((pred_probe[row] - cg_te[row]).abs().mean(0)).cpu(),
            ]
            for col, img in enumerate(imgs):
                ax = axes[row, col]
                if col == 6:
                    ax.imshow(img, cmap="hot", vmin=0, vmax=0.5)
                else:
                    ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
                ax.axis("off")
                if row == 0:
                    ax.set_title(labels[col], fontsize=10)

        plt.tight_layout()
        vis_path = os.path.join(results_dir, f"recon_vis_{probe_name}.png")
        plt.savefig(vis_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n  Visualization saved: {vis_path}")
    except ImportError:
        print("\n  (matplotlib not installed, skipping visualization)")

    results = {
        "probe_name": probe_name,
        "n_test": len(test_idx),
        "gt_bbox_target": {
            "psnr_probe": round(psnr_probe, 4),
            "psnr_copy": round(psnr_copy, 4),
            "psnr_z0": round(psnr_z0, 4),
            "psnr_z_shuffle": round(psnr_zsh, 4),
            "ssim_probe": round(ssim_probe, 4),
            "ssim_copy": round(ssim_copy, 4),
            "delta_psnr_copy": round(d_psnr_copy, 4),
            "delta_psnr_z0": round(d_psnr_z0, 4),
            "delta_psnr_z_shuffle": round(d_psnr_zsh, 4),
        },
        "pred_bbox_target": {
            "psnr_probe": round(psnr_probe_predbox, 4),
            "psnr_copy": round(psnr_copy_predbox, 4),
        },
        "go_no_go": {
            "z0_pass": bool(d_psnr_z0 >= 1.0),
            "z_shuffle_pass": bool(d_psnr_zsh >= 0.5),
            "copy_pass": bool(d_psnr_copy >= 0.5),
            "verdict": "GO" if go else "NO-GO",
        },
    }
    out_path = os.path.join(results_dir, f"eval_{probe_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
