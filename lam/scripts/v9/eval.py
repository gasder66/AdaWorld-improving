"""
V9 统一评估: 聚类质量 + 重建质量 + GO/NO-GO.

指标:
  1. Overall NMI / ARI / Per-Slot NMI / Actor Leakage (同 V8 eval)
  2. PSNR(recon) / PSNR(copy) / PSNR(z=0) / PSNR(z_shuffle)
  3. SSIM(recon) / SSIM(copy)
  4. GO/NO-GO: ΔPSNR(recon-z=0) ≥ 1.0, ΔPSNR(recon-z_shuffle) ≥ 0.5, ΔPSNR(recon-copy) ≥ 0.5

用法:
  PYTHONPATH=lam python lam/scripts/eval_v9.py --name v9a_stage1 --gpu 0
"""
import os, sys, json, argparse
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from lam.modules.v9_model import LatentActionModelV9
from lam.modules.v9_decoder import compute_psnr, compute_ssim_simple
from lam.modules.motion_token_encoder import _crop_resize
from lam.mot_slot_dataset import MOTSlotDataset


def _crop_resize_mask(masks: torch.Tensor, boxes: torch.Tensor, crop_size: int = 32) -> torch.Tensor:
    """对 mask 应用与 _crop_resize 相同的 bbox crop + resize.

    Args:
        masks: (B, T, K, H, W) binary mask per actor
        boxes: (B, T, K, 4) [x1,y1,x2,y2] pixel coords
        crop_size: 输出大小
    Returns:
        crop_masks: (B, T, K, crop_size, crop_size) in [0,1]
    """
    B, T, K, H, W = masks.shape
    # 把 (B,T,K,1,H,W) 当作单通道"视频", 复用 _crop_resize
    # _crop_resize 期望 (B,T,H,W,C), 我们 reshape mask 为 (B,T,K,H,W,1) 然后把 K 当成 batch 维
    # 但 _crop_resize 内部对 K 维做 expand, 所以更简单: 对每个 (b,t,k) 独立 affine_grid
    boxes_flat = boxes.reshape(B * T * K, 4)
    # mask: (B,T,K,H,W) → (B*T*K, 1, H, W)
    masks_flat = masks.reshape(B * T * K, 1, H, W).float()

    x1 = boxes_flat[:, 0] / W
    y1 = boxes_flat[:, 1] / H
    x2 = boxes_flat[:, 2] / W
    y2 = boxes_flat[:, 3] / H
    gx = (x1 + x2) / 2 * 2 - 1
    gy = (y1 + y2) / 2 * 2 - 1
    dx = (x2 - x1) / 2 * 2
    dy = (y2 - y1) / 2 * 2
    theta = torch.zeros(B * T * K, 2, 3, device=masks.device, dtype=masks.dtype)
    theta[:, 0, 0] = dx
    theta[:, 0, 2] = gx
    theta[:, 1, 1] = dy
    theta[:, 1, 2] = gy
    grid = F.affine_grid(theta, [B * T * K, 1, crop_size, crop_size], align_corners=False)
    crops = F.grid_sample(masks_flat, grid, align_corners=False, padding_mode="zeros")
    crops = crops.reshape(B, T, K, crop_size, crop_size)
    return crops


def compute_masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Actor-masked PSNR on 32x32 crops.

    Args:
        pred:   (N, C, H, W) recon
        target: (N, C, H, W) GT crop
        mask:   (N, H, W) binary mask (union of actor pixels in crop space)
    Returns:
        psnr_masked: dB, 只在 mask 区域计算
    """
    if mask.sum().item() < 1e-6:
        return float("nan")
    mask_exp = mask.unsqueeze(1).expand_as(pred)  # (N, C, H, W)
    mse = ((pred - target) ** 2 * mask_exp).sum() / (mask_exp.sum() + 1e-8)
    if mse.item() < 1e-10:
        return 100.0
    return float(10 * torch.log10(1.0 / mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--max_actors", type=int, default=4)
    parser.add_argument("--n_vis", type=int, default=20, help="可视化样本数")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "v9"
    )

    # Load training results for model config
    results_path = os.path.join(RESULTS_DIR, f"results_{args.name}.json")
    with open(results_path) as f:
        train_results = json.load(f)

    variant = train_results.get("variant", "A")
    use_bg = train_results.get("use_bg_slot", True)
    model_dim = train_results.get("model_dim", 256)
    z_dim = train_results.get("z_dim", 16)
    z_bg_dim = train_results.get("z_bg_dim", 16)
    z_app_dim = train_results.get("z_app_dim", 0)
    detach_z = train_results.get("detach_z_actor", False)
    cam_pert = train_results.get("camera_perturbation", False)

    # Load latents
    latents_path = os.path.join(RESULTS_DIR, f"latents_{args.name}.npz")
    data = np.load(latents_path)
    z_actor = data["z_actor"]
    z_bg = data["z_bg"]
    actions = data["actions"]
    actor_ids = data["actor_ids"]
    recon_arr = data["recon"]
    crop_t_arr = data["crop_t"]
    crop_tp1_arr = data["crop_tp1"]
    z_app_arr = data["z_app"] if "z_app" in data else None

    print(f"\n{'='*60}")
    print(f"V9 Evaluation: {args.name} (variant {variant})")
    print(f"  Samples: {len(z_actor)}")
    print(f"  z_actor: {z_actor.shape}, z_bg: {z_bg.shape}")
    if z_app_arr is not None:
        print(f"  z_app: {z_app_arr.shape}")
    print(f"  Actions: {np.unique(actions, return_counts=True)}")
    print(f"  Actors:  {np.unique(actor_ids, return_counts=True)}")
    print(f"{'='*60}")

    results = {
        "model": f"V9-{variant} ({args.name})",
        "variant": variant,
        "n_samples": int(len(z_actor)),
        "n_actions": int(len(np.unique(actions))),
        "n_actors": int(len(np.unique(actor_ids))),
    }

    # === 1. Clustering: Overall NMI / ARI ===
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    pred_all = kmeans.fit_predict(z_actor)
    nmi_overall = normalized_mutual_info_score(actions, pred_all)
    ari_overall = adjusted_rand_score(actions, pred_all)
    print(f"\n[Overall Action Clustering]")
    print(f"  NMI = {nmi_overall:.4f}  (V8: 0.7723, V6c: 0.0525)")
    print(f"  ARI = {ari_overall:.4f}")
    results["overall_nmi"] = round(float(nmi_overall), 4)
    results["overall_ari"] = round(float(ari_overall), 4)

    # === 2. Per-Slot NMI ===
    print(f"\n[Per-Actor NMI]")
    nmi_per_actor = []
    for k in np.unique(actor_ids):
        idx_k = actor_ids == k
        if idx_k.sum() < 50:
            continue
        kmeans_k = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
        pred_k = kmeans_k.fit_predict(z_actor[idx_k])
        true_k = actions[idx_k]
        nmi_k = normalized_mutual_info_score(true_k, pred_k)
        nmi_per_actor.append(nmi_k)
        print(f"  Actor {k}: NMI = {nmi_k:.4f} (n={idx_k.sum()})")
    nmi_per_avg = float(np.mean(nmi_per_actor)) if nmi_per_actor else 0.0
    print(f"  Avg: NMI = {nmi_per_avg:.4f}  (V8: 0.8741)")
    results["per_actor_nmi"] = [round(float(x), 4) for x in nmi_per_actor]
    results["per_actor_nmi_avg"] = round(nmi_per_avg, 4)

    # === 3. Actor Leakage ===
    n_actors = len(np.unique(actor_ids))
    chance = 1.0 / n_actors
    clf = LogisticRegression(max_iter=1000, C=1.0)
    n = len(z_actor)
    idx = np.random.RandomState(args.seed).permutation(n)
    n_train = int(0.8 * n)
    tr, te = idx[:n_train], idx[n_train:]
    clf.fit(z_actor[tr], actor_ids[tr])
    leakage_acc = clf.score(z_actor[te], actor_ids[te])
    print(f"\n[Actor Leakage]")
    print(f"  z_actor -> actor_id acc = {leakage_acc:.4f}  (V8: 0.3350, chance = {chance:.4f})")
    results["actor_leakage_acc"] = round(float(leakage_acc), 4)

    # === 4. Action Probe ===
    clf_act = LogisticRegression(max_iter=1000, C=1.0)
    clf_act.fit(z_actor[tr], actions[tr])
    action_acc = clf_act.score(z_actor[te], actions[te])
    action_chance = 1.0 / args.n_clusters
    print(f"\n[Action Probe]")
    print(f"  z_actor -> action acc = {action_acc:.4f}  (chance = {action_chance:.4f})")
    results["action_probe_acc"] = round(float(action_acc), 4)

    # === 5. Reconstruction: 4-way PSNR ===
    # Need to run model with z=0 and z_shuffle to get those PSNRs
    print(f"\n[Reconstruction Evaluation]")

    # Load model for z=0 and z_shuffle eval
    ckpt_path = os.path.join(RESULTS_DIR, f"model_{args.name}.pt")
    cam_scale = (8.0, 8.0, 0.1, 0.05)
    model = LatentActionModelV9(
        model_dim=model_dim, z_dim=z_dim, z_bg_dim=z_bg_dim,
        max_actors=args.max_actors, crop_size=32,
        use_bg_slot=use_bg, variant=variant,
        z_app_dim=z_app_dim if z_app_dim > 0 else 16,
        detach_z_actor_recon=detach_z,
        camera_param_scale=cam_scale,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    if args.data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    else:
        data_root = args.data_root

    eval_dataset = MOTSlotDataset(
        os.path.join(data_root, "val"),
        max_actors=args.max_actors, num_frames=args.num_frames,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=32, shuffle=False,
    )

    # Collect recon under 4 conditions: probe (normal), z=0, z_shuffle, copy
    all_recon_probe = []
    all_recon_z0 = []
    all_recon_zsh = []
    all_copy = []
    all_target = []
    all_mask_crop = []  # actor mask in 32x32 crop space (target frame t+1)
    n_collected = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            v_mask = batch_gpu["valid_mask"]
            actions_b = batch["actions"]

            # Normal recon
            out = model(batch_gpu)
            recon = out["recon"]       # (B, T-1, K, 3, cs, cs)
            crop_tp1 = out["crop_tp1"] # target
            crop_t = out["crop_t"]     # copy baseline

            # z=0: zero out z_actor in decoder
            z_actor_orig = out["z_actor"]
            z_bg_orig = out.get("z_bg")
            B, T1, K, D = z_actor_orig.shape

            # We need to re-run decoder with z=0
            from lam.modules.motion_token_encoder import _crop_resize
            crop_t_flat = crop_t.reshape(B * T1 * K, 3, 32, 32)
            crop_tp1_flat = crop_tp1.reshape(B * T1 * K, 3, 32, 32)

            z0_flat = torch.zeros(B * T1 * K, z_dim, device=device)
            if z_bg_orig is not None:
                z_bg_flat = z_bg_orig.unsqueeze(2).expand(-1, -1, K, -1).reshape(B * T1 * K, -1)
            else:
                z_bg_flat = torch.zeros(B * T1 * K, 0, device=device)

            if variant == "C":
                z_app = out["z_app"]
                z_app_flat = z_app.reshape(B * T1 * K, -1)
                z_dec_0 = torch.cat([z0_flat, z_app_flat], dim=-1)
                z_dec_sh = torch.cat([z_actor_flat_shuffled, z_app_flat], dim=-1) if False else None
                # shuffle z_actor
                perm = torch.randperm(B * T1 * K)
                z_actor_sh = z_actor_orig.reshape(B * T1 * K, -1)[perm]
                z_dec_sh = torch.cat([z_actor_sh, z_app_flat], dim=-1)
                recon_z0_flat = model.recon_decoder(crop_t_flat, z_dec_0, z_bg_flat)
                recon_zsh_flat = model.recon_decoder(crop_t_flat, z_dec_sh, z_bg_flat)
            else:
                recon_z0_flat = model.recon_decoder(crop_t_flat, z0_flat, z_bg_flat)
                # z_shuffle
                perm = torch.randperm(B * T1 * K)
                z_actor_flat = z_actor_orig.reshape(B * T1 * K, -1)
                z_sh_flat = z_actor_flat[perm]
                recon_zsh_flat = model.recon_decoder(crop_t_flat, z_sh_flat, z_bg_flat)

            recon_flat = recon.reshape(B * T1 * K, 3, 32, 32)
            copy_flat = crop_t.reshape(B * T1 * K, 3, 32, 32)

            # Crop actor masks to 32x32 target-frame space for masked PSNR
            # masks: (B, T, K, H, W), boxes: (B, T, K, 4) — use t+1 (target)
            masks_tp1 = batch_gpu["masks"][:, 1:]   # (B, T1, K, H, W)
            boxes_tp1 = batch_gpu["boxes"][:, 1:]   # (B, T1, K, 4)
            mask_crops = _crop_resize_mask(masks_tp1, boxes_tp1, crop_size=32)
            mask_crops_flat = mask_crops.reshape(B * T1 * K, 32, 32)

            # Only keep valid + labeled samples
            v_np = v_mask[:, 1:].cpu().numpy()
            act_np = actions_b.cpu().numpy()
            for b in range(B):
                for t in range(T1):
                    for k in range(K):
                        if v_np[b, t, k] and act_np[b, t, k] >= 0:
                            idx_flat = b * T1 * K + t * K + k
                            all_recon_probe.append(recon_flat[idx_flat].cpu().numpy())
                            all_recon_z0.append(recon_z0_flat[idx_flat].cpu().numpy())
                            all_recon_zsh.append(recon_zsh_flat[idx_flat].cpu().numpy())
                            all_copy.append(copy_flat[idx_flat].cpu().numpy())
                            all_target.append(crop_tp1_flat[idx_flat].cpu().numpy())
                            all_mask_crop.append(mask_crops_flat[idx_flat].cpu().numpy())
            n_collected += 1
            if n_collected >= 30:
                break

    recon_probe = torch.from_numpy(np.array(all_recon_probe))
    recon_z0 = torch.from_numpy(np.array(all_recon_z0))
    recon_zsh = torch.from_numpy(np.array(all_recon_zsh))
    copy_f = torch.from_numpy(np.array(all_copy))
    target_f = torch.from_numpy(np.array(all_target))

    psnr_probe = compute_psnr(recon_probe, target_f)
    psnr_z0 = compute_psnr(recon_z0, target_f)
    psnr_zsh = compute_psnr(recon_zsh, target_f)
    psnr_copy = compute_psnr(copy_f, target_f)
    ssim_probe = compute_ssim_simple(recon_probe, target_f)
    ssim_copy = compute_ssim_simple(copy_f, target_f)

    d_z0 = psnr_probe - psnr_z0
    d_zsh = psnr_probe - psnr_zsh
    d_copy = psnr_probe - psnr_copy

    print(f"  PSNR(recon):     {psnr_probe:.2f} dB")
    print(f"  PSNR(copy):      {psnr_copy:.2f} dB")
    print(f"  PSNR(z=0):       {psnr_z0:.2f} dB")
    print(f"  PSNR(z_shuffle): {psnr_zsh:.2f} dB")
    print(f"  SSIM(recon):     {ssim_probe:.4f}")
    print(f"  SSIM(copy):      {ssim_copy:.4f}")
    print(f"  ΔPSNR(recon-copy):     {d_copy:+.2f} dB")
    print(f"  ΔPSNR(recon-z=0):      {d_z0:+.2f} dB")
    print(f"  ΔPSNR(recon-z_shuffle):{d_zsh:+.2f} dB")

    results["psnr_recon"] = round(psnr_probe, 2)
    results["psnr_copy"] = round(psnr_copy, 2)
    results["psnr_z0"] = round(psnr_z0, 2)
    results["psnr_zshuffle"] = round(psnr_zsh, 2)
    results["ssim_recon"] = round(ssim_probe, 4)
    results["ssim_copy"] = round(ssim_copy, 4)
    results["delta_psnr_copy"] = round(d_copy, 2)
    results["delta_psnr_z0"] = round(d_z0, 2)
    results["delta_psnr_zshuffle"] = round(d_zsh, 2)

    # --- Actor-masked PSNR (only within actor pixels in 32x32 crop space) ---
    if len(all_mask_crop) > 0:
        mask_crop_t = torch.from_numpy(np.array(all_mask_crop))  # (N, 32, 32)
        # Threshold to binary (grid_sample may have slight blur)
        mask_crop_t = (mask_crop_t > 0.5).float()

        psnr_mask_probe = compute_masked_psnr(recon_probe, target_f, mask_crop_t)
        psnr_mask_z0 = compute_masked_psnr(recon_z0, target_f, mask_crop_t)
        psnr_mask_zsh = compute_masked_psnr(recon_zsh, target_f, mask_crop_t)
        psnr_mask_copy = compute_masked_psnr(copy_f, target_f, mask_crop_t)

        d_mask_z0 = psnr_mask_probe - psnr_mask_z0
        d_mask_zsh = psnr_mask_probe - psnr_mask_zsh
        d_mask_copy = psnr_mask_probe - psnr_mask_copy

        mask_coverage = mask_crop_t.sum().item() / mask_crop_t.numel()

        print(f"\n  [Actor-Masked PSNR (32x32 crop)]")
        print(f"    Mask coverage:  {mask_coverage:.3f} ({mask_coverage*100:.1f}%)")
        print(f"    PSNR(recon):    {psnr_mask_probe:.2f} dB")
        print(f"    PSNR(copy):     {psnr_mask_copy:.2f} dB")
        print(f"    PSNR(z=0):      {psnr_mask_z0:.2f} dB")
        print(f"    PSNR(z_shuffle):{psnr_mask_zsh:.2f} dB")
        print(f"    ΔPSNR(recon-copy):     {d_mask_copy:+.2f} dB")
        print(f"    ΔPSNR(recon-z=0):      {d_mask_z0:+.2f} dB")
        print(f"    ΔPSNR(recon-z_shuffle):{d_mask_zsh:+.2f} dB")

        results["psnr_actor_masked"] = {
            "recon": round(psnr_mask_probe, 2),
            "copy": round(psnr_mask_copy, 2),
            "z0": round(psnr_mask_z0, 2),
            "zshuffle": round(psnr_mask_zsh, 2),
            "delta_copy": round(d_mask_copy, 2),
            "delta_z0": round(d_mask_z0, 2),
            "delta_zshuffle": round(d_mask_zsh, 2),
            "mask_coverage": round(mask_coverage, 4),
        }

    # === 6. GO/NO-GO ===
    go_z0 = d_z0 >= 1.0
    go_zsh = d_zsh >= 0.5
    go_copy = d_copy >= 0.5
    go = go_z0 and go_zsh and go_copy
    print(f"\n[GO/NO-GO]")
    print(f"  z=0 ≥ 1.0:       {'✓' if go_z0 else '✗'} ({d_z0:+.2f})")
    print(f"  z_shuffle ≥ 0.5: {'✓' if go_zsh else '✗'} ({d_zsh:+.2f})")
    print(f"  copy ≥ 0.5:      {'✓' if go_copy else '✗'} ({d_copy:+.2f})")
    print(f"  → {'GO' if go else 'NO-GO'}")
    results["go_verdict"] = "GO" if go else "NO-GO"
    results["go_criteria"] = {
        "z0_ge_1": bool(go_z0), "zshuffle_ge_05": bool(go_zsh), "copy_ge_05": bool(go_copy),
    }

    # === 7. Visualization ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_show = min(args.n_vis, len(all_target))
        fig, axes = plt.subplots(n_show, 5, figsize=(20, 4 * n_show))
        if n_show == 1:
            axes = axes.unsqueeze(0)
        col_titles = ["crop_t", "GT (crop_{t+1})", "recon", "z=0", "z_shuffle"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=14)
        for i in range(n_show):
            for col, img in enumerate([
                all_copy[i], all_target[i], all_recon_probe[i], all_recon_z0[i], all_recon_zsh[i]
            ]):
                img_show = np.transpose(img, (1, 2, 0))
                axes[i, col].imshow(np.clip(img_show, 0, 1))
                axes[i, col].axis("off")
        plt.tight_layout()
        vis_path = os.path.join(RESULTS_DIR, f"recon_vis_{args.name}.png")
        plt.savefig(vis_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"\n  Visualization saved: {vis_path}")
        results["vis_path"] = vis_path
    except Exception as e:
        print(f"\n  (visualization failed: {e})")

    # === 8. Comparison Table ===
    print(f"\n{'='*80}")
    print(f"Comparison: V8 vs V9-{variant}")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'V6c':>10} {'V8':>10} {'V9-'+variant:>10} {'Target':>10}")
    print(f"{'-'*70}")
    print(f"{'Overall NMI':<30} {'0.0525':>10} {'0.7723':>10} {nmi_overall:>10.4f} {'≥0.20':>10}")
    print(f"{'Per-Slot NMI (avg)':<30} {'0.3885':>10} {'0.7684':>10} {nmi_per_avg:>10.4f} {'≥0.30':>10}")
    print(f"{'Actor Leakage':<30} {'1.0000':>10} {'0.3350':>10} {leakage_acc:>10.4f} {'≤0.50':>10}")
    print(f"{'Action Probe':<30} {'N/A':>10} {'0.8741':>10} {action_acc:>10.4f} {'-':>10}")
    print(f"{'PSNR(recon) dB':<30} {'27.35':>10} {'NO-GO':>10} {psnr_probe:>10.2f} {'-':>10}")
    print(f"{'PSNR(copy) dB':<30} {'N/A':>10} {'22.68':>10} {psnr_copy:>10.2f} {'-':>10}")
    print(f"{'ΔPSNR(recon-z=0)':<30} {'N/A':>10} {'0.00':>10} {d_z0:>10.2f} {'≥1.0':>10}")
    print(f"{'ΔPSNR(recon-z_shuffle)':<30} {'N/A':>10} {'0.00':>10} {d_zsh:>10.2f} {'≥0.5':>10}")
    print(f"{'ΔPSNR(recon-copy)':<30} {'N/A':>10} {'+0.66':>10} {d_copy:>10.2f} {'≥0.5':>10}")
    print(f"{'Verdict':<30} {'':>10} {'NO-GO':>10} {results['go_verdict']:>10} {'':>10}")
    print(f"{'='*70}")

    out_path = os.path.join(RESULTS_DIR, f"eval_{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
