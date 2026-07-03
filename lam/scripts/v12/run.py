"""
V12 training script — supports Phase A / B / C.

Usage:
  PYTHONPATH=lam python lam/scripts/v12/run.py \
      --config phaseA --dataset synthetic_minimal_nooverlap \
      --phase A --batch_size 16 --steps 3000 --gpu 4

Output: result/v12/{dataset}/{config}/
"""
import os, sys, json, time, argparse
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lam.modules.v12_model import LatentActionModelV12
from lam.v12_dataset import V12ObjectVideoDataset

VERSION = "v12"


def collate_v12(batch):
    """Collate V12 samples — keep tensors, ignore None."""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], dict):
            out[k] = vals[0]  # metadata shared
        else:
            out[k] = vals
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="synthetic_minimal_nooverlap")
    parser.add_argument("--phase", type=str, default="C", choices=["A", "B", "C"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load pretrained weights (e.g. Phase A -> Phase B/C)")
    # Model
    parser.add_argument("--max_actors", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--content_dim", type=int, default=128)
    parser.add_argument("--struct_dim", type=int, default=128)
    parser.add_argument("--mask_grid", type=int, default=16)
    parser.add_argument("--mask_feat_dim", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--dec_dim", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--dec_blocks", type=int, default=4)
    parser.add_argument("--free_bits", type=float, default=0.05)
    # Loss weights
    parser.add_argument("--lambda_box", type=float, default=10.0)
    parser.add_argument("--lambda_mask", type=float, default=1.0)
    parser.add_argument("--lambda_mom", type=float, default=1.0)
    parser.add_argument("--lambda_actor_masked", type=float, default=2.0)
    parser.add_argument("--recon_weight", type=float, default=0.1)
    parser.add_argument("--kl_beta", type=float, default=0.01)
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
        data_root = os.path.join(ROOT, "data", "v12", args.dataset)
    else:
        data_root = args.data_root

    print(f"\n{'='*60}")
    print(f"V12: Object-Centric Structure-Action World Model")
    print(f"  GPU={args.gpu}, phase={args.phase}, dataset={args.dataset}, config={args.config}")
    print(f"  result => {RESULTS_DIR}")
    print(f"  batch={args.batch_size}, steps={args.steps}, lr={args.lr}")
    print(f"  data: {data_root}")
    print(f"{'='*60}")

    train_dataset = V12ObjectVideoDataset(
        os.path.join(data_root, "train"), output_format="t h w c",
    )
    eval_dataset = V12ObjectVideoDataset(
        os.path.join(data_root, "val"), output_format="t h w c",
    )

    model = LatentActionModelV12(
        image_size=args.image_size, max_actors=args.max_actors,
        crop_size=args.crop_size, content_dim=args.content_dim,
        mask_grid=args.mask_grid, mask_feat_dim=args.mask_feat_dim,
        struct_dim=args.struct_dim, latent_dim=args.latent_dim,
        dec_dim=args.dec_dim, patch_size=args.patch_size,
        dec_blocks=args.dec_blocks, free_bits=args.free_bits,
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded checkpoint: {args.checkpoint}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        collate_fn=collate_v12, pin_memory=(args.num_workers > 0),
    )

    loss_keys = ["total", "recon", "struct", "kl", "box", "mask", "mom"]
    losses = {k: [] for k in loss_keys}
    step = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break
            videos = batch["videos"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            boxes = batch["bboxes"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            # actor_labels/object_types NEVER enter forward.
            batch_input = {
                "videos": videos, "masks": masks,
                "bboxes": boxes, "valid_mask": valid,
            }

            outputs = model(
                batch_input, phase=args.phase,
                lambda_box=args.lambda_box, lambda_mask=args.lambda_mask,
                lambda_mom=args.lambda_mom,
                lambda_actor_masked=args.lambda_actor_masked,
            )
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses["total"].append(float(loss))
            losses["recon"].append(float(outputs.get("recon_loss", 0.0)))
            losses["struct"].append(float(outputs.get("struct_loss", 0.0)))
            losses["kl"].append(float(outputs.get("kl_loss", 0.0)))

            if step % 50 == 0:
                elapsed = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                parts = [f"loss={float(loss):.4f}"]
                if "recon_loss" in outputs:
                    mse = float(outputs.get("recon_loss", 0))
                    parts.append(f"recon={mse:.4f}")
                if "struct_loss" in outputs:
                    parts.append(f"struct={float(outputs['struct_loss']):.4f}")
                if "kl_loss" in outputs:
                    parts.append(f"kl={float(outputs['kl_loss']):.4f}")
                parts.append(f"mem={mem:.1f}GB")
                parts.append(f"{elapsed:.0f}s")
                print(f"  Step {step:4d}/{args.steps}: " + ", ".join(parts))

            if (step + 1) % args.checkpoint_every == 0:
                ckpt = os.path.join(CKT_DIR, f"step{step+1}.pt")
                torch.save(model.state_dict(), ckpt)

            step += 1

    training_time = time.time() - t0
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"\n  训练完成. 峰值内存: {mem_peak:.1f}GB, 耗时: {training_time:.0f}s")

    # Save final model.
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))

    for key, vals in losses.items():
        np.savetxt(os.path.join(LOSS_DIR, f"{key}.txt"), np.array(vals))

    # === Eval: collect latents + metrics ===
    model.eval()
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=8, num_workers=0, shuffle=False,
        collate_fn=collate_v12,
    )

    all_z, all_actor, all_action = [], [], []
    all_recon_mse, all_copy_mse, all_masked_mse = [], [], []
    all_box_iou, all_inertia_iou = [], []
    n_collected = 0

    with torch.no_grad():
        for batch in eval_loader:
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["bboxes"].to(device)
            valid = batch["valid_mask"].to(device)
            actions = batch["actions"].cpu()
            batch_input = {
                "videos": videos, "masks": masks,
                "bboxes": boxes, "valid_mask": valid,
            }

            # Use Phase C forward for full eval.
            eval_phase = "C" if args.phase in ("B", "C") else "A"
            outputs = model(batch_input, phase=eval_phase)

            if "recon" in outputs:
                target = videos[:, 1:]
                recon = outputs["recon"]
                all_recon_mse.extend(((recon - target) ** 2).mean(dim=[2, 3, 4]).cpu().reshape(-1).tolist())
                all_copy_mse.extend(((videos[:, :-1] - target) ** 2).mean(dim=[2, 3, 4]).cpu().reshape(-1).tolist())
                actor_mask = masks[:, 1:].sum(dim=2).clamp(0, 1).unsqueeze(-1)
                masked_err = ((recon - target) ** 2) * actor_mask.float()
                denom = actor_mask.float().sum(dim=[2, 3, 4]).clamp(min=1.0)
                all_masked_mse.extend((masked_err.sum(dim=[2, 3, 4]) / denom).cpu().reshape(-1).tolist())

            if "pred_struct" in outputs:
                # Box IoU: compare predicted bbox to GT bbox at t+1.
                pred_bbox = outputs["pred_struct"]["bbox"].cpu()  # normalized cxcywh
                gt_boxes = boxes[:, 1:].cpu().float()  # (B, T1, K, 4) pixel xyxy
                B, T1, K, _ = gt_boxes.shape
                H = W = args.image_size
                gx1, gy1, gx2, gy2 = gt_boxes.unbind(-1)
                gt_cxcywh = torch.stack([(gx1+gx2)/2/W, (gy1+gy2)/2/H, (gx2-gx1)/W, (gy2-gy1)/H], -1)
                pcx, pcy, pw, ph = pred_bbox.unbind(-1)
                pred_xyxy = torch.stack([pcx-pw/2, pcy-ph/2, pcx+pw/2, pcy+ph/2], -1)
                gcx, gcy, gw, gh = gt_cxcywh.unbind(-1)
                gt_xyxy = torch.stack([gcx-gw/2, gcy-gh/2, gcx+gw/2, gcy+gh/2], -1)
                iou = _box_iou(pred_xyxy, gt_xyxy)
                all_box_iou.extend(iou.reshape(-1).tolist())
                # Inertia: use t box as prediction for t+1.
                inertia_boxes = boxes[:, :-1].cpu().float()
                ix1, iy1, ix2, iy2 = inertia_boxes.unbind(-1)
                inertia_cxcywh = torch.stack([(ix1+ix2)/2/W, (iy1+iy2)/2/H, (ix2-ix1)/W, (iy2-iy1)/H], -1)
                icx, icy, iw, ih = inertia_cxcywh.unbind(-1)
                inertia_xyxy = torch.stack([icx-iw/2, icy-ih/2, icx+iw/2, icy+ih/2], -1)
                inertia_iou = _box_iou(inertia_xyxy, gt_xyxy)
                all_inertia_iou.extend(inertia_iou.reshape(-1).tolist())

            if "mu" in outputs:
                z_mu = outputs["mu"].detach().cpu().numpy()
                act_np = actions.numpy()
                valid_np = valid[:, :-1].cpu().numpy()
                obj_types = batch.get("object_types", None)
                obj_np = obj_types.numpy() if obj_types is not None else None
                B, T1, K, _ = z_mu.shape
                for b in range(B):
                    for t in range(T1):
                        for k in range(K):
                            if k < valid_np.shape[2] and valid_np[b, t, k] and act_np[b, t, k] >= 0:
                                all_z.append(z_mu[b, t, k])
                                all_actor.append(int(obj_np[b, k]) if obj_np is not None else k)
                                all_action.append(int(act_np[b, t, k]))

            n_collected += videos.shape[0]
            if n_collected >= 500:
                break

    results = {"phase": args.phase, "n_eval_samples": n_collected, "total_params": total_params}

    if all_recon_mse:
        results["rgb_psnr"] = -10 * np.log10(np.mean(all_recon_mse) + 1e-10)
        results["copy_psnr"] = -10 * np.log10(np.mean(all_copy_mse) + 1e-10)
        results["masked_psnr"] = -10 * np.log10(np.mean(all_masked_mse) + 1e-10)

    if all_box_iou:
        results["pred_box_iou"] = float(np.mean(all_box_iou))
        results["inertia_box_iou"] = float(np.mean(all_inertia_iou))

    if len(all_z) >= 10 and len(np.unique(all_action)) >= 2:
        from sklearn.cluster import KMeans
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import normalized_mutual_info_score
        z = np.asarray(all_z)
        actor = np.asarray(all_actor)
        action = np.asarray(all_action)
        n_clusters = len(np.unique(action))
        pred = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10).fit_predict(z)
        results["overall_nmi"] = float(normalized_mutual_info_score(action, pred))

        per_slot = []
        for slot in np.unique(actor):
            idx = actor == slot
            if idx.sum() >= max(10, n_clusters):
                ps = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10).fit_predict(z[idx])
                per_slot.append(float(normalized_mutual_info_score(action[idx], ps)))
        results["per_slot_nmi"] = per_slot
        results["per_slot_nmi_avg"] = float(np.mean(per_slot)) if per_slot else None

        order = np.random.RandomState(args.seed).permutation(len(z))
        n_train = max(1, int(0.8 * len(z)))
        if len(z) - n_train >= 1:
            clf_a = LogisticRegression(max_iter=1000)
            clf_a.fit(z[order[:n_train]], action[order[:n_train]])
            results["action_probe_acc"] = float(clf_a.score(z[order[n_train:]], action[order[n_train:]]))
            clf_r = LogisticRegression(max_iter=1000)
            clf_r.fit(z[order[:n_train]], actor[order[:n_train]])
            results["actor_leakage_acc"] = float(clf_r.score(z[order[n_train:]], actor[order[n_train:]]))

    with open(os.path.join(RESULTS_DIR, "eval.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save latents.
    if all_z:
        np.savez(
            os.path.join(RESULTS_DIR, "latents.npz"),
            z=np.asarray(all_z), actor=np.asarray(all_actor), action=np.asarray(all_action),
        )

    print(f"\n  === Eval results ===")
    for k, v in results.items():
        if k not in ("phase",):
            print(f"  {k}: {v}")
    print(f"  Saved => {RESULTS_DIR}")


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """boxes: (..., 4) normalized xyxy. Returns IoU (...,)."""
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


if __name__ == "__main__":
    main()
