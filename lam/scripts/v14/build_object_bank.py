"""
Build object bank from A2D annotations — extract real object crops + masks.
Supports --clean mode for filtering low-quality masks.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/build_object_bank.py \
      --a2d_root data/a2d --release_root Release \
      --output data/bridgebench/object_bank_clean.pt \
      --clean --min_mask_area 50 --min_bbox_area 64 \
      --save_stats data/bridgebench/object_bank_clean_stats.json \
      --save_preview data/bridgebench/object_bank_clean_preview.png
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.a2d_dataset import parse_a2d_label

ACTOR_NAMES = {1: "adult", 2: "baby", 3: "ball", 4: "bird", 5: "car", 6: "cat", 7: "dog"}


def _crop_bbox(frame, mask, bbox, crop_size):
    """Crop frame and mask around bbox, resize to crop_size.
    Returns (image_crop (3,cs,cs), mask_crop (1,cs,cs)) or (None, None)."""
    H, W = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(W, int(bbox[2] + 1))
    y2 = min(H, int(bbox[3] + 1))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, None

    crop_img = frame[y1:y2, x1:x2]
    if mask.ndim == 3:
        crop_mask = mask[0, y1:y2, x1:x2]  # (N,H,W) → first instance
    else:
        crop_mask = mask[y1:y2, x1:x2]
    if crop_img.size == 0 or crop_mask.size == 0:
        return None, None

    crop_img = cv2.resize(crop_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    crop_mask = cv2.resize(crop_mask.astype(np.float32).astype(np.uint8),
                           (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    crop_img = torch.from_numpy(crop_img).float().permute(2, 0, 1) / 255.0
    crop_mask = torch.from_numpy(crop_mask).float().unsqueeze(0)
    return crop_img, crop_mask


def _mask_from_bbox(bbox, H, W):
    """Create a rectangular mask from bbox for fallback."""
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(W, int(bbox[2] + 1)); y2 = min(H, int(bbox[3] + 1))
    mask = np.zeros((H, W), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2d_root", default="data/a2d")
    parser.add_argument("--release_root", default="Release")
    parser.add_argument("--output", default="data/bridgebench/object_bank_clean.pt")
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--max_objects", type=int, default=5000)
    parser.add_argument("--max_per_video", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    # Clean filtering.
    parser.add_argument("--clean", action="store_true",
                        help="Enable clean filtering of low-quality masks")
    parser.add_argument("--min_mask_area", type=float, default=50,
                        help="Min pixels of mask area in crop")
    parser.add_argument("--min_bbox_area", type=float, default=64,
                        help="Min pixel area of bbox")
    parser.add_argument("--min_mask_bbox_ratio", type=float, default=0.15,
                        help="Min mask_area / bbox_area ratio")
    parser.add_argument("--min_crop_fg_ratio", type=float, default=0.02,
                        help="Min foreground ratio after crop+resize")
    parser.add_argument("--min_width", type=float, default=8)
    parser.add_argument("--min_height", type=float, default=8)
    parser.add_argument("--save_stats", default=None)
    parser.add_argument("--save_preview", default=None)
    args = parser.parse_args()

    annot_dir = os.path.join(args.release_root, "Annotations", "mat")
    video_dir = os.path.join(args.a2d_root, "train")
    csv_path = os.path.join(args.release_root, "videoset.csv")
    if not os.path.isdir(annot_dir):
        print(f"ERROR: annotation dir not found: {annot_dir}"); sys.exit(1)

    train_videos = []
    with open(csv_path, "r") as f:
        for line in f:
            p = line.strip().split(",")
            if int(p[8]) == 0:
                train_videos.append(p[0])
    print(f"Train videos: {len(train_videos)}")

    rng = np.random.RandomState(args.seed)
    video_dirs = rng.permutation([v for v in sorted(os.listdir(annot_dir)) if v in train_videos])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Stats.
    stats = {"num_raw": 0, "num_kept": 0, "num_dropped_empty_mask": 0,
             "num_dropped_tiny_mask": 0, "num_dropped_bad_bbox": 0,
             "num_dropped_low_ratio": 0, "num_dropped_low_fg": 0,
             "per_category": {}}
    objects, preview_objects = [], []

    for vi, vid in enumerate(video_dirs):
        if len(objects) >= args.max_objects:
            break

        mat_dir = os.path.join(annot_dir, vid)
        mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith(".mat")])
        if not mat_files: continue

        video_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path): continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release(); continue

        selected = rng.choice(mat_files, size=min(args.max_per_video, len(mat_files)),
                              replace=False)
        try:
            for mf in selected:
                if len(objects) >= args.max_objects: break
                frame_num = int(mf.replace(".mat", ""))
                try:
                    with h5py.File(os.path.join(mat_dir, mf), "r") as f:
                        bboxes = f["reBBox"][:]    # (4, N)
                        masks = f["reMask"][:]     # (N, H, W)
                        id_raw = f["id"][:].flatten().astype(int)
                except Exception: continue

                N = bboxes.shape[1]
                if N == 0: continue

                # Seek to frame.
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_num - 1))
                ret, frame = cap.read()
                if not ret: continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                H, W = frame_rgb.shape[:2]

                for n in range(N):
                    if len(objects) >= args.max_objects: break
                    bbox = bboxes[:, n]  # [x1,y1,x2,y2]
                    mask = masks[n] if masks.ndim == 3 else masks
                    if mask.ndim != 2: continue

                    actor_id, _ = parse_a2d_label(id_raw[n])
                    category = ACTOR_NAMES.get(actor_id, "unknown")
                    stats["num_raw"] += 1
                    stats["per_category"].setdefault(category, {"raw": 0, "kept": 0})
                    stats["per_category"][category]["raw"] += 1

                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    bbox_w = bbox[2] - bbox[0]; bbox_h = bbox[3] - bbox[1]

                    # ---- Clean filtering ----
                    if args.clean:
                        mask_sum = mask.sum()
                        # 1. Mask non-empty.
                        if mask_sum < args.min_mask_area:
                            if mask_sum < 1:
                                stats["num_dropped_empty_mask"] += 1
                            else:
                                stats["num_dropped_tiny_mask"] += 1
                            continue
                        # 2. Bbox valid.
                        if bbox_w < args.min_width or bbox_h < args.min_height or bbox_area < args.min_bbox_area:
                            stats["num_dropped_bad_bbox"] += 1
                            continue
                        # 3. Mask/bbox ratio.
                        ratio = mask_sum / (bbox_area + 1e-6)
                        if ratio < args.min_mask_bbox_ratio:
                            stats["num_dropped_low_ratio"] += 1
                            continue

                    crop_img, crop_mask = _crop_bbox(frame_rgb, mask, bbox, args.crop_size)
                    if crop_img is None:
                        stats.setdefault("num_dropped_crop_fail", 0)
                        stats["num_dropped_crop_fail"] += 1
                        continue

                    # 4. Foreground after resize.
                    if args.clean:
                        fg_ratio = crop_mask.sum().item() / (args.crop_size ** 2)
                        if fg_ratio < args.min_crop_fg_ratio:
                            stats["num_dropped_low_fg"] += 1
                            continue

                    cx = (bbox[0] + bbox[2]) / 2 / W
                    cy = (bbox[1] + bbox[3]) / 2 / H
                    w = (bbox[2] - bbox[0]) / W
                    h = (bbox[3] - bbox[1]) / H
                    mask_area_val = crop_mask.sum().item()

                    obj = {
                        "image_crop": crop_img, "mask_crop": crop_mask,
                        "bbox": torch.tensor([cx, cy, w, h]),
                        "category": category, "source_video": vid,
                        "frame_id": frame_num, "instance_id": n,
                        "mask_area": mask_area_val,
                        "bbox_area": bbox_area,
                        "mask_bbox_ratio": mask_area_val / max(bbox_area, 1e-6),
                    }
                    objects.append(obj)
                    stats["num_kept"] += 1
                    stats["per_category"][category]["kept"] += 1
                    if len(preview_objects) < 10:
                        preview_objects.append(obj)

        finally:
            cap.release()

        if vi % 100 == 0:
            print(f"  [{vi}/{len(video_dirs)}] kept={len(objects)}")

    # Save.
    torch.save(objects, args.output)
    print(f"\nObject bank: {len(objects)} objects → {args.output}")

    if args.clean:
        print(f"  Raw: {stats['num_raw']}, Kept: {stats['num_kept']} "
              f"({100*stats['num_kept']/max(1,stats['num_raw']):.1f}%)")
        print(f"  Dropped: empty={stats['num_dropped_empty_mask']} "
              f"tiny={stats['num_dropped_tiny_mask']} bad_bbox={stats['num_dropped_bad_bbox']} "
              f"low_ratio={stats['num_dropped_low_ratio']} low_fg={stats['num_dropped_low_fg']}")

    if args.save_stats:
        stats["mask_areas"] = [o["mask_area"] for o in objects]
        stats["mask_areas_mean"] = float(np.mean(stats["mask_areas"])) if stats["mask_areas"] else 0
        stats["mask_areas_std"] = float(np.std(stats["mask_areas"])) if stats["mask_areas"] else 0
        stats["ratios"] = [o["mask_bbox_ratio"] for o in objects]
        stats["mask_bbox_ratio_mean"] = float(np.mean(stats["ratios"])) if stats["ratios"] else 0
        stats["mask_bbox_ratio_std"] = float(np.std(stats["ratios"])) if stats["ratios"] else 0
        with open(args.save_stats, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"  Stats → {args.save_stats}")

    if args.save_preview and preview_objects:
        _save_preview(preview_objects, args.save_preview, args.crop_size)
        print(f"  Preview → {args.save_preview}")


def _save_preview(objects, out_path, crop_size):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = min(10, len(objects))
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        img = objects[i]["image_crop"].permute(1, 2, 0).clamp(0, 1)
        mask = objects[i]["mask_crop"].squeeze(0)
        overlay = img.clone()
        overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + 0.5
        axes[0, i].imshow(img.numpy()); axes[0, i].set_title(f"img {i}", fontsize=7); axes[0, i].axis("off")
        axes[1, i].imshow(mask.numpy(), cmap="Reds"); axes[1, i].set_title(f"mask", fontsize=7); axes[1, i].axis("off")
    fig.tight_layout(pad=0.3); fig.savefig(out_path, dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
