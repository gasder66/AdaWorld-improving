"""
Build GT Anchor Database from A2D sparse annotations.

Extracts all GT bbox/mask/actor annotations from h5py files.
Filters to videos with >= 5 annotation frames (315 total).
Saves per-anchor data: bbox, mask, actor_id, category, frame info.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/build_a2d_anchor_database.py \
      --a2d_root data/a2d --release_root Release \
      --output data/bridgebench/a2d_dense_completion/anchors \
      --min_frames 5 --save_preview
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.a2d_dataset import parse_a2d_label, ACTOR_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2d_root", default="data/a2d")
    parser.add_argument("--release_root", default="Release")
    parser.add_argument("--output", default="data/bridgebench/a2d_dense_completion/anchors")
    parser.add_argument("--min_frames", type=int, default=5)
    parser.add_argument("--max_anchors", type=int, default=0)
    parser.add_argument("--save_preview", action="store_true")
    args = parser.parse_args()

    annot_dir = os.path.join(args.release_root, "Annotations", "mat")
    video_dir = os.path.join(args.a2d_root, "train")
    csv_path = os.path.join(args.release_root, "videoset.csv")
    if not os.path.isdir(annot_dir):
        print(f"ERROR: {annot_dir} not found"); sys.exit(1)

    train_videos = []
    with open(csv_path, "r") as f:
        for line in f:
            p = line.strip().split(",")
            if int(p[8]) == 0: train_videos.append(p[0])
    print(f"Train videos: {len(train_videos)}")

    os.makedirs(args.output, exist_ok=True)
    anchors = []
    stats = {"videos_total": 0, "videos_valid": 0, "anchors_total": 0,
             "anchors_nonempty_mask": 0, "anchors_bbox_only": 0,
             "categories": {}, "mask_areas": [], "bbox_areas": []}

    all_vids = sorted([v for v in os.listdir(annot_dir) if v in train_videos])

    for vi, vid in enumerate(all_vids):
        mat_dir = os.path.join(annot_dir, vid)
        mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith(".mat")])
        if len(mat_files) < args.min_frames: continue
        stats["videos_total"] += 1
        stats["videos_valid"] += 1

        video_path = os.path.join(video_dir, f"{vid}.mp4")
        video_exists = os.path.exists(video_path)

        for mf in mat_files:
            frame_num = int(mf.replace(".mat", ""))
            try:
                with h5py.File(os.path.join(mat_dir, mf), "r") as f:
                    bboxes = f["reBBox"][:]    # (4, N)
                    masks = f["reMask"][:]     # (N, H, W)
                    ids = f["id"][:].flatten().astype(int)
            except Exception:
                continue

            N = bboxes.shape[1]
            for n in range(N):
                bbox = bboxes[:, n]  # [x1,y1,x2,y2]
                bbox_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                mask = masks[n] if masks.ndim == 3 else masks
                if mask.ndim != 2: continue
                mask_sum = mask.sum()
                has_mask = mask_sum > 0

                actor_id, action_id = parse_a2d_label(ids[n])
                category = ACTOR_NAMES.get(actor_id, "unknown")

                anchor = {
                    "source_video": vid,
                    "frame_id": int(frame_num),
                    "actor_idx": int(n),
                    "actor_id": int(actor_id),
                    "action_id": int(action_id),
                    "category": category,
                    "bbox_xyxy": [float(x) for x in bbox],
                    "mask_sum": float(mask_sum),
                    "bbox_area": float(bbox_area),
                    "has_mask": bool(has_mask),
                    "mask_bbox_ratio": float(mask_sum / max(1, bbox_area)),
                    "video_exists": bool(video_exists),
                }
                anchors.append(anchor)
                stats["anchors_total"] += 1
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
                if has_mask:
                    stats["anchors_nonempty_mask"] += 1
                    stats["mask_areas"].append(float(mask_sum))
                else:
                    stats["anchors_bbox_only"] += 1
                stats["bbox_areas"].append(float(bbox_area))

        if args.max_anchors > 0 and len(anchors) >= args.max_anchors: break
        if vi % 100 == 0:
            print(f"  [{vi}/{len(all_vids)}] {len(anchors)} anchors from {vid[:12]}")

    # Save.
    lines = [json.dumps(a) for a in anchors]
    with open(os.path.join(args.output, "anchors.jsonl"), "w") as f:
        f.write("\n".join(lines))

    stats["mask_areas_mean"] = float(np.mean(stats["mask_areas"])) if stats["mask_areas"] else 0
    stats["mask_areas_median"] = float(np.median(stats["mask_areas"])) if stats["mask_areas"] else 0
    stats["mask_nonempty_ratio"] = stats["anchors_nonempty_mask"] / max(1, stats["anchors_total"])
    stats["bbox_areas_mean"] = float(np.mean(stats["bbox_areas"])) if stats["bbox_areas"] else 0

    with open(os.path.join(args.output, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAnchors: {len(anchors)} from {stats['videos_valid']} videos")
    print(f"  Nonempty mask: {stats['anchors_nonempty_mask']}/{stats['anchors_total']} "
          f"({100*stats['mask_nonempty_ratio']:.1f}%)")
    print(f"  Mask area: mean={stats['mask_areas_mean']:.0f} median={stats['mask_areas_median']:.0f}")
    print(f"  Categories: {stats['categories']}")
    print(f"Saved to {args.output}")

    # Preview.
    if args.save_preview and len(anchors) > 0:
        _save_preview(anchors, annot_dir, video_dir, args.output)


def _save_preview(anchors, annot_dir, video_dir, output):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import random; random.seed(42)
    with_mask = [a for a in anchors if a["has_mask"]]
    without_mask = [a for a in anchors if not a["has_mask"]]
    selected = (random.sample(with_mask, min(25, len(with_mask))) +
                random.sample(without_mask, min(25, len(without_mask))))
    n = min(50, len(selected))
    n_cols = 5; n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))
    axes = axes.flatten() if n_rows*n_cols > 1 else [axes]
    for i in range(n):
        a = selected[i]
        mat_path = os.path.join(annot_dir, a["source_video"], f"{a['frame_id']:05d}.mat")
        try:
            with h5py.File(mat_path, "r") as f:
                mask = f["reMask"][a["actor_idx"]]
            axes[i].imshow(mask, cmap="Blues")
            axes[i].set_title(f"mask={a['mask_sum']:.0f} cat={a['category']}", fontsize=5)
        except Exception:
            axes[i].text(0.5, 0.5, "ERR", ha="center")
        axes[i].axis("off")
    for i in range(n, len(axes)): axes[i].axis("off")
    fig.tight_layout(pad=0.2); fig.savefig(os.path.join(output, "preview.png"), dpi=100); plt.close(fig)
    print(f"Preview saved")


if __name__ == "__main__":
    main()
