"""
Build object bank from A2D annotations — extract real object crops + masks.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/build_object_bank.py \
      --a2d_root data/a2d --release_root Release \
      --output data/bridgebench/object_bank.pt \
      --max_objects 2000 --crop_size 96 --min_area 64
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.a2d_dataset import parse_a2d_label

ACTOR_NAMES = {1: "adult", 2: "baby", 3: "ball", 4: "bird", 5: "car", 6: "cat", 7: "dog"}


def _ascii_to_str(ascii_bytes):
    """Convert uint16 array of ASCII codes to string."""
    return "".join(chr(c) for c in ascii_bytes if c > 0)


def _crop_bbox(frame, mask, bbox, crop_size):
    """Crop frame and mask around bbox, resize to crop_size."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(W, int(x2 + 1))
    y2 = min(H, int(y2 + 1))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, None

    crop_img = frame[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    if crop_img.size == 0 or crop_mask.size == 0:
        return None, None

    crop_img = cv2.resize(crop_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    crop_mask = cv2.resize(crop_mask.astype(np.float32).astype(np.uint8),
                           (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    crop_img = torch.from_numpy(crop_img).float().permute(2, 0, 1) / 255.0
    crop_mask = torch.from_numpy(crop_mask).float().unsqueeze(0)
    return crop_img, crop_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2d_root", default="data/a2d")
    parser.add_argument("--release_root", default="Release")
    parser.add_argument("--output", default="data/bridgebench/object_bank.pt")
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--max_objects", type=int, default=2000)
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--max_per_video", type=int, default=5)
    args = parser.parse_args()

    annot_dir = os.path.join(args.release_root, "Annotations", "mat")
    video_dir = os.path.join(args.a2d_root, "train")
    csv_path = os.path.join(args.release_root, "videoset.csv")

    if not os.path.isdir(annot_dir):
        print(f"ERROR: annotation dir not found: {annot_dir}")
        sys.exit(1)

    # Read videoset.csv for train split.
    train_videos = []
    with open(csv_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            vid, usage = parts[0], int(parts[8])
            if usage == 0:  # train
                train_videos.append(vid)

    print(f"Train videos: {len(train_videos)}")

    objects = []
    video_dirs = sorted(os.listdir(annot_dir))
    rng = np.random.RandomState(42)
    video_dirs = rng.permutation([v for v in video_dirs if v in train_videos])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for vi, vid in enumerate(video_dirs):
        if len(objects) >= args.max_objects:
            break

        mat_dir = os.path.join(annot_dir, vid)
        mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith(".mat")])
        if not mat_files:
            continue

        video_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            continue

        # Open video once.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            continue

        # Process up to max_per_video frames.
        selected = rng.choice(mat_files, size=min(args.max_per_video, len(mat_files)),
                              replace=False)

        for mf in selected:
            if len(objects) >= args.max_objects:
                break
            frame_num = int(mf.replace(".mat", ""))

            try:
                with h5py.File(os.path.join(mat_dir, mf), "r") as f:
                    bboxes = f["reBBox"][:]  # (4, N)
                    masks = f["reMask"][:]   # (N, H, W)
                    id_raw = f["id"][:].flatten().astype(int)  # (N,)
                    # class is stored as HDF5 references; use id to infer actor.
                    # Parse actor_id from the id field (tens digit).
            except Exception as e:
                if vi < 3:
                    print(f"  Warning: {vid}/{mf}: {e}")
                continue

            N = bboxes.shape[1]
            if N == 0:
                continue

            # Read frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # A2D frames are 1-indexed
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for n in range(N):
                if len(objects) >= args.max_objects:
                    break

                bbox = bboxes[:, n]  # [x1, y1, x2, y2]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < args.min_area:
                    continue

                mask = masks[n] if masks.ndim == 3 else masks  # (H, W) or (N, H, W)
                if mask.ndim != 2:
                    continue

                crop_img, crop_mask = _crop_bbox(frame_rgb, mask, bbox, args.crop_size)
                if crop_img is None:
                    continue

                # Get category from actor_id in the id label.
                actor_id, _ = parse_a2d_label(id_raw[n])
                category = ACTOR_NAMES.get(actor_id, "unknown")

                # Normalize bbox to cxcywh [0,1].
                H, W = frame_rgb.shape[:2]
                cx = (bbox[0] + bbox[2]) / 2 / W
                cy = (bbox[1] + bbox[3]) / 2 / H
                w = (bbox[2] - bbox[0]) / W
                h = (bbox[3] - bbox[1]) / H

                objects.append({
                    "image_crop": crop_img,
                    "mask_crop": crop_mask,
                    "bbox": torch.tensor([cx, cy, w, h]),
                    "category": category,
                    "source_video": vid,
                    "frame_id": frame_num,
                    "instance_id": n,
                })

        cap.release()
        if vi % 100 == 0:
            print(f"  [{vi}/{len(video_dirs)}] {len(objects)} objects collected")

    print(f"\nObject bank: {len(objects)} objects")
    cats = {}
    for o in objects:
        cats[o["category"]] = cats.get(o["category"], 0) + 1
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")

    torch.save(objects, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
