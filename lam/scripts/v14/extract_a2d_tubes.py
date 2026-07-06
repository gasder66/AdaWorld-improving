"""
Extract real A2D object tracks for Bridge-3 pilot.

Reads A2D MP4 + h5py annotations, finds valid tracks, extracts T=5 clips.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/extract_a2d_tubes.py \
      --a2d_root data/a2d --release_root Release \
      --output data/bridgebench/bridge3_real_tube_pilot \
      --max_clips 2000 --T 5 --stride 2 --min_mask_area 50
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.a2d_dataset import parse_a2d_label, ACTOR_NAMES

ACTION_DELTA = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
MARGIN = 3  # pixel threshold for pseudo-action direction


def _classify_delta(dcx, dcy, margin=MARGIN):
    ax, ay = abs(dcx), abs(dcy)
    if ax < margin and ay < margin: return 0
    return 1 if (ay > ax and dcy < 0) else (2 if (ay > ax and dcy > 0) else (3 if dcx < 0 else 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2d_root", default="data/a2d")
    parser.add_argument("--release_root", default="Release")
    parser.add_argument("--output", default="data/bridgebench/bridge3_real_tube_pilot")
    parser.add_argument("--max_clips", type=int, default=2000)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--min_mask_area", type=float, default=50)
    parser.add_argument("--min_displacement", type=float, default=2)
    parser.add_argument("--seed", type=int, default=42)
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
            if int(p[8]) == 0: train_videos.append(p[0])
    print(f"Train videos: {len(train_videos)}")

    rng = np.random.RandomState(args.seed)
    video_dirs = rng.permutation([v for v in sorted(os.listdir(annot_dir)) if v in train_videos])

    os.makedirs(args.output, exist_ok=True)
    clips = []
    stats = {"total_tracks": 0, "valid_tracks": 0, "clips_saved": 0,
             "pseudo_actions": {a: 0 for a in range(5)},
             "categories": {}}

    for vi, vid in enumerate(video_dirs):
        if len(clips) >= args.max_clips: break
        mat_dir = os.path.join(annot_dir, vid)
        mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith(".mat")])
        if len(mat_files) < args.T: continue

        video_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path): continue

        frame_nums = [int(mf.replace(".mat", "")) for mf in mat_files]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release(); continue

        try:
            for start_i in range(0, len(mat_files) - args.T * args.stride + 1):
                if len(clips) >= args.max_clips: break
                selected = [start_i + i * args.stride for i in range(args.T)]
                selected_frames = [frame_nums[i] for i in selected]
                selected_mats = [mat_files[i] for i in selected]

                # Load annotations for first frame to get actor count.
                try:
                    with h5py.File(os.path.join(mat_dir, selected_mats[0]), "r") as f:
                        N = f["reBBox"].shape[1]
                except: continue

                if N < 1: continue

                # Focus on the primary actor (first track).
                primary_actor = 0

                # Try to read all T frames for this track.
                frames = []
                skip = False
                for fi, mf in enumerate(selected_mats):
                    frame_num = selected_frames[fi]
                    try:
                        with h5py.File(os.path.join(mat_dir, mf), "r") as f:
                            bboxes = f["reBBox"][:]
                            id_raw = f["id"][:].flatten().astype(int)
                    except:
                        skip = True; break

                    if primary_actor >= bboxes.shape[1]:
                        skip = True; break

                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_num - 1))
                    ret, frame = cap.read()
                    if not ret:
                        skip = True; break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (args.image_size, args.image_size))
                    bbox = bboxes[:, primary_actor]
                    frames.append({
                        "frame": frame_resized,
                        "bbox": bbox,
                        "frame_num": frame_num,
                    })

                if skip or len(frames) < args.T: continue

                # Check mask quality (for first frame).
                try:
                    with h5py.File(os.path.join(mat_dir, selected_mats[0]), "r") as f:
                        mask = f["reMask"][:]
                        mask_area = mask[primary_actor].sum()
                except:
                    mask_area = 0
                if mask_area < args.min_mask_area: continue

                # Compute pseudo actions and bbox stats.
                bboxes_pix = np.array([f["bbox"] for f in frames])
                centers = np.stack([(bboxes_pix[:, 0] + bboxes_pix[:, 2]) / 2,
                                     (bboxes_pix[:, 1] + bboxes_pix[:, 3]) / 2], axis=-1)
                deltas = centers[1:] - centers[:-1]

                total_disp = np.linalg.norm(deltas.sum(axis=0))
                if total_disp < args.min_displacement: continue

                # Build sample.
                H_orig = frames[0]["frame"].shape[0]; W_orig = frames[0]["frame"].shape[1]
                video = torch.stack([torch.from_numpy(f["frame"]).float().permute(2, 0, 1) / 255.0
                                      for f in frames], dim=0)
                K = 1  # Single track per sample for Bridge-3 pilot
                masks_t = torch.zeros(args.T, K, args.image_size, args.image_size)
                boxes_t = torch.zeros(args.T, K, 4)
                actions_t = torch.full((args.T - 1, K,), -1, dtype=torch.long)

                for t in range(args.T):
                    bbox = bboxes_pix[t]
                    x1, y1, x2, y2 = bbox
                    sx = args.image_size / W_orig; sy = args.image_size / H_orig
                    cx = (x1 + x2) / 2 * sx / args.image_size
                    cy = (y1 + y2) / 2 * sy / args.image_size
                    w = (x2 - x1) * sx / args.image_size
                    h = (y2 - y1) * sy / args.image_size
                    boxes_t[t, 0] = torch.tensor([cx, cy, w, h])
                    # Create bbox-based mask.
                    ix1 = max(0, int(x1 * sx)); iy1 = max(0, int(y1 * sy))
                    ix2 = min(args.image_size, int(x2 * sx + 1))
                    iy2 = min(args.image_size, int(y2 * sy + 1))
                    if ix2 > ix1 and iy2 > iy1:
                        masks_t[t, 0, iy1:iy2, ix1:ix2] = 1.0

                for t in range(args.T - 1):
                    dcx = centers[t + 1, 0] - centers[t, 0]
                    dcy = centers[t + 1, 1] - centers[t, 1]
                    actions_t[t, 0] = _classify_delta(dcx, dcy)

                actor_id, _ = parse_a2d_label(int(id_raw[primary_actor]))
                category = ACTOR_NAMES.get(actor_id, "unknown")

                sample = {
                    "video": video,
                    "masks": masks_t,
                    "visible_masks": masks_t,
                    "full_masks": masks_t,
                    "boxes": boxes_t,
                    "visible_boxes": boxes_t,
                    "valid": torch.ones(args.T, K, dtype=torch.bool),
                    "visible_valid": torch.ones(args.T, K, dtype=torch.bool),
                    "actions": actions_t,
                    "actor_id": torch.tensor([primary_actor]),
                    "category": torch.tensor([actor_id] if actor_id else [0]),
                    "depth_order": torch.zeros(args.T, K, dtype=torch.long),
                    "occlusion_ratio": torch.zeros(args.T, K),
                    "is_occluded": torch.zeros(args.T, K, dtype=torch.bool),
                    "is_heavy_occluded": torch.zeros(args.T, K, dtype=torch.bool),
                    "track_id": torch.tensor([0]),
                    "source_video": vid,
                    "frame_ids": torch.tensor(selected_frames),
                    "pseudo_action_conf": torch.ones(args.T - 1, K),
                    "metadata": {"task_name": "Bridge3-RealTube", "track_id": 0, "source_video": vid},
                }
                clips.append(sample)
                stats["clips_saved"] += 1
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
                for t in range(args.T - 1):
                    stats["pseudo_actions"][int(actions_t[t, 0])] += 1

        finally:
            cap.release()

        if vi % 50 == 0:
            print(f"  [{vi}] {len(clips)} clips from {vid[:10]}...")

    # Save.
    out_dir = os.path.join(args.output, "train")
    os.makedirs(out_dir, exist_ok=True)
    for i, clip in enumerate(clips):
        torch.save(clip, os.path.join(out_dir, f"sample_{i:06d}.pt"))

    print(f"\nSaved {len(clips)} clips to {out_dir}")
    print(f"Categories: {stats['categories']}")
    print(f"Pseudo-actions: {stats['pseudo_actions']}")

    stats_path = os.path.join(args.output, "stats.json")
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats → {stats_path}")


if __name__ == "__main__":
    main()
