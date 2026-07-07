"""
Build dense Bridge-3 dataset from YOLO+IoU dense tracks.

Samples T=5 clips from dense tracked frames between GT anchors.
Outputs BridgeBench-format .pt files with dense-per-frame bbox/mask.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/build_dense_bridge3_dataset.py \
      --anchors_jsonl data/bridgebench/a2d_dense_completion/anchors/anchors.jsonl \
      --tracks_dir data/bridgebench/a2d_dense_completion/bbox_tracks \
      --output data/bridgebench/bridge3_dense_real_tubes \
      --max_clips 2000 --train_ratio 0.8 --T 5
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.a2d_dataset import parse_a2d_label, ACTOR_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors_jsonl", default="data/bridgebench/a2d_dense_completion/anchors/anchors.jsonl")
    parser.add_argument("--tracks_dir", default="data/bridgebench/a2d_dense_completion/bbox_tracks")
    parser.add_argument("--output", default="data/bridgebench/bridge3_dense_real_tubes")
    parser.add_argument("--a2d_root", default="data/a2d")
    parser.add_argument("--max_clips", type=int, default=2000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load anchors.
    anchors_by_video = {}
    with open(args.anchors_jsonl) as f:
        for line in f:
            a = json.loads(line)
            anchors_by_video.setdefault(a["source_video"], []).append(a)

    rng = np.random.RandomState(args.seed)
    video_ids = sorted(anchors_by_video.keys())
    n_train_vids = int(len(video_ids) * args.train_ratio)
    rng.shuffle(video_ids)
    train_vids = set(video_ids[:n_train_vids])

    video_dir = os.path.join(args.a2d_root, "train")
    clips = []
    stats = {"clips_total": 0, "clips_train": 0, "clips_val": 0, "videos_used": 0}

    for vid in video_ids:
        if len(clips) >= args.max_clips: break
        track_file = os.path.join(args.tracks_dir, f"{vid}.pt")
        if not os.path.exists(track_file): continue
        track = torch.load(track_file, map_location="cpu", weights_only=False)
        gt_tracks = track.get("gt_tracks", {})
        anchors = sorted(anchors_by_video[vid], key=lambda x: x["frame_id"])
        if len(anchors) < 2: continue

        video_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path): continue

        # Get all tracked frames for this video.
        all_frames = set()
        for actor_key, dets in gt_tracks.items():
            for d in dets:
                all_frames.add(d["frame_num"])
        all_frames = sorted(all_frames)
        if len(all_frames) < args.T: continue

        is_train = vid in train_vids
        stats["videos_used"] += 1

        # Pre-load video frames (only the ones we need).
        cap = cv2.VideoCapture(video_path)
        frame_cache = {}
        for fn in all_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fn - 1))
            ret, frame = cap.read()
            if ret:
                frame_cache[fn] = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    (args.image_size, args.image_size))
        cap.release()
        if len(frame_cache) < args.T: continue

        # Get anchor bboxes for actor mapping.
        anchor_map = {}
        for a in anchors:
            anchor_map[a["frame_id"]] = a

        # Sample T=5 clips from consecutive tracked frames.
        for start_i in range(0, len(all_frames) - args.T + 1):
            if len(clips) >= args.max_clips: break
            selected = all_frames[start_i:start_i + args.T]
            if not all(fn in frame_cache for fn in selected): continue

            # Build clip.
            K_max = 4
            video_t = torch.zeros(args.T, 3, args.image_size, args.image_size)
            boxes_t = torch.zeros(args.T, K_max, 4)
            valid_t = torch.zeros(args.T, K_max, dtype=torch.bool)
            masks_t = torch.zeros(args.T, K_max, args.image_size, args.image_size)
            actions_t = torch.full((args.T-1, K_max), -1, dtype=torch.long)

            # Use first available actor track.
            primary_actor = None
            for ak in gt_tracks:
                primary_actor = int(ak)
                break
            if primary_actor is None: continue

            track_dets = {d["frame_num"]: d for d in gt_tracks[str(primary_actor)]}

            for t, fn in enumerate(selected):
                video_t[t] = torch.from_numpy(frame_cache[fn]).float().permute(2, 0, 1) / 255.0
                if fn in track_dets:
                    bbox = track_dets[fn]["bbox"]
                    H, W = frame_cache[fn].shape[:2]
                    cx = (bbox[0]+bbox[2])/2/W*args.image_size/args.image_size
                    cy = (bbox[1]+bbox[3])/2/H*args.image_size/args.image_size
                    w = (bbox[2]-bbox[0])/W*args.image_size/args.image_size
                    h = (bbox[3]-bbox[1])/H*args.image_size/args.image_size
                    boxes_t[t, 0] = torch.tensor([cx, cy, w, h])
                    x1 = max(0, int(bbox[0]*args.image_size/W))
                    x2 = min(args.image_size, int(bbox[2]*args.image_size/W+1))
                    y1 = max(0, int(bbox[1]*args.image_size/H))
                    y2 = min(args.image_size, int(bbox[3]*args.image_size/H+1))
                    if x2 > x1 and y2 > y1:
                        masks_t[t, 0, y1:y2, x1:x2] = 1.0
                    valid_t[t, 0] = True

            # Pseudo actions from bbox center displacement.
            for t in range(args.T - 1):
                if valid_t[t, 0] and valid_t[t+1, 0]:
                    dcx = boxes_t[t+1,0,0] - boxes_t[t,0,0]
                    dcy = boxes_t[t+1,0,1] - boxes_t[t,0,1]
                    ax, ay = abs(dcx.item()), abs(dcy.item())
                    if ax < 0.005 and ay < 0.005: act = 0
                    elif ay > ax: act = 1 if dcy < 0 else 2
                    else: act = 3 if dcx < 0 else 4
                    actions_t[t, 0] = act

            # Get category from anchors.
            category_id = 0
            for a in anchors:
                actor_id, _ = parse_a2d_label(0)
                category_id = actor_id
                break

            clip = {
                "video": video_t, "masks": masks_t, "visible_masks": masks_t,
                "full_masks": masks_t, "boxes": boxes_t, "visible_boxes": boxes_t,
                "valid": valid_t, "visible_valid": valid_t,
                "actions": actions_t,
                "actor_id": torch.zeros(K_max, dtype=torch.long),
                "category": torch.full((K_max,), category_id, dtype=torch.long),
                "depth_order": torch.zeros(args.T, K_max, dtype=torch.long),
                "occlusion_ratio": torch.zeros(args.T, K_max),
                "is_occluded": torch.zeros(args.T, K_max, dtype=torch.bool),
                "temporal_stride": 1, "source_video": vid,
                "frame_ids": torch.tensor(selected),
            }
            clips.append(clip)
            if is_train: stats["clips_train"] += 1
            else: stats["clips_val"] += 1
            stats["clips_total"] += 1

    # Save.
    train_count, val_count = 0, 0
    for clip in clips:
        is_train = clip["source_video"] in train_vids
        sp = "train" if is_train else "val"
        out_dir = os.path.join(args.output, sp)
        os.makedirs(out_dir, exist_ok=True)
        idx = train_count if is_train else val_count
        torch.save(clip, os.path.join(out_dir, f"sample_{idx:06d}.pt"))
        if is_train: train_count += 1
        else: val_count += 1

    print(f"\nDense Bridge-3 dataset:")
    print(f"  Videos used: {stats['videos_used']}")
    print(f"  Clips: {stats['clips_total']} ({stats['clips_train']} train, {stats['clips_val']} val)")
    print(f"  Saved to {args.output}")

    with open(os.path.join(args.output, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
