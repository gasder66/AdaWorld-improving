"""
Extract real A2D object tracks for Bridge-3 real tube pilot.

Stride support: stride=1/2/4 → different temporal spacing.
Source-disjoint train/val split.
Outputs per-stride clips to {output}/stride{N}/{train,val}/.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/extract_a2d_tubes.py \
      --a2d_root data/a2d --release_root Release \
      --output data/bridgebench/bridge3_real_tube_pilot \
      --max_clips 500 --T 5 --stride 1,2,4 \
      --train_ratio 0.8 --min_mask_area 20 --min_displacement 1
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.a2d_dataset import parse_a2d_label, ACTOR_NAMES

MARGIN = 2


def _classify_delta(dcx, dcy, margin=MARGIN):
    ax, ay = abs(dcx), abs(dcy)
    if ax < margin and ay < margin: return 0
    return 1 if (ay > ax and dcy < 0) else (2 if (ay > ax and dcy > 0) else (3 if dcx < 0 else 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2d_root", default="data/a2d")
    parser.add_argument("--release_root", default="Release")
    parser.add_argument("--output", default="data/bridgebench/bridge3_real_tube_pilot")
    parser.add_argument("--max_clips", type=int, default=500)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--stride", type=str, default="1,2,4")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--min_mask_area", type=float, default=20)
    parser.add_argument("--min_displacement", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_preview", type=int, default=1)
    args = parser.parse_args()

    strides = [int(s) for s in args.stride.split(",")]
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
    all_vids = rng.permutation([v for v in sorted(os.listdir(annot_dir)) if v in train_videos])
    n_train_vids = max(1, int(len(all_vids) * args.train_ratio))
    train_vids_set = set(all_vids[:n_train_vids])
    test_vids_set = set(all_vids[n_train_vids:])
    print(f"Source split: train={len(train_vids_set)} test={len(test_vids_set)}")

    # Collect clips for all strides.
    clips_by_stride = {s: [] for s in strides}
    stats = {"total_frames_processed": 0, "tracks_attempted": 0, "valid_tracks": 0}
    stats["categories"] = {}
    stats["pseudo_actions"] = {str(s): {str(a): 0 for a in range(5)} for s in strides}
    stats["source_split"] = {"train_videos": len(train_vids_set), "test_videos": len(test_vids_set)}
    stats["strides"] = {}

    for vi, vid in enumerate(all_vids):
        all_done = all(len(clips_by_stride[s]) >= args.max_clips for s in strides)
        if all_done: break

        split_label = "train" if vid in train_vids_set else "val"
        mat_dir = os.path.join(annot_dir, vid)
        mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith(".mat")])
        if len(mat_files) < args.T: continue  # need at least T frames

        video_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path): continue

        frame_nums = [int(mf.replace(".mat", "")) for mf in mat_files]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release(); continue

        try:
            # Load all annotations.
            annotations = {}
            for mf in mat_files:
                try:
                    with h5py.File(os.path.join(mat_dir, mf), "r") as f:
                        N = f["reBBox"].shape[1]
                        annotations[mf] = {"bboxes": f["reBBox"][:], "ids": f["id"][:].flatten().astype(int)}
                except: continue

            if not annotations: continue

            # For each primary actor (slot 0), extract clips.
            for actor in range(min(3, N)):
                stats["tracks_attempted"] += 1
                if all_done: break

                # Check first frame actor quality.
                mf0 = mat_files[0]
                if actor >= annotations[mf0]["bboxes"].shape[1]: continue
                # Mask check (from first mat file).
                try:
                    with h5py.File(os.path.join(mat_dir, mf0), "r") as f:
                        mask_area = f["reMask"][actor].sum()
                except: mask_area = 0
                if mask_area < args.min_mask_area: continue

                # For each stride, try to extract clips.
                for stride in strides:
                    if len(clips_by_stride[stride]) >= args.max_clips: continue
                    need_frames = (args.T - 1) * stride + 1
                    # Slide window over mat files.
                    for start_i in range(0, len(mat_files) - need_frames + 1):
                        if len(clips_by_stride[stride]) >= args.max_clips: break
                        selected = [start_i + i * stride for i in range(args.T)]
                        if max(selected) >= len(mat_files): break

                        frames = []
                        skip = False
                        for fi, idx in enumerate(selected):
                            mf = mat_files[idx]
                            if mf not in annotations or actor >= annotations[mf]["bboxes"].shape[1]:
                                skip = True; break
                            bbox = annotations[mf]["bboxes"][:, actor]
                            ids = annotations[mf]["ids"]
                            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_nums[idx] - 1))
                            ret, frame = cap.read()
                            if not ret: skip = True; break
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append({"frame": frame_rgb, "bbox": bbox, "frame_num": frame_nums[idx]})
                        if skip: continue

                        # Compute pseudo actions.
                        pts = []
                        for f in frames:
                            pts.append([(f["bbox"][0]+f["bbox"][2])/2, (f["bbox"][1]+f["bbox"][3])/2])
                        centers = np.array(pts)
                        deltas = centers[1:] - centers[:-1]
                        disp = np.linalg.norm(deltas.sum(axis=0))
                        if disp < args.min_displacement: continue

                        H_orig, W_orig = frames[0]["frame"].shape[:2]
                        sx = args.image_size / W_orig; sy = args.image_size / H_orig

                        # Build sample with K_max=4, K=1 active.
                        K_max = 4
                        video_t = torch.zeros(args.T, 3, args.image_size, args.image_size)
                        masks_t = torch.zeros(args.T, K_max, args.image_size, args.image_size)
                        boxes_t = torch.zeros(args.T, K_max, 4)
                        actions_t = torch.full((args.T-1, K_max), -1, dtype=torch.long)
                        valid_t = torch.zeros(args.T, K_max, dtype=torch.bool)
                        valid_t[:, 0] = True

                        for t in range(args.T):
                            f = frames[t]
                            frame_resized = cv2.resize(f["frame"], (args.image_size, args.image_size))
                            video_t[t] = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
                            x1, y1, x2, y2 = f["bbox"]
                            cx = (x1+x2)/2*sx/args.image_size
                            cy = (y1+y2)/2*sy/args.image_size
                            w = (x2-x1)*sx/args.image_size
                            h = (y2-y1)*sy/args.image_size
                            boxes_t[t, 0] = torch.tensor([cx, cy, w, h])
                            ix1 = max(0, int(x1*sx)); ix2 = min(args.image_size, int(x2*sx+1))
                            iy1 = max(0, int(y1*sy)); iy2 = min(args.image_size, int(y2*sy+1))
                            if ix2 > ix1 and iy2 > iy1:
                                masks_t[t, 0, iy1:iy2, ix1:ix2] = 1.0
                        for t in range(args.T-1):
                            dcx = centers[t+1,0]-centers[t,0]; dcy = centers[t+1,1]-centers[t,1]
                            actions_t[t, 0] = _classify_delta(dcx, dcy)

                        actor_id, _ = parse_a2d_label(int(annotations[mf0]["ids"][actor]))
                        cat = ACTOR_NAMES.get(actor_id, "unknown")

                        clip = {
                            "video": video_t, "masks": masks_t,
                            "visible_masks": masks_t, "full_masks": masks_t,
                            "boxes": boxes_t, "visible_boxes": boxes_t,
                            "valid": valid_t, "visible_valid": valid_t,
                            "actions": actions_t,
                            "actor_id": torch.tensor([0]*K_max),
                            "category": torch.full((K_max,), actor_id if actor_id else 0, dtype=torch.long),
                            "depth_order": torch.zeros(args.T, K_max, dtype=torch.long),
                            "occlusion_ratio": torch.zeros(args.T, K_max),
                            "is_occluded": torch.zeros(args.T, K_max, dtype=torch.bool),
                            "temporal_stride": stride,
                            "source_video": vid,
                            "frame_ids": torch.tensor([frame_nums[i] for i in selected]),
                            "split": split_label,
                            "metadata": {"task_name": "Bridge3-RealTube", "source_video": vid, "temporal_stride": stride},
                        }
                        clips_by_stride[stride].append(clip)
                        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
                        stats["pseudo_actions"][str(stride)][str(int(actions_t[t,0]))] = \
                            stats["pseudo_actions"][str(stride)].get(str(int(actions_t[t,0])), 0) + 1

        finally:
            cap.release()
        if vi % 50 == 0:
            counts = {s: len(clips_by_stride[s]) for s in strides}
            print(f"  [{vi}] {vid[:10]} → {counts}")

    # Save clips per stride.
    for stride in strides:
        clips = clips_by_stride[stride]
        stride_dir = os.path.join(args.output, f"stride{stride}")
        train_count = 0; val_count = 0
        for clip in clips:
            sp = clip.pop("split", "train")
            out_dir = os.path.join(stride_dir, sp)
            os.makedirs(out_dir, exist_ok=True)
            idx = train_count if sp == "train" else val_count
            torch.save(clip, os.path.join(out_dir, f"sample_{idx:06d}.pt"))
            if sp == "train": train_count += 1
            else: val_count += 1
        stats["strides"][str(stride)] = {"total": len(clips), "train": train_count, "val": val_count}
        print(f"  stride{stride}: {len(clips)} clips ({train_count} train, {val_count} val)")

    # Preview.
    if args.output_preview and any(len(clips_by_stride[s]) > 0 for s in strides):
        _save_preview(clips_by_stride, args.output, args.image_size)

    stats_path = os.path.join(args.output, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats → {stats_path}")


def _save_preview(clips_by_stride, output, image_size):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for stride, clips in clips_by_stride.items():
        if len(clips) < 2: continue
        n_cols = 5; n_rows = min(4, len(clips))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))
        if n_rows == 1: axes = axes[np.newaxis, :]
        for i in range(n_rows):
            c = clips[i]; t0 = 0
            rgb_0 = c["video"][t0].permute(1,2,0).cpu().numpy()
            rgb_1 = c["video"][t0+1].permute(1,2,0).cpu().numpy()
            mask_0 = c["masks"][t0,0].cpu().numpy()
            mask_1 = c["masks"][t0+1,0].cpu().numpy()
            act = int(c["actions"][t0,0])
            names = {0:"stay",1:"up",2:"down",3:"left",4:"right"}
            panels = [rgb_0, rgb_1, mask_0, mask_1,
                      np.zeros_like(rgb_0)]  # placeholder for text
            titles = ["frame_t", "frame_t+1", "mask_t", "mask_t+1", f"act={names.get(act,'?')}"]
            for j, (img, title) in enumerate(zip(panels, titles)):
                if img.ndim == 3: axes[i,j].imshow(img)
                else: axes[i,j].imshow(img, cmap="Blues")
                axes[i,j].set_title(title, fontsize=6); axes[i,j].axis("off")
        fig.tight_layout(pad=0.3); fig.savefig(os.path.join(output, f"preview_stride{stride}.png"), dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
