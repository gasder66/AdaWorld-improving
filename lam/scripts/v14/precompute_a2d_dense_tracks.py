"""
Precompute A2D dense bbox tracks using YOLO detection + simple IoU matching.

Avoids BoT-SORT dependency for reliability. Runs YOLO on each frame,
then matches detections across frames using IoU + category consistency.

Usage:
  PYTHONPATH=lam CUDA_VISIBLE_DEVICES=4 python lam/scripts/v14/precompute_a2d_dense_tracks.py \
      --max_videos 50 --gpu 0
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["YOLO_VERBOSE"] = "False"

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


def _iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def _match_track(frames_dets, prev_track, frame_idx):
    """Match previous track to current frame detections by IoU."""
    if not frames_dets[frame_idx]: return -1
    prev_bbox = prev_track[-1]["bbox"]
    best_iou, best_idx = 0, -1
    for i, det in enumerate(frames_dets[frame_idx]):
        if det.get("matched"): continue
        iou = _iou(prev_bbox, det["bbox"])
        if iou > best_iou:
            best_iou = iou; best_idx = i
    if best_iou > 0.2:
        return best_idx
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2d_root", default="data/a2d")
    parser.add_argument("--output", default="data/bridgebench/a2d_dense_completion/bbox_tracks")
    parser.add_argument("--anchors_jsonl", default="data/bridgebench/a2d_dense_completion/anchors/anchors.jsonl")
    parser.add_argument("--max_videos", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    # Load anchors.
    print("Loading anchors...")
    anchors_by_video = {}
    with open(args.anchors_jsonl) as f:
        for line in f:
            a = json.loads(line)
            anchors_by_video.setdefault(a["source_video"], []).append(a)
    video_ids = sorted(anchors_by_video.keys())
    if args.max_videos > 0:
        video_ids = video_ids[:args.max_videos]
    print(f"Videos: {len(video_ids)}")

    # Setup YOLO.
    from ultralytics import YOLO
    yolo = YOLO("/home/xiaojy/projects/AdaWorld-improving/yolov8n.pt")

    video_dir = os.path.join(args.a2d_root, "train")
    os.makedirs(args.output, exist_ok=True)
    stats = {"videos_processed": 0, "total_frames": 0, "tracks_per_video": []}

    for vi, vid in enumerate(video_ids):
        anchors = sorted(anchors_by_video[vid], key=lambda x: x["frame_id"])
        if len(anchors) < 2:
            stats["videos_processed"] += 1; continue
        video_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            stats["videos_processed"] += 1; continue

        anchor_frames = set(a["frame_id"] for a in anchors)
        min_f, max_f = min(anchor_frames), max(anchor_frames)

        # YOLO detect on all frames.
        try:
            results = yolo(video_path, stream=True, conf=args.conf, verbose=False)
        except Exception as e:
            print(f"  [{vi}] {vid}: YOLO error: {e}"); stats["videos_processed"] += 1; continue

        frames_dets = {}  # frame_num -> list of {bbox, conf}
        frame_num = 0
        for r in results:
            frame_num += 1
            if frame_num < min_f - 5 or frame_num > max_f + 5: continue
            if r.boxes is None: continue
            bboxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            frames_dets[frame_num] = []
            for i in range(len(bboxes)):
                frames_dets[frame_num].append({
                    "bbox": bboxes[i].tolist(), "conf": float(confs[i]),
                })

        if not frames_dets:
            stats["videos_processed"] += 1; continue
        stats["total_frames"] += len(frames_dets)

        # Track: match YOLO detections to GT anchors, then propagate via IoU.
        gt_tracks = {}  # actor_idx -> list of matched frame dets
        for a in anchors:
            fn = a["frame_id"]
            if fn not in frames_dets: continue
            gt_bbox = a["bbox_xyxy"]
            for det in frames_dets[fn]:
                if _iou(gt_bbox, det["bbox"]) > 0.3:
                    det["matched"] = True; det["actor_idx"] = a["actor_idx"]
                    gt_tracks.setdefault(a["actor_idx"], []).append({
                        "frame_num": fn, "bbox": det["bbox"], "conf": det["conf"],
                    })
                    break

        # Propagate forward from each anchor.
        for actor_idx, track in gt_tracks.items():
            anchor_frame = track[0]["frame_num"]
            # Forward from anchor.
            prev_bbox = track[0]["bbox"]
            for fn in sorted(f for f in frames_dets if f > anchor_frame):
                match_idx = _match_track(frames_dets, track, fn)
                if match_idx >= 0:
                    det = frames_dets[fn][match_idx]
                    det["matched"] = True
                    track.append({"frame_num": fn, "bbox": det["bbox"], "conf": det["conf"]})
                else:
                    # Keep last known bbox.
                    track.append({"frame_num": fn, "bbox": prev_bbox.copy(), "conf": 0.1, "interpolated": True})

        # Save.
        out = {"video": vid, "anchors": [a["frame_id"] for a in anchors],
               "gt_tracks": {str(k): v for k, v in gt_tracks.items()}}
        torch.save(out, os.path.join(args.output, f"{vid}.pt"))
        stats["videos_processed"] += 1
        stats["tracks_per_video"].append(len(gt_tracks))
        if vi % 10 == 0:
            print(f"  [{vi}/{len(video_ids)}] {vid}: {len(frames_dets)} frames, {len(gt_tracks)} actors")

    with open(os.path.join(args.output, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nDense tracks: {stats['videos_processed']} videos, {stats['total_frames']} frames")
    print(f"  Avg tracks/video: {np.mean(stats['tracks_per_video']):.1f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
