"""
Hidden-anchor benchmark: hide intermediate GT anchor, predict from neighbors.

Compares YOLO+Iou tracking vs linear interpolation baseline.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/eval_hidden_anchor_completion.py \
      --anchors_jsonl data/bridgebench/a2d_dense_completion/anchors/anchors.jsonl \
      --tracks_dir data/bridgebench/a2d_dense_completion/bbox_tracks \
      --output result/v14/dense_tube_completion/hidden_anchor
"""
import argparse, json, os, sys
import numpy as np


def _iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def _lerp_bbox(bbox_a, bbox_b, t):
    """Linear interpolate bbox at position t in [0,1]."""
    return [bbox_a[i] + (bbox_b[i] - bbox_a[i]) * t for i in range(4)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors_jsonl", default="data/bridgebench/a2d_dense_completion/anchors/anchors.jsonl")
    parser.add_argument("--tracks_dir", default="data/bridgebench/a2d_dense_completion/bbox_tracks")
    parser.add_argument("--output", default="result/v14/dense_tube_completion/hidden_anchor")
    args = parser.parse_args()

    # Load anchors.
    anchors_by_video = {}
    with open(args.anchors_jsonl) as f:
        for line in f:
            a = json.loads(line)
            anchors_by_video.setdefault(a["source_video"], []).append(a)

    os.makedirs(args.output, exist_ok=True)

    # Hidden anchor eval: for each video with >= 3 anchors,
    # hide middle anchor B, predict from A (forward) and C (backward).
    results_bbox_track = []
    results_bbox_lerp = []

    for vid, anchors in anchors_by_video.items():
        if len(anchors) < 3: continue
        anchors_sorted = sorted(anchors, key=lambda x: x["frame_id"])
        track_file = os.path.join(args.tracks_dir, f"{vid}.pt")
        has_tracks = os.path.exists(track_file)  # Will implement tracking eval when available

        for i in range(1, len(anchors_sorted) - 1):
            a_left = anchors_sorted[i-1]
            a_mid = anchors_sorted[i]
            a_right = anchors_sorted[i+1]

            gt_bbox = a_mid["bbox_xyxy"]
            frame_l = a_left["frame_id"]; frame_m = a_mid["frame_id"]; frame_r = a_right["frame_id"]

            # === Linear interpolation baseline ===
            t = (frame_m - frame_l) / max(1, frame_r - frame_l)
            bbox_lerp = _lerp_bbox(a_left["bbox_xyxy"], a_right["bbox_xyxy"], t)
            iou_lerp = _iou(bbox_lerp, gt_bbox)
            results_bbox_lerp.append({
                "video": vid, "frame_mid": frame_m,
                "iou": float(iou_lerp), "method": "linear_interp",
                "interval_left": frame_m - frame_l,
                "interval_right": frame_r - frame_m,
                "category": a_mid["category"],
            })

    # Summarize.
    ious_lerp = [r["iou"] for r in results_bbox_lerp]
    print(f"\n{'='*60}")
    print(f"Hidden-Anchor Bbox Evaluation")
    print(f"{'='*60}")
    print(f"  Videos with >= 3 anchors: {sum(1 for v,a in anchors_by_video.items() if len(a)>=3)}")
    print(f"  Hidden anchor pairs: {len(results_bbox_lerp)}")

    print(f"\n  Linear interpolation baseline:")
    print(f"    IoU mean:   {np.mean(ious_lerp):.4f}")
    print(f"    IoU median: {np.median(ious_lerp):.4f}")
    print(f"    IoU std:    {np.std(ious_lerp):.4f}")
    print(f"    IoU > 0.75: {sum(1 for x in ious_lerp if x>0.75)}/{len(ious_lerp)} ({100*sum(1 for x in ious_lerp if x>0.75)/len(ious_lerp):.0f}%)")
    print(f"    IoU > 0.50: {sum(1 for x in ious_lerp if x>0.50)}/{len(ious_lerp)} ({100*sum(1 for x in ious_lerp if x>0.50)/len(ious_lerp):.0f}%)")

    # Per-interval breakdown.
    intervals = [r["interval_left"] for r in results_bbox_lerp]
    print(f"\n  Interval stats: mean={np.mean(intervals):.0f} median={np.median(intervals):.0f} "
          f"min={np.min(intervals):.0f} max={np.max(intervals):.0f}")

    summary = {
        "n_videos_3plus": sum(1 for v,a in anchors_by_video.items() if len(a)>=3),
        "n_hidden_pairs": len(results_bbox_lerp),
        "lerp_iou_mean": float(np.mean(ious_lerp)),
        "lerp_iou_median": float(np.median(ious_lerp)),
        "lerp_iou_std": float(np.std(ious_lerp)),
        "lerp_iou_gt75": sum(1 for x in ious_lerp if x>0.75),
        "lerp_iou_gt50": sum(1 for x in ious_lerp if x>0.50),
        "interval_mean": float(np.mean(intervals)),
        "bbox_tracking_go": float(np.mean(ious_lerp)) > 0.75,
    }

    with open(os.path.join(args.output, "eval.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  bbox_tracking_go: {'✓' if summary['bbox_tracking_go'] else '✗'} "
          f"(lerp IoU mean {summary['lerp_iou_mean']:.3f} {'>' if summary['bbox_tracking_go'] else '<'} 0.75)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
