"""Create demo assets for Synthetic-Minimal-NoOverlap-v0."""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.v12_dataset import ACTION_NAMES, V12ObjectVideoDataset


def _to_uint8(videos):
    arr = videos.detach().cpu().float().clamp(0, 1).numpy()
    return (arr * 255).round().astype(np.uint8)


def _write_mp4(frames_rgb: np.ndarray, out_path: str, fps: int = 3) -> None:
    h, w = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {out_path}")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _draw_rect(img: np.ndarray, box, color) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    img[y1:y1 + 2, x1:x2] = color
    img[max(y2 - 2, y1):y2, x1:x2] = color
    img[y1:y2, x1:x1 + 2] = color
    img[y1:y2, max(x2 - 2, x1):x2] = color


def _overlay(sample) -> np.ndarray:
    frames = _to_uint8(sample["videos"])
    boxes = sample["bboxes"].numpy()
    valid = sample["valid_mask"].numpy()
    colors = [(230, 50, 50), (67, 165, 88), (62, 111, 224), (224, 180, 45)]
    for t in range(frames.shape[0]):
        # Region boundaries make the no-overlap guarantee visible.
        cv2.line(frames[t], (128, 0), (128, 255), (120, 120, 120), 1)
        cv2.line(frames[t], (0, 128), (255, 128), (120, 120, 120), 1)
        for k in range(valid.shape[1]):
            if valid[t, k]:
                _draw_rect(frames[t], boxes[t, k], colors[k % len(colors)])
    return frames


def _make_contact_sheet(sample, out_path: str) -> None:
    raw = _to_uint8(sample["videos"])
    overlay = _overlay(sample)
    rows = []
    for label, frames in (("raw frames", raw), ("bbox + no-overlap regions", overlay)):
        row = np.concatenate(frames, axis=1)
        strip = np.full((28, row.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(strip, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(np.concatenate([strip, row], axis=0))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def _action_summary(ds: V12ObjectVideoDataset, max_samples: int) -> Counter:
    counts = Counter()
    for idx in range(min(len(ds), max_samples)):
        for value in ds[idx]["actions"].reshape(-1).tolist():
            if value >= 0:
                counts[int(value)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/v12/synthetic_minimal_nooverlap")
    parser.add_argument("--out_dir", default="reports/V12/synthetic_demo")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--summary_samples", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = V12ObjectVideoDataset(os.path.join(args.data_root, "train"), task_name="Synthetic-Minimal-NoOverlap-v0")
    val = V12ObjectVideoDataset(os.path.join(args.data_root, "val"), task_name="Synthetic-Minimal-NoOverlap-v0")
    sample = val[args.sample_index]

    raw_mp4 = os.path.join(args.out_dir, "synthetic_sample_raw.mp4")
    overlay_mp4 = os.path.join(args.out_dir, "synthetic_sample_overlay.mp4")
    sheet_png = os.path.join(args.out_dir, "synthetic_contact_sheet.png")
    report_md = os.path.join(args.out_dir, "README.md")
    _write_mp4(_to_uint8(sample["videos"]), raw_mp4)
    _write_mp4(_overlay(sample), overlay_mp4)
    _make_contact_sheet(sample, sheet_png)

    counts = _action_summary(train, args.summary_samples)
    count_lines = "\n".join(
        f"- `{idx} / {ACTION_NAMES[idx]}`: {counts.get(idx, 0)}"
        for idx in range(len(ACTION_NAMES))
    )
    metadata = sample["metadata"]
    text = f"""# Synthetic-Minimal-NoOverlap-v0 Demo

## What Was Generated

- Dataset root: `{args.data_root}`
- Train clips: `{len(train)}`
- Val clips: `{len(val)}`
- Clip shape: `videos={tuple(sample['videos'].shape)}`, `masks={tuple(sample['masks'].shape)}`
- Objects: 4 colored actors with fixed actor ids
- Labels: exact `masks`, `bboxes`, `positions`, `actions`, `valid_mask`
- Action space: `stay / up / down / left / right`

## Design Constraint

- Each actor is restricted to its own quadrant-like region.
- Actor boxes are 2x2 grid cells.
- Regions do not overlap, so collision and occlusion are impossible by construction.
- This is intended as the Stage 0 sanity check, not a realistic scene.

## Demo Assets

- Contact sheet: `synthetic_contact_sheet.png`
- Raw clip: `synthetic_sample_raw.mp4`
- BBox/region overlay clip: `synthetic_sample_overlay.mp4`

## Sample Metadata

- `task_name`: `{metadata.get('task_name')}`
- `no_overlap_guarantee`: `{metadata.get('no_overlap_guarantee')}`
- `episode_id`: `{metadata.get('episode_id')}`

## Action Distribution Preview

Counted over first `{min(len(train), args.summary_samples)}` train samples:

{count_lines}

## Talking Points

- This dataset is intentionally simpler than the earlier synthetic data.
- It removes occlusion/collision as confounders.
- If the model fails here, the issue is likely in representation/reconstruction rather than data complexity.
"""
    with open(report_md, "w") as f:
        f.write(text)
    print(report_md)


if __name__ == "__main__":
    main()
