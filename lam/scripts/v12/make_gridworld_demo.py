"""Create lightweight GridWorld V12 demo assets for meetings."""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np
import torch

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
    colors = [(230, 50, 50), (50, 100, 230), (235, 180, 45), (150, 80, 180)]
    for t in range(frames.shape[0]):
        for k in range(valid.shape[1]):
            if valid[t, k]:
                _draw_rect(frames[t], boxes[t, k], colors[k % len(colors)])
    return frames


def _make_contact_sheet(sample, out_path: str) -> None:
    raw = _to_uint8(sample["videos"])
    overlay = _overlay(sample)
    rows = []
    for label, frames in (("raw", raw), ("bbox/mask slots", overlay)):
        row = np.concatenate(frames, axis=1)
        strip = np.full((28, row.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(strip, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(np.concatenate([strip, row], axis=0))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def _action_summary(ds: V12ObjectVideoDataset, max_samples: int) -> Counter:
    counts = Counter()
    for idx in range(min(len(ds), max_samples)):
        actions = ds[idx]["actions"].reshape(-1)
        for value in actions.tolist():
            if value >= 0:
                counts[int(value)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/v12/gridworld_maze_multiobject")
    parser.add_argument("--out_dir", default="reports/V12/gridworld_demo")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--summary_samples", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = V12ObjectVideoDataset(os.path.join(args.data_root, "train"), task_name="GridWorld-MazeMultiObject-v0")
    val = V12ObjectVideoDataset(os.path.join(args.data_root, "val"), task_name="GridWorld-MazeMultiObject-v0")
    sample = val[args.sample_index]

    raw_mp4 = os.path.join(args.out_dir, "gridworld_sample_raw.mp4")
    overlay_mp4 = os.path.join(args.out_dir, "gridworld_sample_overlay.mp4")
    sheet_png = os.path.join(args.out_dir, "gridworld_contact_sheet.png")
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
    text = f"""# GridWorld-MazeMultiObject-v0 Demo

## What Was Generated

- Dataset root: `{args.data_root}`
- Train clips: `{len(train)}`
- Val clips: `{len(val)}`
- Clip shape: `videos={tuple(sample['videos'].shape)}`, `masks={tuple(sample['masks'].shape)}`
- Objects: player + 2 moving enemies + 1 static object
- Labels: per-object `masks`, `bboxes`, `positions`, `actions`, `valid_mask`
- Action space: `stay / up / down / left / right`

## Demo Assets

- Contact sheet: `gridworld_contact_sheet.png`
- Raw clip: `gridworld_sample_raw.mp4`
- BBox/slot overlay clip: `gridworld_sample_overlay.mp4`

## Sample Metadata

- `task_name`: `{metadata.get('task_name')}`
- `map_id`: `{metadata.get('map_id')}`
- `episode_id`: `{metadata.get('episode_id')}`

## Action Distribution Preview

Counted over first `{min(len(train), args.summary_samples)}` train samples:

{count_lines}

## Talking Points

- This is the Stage 1 bridge between the ultra-clean synthetic sanity check and Atari.
- Compared with Synthetic-Minimal, it adds structured background, walls, goal color, and simple obstacles.
- It still keeps object annotations exact, so failures can be attributed to model behavior rather than detector noise.
"""
    with open(report_md, "w") as f:
        f.write(text)
    print(report_md)


if __name__ == "__main__":
    main()
