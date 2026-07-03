"""Create inspection assets for YOLO-precomputed V12 Atari clips."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.v12_dataset import V12ObjectVideoDataset


COLORS = [
    (230, 50, 50),
    (67, 165, 88),
    (62, 111, 224),
    (224, 180, 45),
    (170, 85, 210),
    (40, 180, 190),
    (240, 120, 50),
    (70, 70, 70),
]


def _to_uint8(videos) -> np.ndarray:
    arr = videos.detach().cpu().float().clamp(0, 1).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _write_mp4(frames_rgb: np.ndarray, out_path: str, fps: int = 3) -> None:
    h, w = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {out_path}")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _draw_box(img: np.ndarray, box, color, label: str) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.putText(img, label, (x1, max(8, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)


def _overlay(sample: Dict) -> np.ndarray:
    frames = _to_uint8(sample["videos"])
    boxes = sample["bboxes"].detach().cpu().numpy()
    valid = sample["valid_mask"].detach().cpu().numpy()
    types = sample["object_types"].detach().cpu().numpy()
    for t in range(frames.shape[0]):
        for k in range(valid.shape[1]):
            if valid[t, k]:
                label = f"s{k}/c{int(types[k])}"
                _draw_box(frames[t], boxes[t, k], COLORS[k % len(COLORS)], label)
    return frames


def _add_title(tile: np.ndarray, text: str) -> np.ndarray:
    strip = np.full((18, tile.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(strip, text[:44], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([strip, tile], axis=0)


def _make_sample_sheet(samples: List[Dict], out_path: str, max_cols: int) -> None:
    tiles = []
    for idx, sample in enumerate(samples):
        overlay = _overlay(sample)
        mid = overlay.shape[0] // 2
        n_boxes = int(sample["valid_mask"].sum().item())
        tile = _add_title(overlay[mid], f"sample {idx:02d}, boxes={n_boxes}")
        tiles.append(tile)

    if not tiles:
        return
    tile_h, tile_w = tiles[0].shape[:2]
    cols = min(max_cols, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _make_sequence_sheet(sample: Dict, out_path: str) -> None:
    raw = _to_uint8(sample["videos"])
    over = _overlay(sample)
    rows = []
    for label, frames in (("raw", raw), ("yolo bbox masks", over)):
        row = np.concatenate(frames, axis=1)
        rows.append(_add_title(row, label))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def _summarize(ds: V12ObjectVideoDataset) -> Dict:
    boxes_per_sample = []
    frames_with_boxes = 0
    class_counts = Counter()
    for i in range(len(ds)):
        sample = ds[i]
        valid = sample["valid_mask"]
        boxes_per_sample.append(int(valid.sum().item()))
        frames_with_boxes += int(valid.any(dim=1).sum().item())
        types = sample["object_types"]
        for k in range(valid.shape[1]):
            if valid[:, k].any():
                class_counts[int(types[k].item())] += 1
    return {
        "samples": len(ds),
        "samples_with_boxes": sum(1 for v in boxes_per_sample if v > 0),
        "total_boxes": sum(boxes_per_sample),
        "avg_boxes_per_sample": sum(boxes_per_sample) / max(len(boxes_per_sample), 1),
        "frames_with_boxes": frames_with_boxes,
        "class_counts": dict(sorted(class_counts.items())),
        "boxes_per_sample": boxes_per_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--num_videos", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = V12ObjectVideoDataset(os.path.join(args.data_root, args.split), output_format="t h w c")
    samples = [ds[i] for i in range(min(len(ds), args.num_samples))]

    _make_sample_sheet(samples, str(out_dir / "sample_midframe_overlay_sheet.png"), args.cols)
    if samples:
        _make_sequence_sheet(samples[0], str(out_dir / "sample_00_sequence_sheet.png"))
    for i, sample in enumerate(samples[: args.num_videos]):
        _write_mp4(_overlay(sample), str(out_dir / f"sample_{i:02d}_overlay.mp4"))

    summary = _summarize(ds)
    with open(out_dir / "detection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    game = os.path.basename(args.data_root).replace("atari_", "")
    text = f"""# Atari YOLO Demo: {game}

## Source

- Dataset root: `{args.data_root}`
- Split shown: `{args.split}`
- Samples in split: `{len(ds)}`
- Annotation source: YOLO + BoT-SORT rectangular bbox masks
- Important: these are detector outputs, not Atari ground-truth object labels.

## Demo Assets

- Multi-sample mid-frame sheet: `sample_midframe_overlay_sheet.png`
- First sampled clip sequence sheet: `sample_00_sequence_sheet.png`
- Overlay videos: `sample_00_overlay.mp4` ... `sample_{max(min(len(samples), args.num_videos) - 1, 0):02d}_overlay.mp4`
- Detection stats: `detection_summary.json`

## Detection Summary

- Samples with at least one box: `{summary['samples_with_boxes']}/{summary['samples']}`
- Total boxes across split: `{summary['total_boxes']}`
- Average boxes per sample: `{summary['avg_boxes_per_sample']:.2f}`
- Frames with at least one box: `{summary['frames_with_boxes']}`
- YOLO class ids used by occupied slots: `{summary['class_counts']}`

## Review Notes

- Use the sheets and videos for manual quality review.
- If boxes miss most sprites or lock onto background/HUD, this detector output should not be treated as object supervision.
- For reliable Atari object labels, the next step should be OCAtari or game-specific rule/RAM extraction rather than generic COCO YOLO.
"""
    with open(out_dir / "README.md", "w") as f:
        f.write(text)
    print(out_dir / "README.md")


if __name__ == "__main__":
    main()
