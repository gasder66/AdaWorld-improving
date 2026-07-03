"""Run a small LocateAnything grounding/tracking demo on V12 Atari clips.

This is an inspection script, not a production precompute path. LocateAnything
is an image VLM, so we query each Atari frame independently and stitch boxes
across time with simple IoU matching.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.v12_dataset import V12ObjectVideoDataset


GAME_PROMPTS = {
    "freeway": ["chicken", "car", "vehicle"],
    "mspacman": ["pacman", "ghost", "pellet", "maze"],
    "spaceinvaders": ["player ship", "alien", "bullet", "spaceship"],
}

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


def _to_uint8(videos: torch.Tensor) -> np.ndarray:
    arr = videos.detach().cpu().float().clamp(0, 1).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _track_boxes(frame_boxes: List[List[float]], max_objects: int, iou_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    t_count = len(frame_boxes)
    bboxes = np.zeros((t_count, max_objects, 4), dtype=np.float32)
    valid = np.zeros((t_count, max_objects), dtype=bool)
    prev_slots: Dict[int, np.ndarray] = {}
    next_slot = 0

    for t, boxes in enumerate(frame_boxes):
        assigned = set()
        for box in boxes:
            box_arr = np.asarray(box, dtype=np.float32)
            best_slot, best_iou = None, 0.0
            for slot, prev_box in prev_slots.items():
                if slot in assigned:
                    continue
                score = _iou(box_arr, prev_box)
                if score > best_iou:
                    best_slot, best_iou = slot, score
            if best_slot is None or best_iou < iou_threshold:
                if next_slot >= max_objects:
                    continue
                best_slot = next_slot
                next_slot += 1
            bboxes[t, best_slot] = box_arr
            valid[t, best_slot] = True
            assigned.add(best_slot)
        prev_slots = {slot: bboxes[t, slot].copy() for slot in np.where(valid[t])[0]}
    return bboxes, valid


def _draw_box(img: np.ndarray, box, color, label: str) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.putText(img, label, (x1, max(8, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)


def _write_mp4(frames_rgb: np.ndarray, out_path: str, fps: int = 2) -> None:
    h, w = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {out_path}")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _make_sheet(raw: np.ndarray, overlay: np.ndarray, out_path: str) -> None:
    def title(row: np.ndarray, text: str) -> np.ndarray:
        strip = np.full((20, row.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(strip, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        return np.concatenate([strip, row], axis=0)

    sheet = np.concatenate(
        [title(np.concatenate(raw, axis=1), "raw"), title(np.concatenate(overlay, axis=1), "LocateAnything + IoU tracking")],
        axis=0,
    )
    cv2.imwrite(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def run_game(args, worker, game: str) -> Dict:
    data_root = Path(args.data_root) / f"atari_{game}" / args.split
    ds = V12ObjectVideoDataset(str(data_root), output_format="t h w c", max_samples=args.sample_index + 1)
    sample = ds[args.sample_index]
    frames = _to_uint8(sample["videos"])[: args.max_frames]
    h, w = frames.shape[1:3]
    categories = args.categories.split(",") if args.categories else GAME_PROMPTS[game]

    frame_boxes: List[List[float]] = []
    answers = []
    for t, frame in enumerate(frames):
        image = Image.fromarray(frame).convert("RGB")
        if args.upscale != 1:
            image = image.resize((w * args.upscale, h * args.upscale), Image.Resampling.NEAREST)
        result = worker.detect(
            image,
            categories,
            generation_mode=args.generation_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            verbose=False,
        )
        boxes = worker.parse_boxes(result["answer"], image.width, image.height)
        scaled = [
            [
                max(0.0, min(w, box["x1"] / args.upscale)),
                max(0.0, min(h, box["y1"] / args.upscale)),
                max(0.0, min(w, box["x2"] / args.upscale)),
                max(0.0, min(h, box["y2"] / args.upscale)),
            ]
            for box in boxes
        ]
        frame_boxes.append(scaled[: args.max_objects])
        answers.append({"frame": t, "answer": result["answer"], "boxes": scaled[: args.max_objects]})

    tracked, valid = _track_boxes(frame_boxes, args.max_objects, args.iou_threshold)
    overlay = frames.copy()
    for t in range(len(frames)):
        for k in range(args.max_objects):
            if valid[t, k]:
                _draw_box(overlay[t], tracked[t, k], COLORS[k % len(COLORS)], f"s{k}")

    out_dir = Path(args.out_dir) / game
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_mp4(overlay, str(out_dir / "locateanything_overlay.mp4"))
    _make_sheet(frames, overlay, str(out_dir / "locateanything_sequence_sheet.png"))

    report = {
        "game": game,
        "data_root": str(data_root),
        "sample_index": args.sample_index,
        "frames": int(len(frames)),
        "categories": categories,
        "total_raw_boxes": int(sum(len(x) for x in frame_boxes)),
        "total_tracked_boxes": int(valid.sum()),
        "answers": answers,
    }
    with open(out_dir / "locateanything_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "README.md", "w") as f:
        f.write(
            f"""# LocateAnything Atari Demo: {game}

- Source split: `{data_root}`
- Sample index: `{args.sample_index}`
- Frames queried: `{len(frames)}`
- Prompt categories: `{categories}`
- Raw boxes: `{report['total_raw_boxes']}`
- Tracked boxes: `{report['total_tracked_boxes']}`

Assets:

- `locateanything_sequence_sheet.png`
- `locateanything_overlay.mp4`
- `locateanything_report.json`

This is model-predicted localization plus simple IoU tracking, not ground truth.
"""
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/v12_atari")
    parser.add_argument("--out_dir", default="reports/V12/atari_locateanything_demo")
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--games", nargs="+", default=["freeway", "mspacman", "spaceinvaders"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=3)
    parser.add_argument("--max_objects", type=int, default=8)
    parser.add_argument("--categories", default="")
    parser.add_argument("--upscale", type=int, default=3)
    parser.add_argument("--iou_threshold", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--generation_mode", choices=["fast", "slow", "hybrid"], default="hybrid")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    args = parser.parse_args()

    import torch

    sys.path.insert(0, os.path.abspath("third_party/Eagle/Embodied"))
    from locateanything_worker import LocateAnythingWorker

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    worker = LocateAnythingWorker(args.model, device=args.device, dtype=dtype)

    reports = [run_game(args, worker, game) for game in args.games]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(reports, f, indent=2)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
