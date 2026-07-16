"""Audit and visualize controlled OCAtari Boxing datasets."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw


FIGHTER_COLORS = ((214, 214, 214), (0, 0, 0))
FIRE_ACTION_IDS = set(range(10, 18)) | {1}


def _as_video(sample: Dict[str, Any]) -> np.ndarray:
    video = sample["videos"].cpu().numpy()
    if video.shape[1] == 3:
        video = np.transpose(video, (0, 2, 3, 1))
    return video.astype(np.uint8)


def _edge_distance(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.tolist()
    bx1, by1, bx2, by2 = b.tolist()
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float(np.hypot(dx, dy))


def _audit_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    video = _as_video(sample)
    masks = sample["masks"].cpu().numpy().astype(bool)
    background = sample["background_masks"].cpu().numpy().astype(bool)
    boxes = sample["bboxes_xyxy"].cpu().numpy()
    if masks.shape[:2] != (video.shape[0], 2):
        raise AssertionError(f"Expected two fighter masks, got {masks.shape}")
    partition = masks.sum(axis=1) + background
    if not np.all(partition == 1):
        raise AssertionError("fighter/background masks do not form an exact partition")
    if int(sample["punch_labels"].sum()) != 0:
        raise AssertionError("stage-1 sample contains a punch")
    if int(sample["contact_labels"].sum()) != 0 or int(sample["occlusion_labels"].sum()) != 0:
        raise AssertionError("stage-1 sample contains contact/occlusion labels")
    if any(int(a) in FIRE_ACTION_IDS for a in sample["env_actions"].tolist()):
        raise AssertionError("stage-1 sample contains a FIRE action")
    if not torch.equal(sample["scores"], sample["scores"][0:1].expand_as(sample["scores"])):
        raise AssertionError("stage-1 sample contains a score change")

    color_precision: List[float] = []
    min_separation = float("inf")
    for t in range(video.shape[0]):
        for k, color in enumerate(FIGHTER_COLORS):
            pixels = video[t][masks[t, k]]
            if len(pixels) == 0:
                raise AssertionError(f"empty fighter mask at t={t}, k={k}")
            color_precision.append(float(np.all(pixels == np.asarray(color), axis=1).mean()))
        min_separation = min(min_separation, _edge_distance(boxes[t, 0], boxes[t, 1]))
    delta = sample["delta_xy"].cpu().numpy()
    return {
        "transitions": int(delta.shape[0] * delta.shape[1]),
        "moving_transitions": int((np.linalg.norm(delta, axis=-1) > 0.5).sum()),
        "max_displacement": float(np.linalg.norm(delta, axis=-1).max()),
        "min_separation": min_separation,
        "mask_color_precision_min": min(color_precision),
        "movement_labels": sample["movement_labels"].flatten().tolist(),
        "env_actions": sample["env_actions"].flatten().tolist(),
        "mask_areas": masks.sum(axis=(-1, -2)).flatten().tolist(),
    }


def _render_sample(sample: Dict[str, Any], output_path: str, scale: int = 2) -> None:
    video = _as_video(sample)
    masks = sample["masks"].cpu().numpy().astype(bool)
    background = sample["background_masks"].cpu().numpy().astype(bool)
    boxes = sample["bboxes_xyxy"].cpu().numpy()
    rows: List[List[Image.Image]] = [[], [], [], []]
    for t, frame in enumerate(video):
        overlay = frame.copy()
        tint = np.zeros_like(overlay)
        tint[masks[t, 0]] = (255, 60, 60)
        tint[masks[t, 1]] = (60, 180, 255)
        active = masks[t].any(axis=0)
        overlay[active] = (0.55 * overlay[active] + 0.45 * tint[active]).astype(np.uint8)
        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image)
        for k, color in enumerate(((255, 60, 60), (60, 180, 255))):
            draw.rectangle(tuple(float(v) for v in boxes[t, k]), outline=color, width=1)
        draw.text((3, 30), f"t={t}", fill=(255, 255, 0))
        rows[0].append(image)
        for k in range(2):
            isolated = np.zeros_like(frame)
            isolated[masks[t, k]] = frame[masks[t, k]]
            rows[k + 1].append(Image.fromarray(isolated))
        bg = np.zeros_like(frame)
        bg[background[t]] = frame[background[t]]
        rows[3].append(Image.fromarray(bg))

    width, height = rows[0][0].size
    canvas = Image.new("RGB", (width * len(rows[0]), height * len(rows)), (32, 32, 32))
    for r, images in enumerate(rows):
        for c, image in enumerate(images):
            canvas.paste(image, (c * width, r * height))
    if scale != 1:
        canvas = canvas.resize((canvas.width * scale, canvas.height * scale), Image.Resampling.NEAREST)
    canvas.save(output_path)


def audit(data_root: str, out_root: str, max_visuals: int) -> Dict[str, Any]:
    os.makedirs(out_root, exist_ok=True)
    report: Dict[str, Any] = {"data_root": data_root, "splits": {}}
    for split in ("train", "val"):
        files = sorted(glob.glob(os.path.join(data_root, split, "*.pt")))
        per_sample = []
        movement_counts: Counter[int] = Counter()
        env_action_counts: Counter[int] = Counter()
        for index, path in enumerate(files):
            sample = torch.load(path, map_location="cpu", weights_only=False)
            stats = _audit_sample(sample)
            per_sample.append(stats)
            movement_counts.update(stats["movement_labels"])
            env_action_counts.update(stats["env_actions"])
            if index < max_visuals:
                _render_sample(sample, os.path.join(out_root, f"{split}_{index:06d}.png"))
        report["splits"][split] = {
            "samples": len(files),
            "transitions": sum(s["transitions"] for s in per_sample),
            "moving_transition_fraction": (
                sum(s["moving_transitions"] for s in per_sample) / max(1, sum(s["transitions"] for s in per_sample))
            ),
            "max_displacement": max((s["max_displacement"] for s in per_sample), default=0.0),
            "min_separation": min((s["min_separation"] for s in per_sample), default=0.0),
            "mask_color_precision_min": min((s["mask_color_precision_min"] for s in per_sample), default=0.0),
            "mask_area_min": min((min(s["mask_areas"]) for s in per_sample), default=0),
            "mask_area_max": max((max(s["mask_areas"]) for s in per_sample), default=0),
            "movement_label_counts": dict(sorted(movement_counts.items())),
            "env_action_counts": dict(sorted(env_action_counts.items())),
        }
    with open(os.path.join(out_root, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--max_visuals", type=int, default=4)
    args = parser.parse_args()
    audit(args.data_root, args.out_root, args.max_visuals)


if __name__ == "__main__":
    main()
