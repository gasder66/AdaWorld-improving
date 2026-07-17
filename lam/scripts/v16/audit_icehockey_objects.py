"""Audit OCAtari Ice Hockey agents, puck tracks, masks, and contact events.

This is deliberately a data audit rather than a training-data generator. It
creates a compact trajectory JSON plus raw and overlay videos so object identity
and interaction quality can be checked before adapting the Boxing LAM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


ENV_NAME = "ALE/IceHockey-v5"
SLOT_CATEGORIES = ("Player", "Player", "Enemy", "Enemy", "Ball")
SLOT_NAMES = ("player_0", "player_1", "enemy_0", "enemy_1", "puck")
OVERLAY_COLORS = (
    (255, 80, 80),
    (255, 170, 50),
    (60, 220, 80),
    (40, 200, 230),
    (255, 40, 220),
)


def _import_ocatari():
    try:
        from ocatari.core import OCAtari
    except Exception as exc:
        raise RuntimeError(
            "OCAtari is unavailable; run this script in the locateanything environment."
        ) from exc
    return OCAtari


def _rgb_frame(env: Any, observation: Any) -> np.ndarray:
    frame = env.render()
    if frame is None:
        frame = observation
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[-1]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * (255.0 if frame.max() <= 1.5 else 1.0), 0, 255).astype(
            np.uint8
        )
    return frame


def _xywh(obj: Any) -> tuple[int, int, int, int]:
    return tuple(int(round(float(value))) for value in obj.xywh)


def _ordered_objects(objects: Iterable[Any]) -> list[Any]:
    grouped: dict[str, list[Any]] = {"Player": [], "Enemy": [], "Ball": []}
    for obj in objects:
        category = str(getattr(obj, "category", obj.__class__.__name__))
        if category in grouped:
            grouped[category].append(obj)
    # OCAtari RAM objects keep stable list slots across frames. Preserve that
    # order; sorting by position would swap identities when two skaters cross.
    ordered = grouped["Player"] + grouped["Enemy"] + grouped["Ball"]
    if len(ordered) != 5:
        counts = {key: len(value) for key, value in grouped.items()}
        raise RuntimeError(f"expected 2 Player, 2 Enemy, and 1 Ball, got {counts}")
    return ordered


def _visible_mask(
    frame: np.ndarray, box: Sequence[int], rgb: Sequence[int]
) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, w, h = (int(value) for value in box)
    x1, y1 = max(0, x - 1), max(0, y - 1)
    x2, y2 = min(width, x + w + 1), min(height, y + h + 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    target = np.asarray(rgb, dtype=np.uint8).reshape(1, 1, 3)
    mask[y1:y2, x1:x2] = np.all(
        frame[y1:y2, x1:x2] == target, axis=-1
    ).astype(np.uint8)
    return mask


def _box_distance(a: Sequence[int], b: Sequence[int]) -> float:
    ax, ay, aw, ah = (float(value) for value in a)
    bx, by, bw, bh = (float(value) for value in b)
    dx = max(bx - (ax + aw), ax - (bx + bw), 0.0)
    dy = max(by - (ay + ah), ay - (by + bh), 0.0)
    return float(np.hypot(dx, dy))


def _open_writer(path: Path, frame_shape: Sequence[int], fps: int) -> cv2.VideoWriter:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    writer = _open_writer(path, frames[0].shape, fps)
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _overlay(
    frame: np.ndarray,
    boxes: Sequence[Sequence[int]],
    masks: Sequence[np.ndarray],
    contact: bool,
) -> np.ndarray:
    result = frame.copy()
    tint = np.zeros_like(result)
    for mask, color in zip(masks, OVERLAY_COLORS):
        tint[mask.astype(bool)] = color
    result = cv2.addWeighted(result, 0.70, tint, 0.30, 0.0)
    for name, box, color in zip(SLOT_NAMES, boxes, OVERLAY_COLORS):
        x, y, w, h = (int(value) for value in box)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 1)
        cv2.putText(
            result,
            name,
            (x, max(10, y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            color,
            1,
            cv2.LINE_AA,
        )
    if contact:
        cv2.putText(
            result,
            "PUCK CONTACT",
            (4, result.shape[0] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 40, 220),
            1,
            cv2.LINE_AA,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="result/v16/icehockey_object_audit_v1")
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action_hold", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--clip_radius", type=int, default=45)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    OCAtari = _import_ocatari()
    env = OCAtari(ENV_NAME, mode="ram", render_mode="rgb_array", obs_mode="ori")
    env.action_space.seed(args.seed)
    observation, _ = env.reset(seed=args.seed)

    raw_frames: list[np.ndarray] = []
    overlay_frames: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    action = 0
    episode = 0
    try:
        for step in range(args.frames):
            if step % args.action_hold == 0:
                action = int(env.action_space.sample())
            observation, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
                episode += 1
            frame = _rgb_frame(env, observation)
            objects = _ordered_objects(env.objects)
            boxes = [_xywh(obj) for obj in objects]
            colors = [tuple(int(v) for v in obj.rgb) for obj in objects]
            masks = [
                _visible_mask(frame, box, color)
                for box, color in zip(boxes, colors)
            ]
            puck_box = boxes[-1]
            distances = [_box_distance(box, puck_box) for box in boxes[:-1]]
            contact = min(distances) <= 1.0
            centers = [
                [box[0] + 0.5 * box[2], box[1] + 0.5 * box[3]] for box in boxes
            ]
            raw_frames.append(frame)
            overlay_frames.append(_overlay(frame, boxes, masks, contact))
            records.append(
                {
                    "step": step,
                    "episode": episode,
                    "action": action,
                    "reward": float(reward),
                    "boxes_xywh": boxes,
                    "centers_xy": centers,
                    "visible_pixels": [int(mask.sum()) for mask in masks],
                    "puck_distances": distances,
                    "puck_contact": contact,
                }
            )
    finally:
        env.close()

    _write_video(output / "icehockey_raw.mp4", raw_frames, args.fps)
    _write_video(output / "icehockey_object_overlay.mp4", overlay_frames, args.fps)
    contact_indices = [
        index for index, record in enumerate(records) if record["puck_contact"]
    ]
    if contact_indices:
        center = contact_indices[len(contact_indices) // 2]
        start = max(0, center - args.clip_radius)
        stop = min(len(overlay_frames), center + args.clip_radius + 1)
        _write_video(
            output / "icehockey_puck_contact.mp4",
            overlay_frames[start:stop],
            args.fps,
        )
    summary = {
        "environment": ENV_NAME,
        "seed": args.seed,
        "frames": len(records),
        "episodes": episode + 1,
        "slot_names": SLOT_NAMES,
        "slot_categories": SLOT_CATEGORIES,
        "contact_frames": len(contact_indices),
        "unique_agent_positions": len(
            {
                tuple(tuple(center) for center in record["centers_xy"][:-1])
                for record in records
            }
        ),
        "puck_x_range": [
            min(record["centers_xy"][-1][0] for record in records),
            max(record["centers_xy"][-1][0] for record in records),
        ],
        "puck_y_range": [
            min(record["centers_xy"][-1][1] for record in records),
            max(record["centers_xy"][-1][1] for record in records),
        ],
    }
    with (output / "trajectory.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "frames": records}, handle, indent=2)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
