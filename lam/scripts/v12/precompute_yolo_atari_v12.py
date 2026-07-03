"""
Precompute YOLO+tracking bbox masks for V12 Atari video-only clips.

This mirrors the older MOT path:
  YOLO.track(frame, tracker='botsort.yaml') -> xyxy + track ids -> fixed slots
  -> rectangular binary masks.

The output is a normal V12 dataset with object fields filled when detections
exist. Generic COCO YOLO is not expected to be reliable on Atari sprites; the
demo script should be used to inspect detection quality before using these
masks for model training/evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.v12_dataset import V12ObjectVideoDataset, validate_v12_sample


def _bbox_to_mask(box: np.ndarray, h: int, w: int) -> torch.Tensor:
    mask = torch.zeros(h, w, dtype=torch.uint8)
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def _sort_detections(xyxy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    if len(xyxy) == 0:
        return np.zeros(0, dtype=np.int64)
    centers_y = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
    centers_x = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
    return np.lexsort((-conf, centers_x, centers_y))


def run_yolo_track_on_sample(
    model,
    sample: Dict,
    *,
    max_objects: int,
    conf: float,
    iou: float,
    tracker: str,
    device: str,
) -> Dict:
    videos = sample["videos"]  # (T,H,W,C) float
    if videos.dtype != torch.uint8:
        frames = (videos.float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    else:
        frames = videos.numpy()
    t_count, h, w = frames.shape[:3]

    bboxes = torch.zeros((t_count, max_objects, 4), dtype=torch.float32)
    masks = torch.zeros((t_count, max_objects, h, w), dtype=torch.uint8)
    valid = torch.zeros((t_count, max_objects), dtype=torch.bool)
    object_types = torch.full((max_objects,), -1, dtype=torch.long)
    actor_ids = torch.arange(max_objects, dtype=torch.long)

    track_to_slot: Dict[int, int] = {}
    next_slot = 0

    for t in range(t_count):
        result = model.track(
            frames[t],
            persist=(t > 0),
            verbose=False,
            conf=conf,
            iou=iou,
            tracker=tracker,
            device=device,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = boxes.conf.cpu().numpy().astype(np.float32) if boxes.conf is not None else np.ones(len(xyxy))
        cls = boxes.cls.cpu().numpy().astype(np.int64) if boxes.cls is not None else np.full(len(xyxy), -1)
        ids = boxes.id.cpu().numpy().astype(np.int64) if boxes.id is not None else None

        order = _sort_detections(xyxy, confs)
        for j in order:
            if ids is not None:
                tid = int(ids[j])
                if tid not in track_to_slot:
                    if next_slot >= max_objects:
                        continue
                    track_to_slot[tid] = next_slot
                    next_slot += 1
                slot = track_to_slot[tid]
            else:
                slot = int(j)
                if slot >= max_objects:
                    continue
            box = xyxy[j]
            bboxes[t, slot] = torch.from_numpy(box)
            masks[t, slot] = _bbox_to_mask(box, h, w)
            valid[t, slot] = True
            object_types[slot] = int(cls[j])

    out = dict(sample)
    out["masks"] = masks
    out["bboxes"] = bboxes
    out["positions"] = torch.full((t_count, max_objects, 2), -1, dtype=torch.long)
    for t in range(t_count):
        for k in range(max_objects):
            if valid[t, k]:
                box = bboxes[t, k]
                out["positions"][t, k, 0] = torch.round((box[1] + box[3]) * 0.5).long()
                out["positions"][t, k, 1] = torch.round((box[0] + box[2]) * 0.5).long()
    out["actions"] = torch.full((max(t_count - 1, 0), max_objects), -1, dtype=torch.long)
    out["actor_ids"] = actor_ids
    out["object_types"] = object_types
    out["valid_mask"] = valid
    out["num_actors"] = int(valid.any(dim=0).sum().item())
    metadata = dict(out.get("metadata") or {})
    metadata["has_object_annotations"] = bool(valid.any().item())
    metadata["annotation_status"] = "yolo_botsort_bbox_masks"
    metadata["yolo_model"] = getattr(model, "ckpt_path", None) or "unknown"
    metadata["yolo_conf"] = float(conf)
    metadata["tracker"] = tracker
    metadata["detected_boxes"] = int(valid.sum().item())
    out["metadata"] = metadata
    return out


def process_split(args, model, split: str) -> Dict:
    in_dir = os.path.join(args.data_root, split)
    out_dir = os.path.join(args.out_root, split)
    os.makedirs(out_dir, exist_ok=True)
    dataset = V12ObjectVideoDataset(in_dir, output_format="t h w c", max_samples=args.max_samples)
    total_boxes = 0
    samples_with_boxes = 0
    for idx in tqdm(range(len(dataset)), desc=f"yolo {os.path.basename(args.data_root)} {split}"):
        sample = dataset[idx]
        out = run_yolo_track_on_sample(
            model,
            sample,
            max_objects=args.max_objects,
            conf=args.conf,
            iou=args.iou,
            tracker=args.tracker,
            device=args.device,
        )
        errors = validate_v12_sample(out)
        if errors:
            raise RuntimeError(f"invalid sample {idx}: {errors}")
        total_boxes += int(out["valid_mask"].sum().item())
        samples_with_boxes += int(out["valid_mask"].any().item())
        torch.save(out, os.path.join(out_dir, f"sample_{idx:06d}.pt"))
    return {
        "split": split,
        "samples": len(dataset),
        "samples_with_boxes": samples_with_boxes,
        "total_boxes": total_boxes,
        "avg_boxes_per_sample": total_boxes / max(len(dataset), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max_objects", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    from ultralytics import YOLO

    os.makedirs(args.out_root, exist_ok=True)
    model = YOLO(args.model)
    model.ckpt_path = args.model
    report = {
        "source": args.data_root,
        "out_root": args.out_root,
        "model": args.model,
        "conf": args.conf,
        "iou": args.iou,
        "tracker": args.tracker,
        "device": args.device,
        "max_objects": args.max_objects,
        "splits": [],
    }
    for split in args.splits:
        report["splits"].append(process_split(args, model, split))
    with open(os.path.join(args.out_root, "precompute_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
