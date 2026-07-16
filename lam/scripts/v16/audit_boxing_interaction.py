"""Audit interaction-labelled OCAtari Boxing clips and render event videos."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from scripts.v16.visualize_boxing_stage1 import _label_frame, _write_video


EVENT_KEYS = {
    "near": "near_labels",
    "contact": "contact_labels",
    "punch_miss": "punch_miss_labels",
    "hit": "hit_labels",
    "occlusion": "occlusion_labels",
    "recovery": "recovery_labels",
}


def _events(sample: Dict) -> set[str]:
    return {name for name, key in EVENT_KEYS.items() if bool(sample[key].any())}


def _validate(sample: Dict, path: str) -> None:
    videos = sample["videos"]
    masks = sample["masks"]
    background = sample["background_masks"]
    time = videos.shape[0]
    if tuple(masks.shape[:2]) != (time, 2):
        raise AssertionError(f"{path}: invalid fighter mask shape {tuple(masks.shape)}")
    if not torch.all(masks.sum(dim=1) + background == 1):
        raise AssertionError(f"{path}: masks do not partition the frame")
    if sample["hit_labels"].shape[0] != time - 1:
        raise AssertionError(f"{path}: hit labels are not transition aligned")
    if sample["punch_miss_labels"].shape[:2] != (time - 1, 2):
        raise AssertionError(f"{path}: punch-miss labels are not transition/object aligned")
    metadata_events = set(sample["metadata"].get("interaction_events", []))
    if metadata_events != _events(sample):
        raise AssertionError(f"{path}: metadata events {metadata_events} != labels {_events(sample)}")


def _render(sample: Dict, stem: str, fps: int) -> None:
    video = sample["videos"].permute(0, 2, 3, 1).numpy().astype(np.uint8)
    frames: List[np.ndarray] = []
    time = len(video)
    for t, frame in enumerate(video):
        transition = min(t, time - 2)
        arms = sample["arm_lengths"][t].int().tolist()
        score = sample["scores"][t].int().tolist()
        occ = sample["occlusion_labels"][t].int().tolist()
        lines = [
            f"frame={t}/{time-1} score={score} arms={arms}",
            f"near={int(sample['near_labels'][t])} contact={int(sample['contact_labels'][t])} occ(P/E)={occ}",
            (
                f"transition={transition} hit={int(sample['hit_labels'][transition])} "
                f"actor={int(sample['hit_actor'][transition])} receiver={int(sample['hit_receiver'][transition])} "
                f"miss(P/E)={sample['punch_miss_labels'][transition].int().tolist()} "
                f"recovery={int(sample['recovery_labels'][transition])}"
            ),
        ]
        frames.append(_label_frame(frame, lines, scale=3))
    _write_video(frames, stem, fps)


def audit(data_root: str, output_dir: str, max_videos: int, fps: int) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    report = {"data_root": data_root, "splits": {}}
    selected_manifest = []
    for split in ("train", "val"):
        files = sorted(glob.glob(os.path.join(data_root, split, "*.pt")))
        sample_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        actor_counts: Counter[str] = Counter()
        candidates = []
        for path in files:
            sample = torch.load(path, map_location="cpu", weights_only=False)
            _validate(sample, path)
            events = _events(sample)
            sample_counts.update(events)
            for event, key in EVENT_KEYS.items():
                label_counts[event] += int(sample[key].sum())
            for actor in sample["hit_actor"][sample["hit_actor"] >= 0].tolist():
                actor_counts["Player" if int(actor) == 0 else "Enemy"] += 1
            candidates.append((path, events))
        report["splits"][split] = {
            "samples": len(files),
            "sample_event_counts": dict(sample_counts),
            "label_counts": dict(label_counts),
            "hit_actor_counts": dict(actor_counts),
        }
        if split == "train":
            selected_paths, covered = [], set()
            for event in EVENT_KEYS:
                for path, events in candidates:
                    if event in events and path not in selected_paths:
                        selected_paths.append(path)
                        covered.update(events)
                        break
                if len(selected_paths) >= max_videos:
                    break
            for path, events in candidates:
                if len(selected_paths) >= max_videos:
                    break
                if path not in selected_paths:
                    selected_paths.append(path)
                    covered.update(events)
            for index, path in enumerate(selected_paths):
                sample = torch.load(path, map_location="cpu", weights_only=False)
                stem = os.path.join(output_dir, f"interaction_{index:02d}_{os.path.basename(path)[:-3]}")
                _render(sample, stem, fps)
                selected_manifest.append(
                    {"path": os.path.abspath(path), "events": sorted(_events(sample)), "stem": os.path.basename(stem)}
                )
    report["videos"] = selected_manifest
    with open(os.path.join(output_dir, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_videos", type=int, default=6)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    audit(args.data_root, args.output_dir, args.max_videos, args.fps)


if __name__ == "__main__":
    main()
