"""Render isolated-punch clips with synchronized OCAtari RAM arm labels."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_object_dataset import BoxingObjectDataset
from scripts.v16.visualize_boxing_stage1 import _label_frame, _write_video


def _signature(sample: Dict) -> tuple:
    active = sample["punch_labels"].any(dim=0).tolist()
    sides = []
    for slot in range(2):
        values = sorted(set(int(v) for v in sample["punch_side_labels"][:, slot].tolist()) - {0})
        sides.append(tuple(values))
    return tuple(active), tuple(sides)


def choose_samples(dataset: BoxingObjectDataset, count: int) -> List[int]:
    ranked = []
    for index in range(len(dataset)):
        sample = dataset[index]
        arm_range = sample["arm_lengths"].amax(dim=0) - sample["arm_lengths"].amin(dim=0)
        ranked.append((index, float(arm_range.max()), _signature(sample)))
    selected, signatures = [], set()
    for index, _score, signature in sorted(ranked, key=lambda row: row[1], reverse=True):
        if signature not in signatures:
            selected.append(index)
            signatures.add(signature)
        if len(selected) >= count:
            break
    for index, _score, _signature_value in sorted(ranked, key=lambda row: row[1], reverse=True):
        if len(selected) >= count:
            break
        if index not in selected:
            selected.append(index)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = BoxingObjectDataset(os.path.join(args.data_root, "train"))
    indices = choose_samples(dataset, args.count)
    manifest = []
    for output_index, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]
        video = (sample["videos"].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        frames = []
        for t, frame in enumerate(video):
            arms = sample["arm_lengths"][t].int().tolist()
            sides = sample["punch_side_labels"][t].int().tolist()
            frames.append(
                _label_frame(
                    frame,
                    [
                        f"sample={dataset_index} frame={t}/{len(video)-1}",
                        f"Player arms L/R={arms[0]} side={sides[0]}",
                        f"Enemy  arms L/R={arms[1]} side={sides[1]}",
                    ],
                    scale=3,
                )
            )
        stem = os.path.join(args.output_dir, f"isolated_punch_{output_index:02d}_sample_{dataset_index:06d}")
        _write_video(frames, stem, args.fps)
        manifest.append(
            {
                "dataset_index": dataset_index,
                "signature": _signature(sample),
                "arm_min": sample["arm_lengths"].amin(dim=0).int().tolist(),
                "arm_max": sample["arm_lengths"].amax(dim=0).int().tolist(),
                "mask_area_min": int(sample["masks"].sum(dim=(-1, -2)).min()),
                "mask_area_max": int(sample["masks"].sum(dim=(-1, -2)).max()),
            }
        )
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"indices": indices, "samples": manifest}, f, indent=2)
    print(json.dumps({"indices": indices, "samples": manifest}, indent=2))


if __name__ == "__main__":
    main()
