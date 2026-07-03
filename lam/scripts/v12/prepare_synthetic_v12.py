"""
Validate or convert the existing synthetic dataset to the V12 schema.

By default this script is read-only and only checks a subset of samples. Pass
--convert to write normalized samples to a separate output root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.v12_dataset import V12ObjectVideoDataset, normalize_v12_sample, validate_v12_sample


def _check_split(root: str, split: str, max_samples: int) -> Dict:
    split_dir = os.path.join(root, split)
    dataset = V12ObjectVideoDataset(split_dir, task_name="Synthetic-Minimal", max_samples=max_samples)
    errors = []
    first_shape = None
    for idx in tqdm(range(len(dataset)), desc=f"check {split}"):
        sample = dataset[idx]
        sample_errors = validate_v12_sample(sample)
        if sample_errors:
            errors.append({"index": idx, "errors": sample_errors})
        if first_shape is None:
            first_shape = {
                "videos": list(sample["videos"].shape),
                "masks": list(sample["masks"].shape),
                "actions": list(sample["actions"].shape),
            }
    return {
        "split": split,
        "checked": len(dataset),
        "first_shape": first_shape,
        "errors": errors[:10],
        "num_errors": len(errors),
    }


def _convert_split(root: str, out_root: str, split: str, max_samples: int) -> int:
    in_dir = os.path.join(root, split)
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".pt"))
    if max_samples > 0:
        files = files[:max_samples]
    for idx, name in enumerate(tqdm(files, desc=f"convert {split}")):
        raw = torch.load(os.path.join(in_dir, name), map_location="cpu")
        sample = normalize_v12_sample(
            raw,
            task_name="Synthetic-Minimal",
            split=split,
            sample_index=idx,
            output_format="t c h w",
        )
        # Store videos as uint8 TCHW to keep disk size comparable to the old dataset.
        sample["videos"] = (sample["videos"].clamp(0, 1) * 255).round().to(torch.uint8)
        sample["masks"] = (sample["masks"] > 0.5).to(torch.uint8)
        torch.save(sample, os.path.join(out_dir, name))
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/synthetic_multi_actor")
    parser.add_argument("--out_root", default="data/v12/synthetic_minimal")
    parser.add_argument("--max_samples", type=int, default=0, help="0 checks/converts all samples")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    report = {"data_root": args.data_root, "splits": []}
    for split in ("train", "val"):
        split_dir = os.path.join(args.data_root, split)
        if not os.path.isdir(split_dir):
            report["splits"].append({"split": split, "error": f"missing {split_dir}"})
            continue
        checked = _check_split(args.data_root, split, args.max_samples)
        report["splits"].append(checked)

    if args.convert:
        converted = {}
        for split in ("train", "val"):
            if os.path.isdir(os.path.join(args.data_root, split)):
                converted[split] = _convert_split(args.data_root, args.out_root, split, args.max_samples)
        report["converted_to"] = args.out_root
        report["converted"] = converted

    if args.summary:
        os.makedirs(os.path.dirname(args.summary), exist_ok=True)
        with open(args.summary, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
