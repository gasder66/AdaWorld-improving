"""Validate V12 dataset splits and print a compact schema report."""
from __future__ import annotations

import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.v12_dataset import V12ObjectVideoDataset, validate_v12_sample


def validate_split(root: str, split: str, max_samples: int) -> dict:
    split_dir = os.path.join(root, split)
    ds = V12ObjectVideoDataset(split_dir, max_samples=max_samples)
    errors = []
    first = None
    for idx in tqdm(range(len(ds)), desc=f"validate {split}"):
        sample = ds[idx]
        if first is None:
            first = {
                "videos": list(sample["videos"].shape),
                "masks": list(sample["masks"].shape),
                "bboxes": list(sample["bboxes"].shape),
                "actions": list(sample["actions"].shape),
                "valid_mask": list(sample["valid_mask"].shape),
                "metadata": sample["metadata"],
            }
        sample_errors = validate_v12_sample(sample)
        if sample_errors:
            errors.append({"index": idx, "errors": sample_errors})
    return {"split": split, "checked": len(ds), "first": first, "num_errors": len(errors), "errors": errors[:10]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = {"data_root": args.data_root, "splits": []}
    for split in args.splits:
        report["splits"].append(validate_split(args.data_root, split, args.max_samples))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if any(s["num_errors"] for s in report["splits"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
