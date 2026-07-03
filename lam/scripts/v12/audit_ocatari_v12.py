"""Audit OCAtari-generated V12 datasets and write a compact report."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
def _iter_files(root: str, splits: Iterable[str]) -> Iterable[Tuple[str, str]]:
    for split in splits:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            if name.endswith(".pt"):
                yield split, os.path.join(split_dir, name)


def _bbox_from_mask(mask: torch.Tensor) -> torch.Tensor:
    ys, xs = torch.where(mask > 0)
    if ys.numel() == 0:
        return torch.zeros(4, dtype=torch.float32)
    return torch.tensor(
        [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1],
        dtype=torch.float32,
    )


def audit_root(root: str, splits: List[str], max_samples: int = 0, exact_mask_bbox_samples: int = 32) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "root": root,
        "splits": {},
        "total_samples": 0,
        "errors": [],
    }
    for split in splits:
        files = [p for s, p in _iter_files(root, [split])]
        if max_samples > 0:
            files = files[:max_samples]
        stats: Dict[str, Any] = {
            "samples": len(files),
            "shape_counts": Counter(),
            "category_counts": Counter(),
            "valid_objects_per_frame": Counter(),
            "slot_action_counts": Counter(),
            "env_action_counts": Counter(),
            "bbox_area": {"min": None, "max": None, "sum": 0.0, "count": 0},
            "mask_pixels": {"min": None, "max": None, "sum": 0.0, "count": 0},
            "ram_length_counts": Counter(),
            "num_schema_errors": 0,
            "num_invalid_bbox": 0,
            "num_mask_bbox_mismatch": 0,
            "num_exact_mask_bbox_mismatch": 0,
            "first_metadata": None,
        }
        for sample_idx, path in enumerate(tqdm(files, desc=f"audit {os.path.basename(root)} {split}")):
            raw = torch.load(path, map_location="cpu")
            sample = raw
            errors = []
            for key in ("videos", "masks", "bboxes", "positions", "actions", "actor_ids", "object_types", "valid_mask", "metadata"):
                if key not in sample:
                    errors.append(f"missing key: {key}")
            if not errors:
                videos = sample["videos"]
                masks = sample["masks"]
                bboxes = sample["bboxes"]
                actions = sample["actions"]
                valid = sample["valid_mask"]
                if videos.ndim != 4 or videos.shape[1] != 3:
                    errors.append(f"videos must be raw (T,3,H,W), got {tuple(videos.shape)}")
                T = int(videos.shape[0])
                K = int(masks.shape[1]) if masks.ndim == 4 else -1
                H = int(videos.shape[2]) if videos.ndim == 4 else -1
                W = int(videos.shape[3]) if videos.ndim == 4 else -1
                if masks.shape != (T, K, H, W):
                    errors.append(f"masks must be {(T, K, H, W)}, got {tuple(masks.shape)}")
                if bboxes.shape != (T, K, 4):
                    errors.append(f"bboxes must be {(T, K, 4)}, got {tuple(bboxes.shape)}")
                if actions.shape != (max(T - 1, 0), K):
                    errors.append(f"actions must be {(max(T - 1, 0), K)}, got {tuple(actions.shape)}")
                if valid.shape != (T, K):
                    errors.append(f"valid_mask must be {(T, K)}, got {tuple(valid.shape)}")
            if errors:
                stats["num_schema_errors"] += 1
                if len(report["errors"]) < 20:
                    report["errors"].append({"path": path, "errors": errors})
                continue
            if stats["first_metadata"] is None:
                metadata = dict(sample["metadata"])
                if "ram_states" in metadata:
                    metadata["ram_states"] = f"{len(metadata['ram_states'])} frames"
                stats["first_metadata"] = metadata

            videos = sample["videos"]
            masks = sample["masks"]
            bboxes = sample["bboxes"]
            valid = sample["valid_mask"]
            actions = sample["actions"]
            env_actions = sample.get("env_actions")
            metadata = sample["metadata"]
            categories = metadata.get("slot_categories", [])

            stats["shape_counts"][str(tuple(videos.shape))] += 1
            for t in range(valid.shape[0]):
                stats["valid_objects_per_frame"][int(valid[t].sum().item())] += 1
            for value in actions[valid[:-1] if valid.shape[0] > 1 else torch.zeros_like(actions, dtype=torch.bool)].reshape(-1):
                stats["slot_action_counts"][int(value.item())] += 1
            if env_actions is not None:
                for value in env_actions.reshape(-1):
                    stats["env_action_counts"][int(value.item())] += 1
            for cat in categories:
                stats["category_counts"][str(cat)] += 1
            ram_states = metadata.get("ram_states", [])
            for ram in ram_states:
                stats["ram_length_counts"][len(ram)] += 1

            widths = bboxes[..., 2] - bboxes[..., 0]
            heights = bboxes[..., 3] - bboxes[..., 1]
            areas = widths * heights
            mask_pixels = masks.sum(dim=(-2, -1)).float()
            valid_areas = areas[valid]
            valid_pixels = mask_pixels[valid]
            stats["num_invalid_bbox"] += int(((widths <= 0) | (heights <= 0))[valid].sum().item())
            stats["num_mask_bbox_mismatch"] += int((valid_areas != valid_pixels).sum().item())
            for key, values in (("bbox_area", valid_areas), ("mask_pixels", valid_pixels)):
                if values.numel() == 0:
                    continue
                bucket = stats[key]
                min_value = float(values.min().item())
                max_value = float(values.max().item())
                sum_value = float(values.sum().item())
                bucket["min"] = min_value if bucket["min"] is None else min(bucket["min"], min_value)
                bucket["max"] = max_value if bucket["max"] is None else max(bucket["max"], max_value)
                bucket["sum"] += sum_value
                bucket["count"] += int(values.numel())

            if sample_idx < exact_mask_bbox_samples:
                T, K = valid.shape
                for t in range(T):
                    for k in range(K):
                        if not bool(valid[t, k]):
                            continue
                        mask_box = _bbox_from_mask(masks[t, k])
                        if not torch.allclose(mask_box, bboxes[t, k], atol=0.0):
                            stats["num_exact_mask_bbox_mismatch"] += 1

        for key in ("bbox_area", "mask_pixels"):
            bucket = stats[key]
            bucket["mean"] = bucket["sum"] / bucket["count"] if bucket["count"] else None
        for key in (
            "shape_counts",
            "category_counts",
            "valid_objects_per_frame",
            "slot_action_counts",
            "env_action_counts",
            "ram_length_counts",
        ):
            stats[key] = dict(sorted(stats[key].items(), key=lambda kv: str(kv[0])))
        report["splits"][split] = stats
        report["total_samples"] += len(files)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--exact_mask_bbox_samples", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    reports = [
        audit_root(root, args.splits, args.max_samples, args.exact_mask_bbox_samples)
        for root in args.roots
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reports": reports}, f, indent=2)
    print(json.dumps({"out": args.out, "total_roots": len(reports), "total_samples": sum(r["total_samples"] for r in reports)}, indent=2))
    if any(r["errors"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
