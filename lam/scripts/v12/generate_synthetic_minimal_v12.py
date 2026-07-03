"""
Generate a deliberately minimal no-overlap synthetic V12 dataset.

Stage 0 should be the easiest sanity check: static background, fixed actor
identities, axis-aligned movement, and hard separation between actor regions so
occlusion/collision cannot become an accidental difficulty source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.v12_dataset import ACTION_NAMES, validate_v12_sample


ACTION_DELTAS = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


@dataclass(frozen=True)
class MinimalConfig:
    task_name: str = "Synthetic-Minimal-NoOverlap-v0"
    image_size: int = 256
    grid_size: int = 16
    num_frames: int = 5
    max_objects: int = 4
    actor_cells: int = 2


COLORS = np.array(
    [
        [230, 65, 65],
        [67, 165, 88],
        [62, 111, 224],
        [224, 180, 45],
    ],
    dtype=np.uint8,
)


REGIONS = [
    (1, 1, 6, 6),
    (1, 9, 6, 14),
    (9, 1, 14, 6),
    (9, 9, 14, 14),
]


def _valid_in_region(pos: Tuple[int, int], region: Tuple[int, int, int, int], size: int) -> bool:
    r, c = pos
    r1, c1, r2, c2 = region
    return r1 <= r and c1 <= c and r + size - 1 <= r2 and c + size - 1 <= c2


def _sample_start(rng: np.random.RandomState, region: Tuple[int, int, int, int], size: int) -> Tuple[int, int]:
    r1, c1, r2, c2 = region
    return int(rng.randint(r1, r2 - size + 2)), int(rng.randint(c1, c2 - size + 2))


def _draw_actor(
    canvas: np.ndarray,
    mask: np.ndarray,
    pos: Tuple[int, int],
    color: np.ndarray,
    cell: int,
    actor_cells: int,
) -> np.ndarray:
    r, c = pos
    y1 = r * cell
    x1 = c * cell
    y2 = (r + actor_cells) * cell
    x2 = (c + actor_cells) * cell
    canvas[y1:y2, x1:x2] = color
    mask[y1:y2, x1:x2] = 1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def generate_sample(idx: int, cfg: MinimalConfig, seed_offset: int, split: str) -> Dict:
    rng = np.random.RandomState(seed_offset + idx)
    positions = [_sample_start(rng, region, cfg.actor_cells) for region in REGIONS]
    all_positions: List[List[Tuple[int, int]]] = [list(positions)]
    all_actions: List[List[int]] = []

    for _ in range(cfg.num_frames - 1):
        prev = all_positions[-1]
        next_pos = []
        actions = []
        for k, pos in enumerate(prev):
            candidates = [0, 1, 2, 3, 4]
            rng.shuffle(candidates)
            chosen_action = 0
            chosen_pos = pos
            for action in candidates:
                dr, dc = ACTION_DELTAS[int(action)]
                proposal = (pos[0] + dr, pos[1] + dc)
                if _valid_in_region(proposal, REGIONS[k], cfg.actor_cells):
                    chosen_action = int(action)
                    chosen_pos = proposal
                    break
            next_pos.append(chosen_pos)
            actions.append(chosen_action)
        all_positions.append(next_pos)
        all_actions.append(actions)

    cell = cfg.image_size // cfg.grid_size
    frames, masks, boxes = [], [], []
    for pos_t in all_positions:
        canvas = np.full((cfg.image_size, cfg.image_size, 3), 238, dtype=np.uint8)
        # Subtle quadrant guides make the "no overlap by design" property obvious.
        canvas[: cfg.image_size // 2, :, :] += np.array([0, 2, 0], dtype=np.uint8)
        canvas[:, : cfg.image_size // 2, :] += np.array([0, 0, 2], dtype=np.uint8)
        mask_t = np.zeros((cfg.max_objects, cfg.image_size, cfg.image_size), dtype=np.uint8)
        box_t = np.zeros((cfg.max_objects, 4), dtype=np.float32)
        for k, pos in enumerate(pos_t):
            box_t[k] = _draw_actor(canvas, mask_t[k], pos, COLORS[k], cell, cfg.actor_cells)
        frames.append(canvas)
        masks.append(mask_t)
        boxes.append(box_t)

    sample = {
        "videos": torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous(),
        "masks": torch.from_numpy(np.stack(masks)).to(torch.uint8),
        "bboxes": torch.from_numpy(np.stack(boxes)).float(),
        "positions": torch.tensor(all_positions, dtype=torch.long),
        "actions": torch.tensor(all_actions, dtype=torch.long),
        "actor_ids": torch.arange(cfg.max_objects, dtype=torch.long),
        "object_types": torch.arange(cfg.max_objects, dtype=torch.long),
        "valid_mask": torch.ones((cfg.num_frames, cfg.max_objects), dtype=torch.bool),
        "event_labels": torch.zeros((cfg.num_frames - 1, cfg.max_objects), dtype=torch.long),
        "num_actors": cfg.max_objects,
        "metadata": {
            "episode_id": int(idx),
            "frame_index": 0,
            "task_name": cfg.task_name,
            "split": split,
            "action_names": ACTION_NAMES,
            "has_object_annotations": True,
            "no_overlap_guarantee": True,
            "regions": REGIONS,
        },
    }
    return sample


def generate_split(root: str, split: str, n: int, cfg: MinimalConfig, seed_offset: int) -> None:
    out_dir = os.path.join(root, split)
    os.makedirs(out_dir, exist_ok=True)
    for idx in tqdm(range(n), desc=f"synthetic-minimal {split}"):
        torch.save(generate_sample(idx, cfg, seed_offset, split), os.path.join(out_dir, f"sample_{idx:06d}.pt"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="data/v12/synthetic_minimal_nooverlap")
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=24000)
    parser.add_argument("--validate_first", action="store_true")
    args = parser.parse_args()

    cfg = MinimalConfig(image_size=args.image_size, num_frames=args.num_frames)
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "dataset_config.json"), "w") as f:
        json.dump({**cfg.__dict__, "train": args.train, "val": args.val}, f, indent=2)
    if args.validate_first:
        from lam.v12_dataset import normalize_v12_sample

        errors = validate_v12_sample(normalize_v12_sample(generate_sample(0, cfg, args.seed, "train")))
        if errors:
            raise RuntimeError(errors)
    generate_split(args.out_root, "train", args.train, cfg, args.seed)
    generate_split(args.out_root, "val", args.val, cfg, args.seed + 1_000_000)
    print(f"Synthetic minimal no-overlap V12 dataset saved to {args.out_root}")


if __name__ == "__main__":
    main()
