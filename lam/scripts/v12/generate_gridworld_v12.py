"""
Generate GridWorld-MazeMultiObject-v0 in the V12 .pt schema.

This generator is self-contained: it uses a deterministic top-down grid renderer
instead of depending on MiniGrid internals, while keeping the MiniGrid-style
world structure of walls, goals, static objects, and moving actors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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
class GridConfig:
    task_name: str = "GridWorld-MazeMultiObject-v0"
    image_size: int = 256
    grid_size: int = 16
    num_frames: int = 5
    max_objects: int = 4
    wall_count: int = 14


COLORS = {
    "floor_a": np.array([232, 236, 231], dtype=np.uint8),
    "floor_b": np.array([219, 225, 221], dtype=np.uint8),
    "wall": np.array([54, 61, 69], dtype=np.uint8),
    "goal": np.array([88, 184, 120], dtype=np.uint8),
    "player": np.array([224, 62, 64], dtype=np.uint8),
    "enemy0": np.array([57, 104, 214], dtype=np.uint8),
    "enemy1": np.array([234, 178, 49], dtype=np.uint8),
    "static": np.array([142, 78, 174], dtype=np.uint8),
    "grid": np.array([188, 196, 192], dtype=np.uint8),
}


def _make_walls(rng: np.random.RandomState, cfg: GridConfig) -> np.ndarray:
    grid = np.zeros((cfg.grid_size, cfg.grid_size), dtype=bool)
    grid[0, :] = True
    grid[-1, :] = True
    grid[:, 0] = True
    grid[:, -1] = True

    # Short wall segments keep the space maze-like without creating hard traps.
    for _ in range(cfg.wall_count):
        horizontal = rng.rand() < 0.5
        length = rng.randint(2, 5)
        r = rng.randint(2, cfg.grid_size - 2)
        c = rng.randint(2, cfg.grid_size - 2)
        for i in range(length):
            rr = r + (0 if horizontal else i)
            cc = c + (i if horizontal else 0)
            if 1 <= rr < cfg.grid_size - 1 and 1 <= cc < cfg.grid_size - 1:
                grid[rr, cc] = True
    return grid


def _free_cells(walls: np.ndarray, reserved: Sequence[Tuple[int, int]] = ()) -> List[Tuple[int, int]]:
    reserved_set = set(reserved)
    cells = []
    for r in range(1, walls.shape[0] - 1):
        for c in range(1, walls.shape[1] - 1):
            if not walls[r, c] and (r, c) not in reserved_set:
                cells.append((r, c))
    return cells


def _sample_positions(rng: np.random.RandomState, walls: np.ndarray, n: int) -> List[Tuple[int, int]]:
    cells = _free_cells(walls)
    if len(cells) < n:
        raise RuntimeError("not enough free cells for GridWorld actors")
    idx = rng.choice(len(cells), size=n, replace=False)
    return [cells[int(i)] for i in idx]


def _try_step(
    pos: Tuple[int, int],
    action: int,
    walls: np.ndarray,
    occupied: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, int], int]:
    dr, dc = ACTION_DELTAS[action]
    nxt = (pos[0] + dr, pos[1] + dc)
    if walls[nxt] or nxt in occupied:
        return pos, 0
    return nxt, action


def _draw_cell(canvas: np.ndarray, cell: Tuple[int, int], color: np.ndarray, cell_size: int, margin: int = 2) -> None:
    r, c = cell
    y1 = r * cell_size + margin
    x1 = c * cell_size + margin
    y2 = (r + 1) * cell_size - margin
    x2 = (c + 1) * cell_size - margin
    canvas[y1:y2, x1:x2] = color


def _render(
    walls: np.ndarray,
    goal: Tuple[int, int],
    positions: Sequence[Tuple[int, int]],
    cfg: GridConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = cfg.image_size // cfg.grid_size
    canvas = np.zeros((cfg.image_size, cfg.image_size, 3), dtype=np.uint8)
    for r in range(cfg.grid_size):
        for c in range(cfg.grid_size):
            base = COLORS["floor_a"] if (r + c) % 2 == 0 else COLORS["floor_b"]
            canvas[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = base
            canvas[r * cell, c * cell : (c + 1) * cell] = COLORS["grid"]
            canvas[r * cell : (r + 1) * cell, c * cell] = COLORS["grid"]
            if walls[r, c]:
                canvas[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = COLORS["wall"]

    _draw_cell(canvas, goal, COLORS["goal"], cell, margin=4)

    masks = np.zeros((cfg.max_objects, cfg.image_size, cfg.image_size), dtype=np.uint8)
    boxes = np.zeros((cfg.max_objects, 4), dtype=np.float32)
    obj_colors = [COLORS["player"], COLORS["enemy0"], COLORS["enemy1"], COLORS["static"]]
    for k, pos in enumerate(positions):
        margin = 3 if k < 3 else 5
        _draw_cell(canvas, pos, obj_colors[k], cell, margin=margin)
        y1 = pos[0] * cell + margin
        x1 = pos[1] * cell + margin
        y2 = (pos[0] + 1) * cell - margin
        x2 = (pos[1] + 1) * cell - margin
        masks[k, y1:y2, x1:x2] = 1
        boxes[k] = np.array([x1, y1, x2, y2], dtype=np.float32)
    return canvas, masks, boxes


def generate_sample(idx: int, cfg: GridConfig, seed_offset: int, split: str) -> Dict:
    rng = np.random.RandomState(seed_offset + idx)
    walls = _make_walls(rng, cfg)
    positions0 = _sample_positions(rng, walls, cfg.max_objects + 1)
    goal = positions0[-1]
    positions = positions0[: cfg.max_objects]

    all_positions = [list(positions)]
    all_actions: List[List[int]] = []
    frames, masks, boxes = [], [], []

    for t in range(cfg.num_frames):
        frame, mask_t, box_t = _render(walls, goal, all_positions[-1], cfg)
        frames.append(frame)
        masks.append(mask_t)
        boxes.append(box_t)
        if t == cfg.num_frames - 1:
            break

        prev = list(all_positions[-1])
        next_pos: List[Tuple[int, int]] = []
        action_t: List[int] = []
        for k, pos in enumerate(prev):
            if k == cfg.max_objects - 1:
                next_pos.append(pos)
                action_t.append(0)
                continue
            occupied = next_pos + prev[k + 1 :]
            action = int(rng.choice([0, 1, 1, 2, 2, 3, 3, 4, 4]))
            pos_new, action_new = _try_step(pos, action, walls, occupied)
            next_pos.append(pos_new)
            action_t.append(action_new)
        all_positions.append(next_pos)
        all_actions.append(action_t)

    videos = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    masks_t = torch.from_numpy(np.stack(masks)).to(torch.uint8)
    bboxes = torch.from_numpy(np.stack(boxes)).float()
    positions_t = torch.tensor(all_positions, dtype=torch.long)
    actions = torch.tensor(all_actions, dtype=torch.long)
    valid_mask = torch.ones((cfg.num_frames, cfg.max_objects), dtype=torch.bool)
    object_types = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    return {
        "videos": videos,
        "masks": masks_t,
        "bboxes": bboxes,
        "positions": positions_t,
        "actions": actions,
        "actor_ids": torch.arange(cfg.max_objects, dtype=torch.long),
        "object_types": object_types,
        "valid_mask": valid_mask,
        "event_labels": torch.zeros((cfg.num_frames - 1, cfg.max_objects), dtype=torch.long),
        "num_actors": cfg.max_objects,
        "metadata": {
            "episode_id": int(idx),
            "frame_index": 0,
            "task_name": cfg.task_name,
            "map_id": int(seed_offset + idx),
            "split": split,
            "action_names": ACTION_NAMES,
            "has_object_annotations": True,
        },
    }


def generate_split(root: str, split: str, n: int, cfg: GridConfig, seed_offset: int) -> None:
    out_dir = os.path.join(root, split)
    os.makedirs(out_dir, exist_ok=True)
    for idx in tqdm(range(n), desc=f"gridworld {split}"):
        sample = generate_sample(idx, cfg, seed_offset, split)
        torch.save(sample, os.path.join(out_dir, f"sample_{idx:06d}.pt"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="data/v12/gridworld_maze_multiobject")
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--max_objects", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--validate_first", action="store_true")
    args = parser.parse_args()

    cfg = GridConfig(
        image_size=args.image_size,
        grid_size=args.grid_size,
        num_frames=args.num_frames,
        max_objects=args.max_objects,
    )
    if cfg.max_objects != 4:
        raise ValueError("V12 GridWorld v0 fixes max_objects=4 for player+2 enemies+static")

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "dataset_config.json"), "w") as f:
        json.dump({**cfg.__dict__, "train": args.train, "val": args.val}, f, indent=2)

    if args.validate_first:
        test = generate_sample(0, cfg, args.seed, "train")
        from lam.v12_dataset import normalize_v12_sample

        errors = validate_v12_sample(normalize_v12_sample(test, task_name=cfg.task_name))
        if errors:
            raise RuntimeError(f"schema validation failed: {errors}")

    generate_split(args.out_root, "train", args.train, cfg, args.seed)
    generate_split(args.out_root, "val", args.val, cfg, args.seed + 1_000_000)
    print(f"GridWorld V12 dataset saved to {args.out_root}")


if __name__ == "__main__":
    main()
