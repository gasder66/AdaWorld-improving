"""
Generate a branching synthetic dataset for V12.1 Experiment D.

Key property: p(s_{t+1} | s_t) is MULTI-MODAL.
All samples start from identical initial positions.
At each step, random actions (stay/up/down/left/right) determine the next state.
Same s_t -> 5^K different s_{t+1}.

Usage:
  PYTHONPATH=lam python lam/scripts/v12/gen_branching_synthetic.py \
      --n_train 5000 --n_val 500 --out data/v12/branching_synthetic
"""
import argparse
import os
import sys

import numpy as np
import torch


GRID = 64
IMG = 256
SCALE = IMG // GRID  # pixels per grid cell
T = 5
K = 4

OBJECT_SIZE = 6  # grid cells per object side

OBJECT_COLORS = np.array([
    [230, 50, 50],    # red
    [50, 100, 230],   # blue
    [235, 180, 45],   # gold
    [150, 80, 180],   # purple
], dtype=np.uint8)

ACTION_DELTA = {
    0: (0, 0),    # stay
    1: (0, -1),   # up
    2: (0, 1),    # down
    3: (-1, 0),   # left
    4: (1, 0),    # right
}

INIT_POSITIONS = np.array([
    [4, 4],              # top-left
    [GRID - 14, 4],      # top-right
    [4, GRID - 14],      # bottom-left
    [GRID - 14, GRID - 14],  # bottom-right
], dtype=np.int32)


def _render_object(frame, mask, px, py, color_idx):
    px = max(0, min(GRID - OBJECT_SIZE, int(px)))
    py = max(0, min(GRID - OBJECT_SIZE, int(py)))
    px_pix = px * SCALE
    py_pix = py * SCALE
    frame[py_pix:py_pix + OBJECT_SIZE * SCALE,
          px_pix:px_pix + OBJECT_SIZE * SCALE] = OBJECT_COLORS[color_idx]
    mask[py_pix:py_pix + OBJECT_SIZE * SCALE,
         px_pix:px_pix + OBJECT_SIZE * SCALE] = 1


def _positions_to_bboxes(positions):
    bboxes = np.zeros((T, K, 4), dtype=np.float32)
    for t in range(T):
        for k in range(K):
            px, py = positions[t, k]
            x1 = px * SCALE
            y1 = py * SCALE
            x2 = (px + OBJECT_SIZE) * SCALE - 1
            y2 = (py + OBJECT_SIZE) * SCALE - 1
            bboxes[t, k] = [x1, y1, x2, y2]
    return bboxes


def generate_sample(rng: np.random.RandomState):
    # All samples start from identical INIT_POSITIONS.
    positions = np.zeros((T, K, 2), dtype=np.int32)
    positions[0] = INIT_POSITIONS

    actions = np.full((T - 1, K), -1, dtype=np.int64)

    for k in range(K):
        px, py = float(positions[0, k, 0]), float(positions[0, k, 1])
        for t in range(1, T):
            act = int(rng.randint(0, 5))
            actions[t - 1, k] = act
            dx, dy = ACTION_DELTA[act]
            px += dx
            py += dy
            px = max(0, min(GRID - OBJECT_SIZE, px))
            py = max(0, min(GRID - OBJECT_SIZE, py))
            positions[t, k] = [int(px), int(py)]

    videos = np.zeros((T, IMG, IMG, 3), dtype=np.uint8)
    masks = np.zeros((T, K, IMG, IMG), dtype=np.uint8)
    bboxes = _positions_to_bboxes(positions)

    for t in range(T):
        for k in range(K):
            _render_object(videos[t], masks[t, k],
                           positions[t, k, 0], positions[t, k, 1], k)

    valid_mask = np.ones((T, K), dtype=bool)
    actor_ids = np.arange(K, dtype=np.int64)
    object_types = np.arange(K, dtype=np.int64)
    event_labels = np.full((T - 1, K), -1, dtype=np.int64)

    sample = {
        "videos": torch.from_numpy(videos),
        "masks": torch.from_numpy(masks),
        "bboxes": torch.from_numpy(bboxes),
        "positions": torch.from_numpy(positions),
        "actions": torch.from_numpy(actions),
        "actor_ids": torch.from_numpy(actor_ids),
        "object_types": torch.from_numpy(object_types),
        "valid_mask": torch.from_numpy(valid_mask),
        "event_labels": torch.from_numpy(event_labels),
        "num_actors": K,
        "metadata": {
            "task_name": "BranchingSynthetic-v0",
            "action_names": ["stay", "up", "down", "left", "right"],
            "has_object_annotations": True,
            "branching": True,
            "same_initial_config": True,
        },
    }
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--out", type=str, default="data/v12/branching_synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    for split_name, n_samples in [("train", args.n_train), ("val", args.n_val)]:
        split_dir = os.path.join(args.out, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for i in range(n_samples):
            sample = generate_sample(rng)
            path = os.path.join(split_dir, f"sample_{i:06d}.pt")
            torch.save(sample, path)
        print(f"  Saved {n_samples} samples to {split_dir}")

    print(f"  Dataset ready: {args.out}")


if __name__ == "__main__":
    main()
