"""
预生成大规模多主体合成数据集，保存到磁盘。

核心设计：
1. A2D 级别规模：4000 训练样本 + 500 验证样本
2. 每样本 5 帧（给时序模型提供连续运动信息）
3. 动态背景：每帧独立随机纹理，消除静态信号主导
4. 像素级标注：每个主体有独立的像素级 binary mask
5. 固定（预生成到磁盘，可复现）
6. 空间标签：每个主体每帧的网格坐标和像素坐标

数据格式：
  data/synthetic_multi_actor/
    train/
      sample_000000.pt  → {
        "videos": (T, 3, H, W),       # float32 [0,1], CHW
        "masks": (T, max_actors, H, W), # float32 binary mask per actor
        "positions": (T, max_actors, 2), # long, grid coordinates (row, col)
        "actions": (T-1, max_actors),   # long, action labels (-1=padding)
        "num_actors": int,              # actual actor count
        "frame_seeds": (T,)             # background seeds for each frame
      }
      ...
    val/
      sample_000000.pt
      ...
"""

import os
import sys
import math
import time
from typing import Dict, List, Tuple
from multiprocessing import Pool
import functools

import torch
import numpy as np
from tqdm import tqdm


# ========== 动作定义 ==========
ACTION_NAMES = ["stay", "up", "down", "left", "right"]
ACTION_DELTAS = {
    0: (0, 0),   # stay
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}

# 主体形状
SHAPE_SQUARE = 0
SHAPE_CIRCLE = 1
SHAPE_TRIANGLE = 2

# 预定义颜色（RGB，高饱和度以便模型区分）
ACTOR_COLORS = [
    (1.0, 0.2, 0.2),  # 红
    (0.2, 0.8, 0.2),  # 绿
    (0.2, 0.4, 1.0),  # 蓝
    (1.0, 0.8, 0.1),  # 黄
    (0.8, 0.3, 0.8),  # 紫
]


# ========== 渲染函数 ==========

def _draw_square(canvas: np.ndarray, mask: np.ndarray, actor_idx: int,
                 grid_r: int, grid_c: int, cell_size: int,
                 color: Tuple[float, ...]) -> None:
    """绘制 2x2 格子的正方形，同时更新 mask。"""
    r0 = grid_r * cell_size
    c0 = grid_c * cell_size
    size = 2 * cell_size
    canvas[r0:r0 + size, c0:c0 + size] = color
    mask[r0:r0 + size, c0:c0 + size] = 1.0


def _draw_circle(canvas: np.ndarray, mask: np.ndarray, actor_idx: int,
                 grid_r: int, grid_c: int, cell_size: int,
                 color: Tuple[float, ...]) -> None:
    """绘制 2x2 格子的内切圆，同时更新 mask。"""
    r0 = grid_r * cell_size
    c0 = grid_c * cell_size
    size = 2 * cell_size
    cy, cx = size / 2 - 0.5, size / 2 - 0.5
    radius = size / 2
    for dr in range(size):
        for dc in range(size):
            dist = math.sqrt((dr - cy) ** 2 + (dc - cx) ** 2)
            if dist <= radius:
                canvas[r0 + dr, c0 + dc] = color
                mask[r0 + dr, c0 + dc] = 1.0


def _draw_triangle(canvas: np.ndarray, mask: np.ndarray, actor_idx: int,
                   grid_r: int, grid_c: int, cell_size: int,
                   color: Tuple[float, ...]) -> None:
    """绘制 2x2 格子的等腰三角形（顶点朝上）。"""
    r0 = grid_r * cell_size
    c0 = grid_c * cell_size
    size = 2 * cell_size
    for dr in range(size):
        half_width = (dr + 0.5) / size * (size / 2)
        center = size / 2 - 0.5
        for dc in range(size):
            if abs(dc - center) <= half_width:
                canvas[r0 + dr, c0 + dc] = color
                mask[r0 + dr, c0 + dc] = 1.0


DRAW_FNS = {
    SHAPE_SQUARE: _draw_square,
    SHAPE_CIRCLE: _draw_circle,
    SHAPE_TRIANGLE: _draw_triangle,
}


def render_frame(positions: List[Tuple[int, int]],
                 shapes: List[int],
                 color_indices: List[int],
                 num_actors: int,
                 max_actors: int,
                 resolution: int,
                 cell_size: int,
                 grid_size: int,
                 bg_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    渲染一帧图像和像素级 mask。

    Returns:
        canvas: (H, W, 3) float32 [0,1]  图像
        masks: (max_actors, H, W) float32 binary  每个 actor 的独立 mask
    """
    # 动态背景：每帧独立噪声
    rng = np.random.RandomState(bg_seed)
    canvas = rng.rand(resolution, resolution, 3).astype(np.float32) * 0.15 + 0.05  # [0.05, 0.20]

    masks = np.zeros((max_actors, resolution, resolution), dtype=np.float32)

    for i in range(num_actors):
        pos = positions[i]
        shape = shapes[i]
        color = ACTOR_COLORS[color_indices[i] % len(ACTOR_COLORS)]
        draw_fn = DRAW_FNS[shape]
        draw_fn(canvas, masks[i], i, pos[0], pos[1], cell_size, color)

    return canvas, masks


def generate_single_sample(args: tuple) -> Dict:
    """生成一个完整的多主体样本。"""
    (idx, resolution, num_frames, min_actors, max_actors,
     grid_size, seed_offset) = args

    cell_size = resolution // grid_size
    max_pos = grid_size - 2  # 2x2 主体，左上角最大坐标
    sample_seed = 1000000 + idx + seed_offset
    rng = np.random.RandomState(sample_seed)

    # 随机选择主体数量
    num_actors = rng.randint(min_actors, max_actors + 1)

    # 为每个主体分配形状和颜色（确保多样性）
    shapes = [rng.randint(0, 3) for _ in range(num_actors)]
    color_indices = list(range(num_actors))  # 每个 actor 不同颜色

    # 随机生成不重叠的初始位置（第 0 帧）
    positions = _place_non_overlapping(num_actors, max_pos, rng)

    # 生成所有帧的位置和动作
    all_positions = [positions]
    all_actions = []

    for t in range(1, num_frames):
        prev_positions = all_positions[-1]
        new_positions = []
        frame_actions = []

        for i in range(num_actors):
            # 随机选择动作，偏好转弯/运动（30% stay, 70% move）
            action = rng.choice([0, 1, 1, 2, 2, 3, 3, 4, 4])
            dr, dc = ACTION_DELTAS[action]
            new_r = max(0, min(max_pos, prev_positions[i][0] + dr))
            new_c = max(0, min(max_pos, prev_positions[i][1] + dc))
            new_pos = (new_r, new_c)

            # 检查是否与已放置的其他主体重叠
            if _check_overlap(new_pos, new_positions, grid_size):
                new_pos = prev_positions[i]  # 保持不动
                action = 0

            new_positions.append(new_pos)
            frame_actions.append(action)

        all_positions.append(new_positions)
        all_actions.append(frame_actions)

    # 渲染所有帧
    frames = []
    masks_list = []
    for t in range(num_frames):
        bg_seed = sample_seed * 100 + t
        canvas, masks = render_frame(
            all_positions[t], shapes, color_indices,
            num_actors, max_actors,
            resolution, cell_size, grid_size, bg_seed,
        )
        frames.append(canvas)
        masks_list.append(masks)

    # 转换为 tensor（使用 uint8 压缩存储）
    # videos: (T, 3, H, W) CHW 格式，uint8 [0, 255]
    videos = torch.from_numpy(np.stack(frames, axis=0))  # (T, H, W, C)
    videos = videos.permute(0, 3, 1, 2).float()  # (T, C, H, W)
    videos = (videos * 255).clamp(0, 255).to(torch.uint8)

    # masks: (T, max_actors, H, W) - bool 压缩为 uint8
    masks = torch.from_numpy(np.stack(masks_list, axis=0)).float()  # (T, max_a, H, W)
    masks = (masks > 0.5).to(torch.uint8)

    # positions: (T, max_actors, 2)  padding = (-1, -1)
    positions_tensor = torch.full((num_frames, max_actors, 2), -1, dtype=torch.long)
    for t in range(num_frames):
        for i in range(num_actors):
            positions_tensor[t, i, 0] = all_positions[t][i][0]
            positions_tensor[t, i, 1] = all_positions[t][i][1]

    # actions: (T-1, max_actors)  padding = -1
    actions_tensor = torch.full((num_frames - 1, max_actors), -1, dtype=torch.long)
    for t in range(num_frames - 1):
        for i in range(num_actors):
            actions_tensor[t, i] = all_actions[t][i]

    return {
        "videos": videos,           # (T, 3, H, W) uint8
        "masks": masks,             # (T, max_actors, H, W) uint8 binary
        "positions": positions_tensor,  # (T, max_actors, 2)
        "actions": actions_tensor,      # (T-1, max_actors)
        "num_actors": num_actors,
    }


def _place_non_overlapping(num_actors: int, max_pos: int, rng: np.random.RandomState) -> List[Tuple[int, int]]:
    """放置不重叠的 2x2 主体。"""
    positions = []
    occupied = set()
    max_attempts = 200
    for _ in range(num_actors):
        for _attempt in range(max_attempts):
            r = rng.randint(0, max_pos + 1)
            c = rng.randint(0, max_pos + 1)
            cells = {(r + dr, c + dc) for dr in range(2) for dc in range(2)}
            if not cells & occupied:
                positions.append((r, c))
                occupied |= cells
                break
        else:
            # 无法放置，回退到随机位置
            r = rng.randint(0, max_pos + 1)
            c = rng.randint(0, max_pos + 1)
            positions.append((r, c))
    return positions


def _check_overlap(new_pos: Tuple[int, int],
                   existing: List[Tuple[int, int]],
                   grid_size: int) -> bool:
    """检查新位置是否与已有主体重叠。"""
    new_cells = {(new_pos[0] + dr, new_pos[1] + dc)
                 for dr in range(2) for dc in range(2)}
    for pos in existing:
        cells = {(pos[0] + dr, pos[1] + dc) for dr in range(2) for dc in range(2)}
        if new_cells & cells:
            return True
    return False


# ========== 主生成函数 ==========

DATASET_CONFIG = {
    "resolution": 256,
    "num_frames": 5,
    "min_actors": 2,
    "max_actors": 4,
    "grid_size": 8,
    # 每个格子 32x32，主体 2x2=64x64 像素
}


def generate_dataset(output_dir: str, num_samples: int, seed_offset: int,
                     num_workers: int = 8) -> None:
    """生成并保存数据集到磁盘。"""
    os.makedirs(output_dir, exist_ok=True)

    args_list = []
    for i in range(num_samples):
        args_list.append((
            i,                                   # idx
            DATASET_CONFIG["resolution"],         # resolution
            DATASET_CONFIG["num_frames"],         # num_frames
            DATASET_CONFIG["min_actors"],         # min_actors
            DATASET_CONFIG["max_actors"],         # max_actors
            DATASET_CONFIG["grid_size"],          # grid_size
            seed_offset,                          # seed_offset
        ))

    # 单进程直接生成（兼容性好）
    print(f"Generating {num_samples} samples to {output_dir} ...")
    t0 = time.time()

    for idx, args in enumerate(tqdm(args_list, desc="Generating")):
        sample = generate_single_sample(args)
        save_path = os.path.join(output_dir, f"sample_{idx:06d}.pt")
        torch.save(sample, save_path)

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            eta = (num_samples - idx - 1) / speed
            print(f"  [{idx + 1}/{num_samples}] {speed:.0f} samples/s, "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

    total_time = time.time() - t0
    print(f"Done! {num_samples} samples in {total_time:.0f}s "
          f"({num_samples / total_time:.0f} samples/s)")
    print(f"Output: {output_dir}")


def main():
    data_root = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_multi_actor")

    # 训练集
    train_dir = os.path.join(data_root, "train")
    generate_dataset(train_dir, num_samples=4000, seed_offset=0)

    # 验证集（固定，用于评估）
    val_dir = os.path.join(data_root, "val")
    generate_dataset(val_dir, num_samples=500, seed_offset=1000000)

    # 保存配置信息
    config_path = os.path.join(data_root, "dataset_config.txt")
    with open(config_path, "w") as f:
        f.write(f"resolution={DATASET_CONFIG['resolution']}\n")
        f.write(f"num_frames={DATASET_CONFIG['num_frames']}\n")
        f.write(f"min_actors={DATASET_CONFIG['min_actors']}\n")
        f.write(f"max_actors={DATASET_CONFIG['max_actors']}\n")
        f.write(f"grid_size={DATASET_CONFIG['grid_size']}\n")
        f.write(f"train_samples=4000\n")
        f.write(f"val_samples=500\n")
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
