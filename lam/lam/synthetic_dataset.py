"""
合成多主体数据集：8x8 网格，2x2 格子大小的主体，动态纹理背景。
用于验证 LAM 在多主体场景下的缺陷及多 slot 架构的改进效果。
"""
import math
from random import randint, random, choice
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# 动作定义: 0=静止, 1=上, 2=下, 3=左, 4=右
ACTION_NAMES = ["stay", "up", "down", "left", "right"]
ACTION_DELTAS = {
    0: (0, 0),   # stay
    1: (-1, 0),  # up (row decreases)
    2: (1, 0),   # down (row increases)
    3: (0, -1),  # left (col decreases)
    4: (0, 1),   # right (col increases)
}

# 主体形状
SHAPE_SQUARE = 0
SHAPE_CIRCLE = 1
SHAPE_TRIANGLE = 2


def _draw_square(canvas: np.ndarray, grid_r: int, grid_c: int,
                 cell_size: int, color: Tuple[float, ...]) -> None:
    """在 canvas 上绘制 2x2 格子的正方形。"""
    r0 = grid_r * cell_size
    c0 = grid_c * cell_size
    size = 2 * cell_size
    canvas[r0:r0 + size, c0:c0 + size] = color


def _draw_circle(canvas: np.ndarray, grid_r: int, grid_c: int,
                 cell_size: int, color: Tuple[float, ...]) -> None:
    """在 canvas 上绘制 2x2 格子的圆形（内切圆）。"""
    r0 = grid_r * cell_size
    c0 = grid_c * cell_size
    size = 2 * cell_size

    for dr in range(size):
        for dc in range(size):
            dist = math.sqrt((dr - size / 2 + 0.5) ** 2 + (dc - size / 2 + 0.5) ** 2)
            if dist <= size / 2:
                canvas[r0 + dr, c0 + dc] = color


def _draw_triangle(canvas: np.ndarray, grid_r: int, grid_c: int,
                   cell_size: int, color: Tuple[float, ...]) -> None:
    """在 canvas 上绘制 2x2 格子的三角形（等腰三角形，顶点朝上）。"""
    r0 = grid_r * cell_size
    c0 = grid_c * cell_size
    size = 2 * cell_size

    for dr in range(size):
        for dc in range(size):
            half_width = (dr + 1) / size * (size / 2)
            center_dc = size / 2 - 0.5
            if abs(dc - center_dc) <= half_width:
                canvas[r0 + dr, c0 + dc] = color


DRAW_FNS = {
    SHAPE_SQUARE: _draw_square,
    SHAPE_CIRCLE: _draw_circle,
    SHAPE_TRIANGLE: _draw_triangle,
}


def _generate_checkerboard_background(resolution: int, cell_size: int,
                                      seed: int) -> np.ndarray:
    """生成棋盘格纹理背景。"""
    bg = np.zeros((resolution, resolution, 3), dtype=np.float32)
    grid_size = resolution // cell_size
    for r in range(grid_size):
        for c in range(grid_size):
            if (r + c) % 2 == 0:
                bg[r * cell_size:(r + 1) * cell_size,
                   c * cell_size:(c + 1) * cell_size] = [0.15, 0.15, 0.15]
            else:
                bg[r * cell_size:(r + 1) * cell_size,
                   c * cell_size:(c + 1) * cell_size] = [0.10, 0.10, 0.10]
    return bg


def _generate_noise_background(resolution: int, seed: int) -> np.ndarray:
    """生成随机噪声纹理背景。"""
    rng = np.random.RandomState(seed)
    noise = rng.rand(resolution, resolution, 1).astype(np.float32) * 0.15 + 0.05
    bg = np.broadcast_to(noise, (resolution, resolution, 3)).copy()
    return bg


class SyntheticMultiActorDataset(Dataset):
    """
    合成多主体数据集。

    每个样本包含 2-5 个主体在 8x8 网格上运动，生成 2 帧视频。
    主体占 2x2 格子（64x64 像素），背景为动态纹理。
    """

    def __init__(
            self,
            resolution: int = 256,
            num_frames: int = 2,
            min_actors: int = 2,
            max_actors: int = 5,
            grid_size: int = 8,
            background_type: str = "checkerboard",
            samples_per_epoch: int = 100000,
            seed: int = 42,
    ) -> None:
        super(SyntheticMultiActorDataset, self).__init__()
        self.resolution = resolution
        self.num_frames = num_frames
        self.min_actors = min_actors
        self.max_actors = max_actors
        self.grid_size = grid_size
        self.cell_size = resolution // grid_size
        self.background_type = background_type
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed

        # 预定义颜色（足够区分 5 个主体）
        self.actor_colors = [
            (0.9, 0.2, 0.2),  # 红
            (0.2, 0.7, 0.2),  # 绿
            (0.2, 0.4, 0.9),  # 蓝
            (0.9, 0.8, 0.1),  # 黄
            (0.8, 0.3, 0.8),  # 紫
        ]

        # 预定义形状
        self.actor_shapes = [SHAPE_SQUARE, SHAPE_CIRCLE, SHAPE_TRIANGLE]

        assert resolution % grid_size == 0, \
            f"Resolution {resolution} must be divisible by grid_size {grid_size}"
        # 2x2 格子主体，网格需要至少 2x2
        assert grid_size >= 4, f"Grid size must be >= 4 for 2x2 actors"

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _random_positions(self, num_actors: int, g: torch.Generator) -> List[Tuple[int, int]]:
        """随机生成不重叠的 2x2 主体位置（左上角坐标）。"""
        max_pos = self.grid_size - 2  # 2x2 主体，左上角最大为 grid_size-2
        positions = []
        occupied = set()

        for _ in range(num_actors):
            for _attempt in range(100):
                r = int(torch.randint(0, max_pos + 1, (1,), generator=g).item())
                c = int(torch.randint(0, max_pos + 1, (1,), generator=g).item())
                # 检查 2x2 区域是否与已有主体重叠
                cells = {(r + dr, c + dc) for dr in range(2) for dc in range(2)}
                if not cells & occupied:
                    positions.append((r, c))
                    occupied |= cells
                    break
            else:
                # 无法放置更多主体，使用已有位置（极端情况）
                r = int(torch.randint(0, max_pos + 1, (1,), generator=g).item())
                c = int(torch.randint(0, max_pos + 1, (1,), generator=g).item())
                positions.append((r, c))

        return positions

    def _move_actor(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """根据动作移动主体，处理边界。"""
        dr, dc = ACTION_DELTAS[action]
        new_r = max(0, min(self.grid_size - 2, pos[0] + dr))
        new_c = max(0, min(self.grid_size - 2, pos[1] + dc))
        return (new_r, new_c)

    def _render_frame(self, positions: List[Tuple[int, int]],
                      shapes: List[int], colors: List[Tuple[float, ...]],
                      bg_seed: int) -> Tensor:
        """渲染一帧图像。"""
        # 生成背景 (numpy)
        if self.background_type == "checkerboard":
            canvas = _generate_checkerboard_background(
                self.resolution, self.cell_size, bg_seed)
        elif self.background_type == "noise":
            canvas = _generate_noise_background(self.resolution, bg_seed)
        else:
            canvas = np.zeros((self.resolution, self.resolution, 3), dtype=np.float32) + 0.05

        # 绘制主体
        for pos, shape, color in zip(positions, shapes, colors):
            draw_fn = DRAW_FNS[shape]
            draw_fn(canvas, pos[0], pos[1], self.cell_size, color)

        return torch.from_numpy(canvas)

    def __getitem__(self, idx: int) -> Dict:
        # 使用 idx + seed 生成可复现的随机数
        sample_seed = self.seed + idx
        g = torch.Generator()
        g.manual_seed(sample_seed)

        # 随机选择主体数量
        num_actors = int(torch.randint(
            self.min_actors, self.max_actors + 1, (1,), generator=g).item())

        # 随机分配形状和颜色
        shapes = [self.actor_shapes[i % len(self.actor_shapes)] for i in range(num_actors)]
        colors = [self.actor_colors[i % len(self.actor_colors)] for i in range(num_actors)]

        # 随机生成初始位置
        positions = self._random_positions(num_actors, g)

        # 随机生成每个主体的动作序列
        all_positions = [positions[:]]  # 第一帧的位置
        all_actions = []

        for t in range(1, self.num_frames):
            frame_actions = []
            new_positions = []
            for i in range(num_actors):
                action = int(torch.randint(0, 5, (1,), generator=g).item())
                new_pos = self._move_actor(all_positions[-1][i], action)
                # 检查是否与其他主体重叠
                occupied = set()
                for j, p in enumerate(new_positions):
                    for dr in range(2):
                        for dc in range(2):
                            occupied.add((p[0] + dr, p[1] + dc))
                new_cells = {(new_pos[0] + dr, new_pos[1] + dc)
                             for dr in range(2) for dc in range(2)}
                if new_cells & occupied:
                    # 重叠则保持不动
                    new_pos = all_positions[-1][i]
                    action = 0

                new_positions.append(new_pos)
                frame_actions.append(action)
            all_positions.append(new_positions)
            all_actions.append(frame_actions)

        # 渲染所有帧
        frames = []
        for t in range(self.num_frames):
            bg_seed = sample_seed * 1000 + t  # 每帧不同的背景种子
            frame = self._render_frame(
                all_positions[t], shapes, colors, bg_seed)
            frames.append(frame)

        video = torch.stack(frames)  # (T, H, W, C)

        # 构建标注信息（padding 到 max_actors 以保证 batch collate 一致性）
        max_a = self.max_actors
        # positions: (T, num_actors, 2) → pad to (T, max_actors, 2)
        pos_tensor = torch.zeros(self.num_frames, max_a, 2, dtype=torch.long)
        for t in range(self.num_frames):
            for i in range(num_actors):
                pos_tensor[t, i] = torch.tensor(all_positions[t][i])

        # actions: (T-1, num_actors) → pad to (T-1, max_actors), padding value = -1
        act_tensor = torch.full((max(self.num_frames - 1, 1), max_a), -1, dtype=torch.long)
        for t in range(len(all_actions)):
            for i in range(num_actors):
                act_tensor[t, i] = all_actions[t][i]

        # ids: (num_actors,) → pad to (max_actors,), padding value = -1
        ids_tensor = torch.full((max_a,), -1, dtype=torch.long)
        ids_tensor[:num_actors] = torch.arange(num_actors)

        return {
            "videos": video,
            "actor_positions": pos_tensor,
            "actor_actions": act_tensor,
            "actor_ids": ids_tensor,
            "num_actors": num_actors,
        }


class SyntheticMultiActorDataModule:
    """LightningDataModule 包装，与 LightningVideoDataset 接口兼容。"""

    def __init__(
            self,
            resolution: int = 256,
            num_frames: int = 2,
            min_actors: int = 2,
            max_actors: int = 5,
            background_type: str = "checkerboard",
            samples_per_epoch: int = 100000,
            batch_size: int = 8,
            num_workers: int = 4,
            seed: int = 42,
    ) -> None:
        self.resolution = resolution
        self.num_frames = num_frames
        self.min_actors = min_actors
        self.max_actors = max_actors
        self.background_type = background_type
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = SyntheticMultiActorDataset(
                resolution=self.resolution,
                num_frames=self.num_frames,
                min_actors=self.min_actors,
                max_actors=self.max_actors,
                background_type=self.background_type,
                samples_per_epoch=self.samples_per_epoch,
                seed=self.seed,
            )
            self.val_dataset = SyntheticMultiActorDataset(
                resolution=self.resolution,
                num_frames=self.num_frames,
                min_actors=self.min_actors,
                max_actors=self.max_actors,
                background_type=self.background_type,
                samples_per_epoch=self.samples_per_epoch // 1000,
                seed=self.seed + 100000,
            )
        elif stage == "test":
            self.test_dataset = SyntheticMultiActorDataset(
                resolution=self.resolution,
                num_frames=self.num_frames,
                min_actors=self.min_actors,
                max_actors=self.max_actors,
                background_type=self.background_type,
                samples_per_epoch=self.samples_per_epoch // 1000,
                seed=self.seed + 200000,
            )

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
