"""
V8 MOT Slot Table 数据接口。

包装 DiskSyntheticDataset, 输出 V8 slot table 格式:
    videos:        (T, H, W, 3) float32 [0,1]
    boxes:         (T, K, 4) float32 [x1,y1,x2,y2] in pixel coords
    track_ids:     (K,) long  # 合成数据: 0..K-1 (slot index 即 track_id)
    actor_labels:  (K,) long  # Stage 1 全 0 (无 actor 条件化)
    valid_mask:    (T, K) bool  # 合成数据 num_actors 以内为 True
    actions:       (T-1, K) long  # GT actions (仅 eval 用)
    num_actors:    int

Stage 1 不重新生成数据。bbox 从 DiskSyntheticDataset 的 positions 推导:
    positions[t,k] = (grid_r, grid_c)
    cell_size = 32, actor 占 2x2 grid = 64x64 像素
    bbox = [grid_c*32, grid_r*32, (grid_c+2)*32, (grid_r+2)*32]
"""
import os
from typing import Dict

import torch
from torch.utils.data import Dataset

from lam.disk_synthetic_dataset import DiskSyntheticDataset


GRID_SIZE = 8
CELL_SIZE = 256 // GRID_SIZE  # 32
ACTOR_GRID_SIZE = 2  # 2x2 grid
ACTOR_PIX_SIZE = CELL_SIZE * ACTOR_GRID_SIZE  # 64


def positions_to_boxes(positions: torch.Tensor) -> torch.Tensor:
    """(T, K, 2) grid positions -> (T, K, 4) pixel bbox [x1,y1,x2,y2].

    positions[t,k] = (grid_r, grid_c) 即 actor 左上角的 grid 坐标。
    bbox 覆盖 2x2 grid = 64x64 像素。
    """
    grid_r = positions[..., 0].long()
    grid_c = positions[..., 1].long()
    x1 = grid_c * CELL_SIZE
    y1 = grid_r * CELL_SIZE
    x2 = x1 + ACTOR_PIX_SIZE
    y2 = y1 + ACTOR_PIX_SIZE
    return torch.stack([x1, y1, x2, y2], dim=-1).float()


class MOTSlotDataset(Dataset):
    """V8 MOT Slot Table 数据集。

    包装底层 DiskSyntheticDataset, 增加 bbox / track_id / valid_mask 字段。
    """

    def __init__(
        self,
        data_dir: str,
        max_actors: int = 4,
        num_frames: int = 5,
    ) -> None:
        super().__init__()
        self.max_actors = max_actors
        self.num_frames = num_frames
        self.base = DiskSyntheticDataset(
            data_dir, max_actors=max_actors,
            num_frames=num_frames, output_format="t h w c",
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict:
        s = self.base[idx]
        T = s["videos"].shape[0]
        K = self.max_actors

        # (T, K, 2) -> (T, K, 4) pixel bbox
        boxes = positions_to_boxes(s["positions"])

        # track_ids: 合成数据 slot index 即 track_id (0..K-1)
        track_ids = torch.arange(K, dtype=torch.long)

        # actor_labels: Stage 1 全 0 (无条件化)
        actor_labels = torch.zeros(K, dtype=torch.long)

        # valid_mask: (T, K), num_actors 以内为 True
        valid_mask = torch.zeros(T, K, dtype=torch.bool)
        for t in range(T):
            valid_mask[t, :int(s["num_actors"])] = True

        return {
            "videos": s["videos"],         # (T, H, W, 3)
            "boxes": boxes,                # (T, K, 4)
            "track_ids": track_ids,        # (K,)
            "actor_labels": actor_labels,  # (K,) all zero
            "valid_mask": valid_mask,      # (T, K)
            "actions": s["actions"],       # (T-1, K)
            "num_actors": s["num_actors"],
        }


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 0,
    num_frames: int = 5,
    max_actors: int = 4,
):
    """创建 V8 训练/验证 DataLoader。"""
    train_ds = MOTSlotDataset(
        os.path.join(data_root, "train"),
        max_actors=max_actors, num_frames=num_frames,
    )
    val_ds = MOTSlotDataset(
        os.path.join(data_root, "val"),
        max_actors=max_actors, num_frames=num_frames,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )
    return train_loader, val_loader
