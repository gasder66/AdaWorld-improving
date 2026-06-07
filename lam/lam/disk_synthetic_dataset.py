"""
从磁盘加载预生成的多主体合成数据集。

存储格式：uint8 压缩（~10GB 总大小）
训练时自动转换为 float32。

输出：
    "videos":    (T, C, H, W) 或 (T, H, W, C) float32 [0,1]
    "masks":     (T, max_actors, H, W) float32 binary
    "positions": (T, max_actors, 2) long
    "actions":   (T-1, max_actors) long (-1=padding)
    "num_actors": int
"""

import os
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader


class DiskSyntheticDataset(Dataset):
    """
    从磁盘加载预生成的多主体数据集。

    每个 .pt 文件包含：
        "videos": (T, 3, H, W) uint8 [0,255]
        "masks": (T, max_actors, H, W) uint8 binary
        "positions": (T, max_actors, 2) long
        "actions": (T-1, max_actors) long (-1 = padding)
        "num_actors": int
    """

    def __init__(
            self,
            data_dir: str,
            max_actors: int = 4,
            num_frames: int = 5,
            output_format: str = "t c h w",
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.max_actors = max_actors
        self.num_frames = num_frames
        self.output_format = output_format

        # 扫描所有 .pt 文件
        self.sample_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pt")
        ])

        print(f"  [DiskSyntheticDataset] {len(self.sample_files)} samples from {data_dir}")
        if len(self.sample_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Dict:
        sample = torch.load(self.sample_files[idx], map_location="cpu")

        # videos: (T, 3, H, W) uint8 → float32 [0,1]
        videos = sample["videos"].float() / 255.0  # (T, 3, H, W)

        # 截取到 num_frames 帧
        videos = videos[:self.num_frames]
        T_actual = videos.shape[0]

        if self.output_format == "t h w c":
            videos = videos.permute(0, 2, 3, 1).contiguous()  # (T, H, W, C)

        # masks: (T, max_actors, H, W) uint8 → float32
        masks = sample["masks"][:T_actual].float()

        # positions: (T, max_actors, 2)
        positions = sample["positions"][:T_actual]

        # actions: (T-1, max_actors), 不足则 padding
        actions = sample["actions"][:T_actual - 1]
        if actions.shape[0] < self.num_frames - 1:
            pad = torch.full((self.num_frames - 1 - actions.shape[0],
                             self.max_actors), -1, dtype=torch.long)
            actions = torch.cat([actions, pad], dim=0)

        return {
            "videos": videos,       # (T, C, H, W) 或 (T, H, W, C)
            "masks": masks,          # (T, max_actors, H, W)
            "positions": positions,  # (T, max_actors, 2)
            "actions": actions,      # (T-1, max_actors)
            "num_actors": sample["num_actors"],
        }


def create_dataloaders(
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        output_format: str = "t h w c",
) -> tuple:
    """创建训练和验证 DataLoader。"""
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_dataset = DiskSyntheticDataset(
        train_dir, output_format=output_format
    )
    val_dataset = DiskSyntheticDataset(
        val_dir, output_format=output_format
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
