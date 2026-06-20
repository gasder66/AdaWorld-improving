"""
MOT Mask 数据集包装器。

用预生成的 YOLO+BoT-SORT masks 替代 GT masks。
输入格式与 DiskSyntheticDataset 兼容，供 V6c 训练使用。

用法:
  dataset = MOTA2DDataset(split="train")
  sample = dataset[0]
  sample["videos"] -> (T, H, W, C) float32
  sample["masks"]  -> (T, max_actors, H, W) float32 (来自 MOT, 非 GT)
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from lam.a2d_dataset import A2DDataset


class MOTA2DDataset(Dataset):
    """用 YOLO+BoT-SORT 生成的 masks 替代 GT masks 的 A2D 数据集。

    Args:
        yolo_source: "a2d" 或 "coco" — 使用哪个 YOLO 生成的 masks
    """

    def __init__(
        self,
        data_root: str,
        release_root: str,
        split: str = "train",
        num_frames: int = 2,
        max_actors: int = 4,
        img_size: int = 256,
        yolo_source: str = "a2d",
        frame_stride: int = 1,
    ):
        super().__init__()
        self.a2d = A2DDataset(
            data_root=data_root, release_root=release_root,
            split=split, num_frames=num_frames,
            max_actors=max_actors, img_size=img_size,
            frame_stride=frame_stride,
        )
        self.split = split

        # MOT masks 目录 (不同 YOLO 来源不同目录)
        dir_map = {"coco": "mot_masks_coco", "reid": "mot_masks_reid"}
        subdir = dir_map.get(yolo_source, "mot_masks")
        self.mot_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "result", "v6_a2d",
            subdir, split
        )
        self.num_frames = num_frames

    def __len__(self):
        return len(self.a2d)

    def __getitem__(self, idx):
        # 获取视频 frames (GT 仅用于视频, 不使用 GT masks)
        sample = self.a2d[idx]

        # 加载预生成的 MOT masks
        mot_path = os.path.join(self.mot_dir, f"mot_masks_{idx:06d}.pt")
        if os.path.exists(mot_path):
            mot_data = torch.load(mot_path, map_location="cpu")
            sample["masks"] = mot_data["mot_masks"]
        else:
            # fallback: 空 masks
            T, H, W = sample["videos"].shape[:3]
            sample["masks"] = torch.zeros(T, 4, H, W)

        return sample
