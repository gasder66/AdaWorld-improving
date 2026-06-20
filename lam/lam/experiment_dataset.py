"""加载预生成的实验数据集 .pt 文件。"""
import os, torch
from torch.utils.data import Dataset


class ExperimentDataset(Dataset):
    """加载预生成的 .pt 实验数据集 (videos + yolo_masks)."""

    def __init__(self, pt_path: str):
        super().__init__()
        data = torch.load(pt_path, map_location="cpu")
        self.samples = data["samples"]
        self.config = data.get("config", {})
        print(f"  [ExpDataset] {len(self.samples)} samples from {pt_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # videos: (T, H, W, C), masks: (T, A, H, W)
        return s
