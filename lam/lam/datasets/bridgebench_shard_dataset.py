"""
BridgeBench Shard Dataset — load pre-packed shards for fast training.

Usage:
  from lam.datasets.bridgebench_shard_dataset import BridgeBenchShardDataset
  ds = BridgeBenchShardDataset("data/bridgebench/bridge1_clean_sharded", "train")
  sample = ds[0]  # identical format to per-file loading
"""
import json, os
from typing import Dict

import torch
from torch.utils.data import Dataset


class BridgeBenchShardDataset(Dataset):

    def __init__(self, shard_dir: str, split: str = "train"):
        super().__init__()
        self.shard_dir = shard_dir
        self.split = split

        meta_path = os.path.join(shard_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {shard_dir}")

        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.shard_files = meta[split]["shard_files"]
        self.shard_size = meta[split]["shard_size"]
        self.total = meta[split]["total"]
        self._loaded = False
        self._data = None

    def _load(self):
        if self._loaded:
            return
        all_data = {}
        for sf in self.shard_files:
            shard = torch.load(os.path.join(self.shard_dir, sf),
                               map_location="cpu", weights_only=False)
            for k, v in shard.items():
                all_data.setdefault(k, []).append(v)
        self._data = {k: torch.cat(v, dim=0) for k, v in all_data.items()}
        self._loaded = True

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Dict:
        self._load()
        return {k: v[idx] for k, v in self._data.items()}
