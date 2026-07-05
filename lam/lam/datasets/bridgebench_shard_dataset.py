"""
BridgeBench Shard Dataset — lazy load individual shards, O(1) indexing.

Usage:
  from lam.datasets.bridgebench_shard_dataset import BridgeBenchShardDataset
  ds = BridgeBenchShardDataset("data/bridgebench/bridge1_clean_sharded", "train")
  sample = ds[0]
"""
import json, os
from typing import Dict

import torch
from torch.utils.data import Dataset


class BridgeBenchShardDataset(Dataset):

    def __init__(self, shard_dir: str, split: str = "train"):
        super().__init__()
        meta_path = os.path.join(shard_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {shard_dir}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.shard_dir = shard_dir
        self.shard_files = meta[split]["shard_files"]
        self.shard_size = meta[split]["shard_size"]
        self.total = meta[split]["total"]
        self._cache = {}  # shard_idx -> loaded dict

    def __len__(self) -> int:
        return self.total

    def _load_shard(self, shard_idx: int) -> Dict:
        sf = self.shard_files[shard_idx]
        shard = torch.load(os.path.join(self.shard_dir, sf),
                           map_location="cpu", weights_only=False)
        return shard

    def __getitem__(self, idx: int) -> Dict:
        shard_idx = idx // self.shard_size
        if shard_idx not in self._cache:
            self._cache[shard_idx] = self._load_shard(shard_idx)
        local = idx % self.shard_size
        return {k: v[local] for k, v in self._cache[shard_idx].items()}
