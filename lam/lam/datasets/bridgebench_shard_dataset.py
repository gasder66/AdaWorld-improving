"""
BridgeBench Shard Dataset — pre-concatenate all shards, index_select for O(1) batching.
"""
import json, os
from typing import Dict

import torch
from torch.utils.data import Dataset


class BridgeBenchShardDataset(Dataset):

    def __init__(self, shard_dir: str, split: str = "train"):
        super().__init__()
        meta_path = os.path.join(shard_dir, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.shard_dir = shard_dir
        self.shard_files = meta[split]["shard_files"]
        self.total = meta[split]["total"]
        self._data = None

    def __len__(self) -> int:
        return self.total

    def _ensure_loaded(self):
        if self._data is not None:
            return
        print(f"  Loading {len(self.shard_files)} shards...", end="", flush=True)
        all_data = {}
        for sf in self.shard_files:
            shard = torch.load(os.path.join(self.shard_dir, sf),
                               map_location="cpu", weights_only=False)
            for k, v in shard.items():
                all_data.setdefault(k, []).append(v)
        self._data = {k: torch.cat(v, dim=0) for k, v in all_data.items()}
        print(f" done ({list(self._data.keys())})")

    def __getitem__(self, idx: int) -> Dict:
        self._ensure_loaded()
        return {k: v[idx] for k, v in self._data.items()}

    def get_batch(self, indices: torch.Tensor) -> Dict:
        """Return batch dict by index_select (no stacking needed)."""
        self._ensure_loaded()
        return {k: v.index_select(0, indices) for k, v in self._data.items()}
