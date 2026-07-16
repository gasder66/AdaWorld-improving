"""Dataset for controlled OCAtari Boxing object-LAM clips."""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


class BoxingObjectDataset(Dataset):
    """Load dense RGB, two sprite masks, and a background context mask."""

    def __init__(self, split_dir: str, max_samples: Optional[int] = None) -> None:
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
        if max_samples is not None:
            self.files = self.files[:max_samples]
        if not self.files:
            raise FileNotFoundError(f"No Boxing samples found in {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = torch.load(self.files[index], map_location="cpu", weights_only=False)
        videos = raw["videos"].float() / 255.0
        masks = raw["masks"].float()
        background_masks = raw["background_masks"].float()
        if videos.ndim != 4 or videos.shape[1] != 3:
            raise ValueError(f"videos must be [T,3,H,W], got {tuple(videos.shape)}")
        if masks.shape[:2] != (videos.shape[0], 2):
            raise ValueError(f"masks must be [T,2,H,W], got {tuple(masks.shape)}")
        if not torch.all(masks.sum(dim=1) + background_masks == 1):
            raise ValueError(f"slot masks do not partition the frame: {self.files[index]}")
        return {
            "videos": videos,
            "masks": masks,
            "background_masks": background_masks,
            "valid_mask": raw["valid_mask"].bool(),
            # The following are evaluation labels only; the model forward ignores them.
            "delta_xy": raw["delta_xy"].float(),
            "movement_labels": raw["movement_labels"].long(),
            "punch_labels": raw["punch_labels"].long(),
            "contact_labels": raw["contact_labels"].long(),
            "occlusion_labels": raw["occlusion_labels"].long(),
            "sample_index": int(raw["metadata"]["sample_index"]),
        }
