"""Dataset for controlled OCAtari Boxing object-LAM clips."""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


class BoxingObjectDataset(Dataset):
    """Load dense RGB, two sprite masks, and a background context mask."""

    def __init__(
        self,
        split_dir: str,
        max_samples: Optional[int] = None,
        target_frames: Optional[int] = None,
    ) -> None:
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
        if max_samples is not None:
            self.files = self.files[:max_samples]
        if not self.files:
            raise FileNotFoundError(f"No Boxing samples found in {split_dir}")
        self.target_frames = target_frames

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = torch.load(self.files[index], map_location="cpu", weights_only=False)
        source_time = raw["videos"].shape[0]
        if self.target_frames is not None and source_time != self.target_frames:
            frame_indices = torch.linspace(0, source_time - 1, self.target_frames).round().long()
        else:
            frame_indices = torch.arange(source_time)
        videos = raw["videos"].index_select(0, frame_indices).float() / 255.0
        masks = raw["masks"].index_select(0, frame_indices).float()
        background_masks = raw["background_masks"].index_select(0, frame_indices).float()
        centers = raw["centers_xy"].index_select(0, frame_indices).float()
        delta_xy = centers[1:] - centers[:-1]
        movement_labels = torch.zeros(delta_xy.shape[:2], dtype=torch.long)
        horizontal = delta_xy[..., 0].abs() >= delta_xy[..., 1].abs()
        moving = delta_xy.norm(dim=-1) > 0.5
        movement_labels[moving & horizontal & (delta_xy[..., 0] < 0)] = 3
        movement_labels[moving & horizontal & (delta_xy[..., 0] > 0)] = 4
        movement_labels[moving & ~horizontal & (delta_xy[..., 1] < 0)] = 1
        movement_labels[moving & ~horizontal & (delta_xy[..., 1] > 0)] = 2
        punch_labels = raw["punch_labels"].index_select(0, frame_indices)
        punch_side = raw.get("punch_side_labels", torch.zeros_like(raw["punch_labels"])).index_select(0, frame_indices)
        arm_lengths = raw["arm_lengths"].index_select(0, frame_indices).float()
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
            "valid_mask": raw["valid_mask"].index_select(0, frame_indices).bool(),
            # The following are evaluation labels only; the model forward ignores them.
            "delta_xy": delta_xy,
            "movement_labels": movement_labels,
            "punch_labels": punch_labels.long(),
            "punch_side_labels": punch_side.long(),
            "arm_lengths": arm_lengths,
            "arm_delta": arm_lengths[1:] - arm_lengths[:-1],
            "contact_labels": raw["contact_labels"].index_select(0, frame_indices).long(),
            "occlusion_labels": raw["occlusion_labels"].index_select(0, frame_indices).long(),
            "sample_index": int(raw["metadata"]["sample_index"]),
        }
