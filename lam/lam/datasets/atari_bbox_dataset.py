"""
V13 AtariBBoxDataset: loads OCAtari .pt files with game-specific normalization.

Returns unified format:
    video:   (T, 3, H, W) float [0,1]  — resized to image_size
    bbox:    (T, K_raw, 4) float       — normalized cxcywh
    valid:   (T, K_raw) bool
    obj_type:(K_raw,) int              — object type IDs
    track_id:(K_raw,) int              — slot indices
    env_action: (T-1,) int             — raw ALE action
    game:    str
"""
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lam.v12_dataset import V12ObjectVideoDataset


OBJECT_TYPE_NAMES_V13 = {
    0: "player",
    1: "enemy",
    2: "static",
    3: "car",
    4: "ghost",
    5: "ship",
    6: "alien",
    7: "bullet",
}

GAME_CONFIGS = {
    "freeway": {
        "image_size": 84,
        "max_objects": 24,
        "crop_size": 24,
        "agent_type": 0,       # chicken=player
        "car_type": 3,
        "num_lanes": 10,
        "num_nearby_cars": 5,
    },
    "spaceinvaders": {
        "image_size": 84,
        "max_objects": 24,
        "crop_size": 20,
        "agent_type": 5,       # ship
        "alien_type": 6,
        "bullet_type": 7,
    },
    "mspacman": {
        "image_size": 84,
        "max_objects": 24,
        "crop_size": 20,
        "agent_type": 0,       # pacman=player
        "ghost_type": 4,
        "pill_type": 2,        # static
    },
}


def _xyxy_to_cxcywh(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return torch.stack([cx, cy, w, h], dim=-1)


class AtariBBoxDataset(Dataset):
    """Load OCAtari V12 .pt files and produce standardized Atari object samples."""

    def __init__(
        self,
        data_dir: str,
        game: str,
        image_size: int = 84,
        sequence_length: int = 5,
        max_objects: int = 24,
    ):
        super().__init__()
        self.game = game
        self.image_size = image_size
        self.T = sequence_length
        self.K_raw = max_objects

        self._inner = V12ObjectVideoDataset(
            data_dir, output_format="t c h w",
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Dict:
        raw = self._inner[idx]

        video = raw["videos"]  # (T, C, H_orig, W_orig)
        T, C, H_orig, W_orig = video.shape

        # Resize to target size.
        if H_orig != self.image_size or W_orig != self.image_size:
            video = F.interpolate(
                video.float(), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).clamp(0, 1)
        if video.shape[0] > self.T:
            video = video[:self.T]

        # Normalize bboxes to cxcywh in [0,1].
        bboxes_raw = raw["bboxes"]  # (T, K_raw, 4) xyxy pixel
        bboxes_norm = _xyxy_to_cxcywh(bboxes_raw[:self.T], H_orig, W_orig)

        valid = raw["valid_mask"][:self.T]  # (T, K_raw) bool
        obj_types = raw["object_types"]      # (K_raw,) int
        track_ids = raw.get("actor_ids", torch.arange(self.K_raw))  # (K_raw,)
        env_action = raw.get("env_actions", None)

        out: Dict = {
            "video": video,                # (T, 3, H, W) float
            "bbox": bboxes_norm,           # (T, K_raw, 4) cxcywh normalized
            "valid": valid,                # (T, K_raw) bool
            "obj_type": obj_types,         # (K_raw,) int
            "track_id": track_ids,         # (K_raw,) int
            "game": self.game,
        }
        if env_action is not None:
            out["env_action"] = env_action[:self.T - 1] if env_action.numel() else env_action

        return out
