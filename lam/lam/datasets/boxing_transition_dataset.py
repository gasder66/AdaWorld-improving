"""Two-frame Boxing transitions with event-balanced sampling weights."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


EVENT_TO_ID = {
    "movement_only": 0,
    "punch_onset": 1,
    "punch_extend": 2,
    "punch_hold": 3,
    "punch_retract": 4,
    "punch_switch": 5,
}

DEFAULT_EVENT_PROBABILITIES = {
    "movement_only": 0.40,
    "punch_onset": 0.15,
    "punch_extend": 0.15,
    "punch_hold": 0.10,
    "punch_retract": 0.15,
    "punch_switch": 0.05,
}


class BoxingTransitionDataset(Dataset):
    def __init__(self, index_path: str) -> None:
        index = torch.load(index_path, map_location="cpu", weights_only=False)
        self.entries: List[Dict[str, Any]] = index["entries"]
        self.stats = index["stats"]
        if not self.entries:
            raise ValueError(f"empty transition index: {index_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        raw = torch.load(entry["path"], map_location="cpu", weights_only=False)
        t = int(entry["transition"])
        frame_indices = torch.tensor([t, t + 1])
        videos = raw["videos"].index_select(0, frame_indices).float() / 255.0
        masks = raw["masks"].index_select(0, frame_indices).float()
        background = raw["background_masks"].index_select(0, frame_indices).float()
        centers = raw["centers_xy"].index_select(0, frame_indices).float()
        arms = raw["arm_lengths"].index_select(0, frame_indices).float()
        return {
            "videos": videos,
            "masks": masks,
            "background_masks": background,
            "valid_mask": raw["valid_mask"].index_select(0, frame_indices).bool(),
            "delta_xy": centers[1:] - centers[:-1],
            "arm_lengths": arms,
            "arm_delta": arms[1:] - arms[:-1],
            "punch_labels": (arms != 0).any(dim=-1).long(),
            "event_id": EVENT_TO_ID[entry["event"]],
            "target_slot": int(entry["target_slot"]),
            "sample_index": index,
        }

    def balanced_sampler(self, num_samples: int | None = None, seed: int = 0) -> WeightedRandomSampler:
        tuple_counts = Counter((entry["event"], entry["fighter"]) for entry in self.entries)
        weights = []
        for entry in self.entries:
            event = entry["event"]
            target_probability = DEFAULT_EVENT_PROBABILITIES[event] * 0.5
            weights.append(target_probability / tuple_counts[(event, entry["fighter"])])
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=num_samples or len(self.entries),
            replacement=True,
            generator=generator,
        )
