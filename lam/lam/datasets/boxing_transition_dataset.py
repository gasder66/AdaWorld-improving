"""Two-frame Boxing transitions with event-balanced sampling weights."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
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

INTERACTION_TO_ID = {
    "non_interaction": 0,
    "near": 1,
    "contact": 2,
    "punch_miss": 3,
    "hit": 4,
    "received_hit": 5,
    "occlusion": 6,
    "recovery": 7,
}

DEFAULT_INTERACTION_PROBABILITIES = {
    "non_interaction": 0.25,
    "near": 0.10,
    "contact": 0.10,
    "punch_miss": 0.15,
    "hit": 0.10,
    "received_hit": 0.15,
    "occlusion": 0.10,
    "recovery": 0.05,
}


class BoxingTransitionDataset(Dataset):
    def __init__(
        self,
        index_path: str,
        temporal_context: int = 1,
        prediction_horizon: int = 1,
    ) -> None:
        index = torch.load(index_path, map_location="cpu", weights_only=False)
        if temporal_context < 1:
            raise ValueError("temporal_context must be positive")
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be positive")
        self.temporal_context = temporal_context
        self.prediction_horizon = prediction_horizon
        available_transitions = {
            (entry["path"], int(entry["transition"])) for entry in index["entries"]
        }
        self.entries: List[Dict[str, Any]] = [
            entry
            for entry in index["entries"]
            if int(entry["transition"]) >= temporal_context - 1
            and all(
                (entry["path"], int(entry["transition"]) + offset)
                in available_transitions
                for offset in range(prediction_horizon)
            )
        ]
        resolved_index = Path(index_path).resolve()
        self.project_root = next(
            (parent for parent in resolved_index.parents if (parent / "data").is_dir()),
            None,
        )
        self._cached_sample_path: Path | None = None
        self._cached_sample: Dict[str, Any] | None = None
        self.stats = index["stats"]
        if not self.entries:
            raise ValueError(f"empty transition index: {index_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        sample_path = Path(entry["path"])
        if not sample_path.exists() and self.project_root is not None:
            parts = sample_path.parts
            if "data" in parts:
                sample_path = self.project_root.joinpath(*parts[parts.index("data") :])
        if sample_path != self._cached_sample_path:
            self._cached_sample = torch.load(
                sample_path, map_location="cpu", weights_only=False
            )
            self._cached_sample_path = sample_path
        if self._cached_sample is None:
            raise RuntimeError(f"failed to load Boxing sample: {sample_path}")
        raw = self._cached_sample
        t = int(entry["transition"])
        frame_indices = torch.arange(
            t - self.temporal_context + 1,
            t + self.prediction_horizon + 1,
        )
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
            "delta_xy": centers[-1:] - centers[-2:-1],
            "arm_lengths": arms[-2:],
            "arm_delta": arms[-1:] - arms[-2:-1],
            "punch_labels": (arms[-2:] != 0).any(dim=-1).long(),
            "event_id": EVENT_TO_ID[entry["event"]],
            "interaction_id": INTERACTION_TO_ID[entry.get("interaction_primary", "non_interaction")],
            "target_slot": int(entry["target_slot"]),
            "sample_index": index,
        }

    def balanced_sampler(
        self, num_samples: int | None = None, seed: int = 0, mode: str = "phase"
    ) -> WeightedRandomSampler:
        if mode == "phase":
            key_name = "event"
            probabilities = DEFAULT_EVENT_PROBABILITIES
            balance_fighter = True
        elif mode == "interaction":
            key_name = "interaction_primary"
            probabilities = DEFAULT_INTERACTION_PROBABILITIES
            # Occlusion is asymmetric in the Atari renderer (the black sprite
            # is usually hidden), so forcing 50/50 fighter balance would repeat
            # a tiny set of rare Player-occlusion transitions.
            balance_fighter = False
        else:
            raise ValueError(f"unknown balance mode: {mode}")
        if balance_fighter:
            counts = Counter((entry.get(key_name, "non_interaction"), entry["fighter"]) for entry in self.entries)
        else:
            counts = Counter(entry.get(key_name, "non_interaction") for entry in self.entries)
        weights = []
        for entry in self.entries:
            event = entry.get(key_name, "non_interaction")
            if balance_fighter:
                weights.append(probabilities[event] * 0.5 / counts[(event, entry["fighter"])])
            else:
                weights.append(probabilities[event] / counts[event])
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=num_samples or len(self.entries),
            replacement=True,
            generator=generator,
        )
