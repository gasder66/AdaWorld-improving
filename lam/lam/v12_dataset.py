"""
V12 unified object-video dataset utilities.

The V12 sample contract is intentionally a superset of the existing synthetic
format. Older samples are normalized on load so V10/V11 experiments can keep
using the clean synthetic data while new GridWorld/Atari datasets share one
schema.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


V12_REQUIRED_KEYS = (
    "videos",
    "masks",
    "bboxes",
    "positions",
    "actions",
    "actor_ids",
    "object_types",
    "valid_mask",
    "metadata",
)


ACTION_NAMES = ["stay", "up", "down", "left", "right"]
OBJECT_TYPE_NAMES = {
    0: "player",
    1: "enemy",
    2: "static",
    3: "car",
    4: "ghost",
    5: "ship",
    6: "alien",
    7: "bullet",
}


def _as_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    return torch.as_tensor(value, dtype=dtype)


def masks_to_bboxes(masks: torch.Tensor) -> torch.Tensor:
    """Convert binary masks (T, K, H, W) to xyxy boxes; invalid boxes are zeros."""
    if masks.numel() == 0:
        T = masks.shape[0] if masks.ndim >= 1 else 0
        K = masks.shape[1] if masks.ndim >= 2 else 0
        return torch.zeros((T, K, 4), dtype=torch.float32)

    masks_bool = masks > 0
    T, K = masks_bool.shape[:2]
    boxes = torch.zeros((T, K, 4), dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            ys, xs = torch.where(masks_bool[t, k])
            if ys.numel() == 0:
                continue
            x1 = xs.min().item()
            y1 = ys.min().item()
            x2 = xs.max().item() + 1
            y2 = ys.max().item() + 1
            boxes[t, k] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
    return boxes


def _infer_positions_from_bboxes(bboxes: torch.Tensor) -> torch.Tensor:
    if bboxes.numel() == 0:
        T = bboxes.shape[0] if bboxes.ndim >= 1 else 0
        K = bboxes.shape[1] if bboxes.ndim >= 2 else 0
        return torch.full((T, K, 2), -1, dtype=torch.long)
    centers = torch.stack(
        [
            (bboxes[..., 1] + bboxes[..., 3]) * 0.5,
            (bboxes[..., 0] + bboxes[..., 2]) * 0.5,
        ],
        dim=-1,
    )
    valid = (bboxes[..., 2] > bboxes[..., 0]) & (bboxes[..., 3] > bboxes[..., 1])
    positions = centers.round().long()
    positions[~valid] = -1
    return positions


def normalize_v12_sample(
    sample: Dict[str, Any],
    *,
    task_name: str = "unknown",
    split: str = "unknown",
    sample_index: Optional[int] = None,
    output_format: str = "t h w c",
) -> Dict[str, Any]:
    """Return a V12-compatible sample without mutating the input dictionary."""
    out = dict(sample)
    videos = _as_tensor(out["videos"])
    if videos.dtype == torch.uint8:
        videos_float = videos.float() / 255.0
    else:
        videos_float = videos.float()
        if videos_float.numel() and videos_float.max() > 1.5:
            videos_float = videos_float / 255.0

    if videos_float.ndim != 4:
        raise ValueError(f"videos must be 4D, got shape {tuple(videos_float.shape)}")
    if videos_float.shape[1] == 3:
        videos_tchw = videos_float.contiguous()
        videos_thwc = videos_tchw.permute(0, 2, 3, 1).contiguous()
    elif videos_float.shape[-1] == 3:
        videos_thwc = videos_float.contiguous()
        videos_tchw = videos_thwc.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"videos must be TCHW or THWC RGB, got {tuple(videos_float.shape)}")

    T, H, W, _ = videos_thwc.shape

    masks = out.get("masks")
    if masks is None:
        masks_t = torch.zeros((T, 0, H, W), dtype=torch.float32)
    else:
        masks_t = _as_tensor(masks).float()
        masks_t = masks_t[:T]
        if masks_t.ndim != 4:
            raise ValueError(f"masks must be (T,K,H,W), got {tuple(masks_t.shape)}")

    K = masks_t.shape[1]

    bboxes = out.get("bboxes", out.get("boxes"))
    if bboxes is None:
        bboxes_t = masks_to_bboxes(masks_t)
    else:
        bboxes_t = _as_tensor(bboxes, dtype=torch.float32)[:T]
        if bboxes_t.shape[:2] != (T, K):
            raise ValueError(
                f"bboxes shape {tuple(bboxes_t.shape)} is incompatible with T={T}, K={K}"
            )

    positions = out.get("positions")
    if positions is None:
        positions_t = _infer_positions_from_bboxes(bboxes_t)
    else:
        positions_t = _as_tensor(positions, dtype=torch.long)[:T]
        if positions_t.shape[:2] != (T, K):
            padded = torch.full((T, K, 2), -1, dtype=torch.long)
            t_lim = min(T, positions_t.shape[0])
            k_lim = min(K, positions_t.shape[1]) if positions_t.ndim >= 2 else 0
            padded[:t_lim, :k_lim] = positions_t[:t_lim, :k_lim]
            positions_t = padded

    actions = out.get("actions")
    if actions is None:
        actions_t = torch.full((max(T - 1, 0), K), -1, dtype=torch.long)
    else:
        actions_t = _as_tensor(actions, dtype=torch.long)
        padded = torch.full((max(T - 1, 0), K), -1, dtype=torch.long)
        if actions_t.ndim == 2:
            t_lim = min(max(T - 1, 0), actions_t.shape[0])
            k_lim = min(K, actions_t.shape[1])
            padded[:t_lim, :k_lim] = actions_t[:t_lim, :k_lim]
        actions_t = padded

    valid_mask = out.get("valid_mask")
    if valid_mask is None:
        valid_t = (masks_t.sum(dim=(-2, -1)) > 0) if K > 0 else torch.zeros((T, 0), dtype=torch.bool)
        if "num_actors" in out and K > 0:
            n = int(out["num_actors"])
            valid_t[:, :n] = True
    else:
        valid_t = _as_tensor(valid_mask).bool()
        padded = torch.zeros((T, K), dtype=torch.bool)
        t_lim = min(T, valid_t.shape[0])
        k_lim = min(K, valid_t.shape[1]) if valid_t.ndim >= 2 else 0
        padded[:t_lim, :k_lim] = valid_t[:t_lim, :k_lim]
        valid_t = padded

    actor_ids = out.get("actor_ids", out.get("track_ids"))
    if actor_ids is None:
        actor_ids_t = torch.arange(K, dtype=torch.long)
    else:
        actor_ids_t = _as_tensor(actor_ids, dtype=torch.long).reshape(-1)
        if actor_ids_t.numel() < K:
            pad = torch.arange(actor_ids_t.numel(), K, dtype=torch.long)
            actor_ids_t = torch.cat([actor_ids_t, pad], dim=0)
        actor_ids_t = actor_ids_t[:K]

    object_types = out.get("object_types", out.get("actor_labels"))
    if object_types is None:
        object_types_t = torch.zeros(K, dtype=torch.long)
    else:
        object_types_t = _as_tensor(object_types, dtype=torch.long).reshape(-1)
        if object_types_t.numel() < K:
            object_types_t = torch.cat(
                [object_types_t, torch.full((K - object_types_t.numel(),), -1, dtype=torch.long)]
            )
        object_types_t = object_types_t[:K]

    metadata = dict(out.get("metadata") or {})
    metadata.setdefault("task_name", task_name)
    metadata.setdefault("split", split)
    if sample_index is not None:
        metadata.setdefault("sample_index", int(sample_index))
    metadata.setdefault("has_object_annotations", bool(K > 0 and valid_t.any().item()))

    event_labels = out.get("event_labels")
    if event_labels is not None:
        event_labels = _as_tensor(event_labels, dtype=torch.long)

    env_actions = out.get("env_actions")
    if env_actions is not None:
        env_actions = _as_tensor(env_actions, dtype=torch.long)

    normalized: Dict[str, Any] = {
        "videos": videos_thwc if output_format == "t h w c" else videos_tchw,
        "masks": masks_t,
        "bboxes": bboxes_t,
        "positions": positions_t,
        "actions": actions_t,
        "actor_ids": actor_ids_t,
        "object_types": object_types_t,
        "valid_mask": valid_t,
        "metadata": metadata,
    }
    if event_labels is not None:
        normalized["event_labels"] = event_labels
    if env_actions is not None:
        normalized["env_actions"] = env_actions
    if "num_actors" in out:
        normalized["num_actors"] = int(out["num_actors"])
    else:
        normalized["num_actors"] = int(valid_t.any(dim=0).sum().item()) if K > 0 else 0
    return normalized


class V12ObjectVideoDataset(Dataset):
    """Load V12 .pt files from a split directory."""

    def __init__(
        self,
        data_dir: str,
        *,
        task_name: Optional[str] = None,
        output_format: str = "t h w c",
        max_samples: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.task_name = task_name or os.path.basename(os.path.dirname(data_dir)) or "unknown"
        self.output_format = output_format
        self.sample_files = sorted(
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")
        )
        if max_samples > 0:
            self.sample_files = self.sample_files[:max_samples]
        if not self.sample_files:
            raise FileNotFoundError(f"No .pt samples found in {data_dir}")

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = torch.load(self.sample_files[idx], map_location="cpu")
        return normalize_v12_sample(
            sample,
            task_name=self.task_name,
            split=os.path.basename(self.data_dir),
            sample_index=idx,
            output_format=self.output_format,
        )


def validate_v12_sample(sample: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in V12_REQUIRED_KEYS:
        if key not in sample:
            errors.append(f"missing key: {key}")
    if errors:
        return errors
    videos = sample["videos"]
    masks = sample["masks"]
    bboxes = sample["bboxes"]
    actions = sample["actions"]
    valid_mask = sample["valid_mask"]
    if videos.ndim != 4 or videos.shape[-1] != 3:
        errors.append(f"videos must be (T,H,W,3), got {tuple(videos.shape)}")
    T = videos.shape[0]
    K = masks.shape[1] if masks.ndim == 4 else -1
    if masks.ndim != 4 or masks.shape[0] != T:
        errors.append(f"masks must be (T,K,H,W), got {tuple(masks.shape)}")
    if bboxes.shape != (T, K, 4):
        errors.append(f"bboxes must be {(T, K, 4)}, got {tuple(bboxes.shape)}")
    if actions.shape != (max(T - 1, 0), K):
        errors.append(f"actions must be {(max(T - 1, 0), K)}, got {tuple(actions.shape)}")
    if valid_mask.shape != (T, K):
        errors.append(f"valid_mask must be {(T, K)}, got {tuple(valid_mask.shape)}")
    if videos.numel() and (videos.min() < 0 or videos.max() > 1):
        errors.append("videos must be float in [0,1] after normalization")
    return errors


def iter_split_files(root: str, splits: Iterable[str] = ("train", "val")) -> Iterable[str]:
    for split in splits:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            if name.endswith(".pt"):
                yield os.path.join(split_dir, name)
