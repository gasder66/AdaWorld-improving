"""Generate V12 Atari clips with OCAtari object annotations.

This script rollouts Atari environments through OCAtari and stores clips in
the V12 object-video schema. Unlike prepare_atari_v12.py, this path stores
RAM/OCAtari-derived object boxes and masks instead of video-only samples.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.v12_dataset import validate_v12_sample


GAME_ENV_NAMES = {
    "freeway": "ALE/Freeway-v5",
    "mspacman": "ALE/MsPacman-v5",
    "spaceinvaders": "ALE/SpaceInvaders-v5",
}

OBJECT_TYPE_TO_ID = {
    "player": 0,
    "pacman": 0,
    "chicken": 0,
    "car": 3,
    "ghost": 4,
    "ship": 5,
    "alien": 6,
    "bullet": 7,
    "missile": 7,
    "projectile": 7,
    "static": 2,
}

OBJECT_PRIORITY = {
    "player": 0,
    "pacman": 0,
    "chicken": 0,
    "car": 1,
    "ghost": 1,
    "alien": 1,
    "ship": 1,
    "bullet": 2,
    "missile": 2,
    "projectile": 2,
    "powerpill": 8,
    "pill": 9,
    "score": 10,
    "life": 10,
    "static": 10,
}

ACTION_TO_ID = {
    "stay": 0,
    "up": 1,
    "down": 2,
    "left": 3,
    "right": 4,
}


@dataclass
class Obj:
    category: str
    xywh: Tuple[int, int, int, int]
    rgb: Optional[Tuple[int, int, int]] = None

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        x, y, w, h = self.xywh
        return x, y, x + w, y + h

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.xywh
        return x + 0.5 * w, y + 0.5 * h


def _import_ocatari():
    try:
        from ocatari.core import OCAtari
    except Exception as exc:  # pragma: no cover - dependency diagnostic path
        raise RuntimeError(
            "OCAtari is not importable. Install it with: "
            'pip install "gymnasium[atari,accept-rom-license]" ocatari'
        ) from exc
    return OCAtari


def _rgb_frame(env: Any, obs: Any) -> np.ndarray:
    frame = None
    if hasattr(env, "get_rgb_state"):
        value = env.get_rgb_state
        frame = value() if callable(value) else value
    if frame is None:
        try:
            frame = env.render()
        except Exception:
            frame = obs
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise RuntimeError(f"Could not obtain RGB frame, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _object_xywh(raw: Any) -> Optional[Tuple[int, int, int, int]]:
    xywh = getattr(raw, "xywh", None)
    if xywh is None:
        xy = getattr(raw, "xy", None)
        wh = getattr(raw, "wh", None)
        if xy is not None and wh is not None:
            xywh = tuple(xy) + tuple(wh)
    if xywh is None:
        x = getattr(raw, "x", None)
        y = getattr(raw, "y", None)
        w = getattr(raw, "w", getattr(raw, "width", None))
        h = getattr(raw, "h", getattr(raw, "height", None))
        if None not in (x, y, w, h):
            xywh = (x, y, w, h)
    if xywh is None or len(xywh) != 4:
        return None
    x, y, w, h = [int(round(float(v))) for v in xywh]
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _object_category(raw: Any) -> str:
    for name in ("category", "name"):
        value = getattr(raw, name, None)
        if value:
            return str(value)
    return raw.__class__.__name__


def _read_objects(env: Any, width: int, height: int, include_hud: bool) -> List[Obj]:
    objects = []
    for raw in getattr(env, "objects", []) or []:
        if not include_hud and bool(getattr(raw, "hud", False)):
            continue
        xywh = _object_xywh(raw)
        if xywh is None:
            continue
        x, y, w, h = xywh
        x1 = max(0, min(width, x))
        y1 = max(0, min(height, y))
        x2 = max(0, min(width, x + w))
        y2 = max(0, min(height, y + h))
        if x2 <= x1 or y2 <= y1:
            continue
        rgb = getattr(raw, "rgb", None)
        rgb_tuple = tuple(int(v) for v in rgb[:3]) if rgb is not None and len(rgb) >= 3 else None
        objects.append(Obj(category=_object_category(raw), xywh=(x1, y1, x2 - x1, y2 - y1), rgb=rgb_tuple))
    objects.sort(
        key=lambda o: (
            _object_priority(o.category),
            o.category,
            o.xywh[1],
            o.xywh[0],
            o.xywh[2],
            o.xywh[3],
        )
    )
    return objects


def _object_type_id(category: str) -> int:
    key = category.lower()
    for token, idx in OBJECT_TYPE_TO_ID.items():
        if token in key:
            return idx
    return 2


def _object_priority(category: str) -> int:
    key = category.lower()
    for token, priority in OBJECT_PRIORITY.items():
        if token in key:
            return priority
    return 5


def _slot_distance(a: Obj, b: Obj) -> float:
    ax, ay = a.center
    bx, by = b.center
    return (ax - bx) ** 2 + (ay - by) ** 2


def _assign_slots(frames_objects: Sequence[List[Obj]], max_objects: int) -> Tuple[List[List[Optional[Obj]]], List[str]]:
    slots: List[Optional[Obj]] = []
    slot_categories: List[str] = []
    assigned_frames: List[List[Optional[Obj]]] = []

    for objects in frames_objects:
        assigned: List[Optional[Obj]] = [None for _ in slots]
        unused = list(range(len(objects)))
        for slot_idx, prev in enumerate(slots):
            if prev is None or not unused:
                continue
            same_class = [i for i in unused if objects[i].category == slot_categories[slot_idx]]
            candidates = same_class or unused
            best_i = min(candidates, key=lambda i: _slot_distance(prev, objects[i]))
            assigned[slot_idx] = objects[best_i]
            unused.remove(best_i)
        for obj_idx in unused:
            if len(slots) >= max_objects:
                break
            slots.append(objects[obj_idx])
            slot_categories.append(objects[obj_idx].category)
            assigned.append(objects[obj_idx])
        slots = [obj for obj in assigned]
        assigned_frames.append(assigned[:max_objects] + [None] * max(0, max_objects - len(assigned)))

    return assigned_frames, slot_categories[:max_objects]


def _delta_action(prev: Optional[Obj], curr: Optional[Obj], deadzone: float) -> int:
    if prev is None or curr is None:
        return ACTION_TO_ID["stay"]
    px, py = prev.center
    cx, cy = curr.center
    dx, dy = cx - px, cy - py
    if abs(dx) <= deadzone and abs(dy) <= deadzone:
        return ACTION_TO_ID["stay"]
    if abs(dx) >= abs(dy):
        return ACTION_TO_ID["right"] if dx > 0 else ACTION_TO_ID["left"]
    return ACTION_TO_ID["down"] if dy > 0 else ACTION_TO_ID["up"]


def _make_sample(
    frames: Sequence[np.ndarray],
    frame_objects: Sequence[List[Obj]],
    env_actions: Sequence[int],
    ram_states: Sequence[Sequence[int]],
    *,
    game: str,
    env_name: str,
    split: str,
    episode_id: int,
    frame_start: int,
    sample_index: int,
    max_objects: int,
    mode: str,
    deadzone: float,
) -> Dict[str, Any]:
    videos = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    T, _, H, W = videos.shape
    assigned, slot_categories = _assign_slots(frame_objects, max_objects)
    K = max_objects

    masks = torch.zeros((T, K, H, W), dtype=torch.uint8)
    bboxes = torch.zeros((T, K, 4), dtype=torch.float32)
    positions = torch.full((T, K, 2), -1, dtype=torch.long)
    valid_mask = torch.zeros((T, K), dtype=torch.bool)
    actions = torch.zeros((max(T - 1, 0), K), dtype=torch.long)

    for t in range(T):
        for k, obj in enumerate(assigned[t]):
            if obj is None or k >= K:
                continue
            x1, y1, x2, y2 = obj.xyxy
            masks[t, k, y1:y2, x1:x2] = 1
            bboxes[t, k] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            cx, cy = obj.center
            positions[t, k] = torch.tensor([round(cy), round(cx)], dtype=torch.long)
            valid_mask[t, k] = True
    for t in range(max(T - 1, 0)):
        for k in range(K):
            actions[t, k] = _delta_action(assigned[t][k], assigned[t + 1][k], deadzone)

    object_types = torch.tensor([_object_type_id(c) for c in slot_categories], dtype=torch.long)
    if object_types.numel() < K:
        object_types = torch.cat([object_types, torch.full((K - object_types.numel(),), -1, dtype=torch.long)])

    sample = {
        "videos": videos,
        "masks": masks,
        "bboxes": bboxes,
        "positions": positions,
        "actions": actions,
        "env_actions": torch.tensor(env_actions[: max(T - 1, 0)], dtype=torch.long),
        "actor_ids": torch.arange(K, dtype=torch.long),
        "object_types": object_types,
        "valid_mask": valid_mask,
        "num_actors": int(valid_mask.any(dim=0).sum().item()),
        "metadata": {
            "task_name": f"OCAtari-{game}",
            "game_name": game,
            "env_name": env_name,
            "split": split,
            "episode_id": int(episode_id),
            "frame_index": int(frame_start),
            "sample_index": int(sample_index),
            "has_object_annotations": True,
            "annotation_status": "ocatari_ram_or_vision_objects",
            "ocatari_mode": mode,
            "slot_categories": slot_categories,
            "ram_states": [list(map(int, r)) for r in ram_states],
        },
    }
    return sample


def _sample_action(env: Any, policy: str) -> int:
    if policy == "noop":
        return 0
    if policy == "minimal_random":
        meanings = getattr(getattr(env, "_env", env), "get_action_meanings", lambda: [])()
        preferred = [i for i, name in enumerate(meanings) if name in {"NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"}]
        if preferred:
            return random.choice(preferred)
    return random.randrange(int(getattr(env, "nb_actions", 1)))


def prepare_game(
    *,
    game: str,
    out_root: str,
    train: int,
    val: int,
    num_frames: int,
    stride: int,
    max_objects: int,
    seed: int,
    mode: str,
    policy: str,
    warmup: int,
    max_steps: int,
    include_hud: bool,
    deadzone: float,
    resume_existing: bool,
) -> Dict[str, Any]:
    OCAtari = _import_ocatari()
    env_name = GAME_ENV_NAMES.get(game, game)
    env = OCAtari(env_name, mode=mode, hud=include_hud, obs_mode="ori", render_mode="rgb_array")

    out_dirs = {
        "train": os.path.join(out_root, f"ocatari_{game}", "train"),
        "val": os.path.join(out_root, f"ocatari_{game}", "val"),
    }
    for path in out_dirs.values():
        os.makedirs(path, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    counts = {"train": 0, "val": 0}
    if resume_existing:
        for split, path in out_dirs.items():
            counts[split] = len([name for name in os.listdir(path) if name.endswith(".pt")])
        counts["train"] = min(counts["train"], train)
        counts["val"] = min(counts["val"], val)
    max_total = train + val
    episode_id = 0
    steps = 0
    obs, _info = env.reset(seed=seed)

    for _ in range(warmup):
        obs, _reward, terminated, truncated, _info = env.step(_sample_action(env, policy))
        if terminated or truncated:
            episode_id += 1
            obs, _info = env.reset(seed=seed + episode_id)

    frames: List[np.ndarray] = []
    objects: List[List[Obj]] = []
    env_actions: List[int] = []
    ram_states: List[Sequence[int]] = []
    frame_indices: List[int] = []

    progress = tqdm(total=max_total, desc=f"ocatari {game}")
    try:
        while sum(counts.values()) < max_total and steps < max_steps:
            frame = _rgb_frame(env, obs)
            H, W = frame.shape[:2]
            frames.append(frame)
            objects.append(_read_objects(env, W, H, include_hud=include_hud))
            ram_states.append(env.get_ram() if hasattr(env, "get_ram") else [])
            frame_indices.append(steps)

            while len(frames) >= num_frames and sum(counts.values()) < max_total:
                split = "train" if counts["train"] < train else "val"
                split_limit = train if split == "train" else val
                if counts[split] >= split_limit:
                    break
                sample = _make_sample(
                    frames[:num_frames],
                    objects[:num_frames],
                    env_actions[: max(num_frames - 1, 0)],
                    ram_states[:num_frames],
                    game=game,
                    env_name=env_name,
                    split=split,
                    episode_id=episode_id,
                    frame_start=frame_indices[0],
                    sample_index=counts[split],
                    max_objects=max_objects,
                    mode=mode,
                    deadzone=deadzone,
                )
                errors = validate_v12_sample(
                    {
                        **sample,
                        "videos": sample["videos"].float().permute(0, 2, 3, 1) / 255.0,
                        "masks": sample["masks"].float(),
                    }
                )
                if errors:
                    raise RuntimeError(f"V12 schema validation failed for {game}: {errors}")
                torch.save(sample, os.path.join(out_dirs[split], f"sample_{counts[split]:06d}.pt"))
                counts[split] += 1
                progress.update(1)
                del frames[:stride]
                del objects[:stride]
                del ram_states[:stride]
                del frame_indices[:stride]
                del env_actions[:stride]

            action = _sample_action(env, policy)
            env_actions.append(action)
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            if terminated or truncated:
                frames.clear()
                objects.clear()
                env_actions.clear()
                ram_states.clear()
                frame_indices.clear()
                episode_id += 1
                obs, _info = env.reset(seed=seed + episode_id)
    finally:
        progress.close()
        env.close()

    report = {
        "game": game,
        "env_name": env_name,
        "counts": counts,
        "num_frames": num_frames,
        "stride": stride,
        "max_objects": max_objects,
        "seed": seed,
        "mode": mode,
        "policy": policy,
        "include_hud": include_hud,
        "max_steps": max_steps,
        "resume_existing": resume_existing,
    }
    if counts["train"] < train or counts["val"] < val:
        report["warning"] = "requested sample count not reached before max_steps"
    with open(os.path.join(out_root, f"ocatari_{game}", "dataset_config.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=["freeway", "mspacman", "spaceinvaders"])
    parser.add_argument("--out_root", default="data/v12_ocatari")
    parser.add_argument("--train", type=int, default=16)
    parser.add_argument("--val", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["ram", "vision", "both"], default="ram")
    parser.add_argument("--policy", choices=["random", "minimal_random", "noop"], default="minimal_random")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--include_hud", action="store_true")
    parser.add_argument("--deadzone", type=float, default=1.0)
    parser.add_argument("--resume_existing", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    reports = []
    for idx, game in enumerate(args.games):
        reports.append(
            prepare_game(
                game=game,
                out_root=args.out_root,
                train=args.train,
                val=args.val,
                num_frames=args.num_frames,
                stride=args.stride,
                max_objects=args.max_objects,
                seed=args.seed + idx * 1000,
                mode=args.mode,
                policy=args.policy,
                warmup=args.warmup,
                max_steps=args.max_steps,
                include_hud=args.include_hud,
                deadzone=args.deadzone,
                resume_existing=args.resume_existing,
            )
        )
    print(json.dumps({"reports": reports}, indent=2))


if __name__ == "__main__":
    main()
