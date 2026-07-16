"""Generate controlled OCAtari Boxing clips for object-wise latent actions.

Stage 1 deliberately contains movement only: no punches, score changes,
geometric contact, or occlusion.  OCAtari RAM objects provide stable fighter
identity and state labels, while pixel masks are extracted from the rendered
sprites inside OCAtari's vision boxes.  Bounding boxes are stored for audit
only; the model input is RGB plus the two sprite masks.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


ENV_NAME = "ALE/Boxing-v5"
FIGHTER_NAMES = ("Player", "Enemy")
FIGHTER_COLORS = ((214, 214, 214), (0, 0, 0))
# Keep both fighters in [left_arm, right_arm] order. OCAtari exposes Enemy
# arms in the opposite source order (right=59, left=61).
ARM_RAM_INDICES = (55, 57, 61, 59)
SCORE_RAM_INDICES = (18, 19)
PUNCH_EVENT_NAMES = ("punch_onset", "punch_extend", "punch_hold", "punch_retract", "punch_switch")
INTERACTION_EVENT_NAMES = ("near", "contact", "punch_miss", "hit", "occlusion", "recovery")


@dataclass(frozen=True)
class FighterObservation:
    name: str
    mask: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[float, float]
    ram_bbox_xywh: Tuple[int, int, int, int]


@dataclass(frozen=True)
class FrameState:
    frame: np.ndarray
    fighters: Tuple[FighterObservation, FighterObservation]
    ram: np.ndarray
    arm_lengths: Tuple[int, int, int, int]
    scores: Tuple[int, int]


def _import_ocatari():
    try:
        from ocatari.core import OCAtari
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError(
            "OCAtari is unavailable. Use the project environment containing "
            "ocatari and ALE ROMs (currently conda env 'locateanything')."
        ) from exc
    return OCAtari


def _rgb_frame(env: Any, obs: Any) -> np.ndarray:
    frame = None
    if hasattr(env, "get_rgb_state"):
        value = env.get_rgb_state
        frame = value() if callable(value) else value
    if frame is None:
        frame = env.render()
    if frame is None:
        frame = obs
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[-1]
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError(f"Expected HWC RGB frame, got {frame.shape}")
    if frame.dtype != np.uint8:
        if frame.max() <= 1.5:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _category(obj: Any) -> str:
    return str(getattr(obj, "category", obj.__class__.__name__))


def _find_object(objects: Iterable[Any], category: str) -> Any:
    for obj in objects:
        if _category(obj).lower() == category.lower():
            return obj
    raise RuntimeError(f"OCAtari did not return required Boxing object: {category}")


def _xywh(obj: Any) -> Tuple[int, int, int, int]:
    values = getattr(obj, "xywh", None)
    if values is None:
        values = tuple(getattr(obj, "xy")) + tuple(getattr(obj, "wh"))
    x, y, w, h = (int(round(float(v))) for v in values)
    return x, y, w, h


def _clip_box_xywh(box: Sequence[int], width: int, height: int, pad: int = 0) -> Tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in box)
    x1 = max(0, min(width, x - pad))
    y1 = max(0, min(height, y - pad))
    x2 = max(0, min(width, x + w + pad))
    y2 = max(0, min(height, y + h + pad))
    return x1, y1, x2, y2


def _sprite_mask(frame: np.ndarray, color: Sequence[int], search_xywh: Sequence[int]) -> np.ndarray:
    """Return exact-color visible sprite pixels inside a trusted vision box."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box_xywh(search_xywh, width, height, pad=1)
    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return mask
    region = frame[y1:y2, x1:x2]
    target = np.asarray(color, dtype=np.uint8).reshape(1, 1, 3)
    mask[y1:y2, x1:x2] = np.all(region == target, axis=-1).astype(np.uint8)
    return mask


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _center_from_bbox(box: Sequence[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _intersection_area(a: Sequence[int], b: Sequence[int]) -> int:
    ax1, ay1, ax2, ay2 = (int(v) for v in a)
    bx1, by1, bx2, by2 = (int(v) for v in b)
    return max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))


def _edge_distance(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float(np.hypot(dx, dy))


def _read_state(env: Any, obs: Any) -> FrameState:
    frame = _rgb_frame(env, obs)
    ram = np.asarray(env.get_ram(), dtype=np.uint8).copy()
    ram_objects = list(getattr(env, "objects", []) or [])
    vision_objects = list(getattr(env, "objects_v", []) or [])
    fighters: List[FighterObservation] = []
    for name, color in zip(FIGHTER_NAMES, FIGHTER_COLORS):
        ram_obj = _find_object(ram_objects, name)
        vision_obj = _find_object(vision_objects, name)
        ram_box = _xywh(ram_obj)
        vision_box = _xywh(vision_obj)
        mask = _sprite_mask(frame, color, vision_box)
        bbox = _bbox_from_mask(mask)
        if mask.sum() == 0:
            raise RuntimeError(f"Empty visible mask for {name}; vision box={vision_box}")
        fighters.append(
            FighterObservation(
                name=name,
                mask=mask,
                bbox_xyxy=bbox,
                center_xy=_center_from_bbox(bbox),
                ram_bbox_xywh=ram_box,
            )
        )
    return FrameState(
        frame=frame,
        fighters=(fighters[0], fighters[1]),
        ram=ram,
        arm_lengths=tuple(int(ram[i]) for i in ARM_RAM_INDICES),
        scores=tuple(int(ram[i]) for i in SCORE_RAM_INDICES),
    )


def _movement_label(dx: float, dy: float, deadzone: float = 0.5) -> int:
    # 0 stay, 1 up, 2 down, 3 left, 4 right; labels are evaluation-only.
    if abs(dx) <= deadzone and abs(dy) <= deadzone:
        return 0
    if abs(dx) >= abs(dy):
        return 4 if dx > 0 else 3
    return 2 if dy > 0 else 1


def _arm_side(arms: Sequence[int]) -> str:
    left = int(arms[0]) != 0
    right = int(arms[1]) != 0
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return "none"


def _classify_punch_transition(previous: Sequence[int], current: Sequence[int]) -> str:
    previous = np.asarray(previous, dtype=np.int16)
    current = np.asarray(current, dtype=np.int16)
    previous_active = bool(np.any(previous != 0))
    current_active = bool(np.any(current != 0))
    if not previous_active and current_active:
        return "punch_onset"
    if previous_active and current_active and _arm_side(previous) != _arm_side(current):
        return "punch_switch"
    delta = current - previous
    if current_active and bool(np.any(delta > 0)):
        return "punch_extend"
    if previous_active and bool(np.any(delta < 0)):
        return "punch_retract"
    if current_active:
        return "punch_hold"
    return "movement_only"


def _interaction_labels(
    states: Sequence[FrameState],
    *,
    near_distance: float,
    occlusion_drop_ratio: float,
    neutral_arm_value: int,
) -> Dict[str, np.ndarray]:
    """Derive evaluation-only interaction labels from oracle geometry and RAM."""
    time = len(states)
    boxes = [[fighter.bbox_xyxy for fighter in state.fighters] for state in states]
    visible_pixels = np.asarray(
        [[int(fighter.mask.sum()) for fighter in state.fighters] for state in states], dtype=np.int32
    )
    overlap_area = np.asarray([_intersection_area(pair[0], pair[1]) for pair in boxes], dtype=np.int32)
    edge_distance = np.asarray([_edge_distance(pair[0], pair[1]) for pair in boxes], dtype=np.float32)
    contact = overlap_area > 0
    near = (edge_distance <= near_distance) & ~contact

    occlusion = np.zeros((time, 2), dtype=np.uint8)
    for t in range(1, time):
        if not contact[t]:
            continue
        previous = np.maximum(visible_pixels[t - 1], 1)
        occlusion[t] = (visible_pixels[t] < previous * (1.0 - occlusion_drop_ratio)).astype(np.uint8)

    scores = np.asarray([state.scores for state in states], dtype=np.int16)
    score_delta = scores[1:] - scores[:-1]
    hit = np.any(score_delta != 0, axis=1).astype(np.uint8)
    hit_actor = np.full(time - 1, -1, dtype=np.int64)
    hit_receiver = np.full(time - 1, -1, dtype=np.int64)
    for t in range(time - 1):
        changed = np.flatnonzero(score_delta[t] != 0)
        if len(changed):
            actor = int(changed[0])
            hit_actor[t] = actor
            hit_receiver[t] = 1 - actor

    arms = np.asarray([state.arm_lengths for state in states], dtype=np.int16).reshape(time, 2, 2)
    punch_active = np.any(arms != neutral_arm_value, axis=-1)
    close_transition = contact[:-1] | contact[1:] | near[:-1] | near[1:]
    punch_miss = (
        (punch_active[:-1] | punch_active[1:])
        & close_transition[:, None]
        & ~hit[:, None].astype(bool)
    ).astype(np.uint8)
    recovery = (contact[:-1] & ~contact[1:]).astype(np.uint8)
    return {
        "near_labels": near.astype(np.uint8),
        "contact_labels": contact.astype(np.uint8),
        "punch_miss_labels": punch_miss,
        "hit_labels": hit,
        "hit_actor": hit_actor,
        "hit_receiver": hit_receiver,
        "score_delta": score_delta,
        "occlusion_labels": occlusion,
        "recovery_labels": recovery,
        "bbox_overlap_area": overlap_area,
        "edge_distance": edge_distance,
        "visible_pixels": visible_pixels,
    }


def _present_interaction_events(labels: Dict[str, np.ndarray]) -> set[str]:
    present = set()
    for event, key in (
        ("near", "near_labels"),
        ("contact", "contact_labels"),
        ("punch_miss", "punch_miss_labels"),
        ("hit", "hit_labels"),
        ("occlusion", "occlusion_labels"),
        ("recovery", "recovery_labels"),
    ):
        if bool(np.asarray(labels[key]).any()):
            present.add(event)
    return present


def _clip_is_stage1(
    states: Sequence[FrameState],
    *,
    min_separation: float,
    min_motion: float,
    neutral_arm_value: int,
) -> Tuple[bool, str]:
    if any(any(v != neutral_arm_value for v in state.arm_lengths) for state in states):
        return False, "punch"
    if any(state.scores != states[0].scores for state in states[1:]):
        return False, "score_change"
    for state in states:
        a, b = (fighter.bbox_xyxy for fighter in state.fighters)
        if _intersection_area(a, b) > 0:
            return False, "bbox_overlap"
        if _edge_distance(a, b) < min_separation:
            return False, "too_close"
        if np.logical_and(state.fighters[0].mask, state.fighters[1].mask).any():
            return False, "mask_overlap"
    displacement = 0.0
    for fighter_idx in range(2):
        for left, right in zip(states[:-1], states[1:]):
            x0, y0 = left.fighters[fighter_idx].center_xy
            x1, y1 = right.fighters[fighter_idx].center_xy
            displacement = max(displacement, float(np.hypot(x1 - x0, y1 - y0)))
    if displacement < min_motion:
        return False, "static"
    return True, "accepted"


def _clip_is_isolated_punch(
    states: Sequence[FrameState],
    *,
    min_separation: float,
    neutral_arm_value: int,
    required_punch_actor: str = "any",
    required_punch_events: Sequence[str] = (),
) -> Tuple[bool, str]:
    arm_values = np.asarray([state.arm_lengths for state in states], dtype=np.int16).reshape(len(states), 2, 2)
    punch_active = arm_values != neutral_arm_value
    if not punch_active.any():
        return False, "no_punch"
    if not np.any(arm_values[1:] != arm_values[:-1]):
        return False, "no_punch_phase_change"
    actor_slots = {"any": (0, 1), "player": (0,), "enemy": (1,)}[required_punch_actor]
    actor_events = {
        _classify_punch_transition(arm_values[t, slot], arm_values[t + 1, slot])
        for slot in actor_slots
        for t in range(len(states) - 1)
    }
    requested = set(required_punch_events)
    if requested:
        if actor_events.isdisjoint(requested):
            return False, "missing_required_punch_event"
    elif not actor_events.intersection(PUNCH_EVENT_NAMES):
        return False, "missing_required_punch_actor"
    if any(state.scores != states[0].scores for state in states[1:]):
        return False, "score_change"
    for state in states:
        a, b = (fighter.bbox_xyxy for fighter in state.fighters)
        if _intersection_area(a, b) > 0:
            return False, "bbox_overlap"
        if _edge_distance(a, b) < min_separation:
            return False, "too_close"
        if np.logical_and(state.fighters[0].mask, state.fighters[1].mask).any():
            return False, "mask_overlap"
    return True, "accepted"


def _clip_is_interaction(
    states: Sequence[FrameState],
    *,
    near_distance: float,
    occlusion_drop_ratio: float,
    neutral_arm_value: int,
    required_interaction_events: Sequence[str],
    required_hit_actor: str,
) -> Tuple[bool, str]:
    labels = _interaction_labels(
        states,
        near_distance=near_distance,
        occlusion_drop_ratio=occlusion_drop_ratio,
        neutral_arm_value=neutral_arm_value,
    )
    present = _present_interaction_events(labels)
    requested = set(required_interaction_events)
    if requested and present.isdisjoint(requested):
        return False, "missing_required_interaction_event"
    if not requested and not present.intersection({"contact", "punch_miss", "hit", "occlusion", "recovery"}):
        return False, "no_interaction"
    if required_hit_actor != "any":
        slot = 0 if required_hit_actor == "player" else 1
        if not bool(np.any(labels["hit_actor"] == slot)):
            return False, "missing_required_hit_actor"
    return True, "accepted"


def _make_sample(
    states: Sequence[FrameState],
    transition_actions: Sequence[int],
    *,
    split: str,
    sample_index: int,
    episode_id: int,
    frame_start: int,
    action_meanings: Sequence[str],
    stage: str,
    near_distance: float,
    occlusion_drop_ratio: float,
    neutral_arm_value: int,
) -> Dict[str, Any]:
    frames = np.stack([state.frame for state in states])
    masks = np.stack([[fighter.mask for fighter in state.fighters] for state in states])
    boxes = np.asarray([[fighter.bbox_xyxy for fighter in state.fighters] for state in states], dtype=np.float32)
    centers = np.asarray([[fighter.center_xy for fighter in state.fighters] for state in states], dtype=np.float32)
    ram_boxes = np.asarray([[fighter.ram_bbox_xywh for fighter in state.fighters] for state in states], dtype=np.float32)
    arm_lengths = np.asarray([state.arm_lengths for state in states], dtype=np.int16).reshape(len(states), 2, 2)
    punch_active = np.any(arm_lengths != 0, axis=-1).astype(np.uint8)
    punch_side = np.zeros((len(states), 2), dtype=np.int64)
    left_active = arm_lengths[..., 0] != 0
    right_active = arm_lengths[..., 1] != 0
    punch_side[left_active] = 1
    punch_side[right_active] = 2
    punch_side[left_active & right_active] = 3
    scores = np.asarray([state.scores for state in states], dtype=np.int16)
    delta_xy = centers[1:] - centers[:-1]
    movement = np.zeros(delta_xy.shape[:2], dtype=np.int64)
    for t in range(delta_xy.shape[0]):
        for k in range(delta_xy.shape[1]):
            movement[t, k] = _movement_label(float(delta_xy[t, k, 0]), float(delta_xy[t, k, 1]))
    background = 1 - np.clip(masks.sum(axis=1), 0, 1)
    interaction = _interaction_labels(
        states,
        near_distance=near_distance,
        occlusion_drop_ratio=occlusion_drop_ratio,
        neutral_arm_value=neutral_arm_value,
    )
    sample = {
        "videos": torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(),
        "masks": torch.from_numpy(masks.astype(np.uint8)),
        "background_masks": torch.from_numpy(background.astype(np.uint8)),
        "bboxes_xyxy": torch.from_numpy(boxes),
        "ram_bboxes_xywh": torch.from_numpy(ram_boxes),
        "centers_xy": torch.from_numpy(centers),
        "delta_xy": torch.from_numpy(delta_xy),
        "movement_labels": torch.from_numpy(movement),
        "actions": torch.from_numpy(movement),
        "env_actions": torch.tensor(transition_actions, dtype=torch.long),
        "arm_lengths": torch.from_numpy(arm_lengths),
        "punch_labels": torch.from_numpy(punch_active),
        "punch_side_labels": torch.from_numpy(punch_side),
        "arm_delta": torch.from_numpy(arm_lengths[1:] - arm_lengths[:-1]),
        **{key: torch.from_numpy(value) for key, value in interaction.items()},
        "scores": torch.from_numpy(scores),
        "valid_mask": torch.ones((len(states), 2), dtype=torch.bool),
        "track_ids": torch.tensor([0, 1], dtype=torch.long),
        "actor_ids": torch.tensor([0, 1], dtype=torch.long),
        "object_types": torch.tensor([0, 1], dtype=torch.long),
        "ram": torch.from_numpy(np.stack([state.ram for state in states])),
        "metadata": {
            "task_name": f"OCAtari-Boxing-{stage}",
            "game_name": "boxing",
            "env_name": ENV_NAME,
            "split": split,
            "sample_index": int(sample_index),
            "episode_id": int(episode_id),
            "frame_index": int(frame_start),
            "slot_categories": list(FIGHTER_NAMES),
            "action_meanings": list(action_meanings),
            "mask_source": "exact_sprite_color_within_ocatari_vision_box",
            "bbox_role": "audit_only_not_model_state",
            "stage": stage,
            "interaction_events": sorted(_present_interaction_events(interaction)),
        },
    }
    return sample


def _movement_action_ids(action_meanings: Sequence[str]) -> List[int]:
    allowed = {"NOOP", "UP", "DOWN", "LEFT", "RIGHT", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"}
    ids = [i for i, name in enumerate(action_meanings) if name in allowed and "FIRE" not in name]
    if not ids:
        raise RuntimeError(f"No movement-only actions found in {list(action_meanings)}")
    return ids


def _generate_split(
    *,
    split: str,
    count: int,
    out_dir: str,
    seed: int,
    num_frames: int,
    stride: int,
    warmup: int,
    max_steps: int,
    min_separation: float,
    min_motion: float,
    neutral_arm_value: int,
    frameskip: int,
    stage: str,
    required_punch_actor: str,
    required_punch_events: Sequence[str],
    required_interaction_events: Sequence[str],
    required_hit_actor: str,
    near_distance: float,
    occlusion_drop_ratio: float,
) -> Dict[str, Any]:
    OCAtari = _import_ocatari()
    env = OCAtari(ENV_NAME, mode="both", hud=False, obs_mode="ori", render_mode="rgb_array", frameskip=frameskip)
    rng = random.Random(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    action_meanings = list(env._env.unwrapped.get_action_meanings())
    movement_ids = _movement_action_ids(action_meanings)
    fire_ids = [i for i, name in enumerate(action_meanings) if "FIRE" in name]
    noop_id = action_meanings.index("NOOP")
    obs, _ = env.reset(seed=seed)
    episode_id = 0
    global_frame = 0
    for _ in range(warmup):
        obs, _, terminated, truncated, _ = env.step(noop_id)
        global_frame += 1
        if terminated or truncated:
            episode_id += 1
            obs, _ = env.reset(seed=seed + episode_id)

    states: List[FrameState] = []
    actions: List[int] = []
    frame_indices: List[int] = []
    rejection_counts: Dict[str, int] = {}
    accepted = 0
    held_action = noop_id
    hold_remaining = 0
    progress = tqdm(total=count, desc=f"boxing {split}")
    try:
        for _step in range(max_steps):
            states.append(_read_state(env, obs))
            frame_indices.append(global_frame)
            while len(states) >= num_frames and accepted < count:
                candidate = states[:num_frames]
                if stage == "movement_no_punch_no_contact_no_occlusion":
                    ok, reason = _clip_is_stage1(
                        candidate,
                        min_separation=min_separation,
                        min_motion=min_motion,
                        neutral_arm_value=neutral_arm_value,
                    )
                elif stage == "isolated_punch_no_contact_no_occlusion":
                    ok, reason = _clip_is_isolated_punch(
                        candidate,
                        min_separation=min_separation,
                        neutral_arm_value=neutral_arm_value,
                        required_punch_actor=required_punch_actor,
                        required_punch_events=required_punch_events,
                    )
                else:
                    ok, reason = _clip_is_interaction(
                        candidate,
                        near_distance=near_distance,
                        occlusion_drop_ratio=occlusion_drop_ratio,
                        neutral_arm_value=neutral_arm_value,
                        required_interaction_events=required_interaction_events,
                        required_hit_actor=required_hit_actor,
                    )
                if ok:
                    sample = _make_sample(
                        candidate,
                        actions[: num_frames - 1],
                        split=split,
                        sample_index=accepted,
                        episode_id=episode_id,
                        frame_start=frame_indices[0],
                        action_meanings=action_meanings,
                        stage=stage,
                        near_distance=near_distance,
                        occlusion_drop_ratio=occlusion_drop_ratio,
                        neutral_arm_value=neutral_arm_value,
                    )
                    torch.save(sample, os.path.join(out_dir, f"sample_{accepted:06d}.pt"))
                    accepted += 1
                    progress.update(1)
                else:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                del states[:stride]
                del frame_indices[:stride]
                del actions[: min(stride, len(actions))]
            if accepted >= count:
                break

            if hold_remaining <= 0:
                if stage in {"isolated_punch_no_contact_no_occlusion", "interaction"} and fire_ids and rng.random() < 0.65:
                    held_action = rng.choice(fire_ids)
                    hold_remaining = rng.randint(1, 3)
                else:
                    held_action = rng.choice(movement_ids)
                    hold_remaining = rng.randint(1, 4)
            action = held_action
            hold_remaining -= 1
            actions.append(action)
            obs, _reward, terminated, truncated, _info = env.step(action)
            global_frame += 1
            if terminated or truncated:
                states.clear()
                actions.clear()
                frame_indices.clear()
                episode_id += 1
                obs, _ = env.reset(seed=seed + episode_id)
        if accepted < count:
            raise RuntimeError(
                f"Only generated {accepted}/{count} {split} clips in {max_steps} steps; "
                f"rejections={rejection_counts}"
            )
    finally:
        progress.close()
        env.close()
    return {
        "split": split,
        "count": accepted,
        "seed": seed,
        "episodes": episode_id + 1,
        "environment_steps": global_frame,
        "rejection_counts": rejection_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="data/v16_boxing_stage1")
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--val", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--min_separation", type=float, default=8.0)
    parser.add_argument("--min_motion", type=float, default=0.5)
    parser.add_argument("--neutral_arm_value", type=int, default=0)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--near_distance", type=float, default=6.0)
    parser.add_argument("--occlusion_drop_ratio", type=float, default=0.15)
    parser.add_argument(
        "--required_punch_actor",
        choices=["any", "player", "enemy"],
        default="any",
        help="For isolated-punch data, retain only clips containing a punch transition for this actor.",
    )
    parser.add_argument(
        "--required_punch_events",
        nargs="*",
        choices=PUNCH_EVENT_NAMES,
        default=[],
        help="Optional isolated-punch phase filter. Labels are used only for data selection/evaluation.",
    )
    parser.add_argument(
        "--required_hit_actor",
        choices=["any", "player", "enemy"],
        default="any",
        help="For interaction data, retain only clips where this fighter scores a hit.",
    )
    parser.add_argument(
        "--required_interaction_events",
        nargs="*",
        choices=INTERACTION_EVENT_NAMES,
        default=[],
        help="For interaction data, retain clips containing at least one requested event.",
    )
    parser.add_argument(
        "--stage",
        choices=["movement_no_punch_no_contact_no_occlusion", "isolated_punch_no_contact_no_occlusion", "interaction"],
        default="movement_no_punch_no_contact_no_occlusion",
    )
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    reports = []
    for split, count, seed_offset in (("train", args.train, 0), ("val", args.val, 100000)):
        reports.append(
            _generate_split(
                split=split,
                count=count,
                out_dir=os.path.join(args.out_root, split),
                seed=args.seed + seed_offset,
                num_frames=args.num_frames,
                stride=args.stride,
                warmup=args.warmup,
                max_steps=args.max_steps,
                min_separation=args.min_separation,
                min_motion=args.min_motion,
                neutral_arm_value=args.neutral_arm_value,
                frameskip=args.frameskip,
                stage=args.stage,
                required_punch_actor=args.required_punch_actor,
                required_punch_events=args.required_punch_events,
                required_interaction_events=args.required_interaction_events,
                required_hit_actor=args.required_hit_actor,
                near_distance=args.near_distance,
                occlusion_drop_ratio=args.occlusion_drop_ratio,
            )
        )
    config = {**vars(args), "env_name": ENV_NAME, "reports": reports}
    with open(os.path.join(args.out_root, "dataset_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
