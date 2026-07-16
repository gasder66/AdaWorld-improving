"""Build transition-level movement/punch indexes for balanced V16 training."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List

import torch


EVENT_NAMES = ("movement_only", "punch_onset", "punch_extend", "punch_hold", "punch_retract", "punch_switch")
INTERACTION_PRIMARY_NAMES = (
    "non_interaction", "near", "contact", "punch_miss", "hit", "received_hit", "occlusion", "recovery"
)


def _side(arms: torch.Tensor) -> str:
    left = bool(arms[0] != 0)
    right = bool(arms[1] != 0)
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return "none"


def classify_transition(previous: torch.Tensor, current: torch.Tensor) -> str:
    previous_active = bool((previous != 0).any())
    current_active = bool((current != 0).any())
    previous_side = _side(previous)
    current_side = _side(current)
    delta = current - previous
    if not previous_active and current_active:
        return "punch_onset"
    if previous_active and current_active and previous_side != current_side:
        return "punch_switch"
    if current_active and bool((delta > 0).any()):
        return "punch_extend"
    if previous_active and bool((delta < 0).any()):
        return "punch_retract"
    if current_active:
        return "punch_hold"
    return "movement_only"


def classify_interaction(sample: Dict, transition: int, slot: int) -> tuple[List[str], str]:
    events: List[str] = []
    for name, key in (("near", "near_labels"), ("contact", "contact_labels")):
        values = sample.get(key)
        if values is not None and bool(values[transition] or values[transition + 1]):
            events.append(name)
    punch_miss = sample.get("punch_miss_labels")
    if punch_miss is not None and bool(punch_miss[transition, slot]):
        events.append("punch_miss")
    hit_actor = sample.get("hit_actor")
    hit_receiver = sample.get("hit_receiver")
    if hit_actor is not None and int(hit_actor[transition]) == slot:
        events.append("hit")
    if hit_receiver is not None and int(hit_receiver[transition]) == slot:
        events.append("received_hit")
    occlusion = sample.get("occlusion_labels")
    if occlusion is not None and bool(occlusion[transition, slot] or occlusion[transition + 1, slot]):
        events.append("occlusion")
    recovery = sample.get("recovery_labels")
    if recovery is not None and bool(recovery[transition]):
        events.append("recovery")
    precedence = ("received_hit", "hit", "occlusion", "recovery", "contact", "punch_miss", "near")
    primary = next((name for name in precedence if name in events), "non_interaction")
    return events, primary


def build_split(roots: List[str], split: str) -> Dict:
    entries = []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, split, "*.pt"))):
            sample = torch.load(path, map_location="cpu", weights_only=False)
            arms = sample["arm_lengths"].to(torch.int16)
            valid = sample["valid_mask"].bool()
            for t in range(arms.shape[0] - 1):
                for slot in range(2):
                    if not bool(valid[t, slot] and valid[t + 1, slot]):
                        continue
                    event = classify_transition(arms[t, slot], arms[t + 1, slot])
                    interaction_events, interaction_primary = classify_interaction(sample, t, slot)
                    entries.append(
                        {
                            "path": os.path.abspath(path),
                            "transition": t,
                            "target_slot": slot,
                            "fighter": "Player" if slot == 0 else "Enemy",
                            "event": event,
                            "side_before": _side(arms[t, slot]),
                            "side_after": _side(arms[t + 1, slot]),
                            "arm_before": arms[t, slot].tolist(),
                            "arm_after": arms[t + 1, slot].tolist(),
                            "arm_delta": (arms[t + 1, slot] - arms[t, slot]).tolist(),
                            "interaction_events": interaction_events,
                            "interaction_primary": interaction_primary,
                        }
                    )
    event_counts = Counter(entry["event"] for entry in entries)
    event_fighter_counts = Counter((entry["event"], entry["fighter"]) for entry in entries)
    side_counts = Counter(entry["side_after"] for entry in entries if entry["event"] != "movement_only")
    interaction_counts = Counter(entry["interaction_primary"] for entry in entries)
    interaction_fighter_counts = Counter((entry["interaction_primary"], entry["fighter"]) for entry in entries)
    return {
        "split": split,
        "roots": [os.path.abspath(root) for root in roots],
        "entries": entries,
        "stats": {
            "transitions": len(entries),
            "event_counts": dict(event_counts),
            "event_fighter_counts": {f"{event}:{fighter}": count for (event, fighter), count in event_fighter_counts.items()},
            "punch_side_counts": dict(side_counts),
            "interaction_primary_counts": dict(interaction_counts),
            "interaction_fighter_counts": {
                f"{event}:{fighter}": count
                for (event, fighter), count in interaction_fighter_counts.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for split in ("train", "val"):
        index = build_split(args.roots, split)
        torch.save(index, os.path.join(args.output_dir, f"{split}.pt"))
        summary[split] = index["stats"]
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
