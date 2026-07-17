"""Compare V16 FDMs and opponent-token interventions by interaction event."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_transition_dataset import BoxingTransitionDataset
from lam.modules.v16_boxing_model import BoxingObjectLAM


EVENTS = ("non_interaction", "near", "contact", "punch_miss", "hit", "received_hit", "occlusion", "recovery")


@torch.no_grad()
def _evaluate_subset(
    model: BoxingObjectLAM,
    loader: DataLoader,
    device: torch.device,
    slot: int,
    *,
    z_ablation: str = "normal",
    opponent_ablation: str = "normal",
) -> tuple[float, int]:
    total, count = 0.0, 0
    for batch in loader:
        model_batch = {key: batch[key].to(device) for key in ("videos", "masks", "background_masks")}
        output = model(
            model_batch,
            ablation=z_ablation,
            opponent_ablation=opponent_ablation,
            target_slot=slot if opponent_ablation != "normal" else None,
        )
        predicted = output["predicted_object_states"][:, 0, slot]
        target = output["object_states"][:, 1, slot]
        error = (predicted - target).square().mean(dim=(-3, -2, -1))
        total += float(error.sum())
        count += int(error.numel())
    return total, count


def evaluate(checkpoint_path: str, index_path: str, max_per_event: int, batch_size: int, gpu: int, seed: int) -> Dict:
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    fdm_type = config.get("fdm_type", "independent")
    model = BoxingObjectLAM(
        config["state_dim"], config["latent_dim"], fdm_type,
        config.get("object_input_mode", "masked_rgb_mask"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = BoxingTransitionDataset(index_path)
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        grouped[(entry.get("interaction_primary", "non_interaction"), int(entry["target_slot"]))].append(index)
    modes = {
        "normal": ("normal", "normal"),
        "zero_z": ("zero", "normal"),
        "shuffle_z": ("shuffle", "normal"),
    }
    if fdm_type == "interaction":
        modes.update(
            mask_opponent_state=("normal", "mask_state"),
            shuffle_opponent_state=("normal", "shuffle_state"),
            mask_opponent_z=("normal", "mask_z"),
            shuffle_opponent_z=("normal", "shuffle_z"),
        )
    sums = {event: {mode: 0.0 for mode in modes} for event in EVENTS}
    counts = {event: {mode: 0 for mode in modes} for event in EVENTS}
    slot_counts = {event: {"Player": 0, "Enemy": 0} for event in EVENTS}
    for event in EVENTS:
        for slot in (0, 1):
            indices = grouped[(event, slot)]
            rng.shuffle(indices)
            indices = indices[:max_per_event]
            if not indices:
                continue
            slot_counts[event]["Player" if slot == 0 else "Enemy"] = len(indices)
            loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
            for mode, (z_ablation, opponent_ablation) in modes.items():
                value, count = _evaluate_subset(
                    model, loader, device, slot,
                    z_ablation=z_ablation,
                    opponent_ablation=opponent_ablation,
                )
                sums[event][mode] += value
                counts[event][mode] += count
    result = {}
    for event in EVENTS:
        metrics = {
            mode: sums[event][mode] / max(1, counts[event][mode])
            for mode in modes
        }
        normal = metrics["normal"]
        result[event] = {
            "count": sum(slot_counts[event].values()),
            "slot_counts": slot_counts[event],
            **metrics,
            **{f"{mode}_minus_normal": value - normal for mode, value in metrics.items() if mode != "normal"},
        }
    return {"checkpoint": checkpoint_path, "fdm_type": fdm_type, "events": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_per_event", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.index, args.max_per_event, args.batch_size, args.gpu, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
