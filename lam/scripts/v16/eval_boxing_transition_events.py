"""Report normal/zero/shuffle state errors for each Boxing event phase."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_transition_dataset import BoxingTransitionDataset, EVENT_TO_ID
from lam.modules.v16_boxing_model import BoxingObjectLAM


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    model = BoxingObjectLAM(
        config["state_dim"], config["latent_dim"], config.get("fdm_type", "independent"),
        config.get("object_input_mode", "masked_rgb_mask"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    base_dataset = BoxingTransitionDataset(args.index)
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(base_dataset), min(args.max_samples, len(base_dataset)), replace=False)
    loader = DataLoader(Subset(base_dataset, indices.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)
    id_to_event = {value: key for key, value in EVENT_TO_ID.items()}
    errors = {mode: defaultdict(list) for mode in ("normal", "zero", "shuffle")}
    for batch in loader:
        model_batch = {key: batch[key].to(device) for key in ("videos", "masks", "background_masks")}
        event_ids = batch["event_id"].numpy()
        target_slots = batch["target_slot"].numpy()
        for mode in errors:
            output = model(model_batch, ablation=mode)
            target = output["object_states"][:, 1:]
            prediction = output["predicted_object_states"]
            per_slot = (prediction - target).square().mean(dim=(-3, -2, -1))[:, 0].cpu().numpy()
            for row, (event_id, slot) in enumerate(zip(event_ids, target_slots)):
                errors[mode][id_to_event[int(event_id)]].append(float(per_slot[row, int(slot)]))
    report = {}
    for event in EVENT_TO_ID:
        normal = float(np.mean(errors["normal"][event])) if errors["normal"][event] else float("nan")
        zero = float(np.mean(errors["zero"][event])) if errors["zero"][event] else float("nan")
        shuffle = float(np.mean(errors["shuffle"][event])) if errors["shuffle"][event] else float("nan")
        report[event] = {
            "count": len(errors["normal"][event]),
            "normal_state_mse": normal,
            "zero_state_mse": zero,
            "shuffle_state_mse": shuffle,
            "zero_minus_normal": zero - normal,
            "shuffle_minus_normal": shuffle - normal,
        }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
