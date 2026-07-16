"""Probe mixed movement/punch V16 latents without training labels in the LAM."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_object_dataset import BoxingObjectDataset
from lam.modules.v16_boxing_model import BoxingObjectLAM


@torch.no_grad()
def collect(model: BoxingObjectLAM, dataset, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    features, arm_changes, displacements, slots = [], [], [], []
    model.eval()
    for batch in loader:
        model_batch = {
            key: batch[key].to(device)
            for key in ("videos", "masks", "background_masks")
        }
        output = model(model_batch)
        z = output["z_mu"].cpu().numpy()
        arm_change = (batch["arm_lengths"][:, 1:] - batch["arm_lengths"][:, :-1]).numpy()
        displacement = batch["delta_xy"].numpy()
        for b in range(z.shape[0]):
            for t in range(z.shape[1]):
                for slot in range(2):
                    features.append(z[b, t, slot])
                    arm_changes.append(arm_change[b, t, slot])
                    displacements.append(displacement[b, t, slot])
                    slots.append(slot)
    return np.asarray(features), np.asarray(arm_changes), np.asarray(displacements), np.asarray(slots)


def r2_per_dim(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    residual = ((target - prediction) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1.0 - residual / np.maximum(total, 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--movement_root", required=True)
    parser.add_argument("--punch_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    model = BoxingObjectLAM(config["state_dim"], config["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    train = ConcatDataset([
        BoxingObjectDataset(os.path.join(args.movement_root, "train"), target_frames=5),
        BoxingObjectDataset(os.path.join(args.punch_root, "train"), target_frames=5),
    ])
    val = ConcatDataset([
        BoxingObjectDataset(os.path.join(args.movement_root, "val"), target_frames=5),
        BoxingObjectDataset(os.path.join(args.punch_root, "val"), target_frames=5),
    ])
    train_x, train_arm, train_displacement, _ = collect(model, train, device)
    val_x, val_arm, val_displacement, val_slots = collect(model, val, device)
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    val_scaled = scaler.transform(val_x)
    train_punch = (np.abs(train_arm).max(axis=1) > 0).astype(np.int64)
    val_punch = (np.abs(val_arm).max(axis=1) > 0).astype(np.int64)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced").fit(train_scaled, train_punch)
    punch_prediction = classifier.predict(val_scaled)
    arm_probe = Ridge(alpha=1.0).fit(train_scaled, train_arm)
    arm_prediction = arm_probe.predict(val_scaled)
    motion_probe = Ridge(alpha=1.0).fit(train_scaled, train_displacement)
    motion_prediction = motion_probe.predict(val_scaled)
    report = {
        "checkpoint": args.checkpoint,
        "train_latents": int(len(train_x)),
        "val_latents": int(len(val_x)),
        "val_punch_fraction": float(val_punch.mean()),
        "punch_balanced_accuracy": float(balanced_accuracy_score(val_punch, punch_prediction)),
        "arm_delta_r2_left": float(r2_per_dim(val_arm, arm_prediction)[0]),
        "arm_delta_r2_right": float(r2_per_dim(val_arm, arm_prediction)[1]),
        "motion_r2_dx": float(r2_per_dim(val_displacement, motion_prediction)[0]),
        "motion_r2_dy": float(r2_per_dim(val_displacement, motion_prediction)[1]),
        "player_punch_fraction": float(val_punch[val_slots == 0].mean()),
        "enemy_punch_fraction": float(val_punch[val_slots == 1].mean()),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
