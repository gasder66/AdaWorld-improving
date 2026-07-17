"""Measure whether object latents distinguish the six Boxing transition phases."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_transition_dataset import BoxingTransitionDataset, EVENT_TO_ID
from lam.modules.v16_boxing_model import BoxingObjectLAM


PHASES = (
    "movement", "onset", "extend", "hold", "retract", "switch",
)


def load_model(path: str, device: torch.device) -> BoxingObjectLAM:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    model = BoxingObjectLAM(
        config["state_dim"],
        config["latent_dim"],
        config.get("fdm_type", "independent"),
        config.get("object_input_mode", "masked_rgb_mask"),
        structure_scale=config.get("structure_scale", 4),
        idm_grid_size=config.get("idm_grid_size", 1),
        idm_type=config.get("idm_type", "conv"),
        idm_token_grid=config.get("idm_token_grid", 8),
        idm_layers=config.get("idm_layers", 2),
        idm_heads=config.get("idm_heads", 4),
        temporal_context=config.get("temporal_context", 1),
        temporal_token_grid=config.get("temporal_token_grid", 8),
        temporal_layers=config.get("temporal_layers", 2),
        temporal_heads=config.get("temporal_heads", 4),
        learned_upsampling=config.get("learned_upsampling", False),
        dynamic_mask_weight=config.get("dynamic_mask_weight", 0.0),
        edge_weight=config.get("edge_weight", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def balanced_indices(
    dataset: BoxingTransitionDataset, per_phase: int, seed: int,
) -> list[int]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        groups[EVENT_TO_ID[entry["event"]]].append(index)
    rng = np.random.RandomState(seed)
    selected = []
    for phase_id in range(len(PHASES)):
        candidates = np.asarray(groups[phase_id])
        count = min(per_phase, len(candidates))
        selected.extend(rng.choice(candidates, count, replace=False).tolist())
    rng.shuffle(selected)
    return selected


@torch.no_grad()
def collect(
    model: BoxingObjectLAM,
    dataset: BoxingTransitionDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0,
    )
    features, labels = [], []
    for raw in loader:
        batch = {
            key: raw[key].to(device)
            for key in ("videos", "masks", "background_masks")
        }
        output = model(batch)
        z = output["z_mu"][:, 0]
        slots = raw["target_slot"].to(device)
        selected = z[torch.arange(z.shape[0], device=device), slots]
        features.append(selected.cpu().numpy())
        labels.append(raw["event_id"].numpy())
    return np.concatenate(features), np.concatenate(labels)


def save_confusions(
    matrices: dict[str, np.ndarray], output: str, title: str,
) -> None:
    figure, axes = plt.subplots(1, len(matrices), figsize=(11, 4.6), constrained_layout=True)
    if len(matrices) == 1:
        axes = [axes]
    for axis, (name, matrix) in zip(axes, matrices.items()):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column, row, f"{matrix[row, column]:.2f}",
                    ha="center", va="center",
                    color="white" if matrix[row, column] > 0.55 else "#162033",
                    fontsize=8,
                )
        axis.set_title(name)
        axis.set_xticks(range(len(PHASES)), PHASES, rotation=45, ha="right")
        axis.set_yticks(range(len(PHASES)), PHASES)
        axis.set_xlabel("predicted phase")
        axis.set_ylabel("true phase")
    figure.suptitle(title)
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_index", required=True)
    parser.add_argument("--val_index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train_per_phase", type=int, default=4000)
    parser.add_argument("--val_per_phase", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    train_dataset = BoxingTransitionDataset(
        args.train_index, temporal_context=model.temporal_context
    )
    val_dataset = BoxingTransitionDataset(
        args.val_index, temporal_context=model.temporal_context
    )
    train_indices = balanced_indices(train_dataset, args.train_per_phase, args.seed)
    val_indices = balanced_indices(val_dataset, args.val_per_phase, args.seed + 1)
    train_x, train_y = collect(
        model, train_dataset, train_indices, device, args.batch_size,
    )
    val_x, val_y = collect(
        model, val_dataset, val_indices, device, args.batch_size,
    )

    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    val_scaled = scaler.transform(val_x)
    classifiers = {
        "linear": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=args.seed,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu", max_iter=600,
            early_stopping=True, random_state=args.seed,
        ),
    }
    scores = {}
    confusions = {}
    for name, classifier in classifiers.items():
        classifier.fit(train_scaled, train_y)
        prediction = classifier.predict(val_scaled)
        scores[name.lower()] = float(balanced_accuracy_score(val_y, prediction))
        confusions[name] = confusion_matrix(
            val_y, prediction, labels=np.arange(len(PHASES)), normalize="true",
        )

    os.makedirs(args.output, exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "train_examples": int(len(train_y)),
        "val_examples": int(len(val_y)),
        "chance_balanced_accuracy": 1.0 / len(PHASES),
        "phase_balanced_accuracy": scores,
        "val_counts": {
            phase: int((val_y == phase_id).sum())
            for phase_id, phase in enumerate(PHASES)
        },
        "confusion_matrices": {
            name: matrix.tolist() for name, matrix in confusions.items()
        },
    }
    with open(os.path.join(args.output, "phase_probe.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    save_confusions(
        confusions,
        os.path.join(args.output, "phase_confusion.png"),
        f"Boxing phase decoding from z ({os.path.basename(os.path.dirname(args.checkpoint))})",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
