"""Probe frozen V16 latents for continuous fighter displacement."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_object_dataset import BoxingObjectDataset
from lam.modules.v16_boxing_model import BoxingObjectLAM


@torch.no_grad()
def collect(model: BoxingObjectLAM, dataset: BoxingObjectDataset, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    features = []
    targets = []
    model.eval()
    for batch in loader:
        indices = torch.tensor([0, batch["videos"].shape[1] - 1])
        model_batch = {
            key: batch[key].index_select(1, indices).to(device)
            for key in ("videos", "masks", "background_masks")
        }
        output = model(model_batch)
        features.append(output["z_mu"][:, 0].cpu().reshape(-1, model.latent_dim))
        displacement = batch["delta_xy"].sum(dim=1)
        targets.append(displacement.reshape(-1, 2))
    return torch.cat(features), torch.cat(targets)


def fit_ridge(train_x: torch.Tensor, train_y: torch.Tensor, ridge: float) -> torch.Tensor:
    ones = torch.ones((train_x.shape[0], 1), dtype=train_x.dtype)
    design = torch.cat([train_x, ones], dim=1)
    eye = torch.eye(design.shape[1], dtype=design.dtype)
    eye[-1, -1] = 0.0
    return torch.linalg.solve(design.T @ design + ridge * eye, design.T @ train_y)


def predict(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype)], dim=1) @ weights


def r2(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    residual = (target - prediction).square().sum(dim=0)
    total = (target - target.mean(dim=0, keepdim=True)).square().sum(dim=0).clamp_min(1e-8)
    return 1.0 - residual / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data/v16_boxing/stage1_movement")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    model = BoxingObjectLAM(
        config["state_dim"], config["latent_dim"], config.get("fdm_type", "independent")
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    train_x, train_y = collect(model, BoxingObjectDataset(os.path.join(args.data_root, "train")), device)
    val_x, val_y = collect(model, BoxingObjectDataset(os.path.join(args.data_root, "val")), device)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    weights = fit_ridge((train_x - mean) / std, train_y, args.ridge)
    train_prediction = predict((train_x - mean) / std, weights)
    val_prediction = predict((val_x - mean) / std, weights)
    train_r2 = r2(train_y, train_prediction)
    val_r2 = r2(val_y, val_prediction)
    report = {
        "checkpoint": args.checkpoint,
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "ridge": args.ridge,
        "train_r2_dx": float(train_r2[0]),
        "train_r2_dy": float(train_r2[1]),
        "val_r2_dx": float(val_r2[0]),
        "val_r2_dy": float(val_r2[1]),
        "val_r2_mean": float(val_r2.mean()),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
