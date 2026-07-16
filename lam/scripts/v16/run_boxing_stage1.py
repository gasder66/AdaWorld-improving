"""Train and ablate the minimal V16 Boxing object-wise IDM/FDM."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_object_dataset import BoxingObjectDataset
from lam.modules.v16_boxing_model import BoxingObjectLAM


@torch.no_grad()
def evaluate(model: BoxingObjectLAM, loader: DataLoader, device: torch.device) -> Dict[str, Dict[str, float]]:
    model.eval()
    result: Dict[str, Dict[str, float]] = {}
    for ablation in ("normal", "zero", "shuffle"):
        sums: Dict[str, float] = {}
        count = 0
        for batch in loader:
            model_batch = {k: batch[k].to(device) for k in ("videos", "masks", "background_masks")}
            out = model(model_batch, ablation=ablation)
            for key in ("loss", "state_loss", "reconstruction_loss", "object_rgb_loss"):
                sums[key] = sums.get(key, 0.0) + float(out[key])
            count += 1
        result[ablation] = {key: value / max(1, count) for key, value in sums.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/v16_boxing/stage1_movement")
    parser.add_argument("--output", default="result/v16/boxing_stage1_smoke")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=32)
    parser.add_argument("--max_val_samples", type=int, default=16)
    parser.add_argument("--state_dim", type=int, default=96)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_ds = BoxingObjectDataset(os.path.join(args.data_root, "train"), args.max_train_samples)
    val_ds = BoxingObjectDataset(os.path.join(args.data_root, "val"), args.max_val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = BoxingObjectLAM(args.state_dim, args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    iterator = iter(train_loader)
    history = []
    model.train()
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        model_batch = {k: batch[k].to(device) for k in ("videos", "masks", "background_masks")}
        out = model(model_batch)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {key: float(out[key].detach()) for key in ("loss", "state_loss", "reconstruction_loss", "object_rgb_loss", "z_variance")}
        history.append(row)
        if step % 20 == 0 or step + 1 == args.steps:
            print(f"step={step:04d} " + " ".join(f"{k}={v:.5f}" for k, v in row.items()))
    metrics = evaluate(model, val_loader, device)
    os.makedirs(args.output, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.output, "model.pt"))
    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history, "ablation": metrics}, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
