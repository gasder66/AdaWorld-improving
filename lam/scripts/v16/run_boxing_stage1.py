"""Train and ablate the minimal V16 Boxing object-wise IDM/FDM."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_object_dataset import BoxingObjectDataset
from lam.datasets.boxing_transition_dataset import BoxingTransitionDataset
from lam.modules.v16_boxing_model import BoxingObjectLAM


@torch.no_grad()
def _model_batch(batch: Dict[str, torch.Tensor], device: torch.device, transition_gap: int) -> Dict[str, torch.Tensor]:
    result = {k: batch[k].to(device) for k in ("videos", "masks", "background_masks")}
    if transition_gap > 1:
        indices = torch.tensor([0, result["videos"].shape[1] - 1], device=device)
        result = {key: value.index_select(1, indices) for key, value in result.items()}
    return result


def _identity_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    videos = batch["videos"].to(device)
    masks = batch["masks"].to(device)
    background = batch["background_masks"].to(device)
    batch_size, time = videos.shape[:2]
    videos = videos.reshape(batch_size * time, *videos.shape[2:])
    masks = masks.reshape(batch_size * time, *masks.shape[2:])
    background = background.reshape(batch_size * time, *background.shape[2:])
    return {
        "videos": videos.unsqueeze(1).expand(-1, 2, -1, -1, -1),
        "masks": masks.unsqueeze(1).expand(-1, 2, -1, -1, -1),
        "background_masks": background.unsqueeze(1).expand(-1, 2, -1, -1),
    }


def _set_phase(model: BoxingObjectLAM, phase: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if phase == "visual":
        modules = (model.object_encoder, model.background_encoder, model.object_decoder, model.background_decoder)
    elif phase == "dynamics":
        modules = (model.idm, model.fdm)
    else:
        raise ValueError(phase)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def evaluate(
    model: BoxingObjectLAM, loader: DataLoader, device: torch.device, transition_gap: int,
    max_batches: int = 0,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    result: Dict[str, Dict[str, float]] = {}
    for ablation in ("normal", "zero", "shuffle"):
        sums: Dict[str, float] = {}
        count = 0
        for batch in loader:
            model_batch = _model_batch(batch, device, transition_gap)
            out = model(model_batch, ablation=ablation)
            for key in ("loss", "state_loss", "identity_state_loss", "reconstruction_loss", "object_rgb_loss", "z_variance"):
                sums[key] = sums.get(key, 0.0) + float(out[key])
            count += 1
            if max_batches and count >= max_batches:
                break
        result[ablation] = {key: value / max(1, count) for key, value in sums.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/v16_boxing/stage1_movement")
    parser.add_argument("--extra_data_root", default=None)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--transition_index_dir", default=None)
    parser.add_argument("--balanced_samples", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--output", default="result/v16/boxing_stage1_smoke")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--pretrain_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=32)
    parser.add_argument("--max_val_samples", type=int, default=16)
    parser.add_argument("--state_dim", type=int, default=96)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transition_gap", type=int, choices=[1, 4], default=4)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_sampler = None
    if args.transition_index_dir:
        train_ds = BoxingTransitionDataset(os.path.join(args.transition_index_dir, "train.pt"))
        val_ds = BoxingTransitionDataset(os.path.join(args.transition_index_dir, "val.pt"))
        train_sampler = train_ds.balanced_sampler(args.balanced_samples or None, args.seed)
    else:
        train_parts = [BoxingObjectDataset(os.path.join(args.data_root, "train"), args.max_train_samples, target_frames=5)]
        val_parts = [BoxingObjectDataset(os.path.join(args.data_root, "val"), args.max_val_samples, target_frames=5)]
        if args.extra_data_root:
            train_parts.append(BoxingObjectDataset(os.path.join(args.extra_data_root, "train"), args.max_train_samples, target_frames=5))
            val_parts.append(BoxingObjectDataset(os.path.join(args.extra_data_root, "val"), args.max_val_samples, target_frames=5))
        train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
        val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = BoxingObjectLAM(args.state_dim, args.latent_dim).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"])
        print(f"initialized from {args.init_checkpoint}")
    history = {"visual": [], "dynamics": []}

    _set_phase(model, "visual")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    iterator = iter(train_loader)
    model.train()
    for step in range(args.pretrain_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        out = model(_identity_batch(batch, device), ablation="zero")
        visual_loss = (
            out["object_rgb_loss"] + 0.5 * out["mask_bce"] + 0.5 * out["mask_dice_loss"]
            + 0.25 * out["background_loss"] + 0.25 * out["reconstruction_loss"]
        )
        optimizer.zero_grad(set_to_none=True)
        visual_loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        row = {"loss": float(visual_loss.detach()), "reconstruction_loss": float(out["reconstruction_loss"])}
        history["visual"].append(row)
        if step % 20 == 0 or step + 1 == args.pretrain_steps:
            print(f"visual step={step:04d} " + " ".join(f"{k}={v:.5f}" for k, v in row.items()))

    _set_phase(model, "dynamics")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    iterator = iter(train_loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        model_batch = _model_batch(batch, device, args.transition_gap)
        out = model(model_batch)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {
            key: float(out[key].detach())
            for key in (
                "loss", "state_loss", "reconstruction_loss", "object_rgb_loss",
                "z_variance", "variance_floor_loss", "z_norm_loss", "action_contrast_loss",
            )
        }
        history["dynamics"].append(row)
        if step % 20 == 0 or step + 1 == args.steps:
            print(f"step={step:04d} " + " ".join(f"{k}={v:.5f}" for k, v in row.items()))
    metrics = evaluate(model, val_loader, device, args.transition_gap, args.max_eval_batches)
    os.makedirs(args.output, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.output, "model.pt"))
    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history, "ablation": metrics}, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
