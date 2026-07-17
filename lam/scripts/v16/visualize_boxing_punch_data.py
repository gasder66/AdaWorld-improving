"""Render isolated-punch clips with synchronized OCAtari RAM arm labels."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_object_dataset import BoxingObjectDataset
from lam.modules.v16_boxing_model import BoxingObjectLAM
from scripts.v16.visualize_boxing_stage1 import _label_frame, _write_video


def _signature(sample: Dict) -> tuple:
    active = sample["punch_labels"].any(dim=0).tolist()
    sides = []
    for slot in range(2):
        values = sorted(set(int(v) for v in sample["punch_side_labels"][:, slot].tolist()) - {0})
        sides.append(tuple(values))
    return tuple(active), tuple(sides)


def choose_samples(dataset: BoxingObjectDataset, count: int) -> List[int]:
    ranked = []
    for index in range(len(dataset)):
        sample = dataset[index]
        arm_range = sample["arm_lengths"].amax(dim=0) - sample["arm_lengths"].amin(dim=0)
        ranked.append((index, float(arm_range.max()), _signature(sample)))
    selected, signatures = [], set()
    for index, _score, signature in sorted(ranked, key=lambda row: row[1], reverse=True):
        if signature not in signatures:
            selected.append(index)
            signatures.add(signature)
        if len(selected) >= count:
            break
    for index, _score, _signature_value in sorted(ranked, key=lambda row: row[1], reverse=True):
        if len(selected) >= count:
            break
        if index not in selected:
            selected.append(index)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = BoxingObjectDataset(os.path.join(args.data_root, "train"))
    indices = choose_samples(dataset, args.count)
    manifest = []
    for output_index, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]
        video = (sample["videos"].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        frames = []
        for t, frame in enumerate(video):
            arms = sample["arm_lengths"][t].int().tolist()
            sides = sample["punch_side_labels"][t].int().tolist()
            frames.append(
                _label_frame(
                    frame,
                    [
                        f"sample={dataset_index} frame={t}/{len(video)-1}",
                        f"Player arms L/R={arms[0]} side={sides[0]}",
                        f"Enemy  arms L/R={arms[1]} side={sides[1]}",
                    ],
                    scale=3,
                )
            )
        stem = os.path.join(args.output_dir, f"isolated_punch_{output_index:02d}_sample_{dataset_index:06d}")
        _write_video(frames, stem, args.fps)
        manifest.append(
            {
                "dataset_index": dataset_index,
                "signature": _signature(sample),
                "arm_min": sample["arm_lengths"].amin(dim=0).int().tolist(),
                "arm_max": sample["arm_lengths"].amax(dim=0).int().tolist(),
                "mask_area_min": int(sample["masks"].sum(dim=(-1, -2)).min()),
                "mask_area_max": int(sample["masks"].sum(dim=(-1, -2)).max()),
            }
        )
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"indices": indices, "samples": manifest}, f, indent=2)
    if args.checkpoint:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["args"]
        model = BoxingObjectLAM(
            config["state_dim"], config["latent_dim"], config.get("fdm_type", "independent"),
            config.get("object_input_mode", "masked_rgb_mask"),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model_dataset = BoxingObjectDataset(os.path.join(args.data_root, "train"), target_frames=5)
        reconstruction_indices = indices[:4]
        model_samples = [model_dataset[index] for index in reconstruction_indices]
        batch = {
            key: torch.stack([sample[key] for sample in model_samples]).to(device)
            for key in ("videos", "masks", "background_masks")
        }
        with torch.no_grad():
            outputs = {
                name: model(batch, ablation=name)["reconstruction"].cpu()
                for name in ("normal", "zero", "shuffle")
            }
        reconstruction_frames = []
        for batch_index, dataset_index in enumerate(reconstruction_indices):
            sample = model_samples[batch_index]
            for t in range(sample["videos"].shape[0] - 1):
                tensors = [
                    ("current", sample["videos"][t]),
                    ("target", sample["videos"][t + 1]),
                    ("normal", outputs["normal"][batch_index, t]),
                    ("zero", outputs["zero"][batch_index, t]),
                    ("shuffle", outputs["shuffle"][batch_index, t]),
                ]
                panels = []
                target = sample["videos"][t + 1]
                losses = {}
                for label, tensor in tensors:
                    array = (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
                    if label in {"normal", "zero", "shuffle"}:
                        losses[label] = float((tensor - target).abs().mean())
                    panels.append(_label_frame(array, [label], scale=2))
                combined = np.concatenate(panels, axis=1)
                reconstruction_frames.append(
                    _label_frame(
                        combined,
                        [f"sample={dataset_index} transition={t}->{t+1}", json.dumps(losses)],
                        scale=1,
                    )
                )
        _write_video(
            reconstruction_frames,
            os.path.join(args.output_dir, "punch_reconstruction_comparison"),
            args.fps,
        )
    print(json.dumps({"indices": indices, "samples": manifest}, indent=2))


if __name__ == "__main__":
    main()
