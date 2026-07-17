"""Phase-aware contour diagnostics for V16 Boxing checkpoints."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import random
import sys
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_transition_dataset import BoxingTransitionDataset
from lam.modules.v16_boxing_model import BoxingObjectLAM


PHASES = (
    "movement_only", "punch_onset", "punch_extend",
    "punch_hold", "punch_retract", "punch_switch",
)


def _load_model(path: str, device: torch.device) -> BoxingObjectLAM:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    model = BoxingObjectLAM(
        config["state_dim"], config["latent_dim"], config.get("fdm_type", "independent"),
        config.get("object_input_mode", "masked_rgb_mask"),
        structure_scale=config.get("structure_scale", 4),
        idm_grid_size=config.get("idm_grid_size", 1),
        idm_type=config.get("idm_type", "conv"),
        idm_token_grid=config.get("idm_token_grid", 8),
        idm_layers=config.get("idm_layers", 2),
        idm_heads=config.get("idm_heads", 4),
        learned_upsampling=config.get("learned_upsampling", False),
        dynamic_mask_weight=config.get("dynamic_mask_weight", 0.0),
        edge_weight=config.get("edge_weight", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _condition_states(model: BoxingObjectLAM, states: Tensor, content: Tensor | None) -> Tensor:
    if content is None:
        return states
    flat_states = states.reshape(-1, model.state_dim, *states.shape[-2:])
    flat_content = content.reshape(-1, model.state_dim)
    gamma, beta = model.content_conditioner(flat_content).chunk(2, dim=-1)
    gamma = 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
    beta = beta.unsqueeze(-1).unsqueeze(-1)
    return (
        model.content_state_norm(flat_states) * (1.0 + gamma) + beta
    ).reshape_as(states)


def _decode_masks(
    model: BoxingObjectLAM,
    states: Tensor,
    content: Tensor | None,
    output_size: tuple[int, int],
) -> Tensor:
    conditioned = _condition_states(model, states, content)
    logits = model.object_decoder(
        conditioned.reshape(-1, model.state_dim, *conditioned.shape[-2:]), output_size
    )[:, 3]
    return torch.sigmoid(logits).reshape(*states.shape[:3], *output_size)


def _edges(mask: Tensor) -> Tensor:
    mask = mask.float()
    dilated = F.max_pool2d(mask.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    eroded = -F.max_pool2d(-mask.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    return (dilated - eroded) > 0


def _per_sample_metrics(probability: Tensor, target: Tensor, current: Tensor) -> Dict[str, Tensor]:
    prediction = probability >= 0.5
    target_binary = target >= 0.5
    current_binary = current >= 0.5
    intersection = (prediction & target_binary).sum(dim=(-2, -1)).float()
    union = (prediction | target_binary).sum(dim=(-2, -1)).float().clamp_min(1.0)
    iou = intersection / union

    pred_edge = _edges(prediction)
    target_edge = _edges(target_binary)
    pred_tolerance = F.max_pool2d(pred_edge.float().unsqueeze(1), 5, stride=1, padding=2).squeeze(1) > 0
    target_tolerance = F.max_pool2d(target_edge.float().unsqueeze(1), 5, stride=1, padding=2).squeeze(1) > 0
    precision = (pred_edge & target_tolerance).sum(dim=(-2, -1)).float() / pred_edge.sum(
        dim=(-2, -1)
    ).float().clamp_min(1.0)
    recall = (target_edge & pred_tolerance).sum(dim=(-2, -1)).float() / target_edge.sum(
        dim=(-2, -1)
    ).float().clamp_min(1.0)
    boundary_f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)

    changed = current_binary ^ target_binary
    dynamic = F.max_pool2d(changed.float().unsqueeze(1), 7, stride=1, padding=3).squeeze(1) > 0
    dynamic_intersection = (prediction & target_binary & dynamic).sum(dim=(-2, -1)).float()
    dynamic_union = ((prediction | target_binary) & dynamic).sum(dim=(-2, -1)).float().clamp_min(1.0)
    dynamic_iou = dynamic_intersection / dynamic_union
    dynamic_l1 = ((probability - target).abs() * dynamic).sum(dim=(-2, -1)) / dynamic.sum(
        dim=(-2, -1)
    ).float().clamp_min(1.0)
    uncertain = ((probability > 0.1) & (probability < 0.9) & target_tolerance).sum(
        dim=(-2, -1)
    ).float() / target_tolerance.sum(dim=(-2, -1)).float().clamp_min(1.0)
    return {
        "mask_iou": iou,
        "boundary_f1": boundary_f1,
        "dynamic_iou": dynamic_iou,
        "dynamic_l1": dynamic_l1,
        "boundary_uncertain_fraction": uncertain,
    }


def _select_slot(value: Tensor, slots: Tensor) -> Tensor:
    return value[torch.arange(value.shape[0], device=value.device), slots]


@torch.no_grad()
def evaluate(
    model: BoxingObjectLAM,
    dataset: BoxingTransitionDataset,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Dict:
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False, num_workers=0)
    sums: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    counts: Dict[str, int] = defaultdict(int)
    for raw in loader:
        batch = {key: raw[key].to(device) for key in ("videos", "masks", "background_masks")}
        output = model(batch)
        height, width = batch["videos"].shape[-2:]
        content = output["content_states"]
        content_t = None if content is None else content[:, :-1]
        oracle_probability = _decode_masks(
            model, output["object_states"][:, 1:], content_t, (height, width)
        )[:, 0]
        predicted_probability = torch.sigmoid(output["object_mask_logits"][:, 0])
        slots = raw["target_slot"].to(device)
        current = _select_slot(batch["masks"][:, 0], slots)
        target = _select_slot(batch["masks"][:, 1], slots)
        probabilities = {
            "oracle_state": _select_slot(oracle_probability, slots),
            "predicted_state": _select_slot(predicted_probability, slots),
        }
        phase_ids = raw["event_id"].tolist()
        for name, probability in probabilities.items():
            values = _per_sample_metrics(probability, target, current)
            for row, phase_id in enumerate(phase_ids):
                phase = PHASES[int(phase_id)]
                for metric, tensor in values.items():
                    sums[phase][name][metric] += float(tensor[row])
        for phase_id in phase_ids:
            counts[PHASES[int(phase_id)]] += 1
    report = {}
    for phase in PHASES:
        count = counts[phase]
        report[phase] = {"count": count}
        for name in ("oracle_state", "predicted_state"):
            report[phase][name] = {
                metric: value / max(1, count)
                for metric, value in sums[phase][name].items()
            }
    aggregate_count = sum(counts.values())
    report["overall"] = {"count": aggregate_count}
    for name in ("oracle_state", "predicted_state"):
        metric_names = next(iter(sums.values()))[name].keys()
        report["overall"][name] = {
            metric: sum(sums[phase][name][metric] for phase in PHASES) / max(1, aggregate_count)
            for metric in metric_names
        }
    return report


def _font(size: int = 13) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _mask_image(mask: Tensor, scale: int = 4) -> Image.Image:
    array = (mask.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L").convert("RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _label(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    header = 43 if subtitle else 25
    canvas = Image.new("RGB", (image.width, image.height + header), (28, 28, 28))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 3), title, fill=(248, 248, 248), font=_font(13))
    if subtitle:
        draw.text((5, 23), subtitle, fill=(190, 190, 190), font=_font(11))
    return canvas


def _bounds(masks: Tensor, pad: int = 10) -> tuple[int, int, int, int]:
    ys, xs = torch.where(masks.any(dim=0))
    if len(xs) == 0:
        return 0, 0, masks.shape[-1], masks.shape[-2]
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(masks.shape[-1], int(xs.max()) + pad + 1),
        min(masks.shape[-2], int(ys.max()) + pad + 1),
    )


@torch.no_grad()
def visualize(
    model: BoxingObjectLAM,
    dataset: BoxingTransitionDataset,
    phase_indices: Dict[str, int],
    device: torch.device,
    output_path: str,
) -> None:
    rows = []
    for phase, index in phase_indices.items():
        sample = dataset[index]
        batch = {
            key: sample[key].unsqueeze(0).to(device)
            for key in ("videos", "masks", "background_masks")
        }
        output = model(batch)
        height, width = batch["videos"].shape[-2:]
        content = output["content_states"]
        content_t = None if content is None else content[:, :-1]
        oracle = _decode_masks(model, output["object_states"][:, 1:], content_t, (height, width))[0, 0]
        predicted = torch.sigmoid(output["object_mask_logits"][0, 0])
        slot = int(sample["target_slot"])
        current = sample["masks"][0, slot]
        target = sample["masks"][1, slot]
        oracle = oracle[slot].cpu()
        predicted = predicted[slot].cpu()
        x0, y0, x1, y1 = _bounds(torch.stack([current, target]), pad=12)
        panels = []
        for title, mask in (
            ("current", current),
            ("target", target),
            ("oracle probability", oracle),
            ("oracle binary", oracle >= 0.5),
            ("predicted probability", predicted),
            ("predicted binary", predicted >= 0.5),
        ):
            crop = mask[y0:y1, x0:x1]
            metrics = ""
            if title.startswith("oracle"):
                values = _per_sample_metrics(oracle.unsqueeze(0), target.unsqueeze(0), current.unsqueeze(0))
                metrics = f"BF1={float(values['boundary_f1'][0]):.3f}"
            elif title.startswith("predicted"):
                values = _per_sample_metrics(predicted.unsqueeze(0), target.unsqueeze(0), current.unsqueeze(0))
                metrics = f"BF1={float(values['boundary_f1'][0]):.3f}"
            panels.append(_label(_mask_image(crop, 5), title, metrics))
        height_px = max(panel.height for panel in panels)
        header = Image.new("RGB", (145, height_px), (28, 28, 28))
        draw = ImageDraw.Draw(header)
        draw.text((6, 7), phase, fill=(248, 248, 248), font=_font(13))
        arms = sample["arm_lengths"].int().tolist()
        draw.text((6, 29), f"arm {arms[0][slot]}", fill=(190, 190, 190), font=_font(10))
        draw.text((6, 46), f"to {arms[1][slot]}", fill=(190, 190, 190), font=_font(10))
        padded = []
        for panel in panels:
            canvas = Image.new("RGB", (panel.width, height_px), (28, 28, 28))
            canvas.paste(panel, (0, 0))
            padded.append(np.asarray(canvas))
        rows.append(np.concatenate([np.asarray(header), *padded], axis=1))
    width = max(row.shape[1] for row in rows)
    rows = [
        np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0)), constant_values=28)
        for row in rows
    ]
    Image.fromarray(np.concatenate(rows, axis=0)).save(output_path)


def _phase_indices(dataset: BoxingTransitionDataset, per_phase: int, seed: int) -> list[int]:
    grouped: Dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        grouped[entry["event"]].append(index)
    rng = random.Random(seed)
    selected = []
    for phase in PHASES:
        rng.shuffle(grouped[phase])
        selected.extend(grouped[phase][:per_phase])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_phase", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = BoxingTransitionDataset(args.index)
    model = _load_model(args.checkpoint, device)
    indices = _phase_indices(dataset, args.per_phase, args.seed)
    report = {
        "checkpoint": args.checkpoint,
        "index": args.index,
        "structure_shape": list(model.object_encoder(torch.zeros(
            1, model.OBJECT_INPUT_CHANNELS[model.object_input_mode], 210, 160, device=device
        )).shape[-2:]),
        "phases": evaluate(model, dataset, indices, device, args.batch_size),
    }
    examples = {}
    for phase in PHASES:
        candidates = [index for index in indices if dataset.entries[index]["event"] == phase]
        if candidates:
            examples[phase] = candidates[0]
    visualize(model, dataset, examples, device, os.path.join(args.output_dir, "phase_contours.png"))
    with open(os.path.join(args.output_dir, "contour_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
