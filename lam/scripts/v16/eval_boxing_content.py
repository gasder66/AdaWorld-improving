"""Evaluate and visualize content conditioning in the E06 Boxing model."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_transition_dataset import BoxingTransitionDataset
from lam.modules.v16_boxing_model import BoxingObjectLAM


MODES = ("normal", "zero", "shuffle", "swap_slots")
EVENT_NAMES = {
    0: "non_interaction", 1: "near", 2: "contact", 3: "punch_miss",
    4: "hit", 5: "received_hit", 6: "occlusion", 7: "recovery",
}


def _font(size: int = 13) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _tensor_image(tensor: Tensor, scale: int = 2) -> Image.Image:
    array = (tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(array)
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
    if model.object_input_mode != "mask_structure_content":
        raise ValueError(f"content evaluation requires mask_structure_content, got {model.object_input_mode}")
    return model


def _recolor(videos: Tensor, masks: Tensor, seed: int) -> tuple[Tensor, Tensor]:
    """Give each slot a stable random color while preserving its mask geometry."""
    generator = torch.Generator(device=videos.device).manual_seed(seed)
    colors = 0.15 + 0.8 * torch.rand(
        videos.shape[0], 1, masks.shape[2], 3, 1, 1,
        generator=generator, device=videos.device, dtype=videos.dtype,
    )
    result = videos.clone()
    for slot in range(masks.shape[2]):
        alpha = masks[:, :, slot].unsqueeze(2)
        result = result * (1.0 - alpha) + colors[:, :, slot] * alpha
    return result, colors[:, 0, :, :, 0, 0]


def _masked_mean_rgb(images: Tensor, masks: Tensor) -> Tensor:
    """images [B,K,3,H,W], masks [B,K,H,W] -> [B,K,3]."""
    weights = masks.unsqueeze(2)
    denominator = weights.sum(dim=(-2, -1)).clamp_min(1.0)
    return (images * weights).sum(dim=(-2, -1)) / denominator


def _batch_metrics(outputs: Dict[str, Dict[str, Tensor]], batch: Dict[str, Tensor]) -> Dict[str, Dict[str, float]]:
    target_video = batch["videos"][:, 1].unsqueeze(1).expand(-1, 2, -1, -1, -1)
    target_masks = batch["masks"][:, 1]
    target_colors = _masked_mean_rgb(target_video, target_masks)
    normal_structure = outputs["normal"]["predicted_object_states"]
    result: Dict[str, Dict[str, float]] = {}
    for mode, output in outputs.items():
        predicted_colors = _masked_mean_rgb(output["object_rgb"][:, 0], target_masks)
        donor_colors = target_colors.flip(dims=(1,))
        structure_delta = (output["predicted_object_states"] - normal_structure).abs()
        result[mode] = {
            "samples": float(batch["videos"].shape[0]),
            "reconstruction_l1_sum": float(
                (output["reconstruction"][:, 0] - batch["videos"][:, 1]).abs().mean(dim=(-3, -2, -1)).sum()
            ),
            "object_rgb_l1_sum": float(
                (
                    (output["object_rgb"][:, 0] - target_video).abs()
                    * target_masks.unsqueeze(2)
                ).sum(dim=(-3, -2, -1)).div(3.0 * target_masks.sum(dim=(-2, -1)).clamp_min(1.0)).sum()
            ),
            "mask_iou_sum": float(output["mask_iou"]) * batch["videos"].shape[0] * 2,
            "mask_iou_count": float(batch["videos"].shape[0] * 2),
            "structure_mean_abs_delta_sum": float(structure_delta.mean(dim=(-4, -3, -2, -1)).sum()),
            "own_color_l1_sum": float((predicted_colors - target_colors).abs().mean(dim=-1).sum()),
            "donor_color_l1_sum": float((predicted_colors - donor_colors).abs().mean(dim=-1).sum()),
            "color_count": float(batch["videos"].shape[0] * 2),
        }
    return result


def _accumulate(total: Dict[str, Dict[str, float]], update: Dict[str, Dict[str, float]]) -> None:
    for mode, values in update.items():
        for key, value in values.items():
            total[mode][key] += value


def _finalize(total: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    result = {}
    for mode, values in total.items():
        samples = max(1.0, values["samples"])
        colors = max(1.0, values["color_count"])
        result[mode] = {
            "samples": int(values["samples"]),
            "reconstruction_l1": values["reconstruction_l1_sum"] / samples,
            "object_rgb_l1": values["object_rgb_l1_sum"] / colors,
            "mask_iou": values["mask_iou_sum"] / max(1.0, values["mask_iou_count"]),
            "structure_mean_abs_delta_vs_normal": values["structure_mean_abs_delta_sum"] / samples,
            "own_color_l1": values["own_color_l1_sum"] / colors,
            "donor_color_l1": values["donor_color_l1_sum"] / colors,
        }
    swap = result["swap_slots"]
    result["swap_slots"]["donor_preference_margin"] = swap["own_color_l1"] - swap["donor_color_l1"]
    return result


@torch.no_grad()
def evaluate(
    model: BoxingObjectLAM,
    dataset: BoxingTransitionDataset,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    recolor: bool,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False, num_workers=0)
    total: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for batch_index, raw_batch in enumerate(loader):
        batch = {key: raw_batch[key].to(device) for key in ("videos", "masks", "background_masks")}
        if recolor:
            batch["videos"], _ = _recolor(batch["videos"], batch["masks"], seed + batch_index)
        outputs = {mode: model(batch, content_ablation=mode) for mode in MODES}
        _accumulate(total, _batch_metrics(outputs, batch))
    return _finalize(total)


def _balanced_indices(dataset: BoxingTransitionDataset, per_event: int, seed: int) -> list[int]:
    grouped: Dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        grouped[entry.get("interaction_primary", "non_interaction")].append(index)
    rng = random.Random(seed)
    selected = []
    for values in grouped.values():
        rng.shuffle(values)
        selected.extend(values[:per_event])
    rng.shuffle(selected)
    return selected


def _choose_examples(dataset: BoxingTransitionDataset, count: int, seed: int) -> list[int]:
    grouped: Dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        grouped[entry.get("interaction_primary", "non_interaction")].append(index)
    rng = random.Random(seed)
    selected = []
    preferred = ("non_interaction", "punch_miss", "hit", "received_hit", "occlusion", "contact")
    for event in preferred:
        rng.shuffle(grouped[event])
        if grouped[event]:
            selected.append(grouped[event][0])
        if len(selected) == count:
            break
    return selected


def _save_rows(rows: Iterable[np.ndarray], path: str) -> None:
    rows = list(rows)
    width = max(row.shape[1] for row in rows)
    padded = [np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0)), constant_values=28) for row in rows]
    Image.fromarray(np.concatenate(padded, axis=0)).save(path)


@torch.no_grad()
def visualize(
    model: BoxingObjectLAM,
    dataset: BoxingTransitionDataset,
    indices: Sequence[int],
    device: torch.device,
    output_dir: str,
    recolor: bool,
    seed: int,
) -> None:
    samples = [dataset[index] for index in indices]
    batch = {
        key: torch.stack([sample[key] for sample in samples]).to(device)
        for key in ("videos", "masks", "background_masks")
    }
    colors = None
    if recolor:
        batch["videos"], colors = _recolor(batch["videos"], batch["masks"], seed)
    outputs = {mode: model(batch, content_ablation=mode) for mode in ("normal", "zero", "swap_slots")}
    rows = []
    for row, sample in enumerate(samples):
        event = EVENT_NAMES[int(sample["interaction_id"])]
        target = batch["videos"][row, 1]
        images = (
            ("current", batch["videos"][row, 0]),
            ("target", target),
            ("normal content", outputs["normal"]["reconstruction"][row, 0]),
            ("zero content", outputs["zero"]["reconstruction"][row, 0]),
            ("swapped content", outputs["swap_slots"]["reconstruction"][row, 0]),
        )
        panels = []
        for title, image in images:
            subtitle = ""
            if title not in {"current", "target"}:
                subtitle = f"full L1={float((image - target).abs().mean()):.4f}"
            panels.append(_label(_tensor_image(image, 2), title, subtitle))
        height = max(panel.height for panel in panels)
        header = Image.new("RGB", (145, height), (28, 28, 28))
        draw = ImageDraw.Draw(header)
        draw.text((6, 8), event, fill=(248, 248, 248), font=_font(13))
        if colors is not None:
            c0 = ",".join(str(int(value * 255)) for value in colors[row, 0])
            c1 = ",".join(str(int(value * 255)) for value in colors[row, 1])
            draw.text((6, 30), f"P {c0}", fill=(190, 190, 190), font=_font(10))
            draw.text((6, 47), f"E {c1}", fill=(190, 190, 190), font=_font(10))
        padded = []
        for panel in panels:
            canvas = Image.new("RGB", (panel.width, height), (28, 28, 28))
            canvas.paste(panel, (0, 0))
            padded.append(np.asarray(canvas))
        rows.append(np.concatenate([np.asarray(header), *padded], axis=1))
    suffix = "recolored" if recolor else "original"
    _save_rows(rows, os.path.join(output_dir, f"content_rebinding_{suffix}.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_event", type=int, default=500)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = BoxingTransitionDataset(args.index)
    model = _load_model(args.checkpoint, device)
    indices = _balanced_indices(dataset, args.per_event, args.seed)
    examples = _choose_examples(dataset, args.examples, args.seed)
    report = {
        "checkpoint": args.checkpoint,
        "index": args.index,
        "examples": examples,
        "original": evaluate(model, dataset, indices, device, args.batch_size, False, args.seed),
        "recolored": evaluate(model, dataset, indices, device, args.batch_size, True, args.seed),
    }
    visualize(model, dataset, examples, device, args.output_dir, False, args.seed)
    visualize(model, dataset, examples, device, args.output_dir, True, args.seed)
    with open(os.path.join(args.output_dir, "content_evaluation.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
