"""Complete reconstruction and latent-space diagnostics for V16 E04."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from umap import UMAP

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_transition_dataset import BoxingTransitionDataset
from lam.modules.v16_boxing_model import BoxingObjectLAM


EVENTS = (
    "non_interaction", "near", "contact", "punch_miss",
    "hit", "received_hit", "occlusion", "recovery",
)
EVENT_NAMES = {
    0: "non_interaction", 1: "near", 2: "contact", 3: "punch_miss",
    4: "hit", 5: "received_hit", 6: "occlusion", 7: "recovery",
}


def _font(size: int = 13) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _label(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    header = 42 if subtitle else 25
    canvas = Image.new("RGB", (image.width, image.height + header), (28, 28, 28))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 3), title, fill=(248, 248, 248), font=_font(13))
    if subtitle:
        draw.text((5, 22), subtitle, fill=(190, 190, 190), font=_font(11))
    return canvas


def _tensor_image(tensor: torch.Tensor, scale: int = 2) -> Image.Image:
    array = (tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(array)
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _direction(dx: float, dy: float) -> str:
    if np.hypot(dx, dy) <= 0.25:
        return "stay"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _load_model(path: str, device: torch.device) -> BoxingObjectLAM:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    model = BoxingObjectLAM(
        config["state_dim"], config["latent_dim"], config.get("fdm_type", "independent"),
        config.get("object_input_mode", "masked_rgb_mask"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _balanced_indices(dataset: BoxingTransitionDataset, per_event: int, seed: int) -> List[int]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        grouped[entry.get("interaction_primary", "non_interaction")].append(index)
    rng = random.Random(seed)
    result: List[int] = []
    for event in EVENTS:
        candidates = grouped[event]
        rng.shuffle(candidates)
        result.extend(candidates[:per_event])
    rng.shuffle(result)
    return result


@torch.no_grad()
def collect_latents(
    model: BoxingObjectLAM,
    dataset: BoxingTransitionDataset,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False, num_workers=0)
    values: Dict[str, List] = defaultdict(list)
    for batch in loader:
        model_batch = {key: batch[key].to(device) for key in ("videos", "masks", "background_masks")}
        output = model(model_batch)
        z = output["z_mu"][:, 0].cpu().numpy()
        motion = batch["delta_xy"][:, 0].numpy()
        arms = batch["arm_lengths"].numpy()
        target_slots = batch["target_slot"].numpy()
        interaction_ids = batch["interaction_id"].numpy()
        for row, slot in enumerate(target_slots):
            slot = int(slot)
            values["z"].append(z[row, slot])
            values["motion"].append(motion[row, slot])
            values["slot"].append(slot)
            values["interaction"].append(int(interaction_ids[row]))
            values["punch"].append(int(np.any(arms[row, 1, slot] != 0)))
            values["arm_delta"].append(float(np.max(np.abs(arms[row, 1, slot] - arms[row, 0, slot]))))
    return {key: np.asarray(value) for key, value in values.items()}


def _probe_metrics(values: Dict[str, np.ndarray], seed: int) -> Dict[str, float]:
    z = values["z"]
    train, test = train_test_split(np.arange(len(z)), test_size=0.25, random_state=seed)
    motion_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    motion_model.fit(z[train], values["motion"][train])
    motion_pred = motion_model.predict(z[test])
    metrics = {
        "motion_dx_r2": float(r2_score(values["motion"][test, 0], motion_pred[:, 0])),
        "motion_dy_r2": float(r2_score(values["motion"][test, 1], motion_pred[:, 1])),
    }
    targets = {
        "punch_balanced_accuracy": values["punch"],
        "identity_balanced_accuracy": values["slot"],
        "interaction_balanced_accuracy": values["interaction"],
        "direction_balanced_accuracy": np.asarray([_direction(*xy) for xy in values["motion"]]),
    }
    for name, labels in targets.items():
        stratify = labels if np.min(np.unique(labels, return_counts=True)[1]) >= 2 else None
        tr, te = train_test_split(
            np.arange(len(z)), test_size=0.25, random_state=seed, stratify=stratify,
        )
        classifier = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        )
        classifier.fit(z[tr], labels[tr])
        metrics[name] = float(balanced_accuracy_score(labels[te], classifier.predict(z[te])))
    return metrics


def make_manifold(values: Dict[str, np.ndarray], output_dir: str, seed: int) -> Dict:
    z = values["z"]
    z_scaled = StandardScaler().fit_transform(z)
    umap_xy = UMAP(n_neighbors=35, min_dist=0.12, metric="euclidean", random_state=seed).fit_transform(z_scaled)
    tsne_xy = TSNE(
        n_components=2, perplexity=min(40, max(5, (len(z) - 1) // 4)),
        init="pca", learning_rate="auto", random_state=seed,
    ).fit_transform(z_scaled)
    directions = np.asarray([_direction(*xy) for xy in values["motion"]])
    direction_order = ("left", "right", "up", "down", "stay")
    direction_colors = dict(zip(direction_order, plt.get_cmap("tab10").colors[:5]))
    event_colors = dict(zip(EVENTS, plt.get_cmap("tab10").colors[:len(EVENTS)]))
    markers = {0: "o", 1: "^"}

    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for direction in direction_order:
        mask = directions == direction
        axes[0, 0].scatter(umap_xy[mask, 0], umap_xy[mask, 1], s=8, alpha=0.55,
                           color=direction_colors[direction], label=direction, linewidths=0)
    speed = np.linalg.norm(values["motion"], axis=1)
    speed_plot = axes[0, 1].scatter(umap_xy[:, 0], umap_xy[:, 1], c=speed, cmap="viridis", s=8, alpha=0.6, linewidths=0)
    figure.colorbar(speed_plot, ax=axes[0, 1], label="adjacent displacement (pixels)")
    punch_colors = np.where(values["punch"] == 1, "#d62728", "#8c8c8c")
    axes[0, 2].scatter(umap_xy[:, 0], umap_xy[:, 1], c=punch_colors, s=8, alpha=0.55, linewidths=0)
    axes[0, 2].scatter([], [], color="#d62728", label="punch")
    axes[0, 2].scatter([], [], color="#8c8c8c", label="no punch")
    for event_id, event in EVENT_NAMES.items():
        mask = values["interaction"] == event_id
        axes[1, 0].scatter(umap_xy[mask, 0], umap_xy[mask, 1], s=8, alpha=0.5,
                           color=event_colors[event], label=event, linewidths=0)
    for slot, name in ((0, "Player"), (1, "Enemy")):
        mask = values["slot"] == slot
        axes[1, 1].scatter(umap_xy[mask, 0], umap_xy[mask, 1], s=9, alpha=0.48,
                           marker=markers[slot], label=name, linewidths=0)
    for direction in direction_order:
        mask = directions == direction
        axes[1, 2].scatter(tsne_xy[mask, 0], tsne_xy[mask, 1], s=8, alpha=0.55,
                           color=direction_colors[direction], label=direction, linewidths=0)

    titles = (
        "UMAP — movement direction", "UMAP — displacement magnitude", "UMAP — punch active (red)",
        "UMAP — interaction event", "UMAP — fighter identity", "t-SNE — movement direction",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("dimension 1")
        axis.set_ylabel("dimension 2")
        axis.grid(alpha=0.12)
    axes[0, 0].legend(loc="best", ncol=2, fontsize=8)
    axes[0, 2].legend(loc="best", fontsize=8)
    axes[1, 0].legend(loc="best", ncol=2, fontsize=7)
    axes[1, 1].legend(loc="best", fontsize=8)
    axes[1, 2].legend(loc="best", ncol=2, fontsize=8)
    figure.suptitle("E04 residual interaction model — target-object latent space")
    path = os.path.join(output_dir, "latent_semantic_space.png")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    np.savez(
        os.path.join(output_dir, "latent_semantic_space.npz"),
        umap=umap_xy, tsne=tsne_xy, **values,
    )
    return {
        "points": int(len(z)),
        "event_counts": {EVENT_NAMES[i]: int((values["interaction"] == i).sum()) for i in EVENT_NAMES},
        "direction_counts": {name: int((directions == name).sum()) for name in direction_order},
        "punch_points": int(values["punch"].sum()),
    }


def make_probe_plot(metrics: Dict[str, float], output_dir: str) -> None:
    labels = ("dx R²", "dy R²", "punch BA", "direction BA", "interaction BA", "identity BA")
    keys = (
        "motion_dx_r2", "motion_dy_r2", "punch_balanced_accuracy",
        "direction_balanced_accuracy", "interaction_balanced_accuracy", "identity_balanced_accuracy",
    )
    values = [metrics[key] for key in keys]
    baselines = [0.0, 0.0, 0.5, 0.2, 0.125, 0.5]
    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bars = axis.bar(labels, values, color=plt.get_cmap("tab10").colors[:len(labels)])
    axis.set_ylim(min(-0.1, min(values) - 0.05), 1.0)
    axis.set_ylabel("held-out score")
    axis.set_title("Linear probes on frozen target-object z")
    axis.grid(axis="y", alpha=0.18)
    for bar, value, baseline in zip(bars, values, baselines):
        axis.hlines(
            baseline, bar.get_x(), bar.get_x() + bar.get_width(),
            color="gray", linewidth=2, linestyles="--",
        )
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    figure.savefig(os.path.join(output_dir, "semantic_probe_scores.png"), dpi=180)
    plt.close(figure)


def _choose_reconstruction_indices(dataset: BoxingTransitionDataset, seed: int) -> List[int]:
    wanted = ("non_interaction", "punch_miss", "hit", "received_hit", "occlusion", "recovery")
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        grouped[entry.get("interaction_primary", "non_interaction")].append(index)
    rng = random.Random(seed)
    selected = []
    for event in wanted:
        candidates = grouped[event]
        rng.shuffle(candidates)
        if candidates:
            selected.append(candidates[0])
    return selected


def _crop_bounds(masks: torch.Tensor, slot: int, pad: int = 12) -> tuple[int, int, int, int]:
    mask = masks[:, slot].any(dim=0)
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        return 0, 0, masks.shape[-1], masks.shape[-2]
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(masks.shape[-1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(masks.shape[-2], int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


@torch.no_grad()
def make_reconstructions(
    model: BoxingObjectLAM,
    dataset: BoxingTransitionDataset,
    indices: Sequence[int],
    device: torch.device,
    output_dir: str,
) -> Dict:
    samples = [dataset[index] for index in indices]
    batch = {
        key: torch.stack([sample[key] for sample in samples]).to(device)
        for key in ("videos", "masks", "background_masks")
    }
    outputs = {mode: model(batch, ablation=mode) for mode in ("normal", "zero", "shuffle")}
    full_rows, zoom_rows, mask_rows, rows = [], [], [], []
    for row, (index, sample) in enumerate(zip(indices, samples)):
        event = EVENT_NAMES[int(sample["interaction_id"])]
        slot = int(sample["target_slot"])
        target = sample["videos"][1]
        images = {
            "current": sample["videos"][0],
            "target": target,
            "normal z": outputs["normal"]["reconstruction"][row, 0].cpu(),
            "zero z": outputs["zero"]["reconstruction"][row, 0].cpu(),
            "shuffle z": outputs["shuffle"]["reconstruction"][row, 0].cpu(),
        }
        metrics = {name: float((image - target).abs().mean()) for name, image in images.items() if name not in {"current", "target"}}
        full_panels = [_label(_tensor_image(image, 2), name, "" if name in {"current", "target"} else f"L1={metrics[name]:.4f}") for name, image in images.items()]
        full_height = max(panel.height for panel in full_panels)
        padded_full = []
        for panel in full_panels:
            canvas = Image.new("RGB", (panel.width, full_height), (28, 28, 28))
            canvas.paste(panel, (0, 0))
            padded_full.append(np.asarray(canvas))
        row_header = Image.new("RGB", (150, full_height), (28, 28, 28))
        draw = ImageDraw.Draw(row_header)
        draw.text((7, 8), event, fill=(248, 248, 248), font=_font(13))
        draw.text((7, 31), "Player" if slot == 0 else "Enemy", fill=(190, 190, 190), font=_font(11))
        full_rows.append(np.asarray(Image.new("RGB", (1, 1))))
        full_rows[-1] = np.concatenate([np.asarray(row_header), *padded_full], axis=1)

        x0, y0, x1, y1 = _crop_bounds(sample["masks"], slot)
        zoom_panels = []
        for name, image in images.items():
            crop = image[:, y0:y1, x0:x1]
            pil = _tensor_image(crop, 5)
            zoom_panels.append(_label(pil, name, "" if name in {"current", "target"} else f"L1={metrics[name]:.4f}"))
        max_h = max(panel.height for panel in zoom_panels)
        padded = []
        for panel in zoom_panels:
            canvas = Image.new("RGB", (panel.width, max_h), (28, 28, 28))
            canvas.paste(panel, (0, 0))
            padded.append(np.asarray(canvas))
        zoom_header = Image.new("RGB", (150, max_h), (28, 28, 28))
        draw = ImageDraw.Draw(zoom_header)
        draw.text((7, 8), event, fill=(248, 248, 248), font=_font(13))
        draw.text((7, 31), "target fighter", fill=(190, 190, 190), font=_font(11))
        zoom_rows.append(np.concatenate([np.asarray(zoom_header), *padded], axis=1))

        mask_images = {
            "current mask": sample["masks"][0, slot],
            "target mask": sample["masks"][1, slot],
            "normal z": torch.sigmoid(outputs["normal"]["object_mask_logits"][row, 0, slot].cpu()),
            "zero z": torch.sigmoid(outputs["zero"]["object_mask_logits"][row, 0, slot].cpu()),
            "shuffle z": torch.sigmoid(outputs["shuffle"]["object_mask_logits"][row, 0, slot].cpu()),
        }
        target_mask = mask_images["target mask"] >= 0.5
        mask_metrics = {}
        mask_panels = []
        for name, mask in mask_images.items():
            if name not in {"current mask", "target mask"}:
                prediction = mask >= 0.5
                intersection = float((prediction & target_mask).sum())
                union = float((prediction | target_mask).sum())
                mask_metrics[name] = intersection / max(1.0, union)
            rgb_mask = mask.unsqueeze(0).repeat(3, 1, 1)[:, y0:y1, x0:x1]
            subtitle = "" if name in {"current mask", "target mask"} else f"IoU={mask_metrics[name]:.3f}"
            mask_panels.append(_label(_tensor_image(rgb_mask, 5), name, subtitle))
        mask_height = max(panel.height for panel in mask_panels)
        padded_masks = []
        for panel in mask_panels:
            canvas = Image.new("RGB", (panel.width, mask_height), (28, 28, 28))
            canvas.paste(panel, (0, 0))
            padded_masks.append(np.asarray(canvas))
        mask_header = Image.new("RGB", (150, mask_height), (28, 28, 28))
        draw = ImageDraw.Draw(mask_header)
        draw.text((7, 8), event, fill=(248, 248, 248), font=_font(13))
        draw.text((7, 31), "target mask", fill=(190, 190, 190), font=_font(11))
        mask_rows.append(np.concatenate([np.asarray(mask_header), *padded_masks], axis=1))
        rows.append({
            "index": index, "event": event, "slot": slot, **metrics,
            **{f"mask_iou_{name.replace(' ', '_')}": value for name, value in mask_metrics.items()},
        })

    def save_rows(rows_array: Sequence[np.ndarray], name: str) -> None:
        width = max(row.shape[1] for row in rows_array)
        padded = [np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0)), constant_values=28) for row in rows_array]
        Image.fromarray(np.concatenate(padded, axis=0)).save(os.path.join(output_dir, name))

    save_rows(full_rows, "reconstruction_full_frames.png")
    save_rows(zoom_rows, "reconstruction_fighter_zooms.png")
    save_rows(mask_rows, "mask_prediction_zooms.png")
    return {"indices": list(indices), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--points_per_event", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = BoxingTransitionDataset(args.index)
    model = _load_model(args.checkpoint, device)
    indices = _balanced_indices(dataset, args.points_per_event, args.seed)
    values = collect_latents(model, dataset, indices, device, args.batch_size)
    manifold = make_manifold(values, args.output_dir, args.seed)
    probes = _probe_metrics(values, args.seed)
    make_probe_plot(probes, args.output_dir)
    reconstructions = make_reconstructions(
        model, dataset, _choose_reconstruction_indices(dataset, args.seed), device, args.output_dir,
    )
    report = {
        "checkpoint": args.checkpoint,
        "index": args.index,
        "manifold": manifold,
        "probes": probes,
        "reconstructions": reconstructions,
    }
    with open(os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
