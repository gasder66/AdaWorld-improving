"""Create raw-motion videos, latent manifolds, and reconstruction comparisons."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from umap import UMAP

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_object_dataset import BoxingObjectDataset
from lam.modules.v16_boxing_model import BoxingObjectLAM


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def _label_frame(frame: np.ndarray, lines: Sequence[str], scale: int = 3) -> np.ndarray:
    image = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (image.width, image.height + 22 * len(lines)), (24, 24, 24))
    canvas.paste(image, (0, 22 * len(lines)))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for index, line in enumerate(lines):
        draw.text((5, index * 22 + 3), line, fill=(245, 245, 245), font=font)
    return np.asarray(canvas)


def _write_video(frames: Sequence[np.ndarray], stem: str, fps: int) -> None:
    imageio.mimsave(stem + ".gif", list(frames), duration=1000 / fps, loop=0)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", stem + ".gif",
            "-vf", f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v", "libx264", stem + ".mp4",
        ],
        check=True,
    )


def _net_displacement(sample: Dict[str, torch.Tensor]) -> np.ndarray:
    return sample["delta_xy"].sum(dim=0).numpy()


def _direction_name(dx: float, dy: float) -> str:
    if np.hypot(dx, dy) < 0.5:
        return "stay"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def choose_motion_samples(dataset: BoxingObjectDataset, count: int) -> List[int]:
    candidates = []
    for index in range(len(dataset)):
        sample = dataset[index]
        displacement = _net_displacement(sample)
        magnitude = np.linalg.norm(displacement, axis=1).max()
        directions = tuple(_direction_name(*xy) for xy in displacement)
        candidates.append((index, float(magnitude), directions))
    selected: List[int] = []
    used = set()
    for index, _magnitude, directions in sorted(candidates, key=lambda row: row[1], reverse=True):
        signature = directions
        if signature not in used:
            selected.append(index)
            used.add(signature)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for index, _magnitude, _directions in sorted(candidates, key=lambda row: row[1], reverse=True):
            if index not in selected:
                selected.append(index)
            if len(selected) >= count:
                break
    return selected


def make_raw_videos(dataset: BoxingObjectDataset, indices: Sequence[int], output_dir: str, fps: int) -> List[Dict]:
    manifest = []
    for output_index, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]
        video = (sample["videos"].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        displacement = _net_displacement(sample)
        labels = [f"P: dx={displacement[0,0]:+.1f} dy={displacement[0,1]:+.1f} ({_direction_name(*displacement[0])})",
                  f"E: dx={displacement[1,0]:+.1f} dy={displacement[1,1]:+.1f} ({_direction_name(*displacement[1])})"]
        frames = [_label_frame(frame, [f"sample={dataset_index} frame={t}/4", *labels]) for t, frame in enumerate(video)]
        stem = os.path.join(output_dir, f"raw_motion_{output_index:02d}_sample_{dataset_index:06d}")
        _write_video(frames, stem, fps)
        manifest.append({"dataset_index": dataset_index, "stem": os.path.basename(stem), "displacement": displacement.tolist()})
    return manifest


@torch.no_grad()
def collect_latents(model: BoxingObjectLAM, dataset: BoxingObjectDataset, device: torch.device):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    z_values, displacements, slots, sample_indices = [], [], [], []
    model.eval()
    for batch in loader:
        endpoints = torch.tensor([0, batch["videos"].shape[1] - 1])
        model_batch = {key: batch[key].index_select(1, endpoints).to(device) for key in ("videos", "masks", "background_masks")}
        output = model(model_batch)
        batch_z = output["z_mu"][:, 0].cpu().numpy()
        batch_displacement = batch["delta_xy"].sum(dim=1).numpy()
        for b in range(batch_z.shape[0]):
            for slot in range(2):
                z_values.append(batch_z[b, slot])
                displacements.append(batch_displacement[b, slot])
                slots.append(slot)
                sample_indices.append(int(batch["sample_index"][b]))
    return np.asarray(z_values), np.asarray(displacements), np.asarray(slots), np.asarray(sample_indices)


def make_manifold_plot(z: np.ndarray, displacement: np.ndarray, slots: np.ndarray, output_path: str, seed: int) -> Dict:
    z_std = (z - z.mean(axis=0, keepdims=True)) / np.maximum(z.std(axis=0, keepdims=True), 1e-6)
    umap_xy = UMAP(n_neighbors=30, min_dist=0.15, metric="euclidean", random_state=seed).fit_transform(z_std)
    tsne_xy = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=seed).fit_transform(z_std)
    dx, dy = displacement[:, 0], displacement[:, 1]
    speed = np.linalg.norm(displacement, axis=1)
    angle = (np.arctan2(-dy, dx) + 2 * np.pi) % (2 * np.pi)
    direction_color = hsv_to_rgb(np.stack([angle / (2 * np.pi), np.full_like(angle, 0.78), np.full_like(angle, 0.88)], axis=1))
    direction_color[speed < 0.5] = (0.55, 0.55, 0.55)
    labels = np.asarray([_direction_name(x, y) for x, y in displacement])
    category_order = ["left", "right", "up", "down", "stay"]
    category_colors = {name: plt.get_cmap("tab10")(index) for index, name in enumerate(category_order)}

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for slot, marker, name in ((0, "o", "Player"), (1, "^", "Enemy")):
        mask = slots == slot
        axes[0, 0].scatter(umap_xy[mask, 0], umap_xy[mask, 1], c=direction_color[mask], s=18, marker=marker, alpha=0.78, label=name)
        axes[0, 1].scatter(tsne_xy[mask, 0], tsne_xy[mask, 1], c=direction_color[mask], s=18, marker=marker, alpha=0.78, label=name)
    for label in category_order:
        mask = labels == label
        axes[1, 0].scatter(umap_xy[mask, 0], umap_xy[mask, 1], color=category_colors[label], s=18, alpha=0.72, label=label)
    speed_plot = axes[1, 1].scatter(umap_xy[:, 0], umap_xy[:, 1], c=speed, cmap="viridis", s=18, alpha=0.78)
    figure.colorbar(speed_plot, ax=axes[1, 1], label="4-frame displacement magnitude (pixels)")
    axes[0, 0].set_title("UMAP — continuous direction hue")
    axes[0, 1].set_title("t-SNE — continuous direction hue")
    axes[1, 0].set_title("UMAP — cardinal direction")
    axes[1, 1].set_title("UMAP — displacement magnitude")
    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    axes[1, 0].legend(loc="best", ncol=3)
    for axis in axes.flat:
        axis.set_xlabel("dimension 1")
        axis.set_ylabel("dimension 2")
        axis.grid(alpha=0.15)
    figure.suptitle("V16 Boxing latent-action manifold (validation set)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    np.savez(output_path.replace(".png", ".npz"), z=z, displacement=displacement, slots=slots, umap=umap_xy, tsne=tsne_xy)
    counts = {name: int((labels == name).sum()) for name in category_order}
    return {"points": len(z), "direction_counts": counts, "speed_min": float(speed.min()), "speed_max": float(speed.max())}


@torch.no_grad()
def make_reconstruction_video(
    model: BoxingObjectLAM,
    dataset: BoxingObjectDataset,
    indices: Sequence[int],
    output_dir: str,
    device: torch.device,
    fps: int,
) -> Dict:
    frames = []
    per_sample = []
    model.eval()
    for dataset_index in indices:
        sample = dataset[dataset_index]
        endpoints = torch.tensor([0, sample["videos"].shape[0] - 1])
        batch = {
            "videos": sample["videos"].index_select(0, endpoints).unsqueeze(0).to(device),
            "masks": sample["masks"].index_select(0, endpoints).unsqueeze(0).to(device),
            "background_masks": sample["background_masks"].index_select(0, endpoints).unsqueeze(0).to(device),
        }
        outputs = {name: model(batch, ablation=name)["reconstruction"][0, 0].cpu() for name in ("normal", "zero", "shuffle")}
        images = {
            "current t": sample["videos"][0],
            "target t+4": sample["videos"][-1],
            "normal z": outputs["normal"],
            "zero z": outputs["zero"],
            "shuffle z": outputs["shuffle"],
        }
        panels = []
        metrics = {}
        target = sample["videos"][-1]
        for label, tensor in images.items():
            array = (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            if label not in {"current t", "target t+4"}:
                metrics[label] = float((tensor - target).abs().mean())
            panels.append(_label_frame(array, [label], scale=2))
        height = max(panel.shape[0] for panel in panels)
        frame = np.concatenate([np.pad(panel, ((0, height-panel.shape[0]), (0, 0), (0, 0))) for panel in panels], axis=1)
        frame = _label_frame(frame, [f"validation sample={dataset_index} | lower reconstruction L1 is better", json.dumps(metrics)], scale=1)
        frames.extend([frame] * max(1, fps))
        per_sample.append({"dataset_index": dataset_index, **metrics})
    stem = os.path.join(output_dir, "reconstruction_comparison")
    _write_video(frames, stem, fps)
    imageio.imwrite(stem + ".png", frames[0])
    return {"stem": os.path.basename(stem), "samples": per_sample}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data/v16_boxing/stage1_movement")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_videos", type=int, default=6)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    model = BoxingObjectLAM(config["state_dim"], config["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = BoxingObjectDataset(os.path.join(args.data_root, "val"))
    indices = choose_motion_samples(dataset, args.num_videos)
    raw_manifest = make_raw_videos(dataset, indices, args.output_dir, args.fps)
    z, displacement, slots, _sample_indices = collect_latents(model, dataset, device)
    manifold = make_manifold_plot(z, displacement, slots, os.path.join(args.output_dir, "latent_manifold.png"), args.seed)
    reconstruction = make_reconstruction_video(model, dataset, indices, args.output_dir, device, args.fps)
    manifest = {"checkpoint": args.checkpoint, "selected_indices": indices, "raw_videos": raw_manifest,
                "manifold": manifold, "reconstruction": reconstruction}
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
