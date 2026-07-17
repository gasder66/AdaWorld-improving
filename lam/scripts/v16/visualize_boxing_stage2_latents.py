"""UMAP/t-SNE views of adjacent movement and punch latent transitions."""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import ConcatDataset, DataLoader
from umap import UMAP

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.boxing_object_dataset import BoxingObjectDataset
from lam.modules.v16_boxing_model import BoxingObjectLAM


def _motion_name(dx: float, dy: float) -> str:
    if np.hypot(dx, dy) <= 0.5:
        return "stay"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


@torch.no_grad()
def collect(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    values = {key: [] for key in ("z", "next_arm", "arm_delta", "motion", "slot")}
    model.eval()
    for batch in loader:
        model_batch = {key: batch[key].to(device) for key in ("videos", "masks", "background_masks")}
        z = model(model_batch)["z_mu"].cpu().numpy()
        next_arm = batch["arm_lengths"][:, 1:].numpy()
        arm_delta = (batch["arm_lengths"][:, 1:] - batch["arm_lengths"][:, :-1]).numpy()
        motion = batch["delta_xy"].numpy()
        for b in range(z.shape[0]):
            for t in range(z.shape[1]):
                for slot in range(2):
                    values["z"].append(z[b, t, slot])
                    values["next_arm"].append(next_arm[b, t, slot])
                    values["arm_delta"].append(arm_delta[b, t, slot])
                    values["motion"].append(motion[b, t, slot])
                    values["slot"].append(slot)
    return {key: np.asarray(value) for key, value in values.items()}


def _scatter_categorical(axis, xy, labels, categories, colors, title, slots=None):
    markers = {0: "o", 1: "^"}
    for category in categories:
        for slot in (0, 1):
            mask = labels == category
            if slots is not None:
                mask &= slots == slot
            if mask.any():
                label = category if slot == 0 else None
                axis.scatter(xy[mask, 0], xy[mask, 1], s=9, alpha=0.55, marker=markers[slot],
                             color=colors[category], label=label, linewidths=0)
    axis.set_title(title)
    axis.set_xlabel("dimension 1")
    axis.set_ylabel("dimension 2")
    axis.grid(alpha=0.12)
    axis.legend(loc="best", markerscale=1.8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--movement_root", required=True)
    parser.add_argument("--punch_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_umap_points", type=int, default=5000)
    parser.add_argument("--max_tsne_points", type=int, default=2500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    model = BoxingObjectLAM(
        config["state_dim"], config["latent_dim"], config.get("fdm_type", "independent"),
        config.get("object_input_mode", "masked_rgb_mask"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = ConcatDataset([
        BoxingObjectDataset(os.path.join(args.movement_root, "val"), target_frames=5),
        BoxingObjectDataset(os.path.join(args.punch_root, "val"), target_frames=5),
    ])
    values = collect(model, dataset, device)
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(values["z"]), min(args.max_umap_points, len(values["z"])), replace=False)
    z = values["z"][indices]
    z = (z - z.mean(axis=0, keepdims=True)) / np.maximum(z.std(axis=0, keepdims=True), 1e-6)
    next_arm = values["next_arm"][indices]
    arm_delta = values["arm_delta"][indices]
    motion = values["motion"][indices]
    slots = values["slot"][indices]
    punch_active = np.where(np.abs(next_arm).max(axis=1) > 0, "punch", "no punch")
    left = np.abs(next_arm[:, 0]) > 0
    right = np.abs(next_arm[:, 1]) > 0
    punch_side = np.full(len(z), "none", dtype=object)
    punch_side[left & ~right] = "left"
    punch_side[right & ~left] = "right"
    punch_side[left & right] = "both"
    extension = np.abs(next_arm).max(axis=1)
    phase_change = np.abs(arm_delta).max(axis=1)
    motion_labels = np.asarray([_motion_name(dx, dy) for dx, dy in motion])
    umap_xy = UMAP(n_neighbors=35, min_dist=0.12, random_state=args.seed).fit_transform(z)
    tsne_count = min(args.max_tsne_points, len(z))
    tsne_indices = rng.choice(len(z), tsne_count, replace=False)
    tsne_xy = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=args.seed).fit_transform(z[tsne_indices])

    figure, axes = plt.subplots(2, 3, figsize=(16, 10.5), constrained_layout=True)
    _scatter_categorical(
        axes[0, 0], umap_xy, punch_active, ["no punch", "punch"],
        {"no punch": "#8c8c8c", "punch": "#d62728"}, "UMAP — punch active", slots,
    )
    _scatter_categorical(
        axes[0, 1], umap_xy, punch_side, ["none", "left", "right", "both"],
        {"none": "#9a9a9a", "left": "#1f77b4", "right": "#ff7f0e", "both": "#9467bd"},
        "UMAP — active arm", slots,
    )
    extension_plot = axes[0, 2].scatter(umap_xy[:, 0], umap_xy[:, 1], c=extension, cmap="viridis", s=9, alpha=0.6, linewidths=0)
    axes[0, 2].set_title("UMAP — arm extension value")
    figure.colorbar(extension_plot, ax=axes[0, 2], label="max(left, right) RAM value")
    phase_plot = axes[1, 0].scatter(umap_xy[:, 0], umap_xy[:, 1], c=phase_change, cmap="plasma", s=9, alpha=0.6, linewidths=0)
    axes[1, 0].set_title("UMAP — adjacent punch phase change")
    figure.colorbar(phase_plot, ax=axes[1, 0], label="max absolute arm delta")
    motion_categories = ["left", "right", "up", "down", "stay"]
    _scatter_categorical(
        axes[1, 1], umap_xy, motion_labels, motion_categories,
        dict(zip(motion_categories, plt.get_cmap("tab10").colors[:5])), "UMAP — movement direction", slots,
    )
    _scatter_categorical(
        axes[1, 2], tsne_xy, punch_active[tsne_indices], ["no punch", "punch"],
        {"no punch": "#8c8c8c", "punch": "#d62728"}, "t-SNE — punch active", slots[tsne_indices],
    )
    for axis in (axes[0, 2], axes[1, 0]):
        axis.set_xlabel("dimension 1")
        axis.set_ylabel("dimension 2")
        axis.grid(alpha=0.12)
    figure.suptitle("V16 Boxing Stage 2 adjacent-transition latent manifold")
    output_path = os.path.join(args.output_dir, "stage2_latent_manifold.png")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    np.savez(
        os.path.join(args.output_dir, "stage2_latent_manifold.npz"),
        umap=umap_xy, z=z, next_arm=next_arm, arm_delta=arm_delta,
        motion=motion, slots=slots, punch_active=punch_active,
    )
    report = {
        "all_validation_latents": int(len(values["z"])),
        "umap_points": int(len(z)),
        "tsne_points": int(tsne_count),
        "punch_points": int((punch_active == "punch").sum()),
        "no_punch_points": int((punch_active == "no punch").sum()),
        "left_points": int((punch_side == "left").sum()),
        "right_points": int((punch_side == "right").sum()),
        "both_points": int((punch_side == "both").sum()),
    }
    with open(os.path.join(args.output_dir, "stage2_latent_manifold.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
