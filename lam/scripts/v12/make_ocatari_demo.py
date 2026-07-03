"""Create visual QA assets for OCAtari-generated V12 clips."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.v12_dataset import V12ObjectVideoDataset


PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
]


def _to_uint8_video(videos: torch.Tensor) -> np.ndarray:
    arr = videos.detach().cpu()
    if arr.ndim == 4 and arr.shape[1] == 3:
        arr = arr.permute(0, 2, 3, 1)
    arr = arr.numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _draw_frame(frame: np.ndarray, sample: Dict, t: int, scale: int) -> Image.Image:
    image = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    bboxes = sample["bboxes"][t]
    valid = sample["valid_mask"][t]
    categories = sample["metadata"].get("slot_categories", [])
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for k in range(bboxes.shape[0]):
        if not bool(valid[k]):
            continue
        x1, y1, x2, y2 = [int(round(float(v))) * scale for v in bboxes[k]]
        color = PALETTE[k % len(PALETTE)]
        label = categories[k] if k < len(categories) else f"slot{k}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(1, scale))
        draw.text((x1 + 1, max(0, y1 - 10)), f"{k}:{label}", fill=color, font=font)
    draw.text((4, 4), f"t={t}", fill=(255, 255, 255), font=font)
    return image


def _make_grid(frames: List[Image.Image], cols: int) -> Image.Image:
    if not frames:
        raise ValueError("no frames")
    w, h = frames[0].size
    rows = (len(frames) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for idx, frame in enumerate(frames):
        canvas.paste(frame, ((idx % cols) * w, (idx // cols) * h))
    return canvas


def make_demo(
    data_root: str,
    out_root: str,
    max_samples: int,
    scale: int,
    fps: int,
    sample_indices: List[int] | None = None,
) -> None:
    os.makedirs(out_root, exist_ok=True)
    ds = V12ObjectVideoDataset(os.path.join(data_root, "train"), output_format="t h w c")
    indices = sample_indices if sample_indices else list(range(min(max_samples, len(ds))))
    indices = [idx for idx in indices if 0 <= idx < len(ds)]
    lines = [
        f"# OCAtari Demo: {os.path.basename(data_root)}",
        "",
        f"- Source: `{data_root}`",
        f"- Samples shown: {len(indices)}",
        f"- Sample indices: {indices}",
        "- Boxes/masks come from OCAtari objects, not generic image detectors.",
        "",
        "## Assets",
        "",
    ]
    for out_i, dataset_i in enumerate(indices):
        sample = ds[dataset_i]
        video = _to_uint8_video(sample["videos"])
        frames = [_draw_frame(video[t], sample, t, scale) for t in range(video.shape[0])]
        gif_name = f"sample_{dataset_i:06d}.gif"
        grid_name = f"sample_{dataset_i:06d}_grid.png"
        frames[0].save(
            os.path.join(out_root, gif_name),
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        _make_grid(frames, cols=len(frames)).save(os.path.join(out_root, grid_name))
        valid_counts = sample["valid_mask"].sum(dim=1).tolist()
        categories = sample["metadata"].get("slot_categories", [])
        lines.append(f"- `{gif_name}` / `{grid_name}`: valid objects per frame = {valid_counts}; slots = {categories}")
    with open(os.path.join(out_root, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--sample_indices", nargs="*", type=int, default=None)
    args = parser.parse_args()
    make_demo(args.data_root, args.out_root, args.max_samples, args.scale, args.fps, args.sample_indices)


if __name__ == "__main__":
    main()
