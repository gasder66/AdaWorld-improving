"""
V13 slot visualization: debug SlotBuilder output.

Usage:
  PYTHONPATH=lam python lam/scripts/v13/visualize_slots.py \
      --game freeway --num_samples 16 --gpu 0
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.atari_bbox_dataset import AtariBBoxDataset, GAME_CONFIGS
from lam.modules.v13_slot_builder import build_slot_builder, ROLE_AGENT, ROLE_LANE_GROUP, ROLE_NEARBY_CAR


ROLE_COLORS = {
    ROLE_AGENT: (255, 50, 50),
    ROLE_LANE_GROUP: (50, 150, 255),
    ROLE_NEARBY_CAR: (50, 255, 50),
}


def _draw_rect(img, cx, cy, w, h, color):
    H, W = img.shape[:2]
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    x1, x2 = max(0, x1), min(W - 1, x2)
    y1, y2 = max(0, y1), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    img[y1:y1 + 2, x1:x2] = color
    img[y2 - 2:y2, x1:x2] = color
    img[y1:y2, x1:x1 + 2] = color
    img[y1:y2, x2 - 2:x2] = color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="freeway")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = GAME_CONFIGS[args.game]
    data_root = f"data/v12_ocatari/ocatari_{args.game}/train"

    if not os.path.isdir(data_root):
        print(f"ERROR: data not found at {data_root}")
        sys.exit(1)

    ds = AtariBBoxDataset(data_root, game=args.game, image_size=cfg["image_size"],
                          max_objects=cfg["max_objects"])
    slot_builder = build_slot_builder(args.game)
    slot_builder.eval()

    print(f"Dataset: {len(ds)} samples, game={args.game}, image_size={cfg['image_size']}")
    print(f"Slot builder: K_focus={slot_builder.K_focus}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(args.num_samples, len(ds))
    if args.out_dir is None:
        args.out_dir = f"result/v13/{args.game}/slot_debug"
    os.makedirs(args.out_dir, exist_ok=True)

    for sample_idx in range(n):
        raw = ds[sample_idx]

        # Create a pseudo-batch.
        batch = {
            "bbox": raw["bbox"].unsqueeze(0),
            "valid": raw["valid"].unsqueeze(0),
            "obj_type": raw["obj_type"].unsqueeze(0),
        }
        slots = slot_builder(batch)

        video = raw["video"]  # (T, 3, H, W)
        T, C, H, W = video.shape

        fig, axes = plt.subplots(2, max(T, 2), figsize=(max(T, 2) * 2.5, 5))
        if axes.ndim == 1:
            axes = axes[np.newaxis, :]

        for t in range(T):
            frame = (video[t].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8).copy()
            frame_rgb = frame.copy()
            frame_slots = frame.copy()

            # Draw raw objects.
            bw = raw["valid"]
            bb = raw["bbox"]
            for k in range(raw["valid"].shape[-1]):
                if bw[t, k]:
                    cx, cy, w_, h_ = bb[t, k].tolist()
                    _draw_rect(frame_rgb, cx, cy, w_, h_, (255, 255, 255))

            # Draw focus slots.
            sv = slots["slot_valid"][0, t]
            sb = slots["slot_bbox"][0, t]
            sr = slots["slot_role"][0, t]
            for k in range(slot_builder.K_focus):
                if sv[k]:
                    cx, cy, w_, h_ = sb[k].tolist()
                    color = ROLE_COLORS.get(sr[k].item(), (200, 200, 200))
                    _draw_rect(frame_slots, cx, cy, w_, h_, color)

            axes[0, t].imshow(frame_rgb)
            axes[0, t].set_title(f"Raw t={t}", fontsize=7)
            axes[0, t].axis("off")
            axes[1, t].imshow(frame_slots)
            axes[1, t].set_title(f"Focus slots t={t}", fontsize=7)
            axes[1, t].axis("off")

        # Fill empty columns.
        for t in range(T, axes.shape[1]):
            for row in range(2):
                axes[row, t].axis("off")

        fig.tight_layout(pad=0.3)
        out_path = os.path.join(args.out_dir, f"sample_{sample_idx:04d}.png")
        fig.savefig(out_path, dpi=100)
        plt.close(fig)

        if sample_idx % 4 == 0:
            print(f"  [{sample_idx}/{n}] saved {out_path}")

    print(f"  Done. {n} images saved to {args.out_dir}")


if __name__ == "__main__":
    main()
