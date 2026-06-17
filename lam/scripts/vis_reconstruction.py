"""
加载 V4 模型并展示重建效果。

用法:
  CUDA_VISIBLE_DEVICES=2 python vis_reconstruction.py \\
      --checkpoint ../result/v4_dualstream/model_v4_dino_frozen_full.pt \\
      --config dinov2  (或 trainable)
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_model_config(name):
    configs = {
        "dinov2": dict(stream_a_mode="dinov2", freeze_dino=True,
                       pretrained_dino=True, stream_b_mode="linear",
                       decoder_residual=True, enc_blocks=2, dec_blocks=4),
        "trainable": dict(stream_a_mode="trainable", freeze_dino=False,
                          pretrained_dino=False, stream_b_mode="linear",
                          decoder_residual=True, enc_blocks=2, dec_blocks=4),
        "v3equiv": dict(stream_a_mode="trainable", freeze_dino=False,
                        pretrained_dino=False, stream_b_mode="linear",
                        decoder_residual=True, enc_blocks=4, dec_blocks=4),
    }
    return configs[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="dinov2")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "..", "result", "v4_dualstream", "vis"))
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Build model
    cfg = get_model_config(args.config)
    model = LatentActionModel(**cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Model loaded: {args.checkpoint}")
    print(f"  stream_a={cfg['stream_a_mode']}, freeze_dino={cfg['freeze_dino']}")

    # Data
    if args.data_root is None:
        args.data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    dataset = DiskSyntheticDataset(
        os.path.join(args.data_root, "val"),
        num_frames=5, output_format="t h w c",
    )
    print(f"Dataset: {len(dataset)} samples")

    # Inference & visualization
    for idx in range(min(args.num_samples, len(dataset))):
        sample = dataset[idx]
        videos = sample["videos"].unsqueeze(0).to(device)  # (1, T, H, W, C)
        masks = sample["masks"].unsqueeze(0).to(device)    # (1, T, A, H, W)
        num_actors = int(sample["num_actors"])

        with torch.no_grad():
            out = model({"videos": videos, "masks": masks})

        # Prepare images
        gt_0 = videos[0, 0].cpu().numpy()                # frame 0
        gt_1 = videos[0, 1].cpu().numpy()                # frame 1 (target)
        recon = out["recon"][0, 0].cpu().numpy()          # reconstructed frame 1
        diff = np.abs(gt_1 - recon)
        diff_mse = ((gt_1 - recon) ** 2).mean()

        # Build figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        titles = [f"Frame t", "Frame t+1 (GT)", f"Reconstructed\n(diff diff_mse={diff_mse:.4f})",
                  "|GT - Recon| Heatmap",
                  "Mask (actors)", "Mask (bg + actors)", "Recon overlayed", "Subject features"]

        # Row 1: Reconstruction comparison
        images_row1 = [gt_0, gt_1, recon, diff]
        for i, (img, ax) in enumerate(zip(images_row1, axes[0])):
            ax.imshow(img)
            ax.set_title(titles[i], fontsize=10)
            ax.axis("off")

        # Row 2: Mask visualization
        mask_actors = masks[0, 0].cpu().numpy()  # (A, H, W)
        # Show individual actor masks
        ax = axes[1, 0]
        mask_rgb = np.zeros((*mask_actors.shape[1:], 3))
        colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0)]
        for a in range(mask_actors.shape[0]):
            for c in range(3):
                mask_rgb[..., c] += mask_actors[a] * colors[a][c]
        ax.imshow(np.clip(mask_rgb, 0, 1))
        ax.set_title(titles[4], fontsize=10)
        ax.axis("off")

        # Full mask including background
        with torch.no_grad():
            all_masks = model._build_masks_with_background(masks)
        all_masks_np = all_masks[0, 0].cpu().numpy()
        ax = axes[1, 1]
        mask_rgb2 = np.zeros((*all_masks_np.shape[1:], 3))
        bg_color = (0.5, 0.5, 0.5)
        for c in range(3):
            mask_rgb2[..., c] += all_masks_np[0] * bg_color[c]  # bg
        for a in range(1, all_masks_np.shape[0]):
            for c in range(3):
                mask_rgb2[..., c] += all_masks_np[a] * colors[(a-1) % len(colors)][c]
        ax.imshow(np.clip(mask_rgb2, 0, 1))
        ax.set_title(titles[5], fontsize=10)
        ax.axis("off")

        # Reconstruction overlay
        ax = axes[1, 2]
        ax.imshow(gt_1 * 0.5 + recon * 0.5)
        ax.set_title(titles[6], fontsize=10)
        ax.axis("off")

        # obj_feats info
        ax = axes[1, 3]
        vm = out["valid_mask"][0, 0].cpu().numpy()
        obj_feats = out["obj_feats"][0, 0].cpu().numpy() if "obj_feats" in out else None
        ax.text(0.1, 0.5,
                f"Valid masks (t=0):\n  bg={vm[0]}\n  actors: {vm[1:]}\n\n"
                f"num_actors={num_actors}\n"
                f"latent_dim=32\n"
                f"z_mu norm: mean={out['z_mu'].norm(dim=-1).mean().item():.2f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='center')
        ax.set_title(titles[7], fontsize=10)
        ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(args.out_dir, f"recon_sample_{idx:03d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{idx+1}/{args.num_samples}] Saved {save_path}")

    print(f"\nDone. Images saved to {args.out_dir}")


if __name__ == "__main__":
    main()
