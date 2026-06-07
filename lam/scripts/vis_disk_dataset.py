"""
可视化预生成的多主体合成数据集。

用法：
  python vis_disk_dataset.py [num_samples=5] [output_dir=vis_output]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from lam.disk_synthetic_dataset import DiskSyntheticDataset

ACTION_NAMES = ["stay", "up", "down", "left", "right"]

def visualize_dataset():
    data_root = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_multi_actor")

    # 加载验证集
    val_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"),
        output_format="t h w c",
        num_frames=5,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "results", "dataset_vis")
    os.makedirs(output_dir, exist_ok=True)

    num_vis = min(5, len(val_dataset))

    for idx in range(num_vis):
        sample = val_dataset[idx]

        videos = sample["videos"]  # (T, H, W, C) float32 [0,1]
        masks = sample["masks"]    # (T, max_actors, H, W) float32 binary
        positions = sample["positions"]  # (T, max_actors, 2)
        actions = sample["actions"]      # (T-1, max_actors)
        num_actors = sample["num_actors"]

        T = videos.shape[0]
        cell_size = 256 // 8

        # 每帧：原图 + 每个 actor 的 mask 叠加
        fig, axes = plt.subplots(2, T, figsize=(5 * T, 10))
        if T == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(f"Sample {idx}: {num_actors} actors, {T} frames", fontsize=14)

        actor_colors = [
            (1.0, 0.2, 0.2, 0.5),
            (0.2, 0.8, 0.2, 0.5),
            (0.2, 0.4, 1.0, 0.5),
            (1.0, 0.8, 0.1, 0.5),
        ]

        for t in range(T):
            # 第1行：原图
            img = videos[t].cpu().numpy()
            axes[0, t].imshow(img)
            axes[0, t].set_title(f"Frame {t}", fontsize=12)
            axes[0, t].axis("off")

            # 叠加网格线
            for gi in range(9):
                axes[0, t].axhline(gi * cell_size - 0.5, color='white', lw=0.3, alpha=0.5)
                axes[0, t].axvline(gi * cell_size - 0.5, color='white', lw=0.3, alpha=0.5)

            # 标注每个 actor 的位置
            for a in range(num_actors):
                r, c = positions[t, a].tolist()
                if r >= 0:
                    rect = Rectangle(
                        (c * cell_size - 0.5, r * cell_size - 0.5),
                        2 * cell_size, 2 * cell_size,
                        fill=False, edgecolor=actor_colors[a][:3], linewidth=2
                    )
                    axes[0, t].add_patch(rect)
                    axes[0, t].text(
                        c * cell_size + 2, r * cell_size - 2,
                        f"A{a}", color=actor_colors[a][:3],
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.6, edgecolor='none')
                    )

            # 第2行：mask 叠加
            overlay = np.zeros((256, 256, 4), dtype=np.float32)
            for a in range(num_actors):
                mask = masks[t, a].cpu().numpy()  # (H, W)
                for c_idx in range(3):
                    overlay[..., c_idx] += mask * actor_colors[a][c_idx]
                overlay[..., 3] += mask * actor_colors[a][3]

            overlay = np.clip(overlay, 0, 1)
            axes[1, t].imshow(overlay)
            axes[1, t].set_title(f"Masks t={t}", fontsize=12)
            axes[1, t].axis("off")

            # 动作文字
            if t > 0 and t - 1 < actions.shape[0]:
                action_text = ""
                for a in range(num_actors):
                    act = actions[t - 1, a].item()
                    if act >= 0:
                        action_text += f"A{a}: {ACTION_NAMES[act]}, "
                axes[0, t].text(5, 20, action_text, color='white', fontsize=9,
                                bbox=dict(boxstyle='round,pad=0.1',
                                         facecolor='black', alpha=0.6, edgecolor='none'))

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{idx:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    print(f"\nDone! Visualized {num_vis} samples in {output_dir}")


if __name__ == "__main__":
    visualize_dataset()
