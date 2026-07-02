"""
将合成数据集转换为 YOLO训练格式。

每个 .pt 样本的每一帧导出为一张 PNG 图像 + 一个 YOLO 标注文件。
标注格式: class_id cx cy w h (归一化到 [0,1])

用法:
  python prepare_yolo_data.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from pathlib import Path
from PIL import Image


def masks_to_yolo_labels(masks, num_actors, img_size=256):
    """将 mask 转换为 YOLO 标注格式。

    Args:
        masks: (T, max_actors, H, W) binary
        num_actors: 实际主体数
        img_size: 图像尺寸

    Returns:
        list of str: 每帧的 YOLO 标注行 (class cx cy w h)
    """
    labels = []
    for a in range(num_actors):
        mask = masks[a]  # (H, W)
        if mask.sum() < 10:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        y1, y2 = rows[0], rows[-1] + 1
        x1, x2 = cols[0], cols[-1] + 1
        cx = (x1 + x2) / 2.0 / img_size
        cy = (y1 + y2) / 2.0 / img_size
        w = (x2 - x1) / img_size
        h = (y2 - y1) / img_size
        # 所有主体统一为 class 0 (single class detection)
        labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return labels


def main():
    data_root = os.path.join(os.path.dirname(__file__), "..", "..",
                             "data", "synthetic_multi_actor")
    yolo_root = os.path.join(os.path.dirname(__file__), "..", "..",
                             "data", "yolo_synthetic")

    for split in ["train", "val"]:
        src_dir = os.path.join(data_root, split)
        img_dir = os.path.join(yolo_root, split, "images")
        lbl_dir = os.path.join(yolo_root, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(src_dir) if f.endswith(".pt")])
        total_boxes = 0
        total_frames = 0

        for fname in files:
            sample = torch.load(os.path.join(src_dir, fname), map_location="cpu")
            videos = sample["videos"]  # (T, 3, H, W) uint8
            masks = sample["masks"]    # (T, max_actors, H, W)
            num_actors = int(sample["num_actors"])
            T = videos.shape[0]

            base = fname.replace(".pt", "")
            for t in range(T):
                img = videos[t]  # (3, H, W) uint8
                img_np = img.permute(1, 2, 0).numpy()  # (H, W, 3)
                if img_np.dtype != np.uint8:
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

                img_path = os.path.join(img_dir, f"{base}_t{t}.png")
                Image.fromarray(img_np).save(img_path)

                yolo_labels = masks_to_yolo_labels(
                    masks[t].numpy(), num_actors
                )
                lbl_path = os.path.join(lbl_dir, f"{base}_t{t}.txt")
                with open(lbl_path, "w") as f:
                    f.write("\n".join(yolo_labels))

                total_boxes += len(yolo_labels)
                total_frames += 1

        print(f"  [{split}] {len(files)} samples, {total_frames} frames, "
              f"{total_boxes} boxes → {img_dir}")

    # 生成 yaml 配置
    yaml_path = os.path.join(yolo_root, "dataset.yaml")
    abs_yolo = os.path.abspath(yolo_root)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_yolo}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['object']\n")
    print(f"\n  YOLO config: {yaml_path}")
    print("Done!")


if __name__ == "__main__":
    main()
