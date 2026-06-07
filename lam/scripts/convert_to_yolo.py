"""
将合成数据集转换为 YOLO 训练格式，用于微调 YOLOv8 检测彩色方块。

YOLO 格式：
- images/: *.png 图像
- labels/: *.txt 标注 (class x_center y_center width height，归一化坐标)
- data.yaml: 数据集配置

用法:
  python convert_to_yolo.py --split train --max_samples 1000
  python convert_to_yolo.py --split val --max_samples 200
"""
import os
import sys
import argparse

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.disk_synthetic_dataset import DiskSyntheticDataset


def convert_split(dataset, output_dir, max_samples=None):
    """将数据集的一个 split 转换为 YOLO 格式。"""
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    total_boxes = 0

    for idx in tqdm(range(n), desc=f"Converting {output_dir}"):
        sample = dataset[idx]
        videos = sample["videos"]  # (T, H, W, C) float32 [0,1]
        masks = sample["masks"]    # (T, A, H, W)
        T, H, W, C = videos.shape
        A = masks.shape[1]

        for t in range(T):
            # 保存图像
            frame = videos[t].numpy()  # (H, W, C) float32
            frame_uint8 = (frame * 255).astype(np.uint8)
            img = Image.fromarray(frame_uint8)
            img_name = f"sample_{idx:06d}_frame_{t:02d}.png"
            img_path = os.path.join(images_dir, img_name)
            img.save(img_path)

            # 生成 YOLO 标注
            label_name = f"sample_{idx:06d}_frame_{t:02d}.txt"
            label_path = os.path.join(labels_dir, label_name)

            lines = []
            for a in range(A):
                ys, xs = torch.where(masks[t, a] > 0.5)
                if len(ys) == 0:
                    continue

                x1 = xs.min().item()
                y1 = ys.min().item()
                x2 = xs.max().item()
                y2 = ys.max().item()

                # YOLO 格式：class x_center y_center width height (归一化)
                x_center = ((x1 + x2) / 2) / W
                y_center = ((y1 + y2) / 2) / H
                width = (x2 - x1) / W
                height = (y2 - y1) / H

                lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                total_boxes += 1

            with open(label_path, "w") as f:
                f.write("\n".join(lines))

    return n, total_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--max_train", type=int, default=1000,
                        help="最多转换多少训练样本")
    parser.add_argument("--max_val", type=int, default=200,
                        help="最多转换多少验证样本")
    args = parser.parse_args()

    if args.data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    else:
        data_root = args.data_root

    if args.output_root is None:
        output_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "yolo_synthetic"
        )

    print(f"Converting synthetic dataset to YOLO format...")
    print(f"  Source: {data_root}")
    print(f"  Output: {output_root}")

    # 转换训练集
    train_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "train"), num_frames=5, output_format="t h w c",
    )
    n_train, boxes_train = convert_split(
        train_dataset,
        os.path.join(output_root, "train"),
        max_samples=args.max_train,
    )

    # 转换验证集
    val_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"), num_frames=5, output_format="t h w c",
    )
    n_val, boxes_val = convert_split(
        val_dataset,
        os.path.join(output_root, "val"),
        max_samples=args.max_val,
    )

    # 生成 data.yaml
    yaml_content = f"""# Synthetic Multi-Actor Dataset for YOLO
path: {os.path.abspath(output_root)}
train: train/images
val: val/images

nc: 1
names: ['colored_block']
"""
    yaml_path = os.path.join(output_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n  Train: {n_train} samples, {boxes_train} boxes")
    print(f"  Val: {n_val} samples, {boxes_val} boxes")
    print(f"  data.yaml: {yaml_path}")
    print(f"  Done!")


if __name__ == "__main__":
    main()
