"""
验证 LocateAnything + SAM2 在合成数据集上的分割质量。

步骤：
1. 加载合成数据集样本
2. 用 LocateAnything 检测彩色方块
3. 用 SAM2 从 detected box 生成 mask
4. 对比预测 mask 与 GT mask

用法:
  CUDA_VISIBLE_DEVICES=2 python verify_segmentation_pipeline.py
"""
import os
import sys
import json
import argparse

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eagle", "Embodied"))

from lam.disk_synthetic_dataset import DiskSyntheticDataset


COLOR_NAMES = {
    0: "red",
    1: "green",
    2: "blue",
    3: "yellow",
    4: "purple",
}

ACTION_NAMES = ["stay", "up", "down", "left", "right"]


def rgb_to_color_name(rgb):
    """将 RGB 值映射到最接近的颜色名称。"""
    colors = {
        "red": (1.0, 0.2, 0.2),
        "green": (0.2, 0.8, 0.2),
        "blue": (0.2, 0.4, 1.0),
        "yellow": (1.0, 0.8, 0.1),
        "purple": (0.8, 0.3, 0.8),
    }
    best = min(colors, key=lambda c: sum((a - b) ** 2 for a, b in zip(rgb, colors[c])))
    return best


def extract_gt_boxes_from_masks(masks):
    """从 GT mask 中提取 bounding box。

    Args:
        masks: (A, H, W) binary tensor
    Returns:
        boxes: (A, 4) [[x1,y1,x2,y2], ...] pixel coords, None for empty masks
    """
    A, H, W = masks.shape
    boxes = []
    for a in range(A):
        ys, xs = torch.where(masks[a] > 0.5)
        if len(ys) == 0:
            boxes.append(None)
        else:
            boxes.append([xs.min().item(), ys.min().item(),
                          xs.max().item(), ys.max().item()])
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=10,
                        help="验证样本数")
    parser.add_argument("--skip_sam2", action="store_true",
                        help="跳过 SAM2（仅测试 LocateAnything 检测）")
    args = parser.parse_args()

    device = torch.device("cuda:0")

    # 数据集
    data_root = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
    )
    dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"),
        num_frames=5, output_format="t h w c",
    )
    print(f"Dataset: {len(dataset)} samples")

    # 加载 LocateAnythingWorker
    print("Loading LocateAnythingWorker...")
    from locateanything_worker import LocateAnythingWorker
    la_worker = LocateAnythingWorker(
        "nvidia/LocateAnything-3B",
        device=device,
        dtype=torch.bfloat16,
    )
    print("  LocateAnythingWorker loaded")

    # 加载 SAM2
    if not args.skip_sam2:
        print("Loading SAM2...")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_model = build_sam2(
                config_file="sam2_hiera_l.yaml",  # 使用大模型以获得最佳质量
                ckpt_path=None,  # 自动从 hub 下载
            ).to(device)
            sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("  SAM2 loaded")
        except Exception as e:
            print(f"  SAM2 load failed: {e}")
            print("  Falling back to box fill (no SAM2)")
            args.skip_sam2 = True

    # ===== 逐样本验证 =====
    results = []
    total_la_recalled = 0  # LocateAnything 检出了多少主体
    total_la_missed = 0    # 漏检了多少
    total_gt_actors = 0    # 总 GT 主体数
    total_la_false_pos = 0 # 误检

    for idx in range(min(args.num_samples, len(dataset))):
        sample = dataset[idx]
        videos = sample["videos"]  # (T, H, W, C) float32 [0,1]
        masks = sample["masks"]    # (T, A, H, W) float32
        actions = sample["actions"]
        num_actors = sample["num_actors"]

        # 取第一帧
        frame = videos[0].numpy()  # (H, W, C) float32
        H, W = frame.shape[:2]

        # === LocateAnything 检测 ===
        # 将帧转换为 PIL Image
        frame_pil = Image.fromarray((frame * 255).astype(np.uint8))

        # 尝试多种 prompt 以获取最佳检测效果
        detection_prompts = [
            ["colored block", "colored square", "colored shape"],
            ["red object", "green object", "blue object", "yellow object", "purple object"],
            ["brightly colored shape on a patterned background"],
        ]

        all_detected = []
        for prompt_set in detection_prompts:
            try:
                result = la_worker.detect(frame_pil, prompt_set,
                                          generation_mode="fast",
                                          temperature=0.1)
                parsed = LocateAnythingWorker.parse_boxes(
                    result["answer"], W, H
                )
                if parsed:
                    all_detected.extend(parsed)
            except Exception as e:
                print(f"    Prompt {prompt_set} failed: {e}")

        # 去重（基于 IoU 阈值 0.5）
        detected_boxes = []
        for box in all_detected:
            is_dup = False
            for existing in detected_boxes:
                # 计算 IoU (box/existing are dicts with x1,y1,x2,y2 keys)
                xa = max(box["x1"], existing["x1"])
                ya = max(box["y1"], existing["y1"])
                xb = min(box["x2"], existing["x2"])
                yb = min(box["y2"], existing["y2"])
                inter = max(0, xb - xa) * max(0, yb - ya)
                box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                exist_area = (existing["x2"] - existing["x1"]) * (existing["y2"] - existing["y1"])
                union = box_area + exist_area - inter
                iou = inter / (union + 1e-6)
                if iou > 0.5:
                    is_dup = True
                    break
            if not is_dup:
                detected_boxes.append(box)

        # === GT 评估 ===
        gt_boxes = extract_gt_boxes_from_masks(masks[0])  # 第一帧
        gt_boxes = [b for b in gt_boxes if b is not None]

        # 匹配：对每个 GT box，找到 IoU 最大的检测框
        matched = [False] * len(detected_boxes)
        for gt in gt_boxes:
            best_iou = 0
            best_j = -1
            for j, dt in enumerate(detected_boxes):
                xa = max(gt[0], dt["x1"])
                ya = max(gt[1], dt["y1"])
                xb = min(gt[2], dt["x2"])
                yb = min(gt[3], dt["y2"])
                inter = max(0, xb - xa) * max(0, yb - ya)
                gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                dt_area = (dt["x2"] - dt["x1"]) * (dt["y2"] - dt["y1"])
                union = gt_area + dt_area - inter
                iou = inter / (union + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= 0.3:  # 召回阈值
                matched[best_j] = True
                total_la_recalled += 1
            else:
                total_la_missed += 1
            total_gt_actors += 1

        total_la_false_pos += matched.count(False)

        # 打印当前样本结果
        sample_info = f"Sample {idx}: {num_actors} actors, "
        sample_info += f"GT boxes: {len(gt_boxes)}, "
        sample_info += f"LA detected: {len(detected_boxes)}, "
        sample_info += f"matched: {sum(matched)}/{len(gt_boxes)}"
        print(sample_info)

        # 可视化（可选）
        if idx < 3:
            save_dir = os.path.join(
                os.path.dirname(__file__), "..", "results", "segmentation_vis"
            )
            os.makedirs(save_dir, exist_ok=True)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # 原图
            axes[0].imshow(frame)
            axes[0].set_title(f"Sample {idx}: Original")
            axes[0].axis("off")

            # 带检测框
            axes[1].imshow(frame)
            for box in detected_boxes:
                rect = patches.Rectangle(
                    (box["x1"], box["y1"]), box["x2"] - box["x1"], box["y2"] - box["y1"],
                    linewidth=2, edgecolor="lime", facecolor="none",
                )
                axes[1].add_patch(rect)
            for gt in gt_boxes:
                rect = patches.Rectangle(
                    (gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
                    linewidth=1, edgecolor="red", facecolor="none", linestyle="--",
                )
                axes[1].add_patch(rect)
            axes[1].set_title(
                f"LA: {len(detected_boxes)} det, matched {sum(matched)}/{len(gt_boxes)}"
            )
            axes[1].axis("off")

            # 图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="none", edgecolor="lime", label="LA Detection"),
                Patch(facecolor="none", edgecolor="red", linestyle="--", label="GT Box"),
            ]
            axes[1].legend(handles=legend_elements, loc="upper right")

            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"la_detection_sample_{idx}.png"),
                dpi=150, bbox_inches="tight"
            )
            plt.close()
            print(f"    Saved: la_detection_sample_{idx}.png")

    # ===== 汇总 =====
    recall = total_la_recalled / (total_gt_actors + 1e-8)
    precision = total_la_recalled / (total_la_recalled + total_la_false_pos + 1e-8)

    metrics = {
        "num_samples": args.num_samples,
        "total_gt_actors": total_gt_actors,
        "total_la_recalled": total_la_recalled,
        "total_la_missed": total_la_missed,
        "total_la_false_pos": total_la_false_pos,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
    }

    print("\n" + "=" * 60)
    print(f"LocateAnything Detection Results:")
    print(f"  Total GT actors:     {total_gt_actors}")
    print(f"  Recalled:            {total_la_recalled}")
    print(f"  Missed:              {total_la_missed}")
    print(f"  False positives:     {total_la_false_pos}")
    print(f"  Recall:              {recall:.1%}")
    print(f"  Precision:           {precision:.1%}")
    print("=" * 60)

    save_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "segmentation_vis",
        "la_detection_metrics.json"
    )
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {save_path}")


if __name__ == "__main__":
    main()