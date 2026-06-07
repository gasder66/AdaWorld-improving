"""
验证 SAM2 的 box→mask 质量。

用 GT bounding box 作为 prompt，让 SAM2 生成 mask，
然后与 GT mask 比较 IoU。

用法:
  CUDA_VISIBLE_DEVICES=2 python verify_sam2_mask.py
"""
import os
import sys
import json

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lam.disk_synthetic_dataset import DiskSyntheticDataset
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def extract_gt_boxes_from_masks(masks):
    """从 GT mask 中提取 bounding box。"""
    A, H, W = masks.shape
    boxes = []
    for a in range(A):
        ys, xs = torch.where(masks[a] > 0.5)
        if len(ys) == 0:
            boxes.append(None)
        else:
            boxes.append([xs.min().item(), ys.min().item(),
                          xs.max().item() + 1, ys.max().item() + 1])
    return boxes


def compute_mask_iou(pred_mask, gt_mask):
    """计算两个 binary mask 的 IoU。"""
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return intersection / (union + 1e-8)


def main():
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

    # 加载 SAM2
    print("Loading SAM2...")
    sam2_model = build_sam2("sam2_hiera_l.yaml").to(device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("  SAM2 loaded")

    # ===== 验证 =====
    num_samples = 20
    all_ious = []
    all_box_fill_ious = []  # 对比：矩形填充的 IoU

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        videos = sample["videos"]  # (T, H, W, C) float32 [0,1]
        masks = sample["masks"]    # (T, A, H, W) float32
        num_actors = sample["num_actors"]

        # 取第一帧
        frame = videos[0].numpy()  # (H, W, C) float32 [0,1]
        frame_uint8 = (frame * 255).astype(np.uint8)
        H, W = frame.shape[:2]

        # GT boxes
        gt_boxes = extract_gt_boxes_from_masks(masks[0])
        gt_masks = masks[0]  # (A, H, W)

        # SAM2 预测
        predictor.set_image(frame_uint8)

        for a in range(num_actors):
            if gt_boxes[a] is None:
                continue

            box = np.array(gt_boxes[a])  # [x1, y1, x2, y2]

            # SAM2 从 box prompt 生成 mask
            with torch.no_grad():
                pred_masks, scores, _ = predictor.predict(
                    box=box,
                    multimask_output=True,
                )
            # 选择最佳 mask（最高分数）
            best_idx = scores.argmax()
            pred_mask = pred_masks[best_idx]  # (H, W) bool

            # GT mask
            gt_mask = gt_masks[a].numpy() > 0.5  # (H, W) bool

            # IoU
            iou = compute_mask_iou(pred_mask, gt_mask)
            all_ious.append(iou)

            # 矩形填充 IoU（对比基线）
            box_fill = np.zeros((H, W), dtype=bool)
            x1, y1, x2, y2 = [int(v) for v in gt_boxes[a]]
            box_fill[y1:y2, x1:x2] = True
            box_fill_iou = compute_mask_iou(box_fill, gt_mask)
            all_box_fill_ious.append(box_fill_iou)

        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{num_samples} samples, "
                  f"SAM2 IoU: {np.mean(all_ious):.3f}, "
                  f"Box fill IoU: {np.mean(all_box_fill_ious):.3f}")

    # ===== 汇总 =====
    sam2_mean = np.mean(all_ious)
    sam2_std = np.std(all_ious)
    box_mean = np.mean(all_box_fill_ious)
    box_std = np.std(all_box_fill_ious)

    print("\n" + "=" * 60)
    print(f"SAM2 Box→Mask Quality ({len(all_ious)} masks):")
    print(f"  SAM2 IoU:     {sam2_mean:.3f} ± {sam2_std:.3f}")
    print(f"  Box fill IoU: {box_mean:.3f} ± {box_std:.3f}")
    print(f"  Improvement:  +{(sam2_mean - box_mean):.3f}")
    print("=" * 60)

    # 保存结果
    results = {
        "num_samples": num_samples,
        "num_masks": len(all_ious),
        "sam2_iou_mean": round(float(sam2_mean), 4),
        "sam2_iou_std": round(float(sam2_std), 4),
        "box_fill_iou_mean": round(float(box_mean), 4),
        "box_fill_iou_std": round(float(box_std), 4),
    }

    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "results", "segmentation_vis"
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "sam2_mask_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {os.path.join(save_dir, 'sam2_mask_metrics.json')}")


if __name__ == "__main__":
    main()