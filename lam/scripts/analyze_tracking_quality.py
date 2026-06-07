"""
分析 YOLOv8 + ByteTrack 的跟踪质量问题。

检查：
1. 检测率：每帧检测到多少对象 vs GT
2. ID 一致性：同一 GT 主体在不同帧是否保持相同 track ID
3. ID 切换：哪些帧发生了 ID 切换
4. 漏检分析：哪些主体被漏检，什么条件下漏检

用法:
  CUDA_VISIBLE_DEVICES=3 python analyze_tracking_quality.py --gpu 0 --num_samples 50
"""
import os
import sys
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.disk_synthetic_dataset import DiskSyntheticDataset


def compute_iou(box1, box2):
    """计算两个 bbox 的 IoU。box: [x1, y1, x2, y2]"""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def extract_gt_boxes(masks):
    """从 GT mask 提取 bbox。masks: (A, H, W)"""
    A, H, W = masks.shape
    boxes = []
    for a in range(A):
        ys, xs = torch.where(masks[a] > 0.5)
        if len(ys) > 0:
            boxes.append([xs.min().item(), ys.min().item(),
                          xs.max().item(), ys.max().item()])
        else:
            boxes.append(None)
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--yolo_weights", type=str, default=None)
    args = parser.parse_args()

    data_root = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
    )

    if args.yolo_weights is None:
        args.yolo_weights = os.path.join(
            os.path.dirname(__file__), "..", "results", "yolo_tracking",
            "yolov8n_synthetic", "weights", "best.pt"
        )

    # 加载 YOLO
    from ultralytics import YOLO
    yolo_model = YOLO(args.yolo_weights)

    # 加载数据集
    dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"), num_frames=5, output_format="t h w c",
    )

    print(f"\n{'='*60}")
    print(f"Tracking Quality Analysis")
    print(f"  YOLO weights: {args.yolo_weights}")
    print(f"  Conf threshold: {args.conf}")
    print(f"  Num samples: {args.num_samples}")
    print(f"{'='*60}")

    # 统计
    total_frames = 0
    total_gt_boxes = 0
    total_det_boxes = 0
    total_matched = 0  # GT 被 det 匹配到的数量
    total_id_switches = 0  # ID 切换次数
    total_id_consistent = 0  # ID 一致跟踪数

    # 逐样本分析
    sample_details = []

    for idx in tqdm(range(min(args.num_samples, len(dataset))),
                    desc="Analyzing tracking"):
        sample = dataset[idx]
        videos = sample["videos"]  # (T, H, W, C)
        gt_masks = sample["masks"]  # (T, A, H, W)
        T, H, W, C = videos.shape
        A = gt_masks.shape[1]
        num_actors = sample["num_actors"]

        # 逐帧跟踪
        # gt_id_to_track_id[t][gt_a] = track_id (如果匹配)
        gt_id_to_track_id = {}
        prev_gt_to_track = {}

        for t in range(T):
            frame = videos[t].numpy()
            frame_uint8 = (frame * 255).astype(np.uint8)

            # YOLO + ByteTrack
            results = yolo_model.track(
                frame_uint8,
                tracker="bytetrack.yaml",
                persist=True,
                conf=args.conf,
                iou=0.45,
                verbose=False,
            )

            # GT boxes
            gt_boxes = extract_gt_boxes(gt_masks[t])
            gt_boxes_valid = [(a, b) for a, b in enumerate(gt_boxes) if b is not None]

            # 检测结果
            if results[0].boxes.id is not None:
                det_boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
            else:
                det_boxes = np.array([]).reshape(0, 4)
                track_ids = np.array([])

            total_frames += 1
            total_gt_boxes += len(gt_boxes_valid)
            total_det_boxes += len(det_boxes)

            # 匹配 GT → det (基于 IoU)
            gt_id_to_track_id[t] = {}
            used_det = set()

            for gt_a, gt_box in gt_boxes_valid:
                best_iou = 0
                best_j = -1
                for j in range(len(det_boxes)):
                    if j in used_det:
                        continue
                    iou = compute_iou(gt_box, det_boxes[j].tolist())
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_iou >= 0.3 and best_j >= 0:
                    gt_id_to_track_id[t][gt_a] = int(track_ids[best_j])
                    used_det.add(best_j)
                    total_matched += 1

            # 检查 ID 一致性
            if t > 0:
                for gt_a in gt_id_to_track_id[t]:
                    if gt_a in prev_gt_to_track:
                        prev_tid = prev_gt_to_track[gt_a]
                        curr_tid = gt_id_to_track_id[t][gt_a]
                        if prev_tid == curr_tid:
                            total_id_consistent += 1
                        else:
                            total_id_switches += 1

            prev_gt_to_track = gt_id_to_track_id[t].copy()

        # 样本统计
        sample_det_rate = total_matched / (total_gt_boxes + 1e-8) if total_gt_boxes > 0 else 0
        sample_details.append({
            "idx": idx,
            "num_actors": num_actors,
            "gt_boxes": total_gt_boxes,
            "det_boxes": total_det_boxes,
            "matched": total_matched,
        })

    # 汇总
    det_rate = total_matched / (total_gt_boxes + 1e-8)
    id_switch_rate = total_id_switches / (total_id_consistent + total_id_switches + 1e-8)

    print(f"\n{'='*60}")
    print(f"Tracking Quality Results")
    print(f"{'='*60}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total GT boxes: {total_gt_boxes}")
    print(f"  Total detections: {total_det_boxes}")
    print(f"  Matched (IoU≥0.3): {total_matched}")
    print(f"  Detection rate: {det_rate:.1%}")
    print(f"  ID consistent: {total_id_consistent}")
    print(f"  ID switches: {total_id_switches}")
    print(f"  ID switch rate: {id_switch_rate:.1%}")
    print(f"  Avg det per frame: {total_det_boxes/total_frames:.1f}")
    print(f"  Avg GT per frame: {total_gt_boxes/total_frames:.1f}")

    # 保存
    results = {
        "conf_threshold": args.conf,
        "num_samples": args.num_samples,
        "total_frames": total_frames,
        "total_gt_boxes": total_gt_boxes,
        "total_det_boxes": total_det_boxes,
        "total_matched": total_matched,
        "detection_rate": round(det_rate, 4),
        "id_consistent": total_id_consistent,
        "id_switches": total_id_switches,
        "id_switch_rate": round(id_switch_rate, 4),
    }

    save_dir = os.path.join(os.path.dirname(__file__), "..", "results", "yolo_tracking")
    save_path = os.path.join(save_dir, f"tracking_quality_conf{args.conf}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
