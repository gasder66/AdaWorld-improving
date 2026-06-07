"""
微调 YOLOv8n 检测合成数据中的彩色方块，然后测试 ByteTrack 跟踪。

用法:
  CUDA_VISIBLE_DEVICES=3 python train_yolo_and_track.py --gpu 0 --epochs 10
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    parser.add_argument("--skip_train", action="store_true",
                        help="跳过训练，直接测试跟踪")
    args = parser.parse_args()

    data_yaml = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "yolo_synthetic", "data.yaml"
    )
    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "results", "yolo_tracking"
    )
    os.makedirs(results_dir, exist_ok=True)

    if not args.skip_train:
        # ===== 训练 YOLOv8n =====
        from ultralytics import YOLO

        print(f"\n{'='*60}")
        print(f"Training YOLOv8n on synthetic data")
        print(f"  Model: {args.yolo_model}")
        print(f"  Data: {data_yaml}")
        print(f"  Epochs: {args.epochs}")
        print(f"  GPU: {args.gpu}")
        print(f"{'='*60}")

        model = YOLO(args.yolo_model)
        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=256,
            batch=32,
            device=args.gpu,
            project=results_dir,
            name="yolov8n_synthetic",
            exist_ok=True,
            verbose=True,
        )

        best_model_path = os.path.join(
            results_dir, "yolov8n_synthetic", "weights", "best.pt"
        )
        print(f"\n  Best model: {best_model_path}")
    else:
        best_model_path = os.path.join(
            results_dir, "yolov8n_synthetic", "weights", "best.pt"
        )

    # ===== 测试检测 + ByteTrack 跟踪 =====
    print(f"\n{'='*60}")
    print(f"Testing YOLOv8n + ByteTrack tracking")
    print(f"  Model: {best_model_path}")
    print(f"{'='*60}")

    from ultralytics import YOLO
    import torch
    import numpy as np
    from PIL import Image

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from lam.disk_synthetic_dataset import DiskSyntheticDataset

    model = YOLO(best_model_path)

    # 加载验证集
    data_root = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
    )
    val_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"), num_frames=5, output_format="t h w c",
    )

    # 逐样本测试检测 + 跟踪
    total_gt_actors = 0
    total_detected = 0
    total_tracked = 0
    total_track_switches = 0

    num_test = min(20, len(val_dataset))

    for idx in range(num_test):
        sample = val_dataset[idx]
        videos = sample["videos"]  # (T, H, W, C)
        masks = sample["masks"]    # (T, A, H, W)
        T, H, W, C = videos.shape
        A = masks.shape[1]
        num_actors = sample["num_actors"]

        # GT actors
        total_gt_actors += num_actors

        # 逐帧检测 + 跟踪
        prev_track_ids = None
        for t in range(T):
            frame = videos[t].numpy()  # (H, W, C) float32
            frame_uint8 = (frame * 255).astype(np.uint8)

            # YOLO 检测 + ByteTrack 跟踪
            results = model.track(
                frame_uint8,
                tracker="bytetrack.yaml",
                persist=True,  # 保持跟踪状态
                conf=0.25,
                iou=0.45,
                verbose=False,
            )

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy()
                n_tracked = len(track_ids)
                total_tracked += n_tracked

                # 检查 ID 切换
                if prev_track_ids is not None:
                    # 简单统计：跟踪 ID 数量变化
                    if len(set(track_ids)) != len(set(prev_track_ids)):
                        total_track_switches += 1

                prev_track_ids = track_ids
            else:
                prev_track_ids = None

            # 检测数量（不含跟踪）
            total_detected += len(results[0].boxes)

    avg_det_per_frame = total_detected / (num_test * T)
    avg_tracked_per_frame = total_tracked / (num_test * T)

    print(f"\n  Results ({num_test} samples, {T} frames each):")
    print(f"  Avg GT actors per sample: {total_gt_actors / num_test:.1f}")
    print(f"  Avg detections per frame: {avg_det_per_frame:.1f}")
    print(f"  Avg tracked objects per frame: {avg_tracked_per_frame:.1f}")
    print(f"  Track ID switches: {total_track_switches}")

    # 保存结果
    import json
    track_results = {
        "num_test_samples": num_test,
        "frames_per_sample": T,
        "avg_gt_actors": total_gt_actors / num_test,
        "avg_detections_per_frame": avg_det_per_frame,
        "avg_tracked_per_frame": avg_tracked_per_frame,
        "track_id_switches": total_track_switches,
    }
    with open(os.path.join(results_dir, "tracking_results.json"), "w") as f:
        json.dump(track_results, f, indent=2)

    print(f"  Results saved: {os.path.join(results_dir, 'tracking_results.json')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
