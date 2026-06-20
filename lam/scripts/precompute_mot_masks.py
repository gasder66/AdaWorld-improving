"""
预生成 A2D 的 MOT masks (YOLO+BoT-SORT, 无需 GT 标注)。

对训练集和测试集的每个样本，运行 YOLO+BoT-SORT → bbox → filled mask，
保存为 .pt 文件，供后续 V6c 训练使用。

用法:
  CUDA_VISIBLE_DEVICES=2 python precompute_mot_masks.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from ultralytics import YOLO
from lam.a2d_dataset import A2DDataset


def bbox_to_mask(bbox_xyxy, img_size=256):
    mask = torch.zeros(img_size, img_size)
    x1, y1, x2, y2 = bbox_xyxy.round().astype(int).tolist()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_size, x2), min(img_size, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return mask


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # YOLO (A2D-trained)
    yolo = YOLO('/home/xiaojy/projects/AdaWorld-improving/result/yolo_a2d/yolov8n_a2d/weights/best.pt')

    # 优化的 BoT-SORT 配置
    import yaml
    cfg = {
        'tracker_type': 'botsort',
        'track_high_thresh': 0.5,
        'track_low_thresh': 0.3,
        'new_track_thresh': 0.6,
        'track_buffer': 15,
        'match_thresh': 0.85,
        'fuse_score': True,
        'gmc_method': 'sparseOptFlow',
        'proximity_thresh': 0.5,
        'appearance_thresh': 0.8,
        'with_reid': False,
    }
    cfg_path = '/tmp/botsort_optimized.yaml'
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f)

    out_dir = '/home/xiaojy/projects/AdaWorld-improving/result/v6_a2d/mot_masks'
    os.makedirs(out_dir, exist_ok=True)

    for split in ['train', 'test']:
        dataset = A2DDataset(
            data_root='/home/xiaojy/projects/AdaWorld-improving/data/a2d',
            release_root='/home/xiaojy/projects/AdaWorld-improving/Release',
            split=split, num_frames=2, max_actors=4, img_size=256,
        )

        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        total_time = 0
        for idx in range(len(dataset)):
            t0 = time.time()
            sample = dataset[idx]
            video_np = sample['videos'].numpy()  # (T, H, W, C)
            T = video_np.shape[0]
            vid = sample.get('video_id', f'sample_{idx}')

            # YOLO+BoT-SORT for all frames
            all_boxes = []
            for t in range(T):
                img = (video_np[t] * 255).clip(0, 255).astype(np.uint8)
                results = yolo.track(img, persist=(t > 0), verbose=False,
                                     conf=0.5, tracker=cfg_path)
                boxes = results[0].boxes
                if boxes is not None:
                    all_boxes.append(boxes)
                else:
                    all_boxes.append(None)

            # Convert to masks
            H, W = 256, 256
            pred_masks = torch.zeros(T, 4, H, W)
            if len(all_boxes) > 0 and all_boxes[0] is not None and all_boxes[0].id is not None:
                ids_0 = all_boxes[0].id.cpu().numpy().astype(int)
                xyxy_0 = all_boxes[0].xyxy.cpu().numpy()
                for i, tid in enumerate(ids_0):
                    slot = tid - 1
                    if 0 <= slot < 4:
                        pred_masks[0, slot] = bbox_to_mask(xyxy_0[i], H)

            for t in range(1, T):
                if all_boxes[t] is not None and all_boxes[t].id is not None:
                    ids_t = all_boxes[t].id.cpu().numpy().astype(int)
                    xyxy_t = all_boxes[t].xyxy.cpu().numpy()
                    for i, tid in enumerate(ids_t):
                        slot = tid - 1
                        if 0 <= slot < 4:
                            pred_masks[t, slot] = bbox_to_mask(xyxy_t[i], H)

            # Save
            save_path = os.path.join(split_dir, f"mot_masks_{idx:06d}.pt")
            torch.save({
                "mot_masks": pred_masks,
                "video_id": vid,
                "num_actors": sample['num_actors'],
            }, save_path)

            elapsed = time.time() - t0
            total_time += elapsed
            if idx % 50 == 0:
                avg = total_time / max(idx + 1, 1)
                remaining = avg * (len(dataset) - idx - 1)
                print(f"  [{split}] {idx}/{len(dataset)} - {avg:.2f}s/sample, "
                      f"ETA {remaining/60:.1f}min")

        print(f"  [{split}] Done. {len(dataset)} samples, {total_time:.0f}s total")


if __name__ == "__main__":
    main()
