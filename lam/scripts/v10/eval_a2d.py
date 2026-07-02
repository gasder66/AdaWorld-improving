"""
A2D MOT → V6c 评估流水线。

流程:
  1. A2DDataset 加载视频帧 + GT bbox masks
  2. YOLOv8n (A2D-trained) + BoT-SORT 检测每帧 → track IDs
  3. bbox → filled rectangle mask
  4. V6c 用 GT masks 和 MOT masks 分别重建
  5. 对比 PSNR

用法:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python eval_a2d_mot.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
from ultralytics import YOLO
from lam.modules import LatentActionModel
from lam.a2d_dataset import A2DDataset


def bbox_to_mask(bbox_xyxy, img_size=256):
    mask = torch.zeros(img_size, img_size)
    x1, y1, x2, y2 = bbox_xyxy.round().astype(int).tolist()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_size, x2), min(img_size, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return mask


def run_mot_on_video(yolo_model, video_frames_np, max_actors=4):
    """对一段视频执行 YOLO+BoT-SORT，返回预测 masks (T, max_actors, H, W)。"""
    T, H, W, C = video_frames_np.shape
    pred_masks = torch.zeros(T, max_actors, H, W)

    for t in range(T):
        img = (video_frames_np[t] * 255).clip(0, 255).astype(np.uint8)
        results = yolo_model.track(img, persist=(t > 0), verbose=False,
                                    conf=0.3, tracker='botsort.yaml')
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        for i, tid in enumerate(ids):
            slot = tid - 1
            if 0 <= slot < max_actors:
                pred_masks[t, slot] = bbox_to_mask(xyxy[i], H)

    return pred_masks


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # YOLO (A2D-trained)
    yolo = YOLO('/home/xiaojy/projects/AdaWorld-improving/result/yolo_a2d/yolov8n_a2d/weights/best.pt')

    # V6c model
    lam = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8, max_actors=4,
        keep_background=True, use_obj_st_attention=True,
        free_bits_lambda=0.1,
    ).to(device)
    ckpt = '/home/xiaojy/projects/AdaWorld-improving/result/v6_a2d/model_v6c_a2d.pt'
    lam.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    lam.eval()
    for p in lam.parameters():
        p.requires_grad_(False)

    # A2D dataset
    dataset = A2DDataset(
        data_root='/home/xiaojy/projects/AdaWorld-improving/data/a2d',
        release_root='/home/xiaojy/projects/AdaWorld-improving/Release',
        split='test', num_frames=5, max_actors=4, img_size=256,
    )

    print(f"\n{'='*60}")
    print(f"A2D MOT → V6c Evaluation")
    print(f"  Samples: {len(dataset)}")
    print(f"  YOLO: A2D-trained yolov8n (mAP50=0.625)")
    print(f"  Tracker: BoT-SORT")
    print(f"  LAM: V6c A2D-trained (PSNR=16.2 dB on A2D)")
    print(f"{'='*60}")

    results = []
    for idx in range(min(len(dataset), 50)):
        sample = dataset[idx]
        videos = sample["videos"].unsqueeze(0).to(device)  # (1, T, H, W, C)
        masks_gt = sample["masks"].unsqueeze(0).to(device)  # (1, T, A, H, W)
        num_actors = int(sample["num_actors"])
        video_id = sample.get("video_id", str(idx))

        # Run MOT
        video_np = sample["videos"].numpy()  # (T, H, W, C)
        pred_masks = run_mot_on_video(yolo, video_np, max_actors=4).to(device)
        pred_masks = pred_masks.unsqueeze(0)  # (1, T, A, H, W)

        # V6c: GT masks reconstruction
        with torch.no_grad():
            out_gt = lam({"videos": videos, "masks": masks_gt})
            out_mot = lam({"videos": videos, "masks": pred_masks})

        gt = videos[:, 1:]
        mse_gt = float(((gt - out_gt["recon"]) ** 2).mean())
        mse_mot = float(((gt - out_mot["recon"]) ** 2).mean())
        psnr_gt = -10 * np.log10(mse_gt + 1e-10)
        psnr_mot = -10 * np.log10(mse_mot + 1e-10)

        # Detection stats
        gt_det = (masks_gt[0, :, :num_actors].sum(dim=(-2, -1)) > 0.5).sum().item()
        mot_det = (pred_masks[0, :, :num_actors].sum(dim=(-2, -1)) > 0.5).sum().item()

        results.append({
            "idx": idx, "video_id": video_id, "num_actors": num_actors,
            "psnr_gt": psnr_gt, "psnr_mot": psnr_mot,
            "gt_det": gt_det, "mot_det": mot_det,
        })

        if idx % 10 == 0:
            print(f"  [{idx+1}/{min(len(dataset),50)}] {video_id}: "
                  f"PSNR GT={psnr_gt:.2f}, MOT={psnr_mot:.2f}, "
                  f"det GT={gt_det}, MOT={mot_det}, actors={num_actors}")

    # Summary
    psnr_gt_list = [r["psnr_gt"] for r in results]
    psnr_mot_list = [r["psnr_mot"] for r in results]
    total_gt_det = sum(r["gt_det"] for r in results)
    total_mot_det = sum(r["mot_det"] for r in results)

    print(f"\n{'='*60}")
    print(f"A2D MOT Pipeline Summary (n={len(results)})")
    print(f"  GT masks PSNR:    {np.mean(psnr_gt_list):.2f} ± {np.std(psnr_gt_list):.2f} dB")
    print(f"  MOT masks PSNR:   {np.mean(psnr_mot_list):.2f} ± {np.std(psnr_mot_list):.2f} dB")
    det_rate = total_mot_det / max(total_gt_det, 1) * 100
    print(f"  Detection rate:   {det_rate:.1f}% (MOT={total_mot_det}, GT={total_gt_det})")
    print(f"{'='*60}")

    out_dir = '/home/xiaojy/projects/AdaWorld-improving/result/v6_a2d/mot_eval'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "a2d_mot_results.json"), "w") as f:
        json.dump({
            "n_samples": len(results),
            "psnr_gt_mean": float(np.mean(psnr_gt_list)),
            "psnr_gt_std": float(np.std(psnr_gt_list)),
            "psnr_mot_mean": float(np.mean(psnr_mot_list)),
            "psnr_mot_std": float(np.std(psnr_mot_list)),
            "detection_rate": float(det_rate),
            "per_sample": results,
        }, f, indent=2, default=str)
    print(f"Saved: {out_dir}/a2d_mot_results.json")
    print("Done!")


if __name__ == "__main__":
    main()
