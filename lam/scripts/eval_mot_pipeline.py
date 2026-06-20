"""
MOT → V5 重建完整评估流水线。

流程:
  1. YOLO+ByteTrack 检测每帧主体 → track IDs
  2. bbox → filled rectangle mask
  3. 按 track ID 对齐到 V5 的 slot 索引
  4. V5 用预测 masks 重建
  5. 对比 GT masks 与预测 masks 的 PSNR

用法:
  CUDA_VISIBLE_DEVICES=3 python eval_mot_pipeline.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from ultralytics import YOLO
from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset


def bbox_to_mask(bbox, img_size=256):
    """将 YOLO bbox (x1,y1,x2,y2) 转换为 filled rectangle mask."""
    mask = torch.zeros(img_size, img_size, device='cpu')
    x1, y1, x2, y2 = bbox.round().astype(int).tolist()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_size, x2), min(img_size, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return mask


def run_mot_pipeline(model, yolo_model, videos, masks_gt, num_actors_gt):
    """
    对一段视频执行 MOT 检测 + V5 重建。

    Returns:
        recon_pred: 用预测 masks 的 V5 重建
        recon_gt:   用 GT masks 的 V5 重建
        mot_metrics: 检测统计
    """
    B, T, H, W, C = videos.shape  # (B, T, H, W, C) from dataset
    max_actors = masks_gt.shape[2]
    pred_masks = torch.zeros(B, T, max_actors, H, W, device=videos.device)

    # YOLO+BoT-SORT 逐帧检测 (比 ByteTrack 更稳定)
    for t in range(T):
        img = (videos[0, t].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        results = yolo_model.track(img, persist=(t > 0), verbose=False,
                                    conf=0.5, tracker='botsort.yaml')
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        for i, tid in enumerate(ids):
            slot = tid - 1
            if slot < max_actors:
                pred_masks[0, t, slot] = bbox_to_mask(xyxy[i], H).to(videos.device)

    # V5: 用预测 masks 重建
    with torch.no_grad():
        recon_pred = model({"videos": videos, "masks": pred_masks})
        recon_gt = model({"videos": videos, "masks": masks_gt})

    mse_pred = float(((videos[:, 1:] - recon_pred["recon"]) ** 2).mean())
    mse_gt = float(((videos[:, 1:] - recon_gt["recon"]) ** 2).mean())
    psnr_pred = -10 * np.log10(mse_pred + 1e-10)
    psnr_gt = -10 * np.log10(mse_gt + 1e-10)

    # 检测统计
    total_gt = (masks_gt[:, :, :num_actors_gt[0]].sum(dim=(-2, -1)) > 0.5).sum().item()
    total_pred = (pred_masks[:, :, :num_actors_gt[0]].sum(dim=(-2, -1)) > 0.5).sum().item()
    missed = max(0, total_gt - total_pred)

    return recon_pred, recon_gt, {
        "psnr_pred": psnr_pred, "psnr_gt": psnr_gt,
        "total_gt": total_gt, "total_pred": total_pred, "missed": missed,
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # YOLO
    yolo = YOLO('/home/xiaojy/projects/AdaWorld-improving/result/yolo_synthetic/yolov8n_synthetic/weights/best.pt')

    # V6 (with background slot + structured losses)
    lam = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8, max_actors=4,
        keep_background=True, use_obj_st_attention=True,
        free_bits_lambda=0.1,
    ).to(device)
    ckpt = '/home/xiaojy/projects/AdaWorld-improving/result/v6_structured/model_v6_conservative.pt'
    lam.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    lam.eval()
    for p in lam.parameters():
        p.requires_grad_(False)

    dataset = DiskSyntheticDataset(
        os.path.join(os.path.dirname(__file__), "..", "..", "data",
                     "synthetic_multi_actor", "val"),
        num_frames=5, output_format="t h w c",
    )

    results = []
    for idx in range(500):
        sample = dataset[idx]
        videos = sample["videos"].unsqueeze(0).to(device)
        masks = sample["masks"].unsqueeze(0).to(device)
        num_actors = torch.tensor([sample["num_actors"]])

        r = run_mot_pipeline(lam, yolo, videos, masks, num_actors)
        results.append(r[2])

        if idx % 100 == 0:
            print(f"  [{idx}/500] PSNR: GT={r[2]['psnr_gt']:.2f}, MOT={r[2]['psnr_pred']:.2f}, "
                  f"missed={r[2]['missed']}/{r[2]['total_gt']}")

    # 汇总
    psnr_gt_list = [r["psnr_gt"] for r in results]
    psnr_pred_list = [r["psnr_pred"] for r in results]
    total_missed = sum(r["missed"] for r in results)
    total_gt = sum(r["total_gt"] for r in results)

    print(f"\n{'='*50}")
    print(f"MOT Pipeline Evaluation (n=500)")
    print(f"  GT masks PSNR:      {np.mean(psnr_gt_list):.2f} ± {np.std(psnr_gt_list):.2f} dB")
    print(f"  MOT masks PSNR:     {np.mean(psnr_pred_list):.2f} ± {np.std(psnr_pred_list):.2f} dB")
    print(f"  Detection rate:     {(1 - total_missed/total_gt)*100:.2f}%")
    print(f"  Missed:             {total_missed}/{total_gt}")
    print(f"{'='*50}")

    out_dir = '/home/xiaojy/projects/AdaWorld-improving/result/v5_maskedpool/mot_eval'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mot_eval_results.json"), "w") as f:
        json.dump({
            "psnr_gt_mean": float(np.mean(psnr_gt_list)),
            "psnr_gt_std": float(np.std(psnr_gt_list)),
            "psnr_mot_mean": float(np.mean(psnr_pred_list)),
            "psnr_mot_std": float(np.std(psnr_pred_list)),
            "detection_rate": float((1 - total_missed/total_gt)*100),
        }, f, indent=2)
    print(f"Saved: {out_dir}/mot_eval_results.json")
    print("Done!")


if __name__ == "__main__":
    main()
