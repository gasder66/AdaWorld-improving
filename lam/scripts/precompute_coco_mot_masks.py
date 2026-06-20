"""预生成 COCO 预训练 YOLO 的 MOT masks (零样本, 无需任何 A2D 标注)。"""
import os, sys, time, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import numpy as np
from ultralytics import YOLO
from lam.a2d_dataset import A2DDataset

yolo = YOLO("/home/xiaojy/projects/AdaWorld-improving/yolov8n.pt")
cfg = {"tracker_type":"botsort","track_high_thresh":0.5,"track_low_thresh":0.3,
       "new_track_thresh":0.6,"track_buffer":15,"match_thresh":0.85,
       "fuse_score":True,"gmc_method":"sparseOptFlow",
       "proximity_thresh":0.5,"appearance_thresh":0.8,"with_reid":False}
cfg_path = "/tmp/botsort_opt.yaml"
with open(cfg_path, "w") as f: yaml.dump(cfg, f)

for split in ["train", "test"]:
    dataset = A2DDataset(data_root="data/a2d", release_root="Release",
                         split=split, num_frames=2, max_actors=4, img_size=256)
    out_dir = f"/home/xiaojy/projects/AdaWorld-improving/result/v6_a2d/mot_masks_coco/{split}"
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        video_np = sample["videos"].numpy()
        T = video_np.shape[0]
        pred_masks = torch.zeros(T, 4, 256, 256)

        for t in range(T):
            img = (video_np[t]*255).clip(0,255).astype(np.uint8)
            results = yolo.track(img, persist=(t>0), verbose=False, conf=0.5, tracker=cfg_path)
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                for i, tid in enumerate(ids):
                    slot = tid - 1
                    if 0 <= slot < 4:
                        x1,y1,x2,y2 = xyxy[i].round().astype(int).tolist()
                        x1,y1=max(0,x1),max(0,y1); x2,y2=min(256,x2),min(256,y2)
                        if x2>x1 and y2>y1:
                            pred_masks[t, slot, y1:y2, x1:x2] = 1.0

        torch.save({"mot_masks": pred_masks}, os.path.join(out_dir, f"mot_masks_{idx:06d}.pt"))
        if idx % 200 == 0:
            print(f"[{split}] {idx}/{len(dataset)}")
    print(f"[{split}] Done. {len(dataset)} samples")
