"""生成指定实验配置的 A2D YOLO MOT 数据集 (one-time precompute)."""
import sys, os, pickle, torch, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultralytics import YOLO
import yaml

yolo = YOLO("/home/xiaojy/projects/AdaWorld-improving/yolov8n.pt")
cfg = {"tracker_type":"botsort","track_high_thresh":0.5,"track_low_thresh":0.3,
       "new_track_thresh":0.6,"track_buffer":15,"match_thresh":0.85,
       "fuse_score":True,"gmc_method":"sparseOptFlow",
       "proximity_thresh":0.5,"appearance_thresh":0.8,"with_reid":False}
cfg_path = "/tmp/botsort_gen.yaml"
with open(cfg_path, "w") as f: yaml.dump(cfg, f)

video_dir = "/home/xiaojy/projects/AdaWorld-improving/data/a2d/train"
video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
out_dir = "/home/xiaojy/projects/AdaWorld-improving/result/v6_a2d/experiments"
os.makedirs(out_dir, exist_ok=True)

# Experiments: (name, T, stride)
experiments = [("E1_T2_S30", 2, 30), ("E2_T5_S10", 5, 10), ("E3_T10_S5", 10, 5)]

for exp_name, T, stride in experiments:
    print(f"\n=== Generating {exp_name}: T={T}, stride={stride} ===")
    all_samples = []
    used_videos = 0

    for vi, vf in enumerate(video_files):
        vpath = os.path.join(video_dir, vf)
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total is None or total <= 0:
            continue

        max_start = total - 1 - (T - 1) * stride
        if max_start < 0:
            continue

        # Take up to 3 windows per video
        windows = 0
        for start in range(0, max_start + 1, stride):
            if windows >= 3:
                break
            frames_bgr = []
            prev_masks = None
            all_masks = None

            for t in range(T):
                fn = start + t * stride
                cap = cv2.VideoCapture(vpath)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ret, frame_bgr = cap.read()
                cap.release()
                if not ret:
                    break
                frame_bgr = cv2.resize(frame_bgr, (256, 256))
                frames_bgr.append(frame_bgr)

                # YOLO detection with IoU tracking
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = yolo.track(frame_rgb, persist=(t > 0), verbose=False,
                                      conf=0.5, tracker=cfg_path)
                masks = torch.zeros(4, 256, 256)
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
                                masks[slot, y1:y2, x1:x2] = 1.0
                elif boxes is not None:
                    # No IDs: per-frame assignment
                    xyxy = boxes.xyxy.cpu().numpy()
                    for i, box in enumerate(xyxy[:4]):
                        x1,y1,x2,y2 = box.round().astype(int).tolist()
                        x1,y1=max(0,x1),max(0,y1); x2,y2=min(256,x2),min(256,y2)
                        if x2>x1 and y2>y1:
                            masks[i, y1:y2, x1:x2] = 1.0

                if t == 0:
                    all_masks = torch.zeros(T, 4, 256, 256)
                all_masks[t] = masks

            if len(frames_bgr) == T and all_masks is not None:
                frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
                videos = torch.from_numpy(np.stack(frames_rgb)).float() / 255.0
                all_samples.append({"videos": videos, "masks": all_masks})
                windows += 1

            if vi % 200 == 0 and windows == 0:
                print(f"  [{vi}/{len(video_files)}] {vf}: {total} frames, max_start={max_start}")

        if windows > 0:
            used_videos += 1

    out_path = os.path.join(out_dir, f"{exp_name}.pt")
    torch.save({"samples": all_samples, "config": {"T": T, "stride": stride}}, out_path)
    print(f"  Saved {len(all_samples)} samples from {used_videos} videos -> {out_path}")

print("\nDone!")
