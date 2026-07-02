"""生成指定 stride 的 A2D T=2 YOLO MOT 数据集。"""

import sys, os, torch, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultralytics import YOLO; import yaml

STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 30
print(f"=== Generating stride={STRIDE} dataset ===")

yolo = YOLO("/home/xiaojy/projects/AdaWorld-improving/yolov8n.pt")
cfg = {"tracker_type":"botsort","track_high_thresh":0.5,"track_low_thresh":0.3,
       "new_track_thresh":0.6,"track_buffer":15,"match_thresh":0.85,
       "fuse_score":True,"gmc_method":"sparseOptFlow",
       "proximity_thresh":0.5,"appearance_thresh":0.8,"with_reid":False}
with open("/tmp/bts_gen.yaml","w") as f: yaml.dump(cfg, f)

video_dir = "/home/xiaojy/projects/AdaWorld-improving/data/a2d/train"
videos = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])[:200]

out_dir = "/home/xiaojy/projects/AdaWorld-improving/result/v6_a2d/experiments"
os.makedirs(out_dir, exist_ok=True)

samples = []
for vi, vf in enumerate(videos):
    cap = cv2.VideoCapture(os.path.join(video_dir, vf))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()

    # Need at least start + stride < total
    max_windows = (total - 1) // STRIDE
    if max_windows < 1: continue

    for start in [0, max_windows // 2 * STRIDE]:
        fn0 = start; fn1 = start + STRIDE
        if fn1 >= total: continue

        frames = []
        masks_list = []
        for t, fn in enumerate([fn0, fn1]):
            cap = cv2.VideoCapture(os.path.join(video_dir, vf))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, f = cap.read(); cap.release()
            if not ret: break
            f = cv2.cvtColor(cv2.resize(f, (256,256)), cv2.COLOR_BGR2RGB)
            frames.append(f)

            r = yolo(f, verbose=False, conf=0.5)
            m = torch.zeros(4, 256, 256)
            if r[0].boxes is not None:
                xyxy = r[0].boxes.xyxy.cpu().numpy()
                for i, b in enumerate(xyxy[:4]):
                    x1,y1,x2,y2 = b.round().astype(int).tolist()
                    x1,y1=max(0,x1),max(0,y1); x2,y2=min(256,x2),min(256,y2)
                    if x2>x1 and y2>y1: m[i,y1:y2,x1:x2]=1.0
            masks_list.append(m)

        if len(frames) == 2:
            v = torch.from_numpy(np.stack(frames)).float()/255.0
            masks = torch.stack(masks_list)
            samples.append({"videos": v, "masks": masks})

    if vi % 50 == 0:
        print(f"  [{vi}/{len(videos)}] {len(samples)} samples")

path = os.path.join(out_dir, f"exp_stride{STRIDE}.pt")
torch.save({"samples": samples, "config": {"T": 2, "stride": STRIDE}}, path)
print(f"Saved {len(samples)} samples -> {path}")
