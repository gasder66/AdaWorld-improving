"""
V8 Stage 4: YOLO-detected A2D Box Dataset.

用 YOLOv8n 检测 A2D 视频任意帧,不受 GT 标注帧限制。
大幅增加训练样本 (~28000 vs 248)。

输出 V8 slot table 格式:
    videos:        (T, H, W, 3) float32 [0,1]
    boxes:         (T, K, 4) float32 [x1,y1,x2,y2]
    track_ids:     (K,) long
    actor_labels:  (K,) long  # A2D actor type (1-7, mapped from COCO)
    valid_mask:    (T, K) bool
    actions:       (T-1, K) long  # A2D action (0-7), -1=无标注/invalid
    num_actors:    int

Action labels:
  - 如果采样帧恰为 A2D 标注帧,从 .mat 读取 GT action
  - 否则 action=-1 (训练用,不用于 NMI eval)
"""
import os
import csv
import pickle
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

# COCO → A2D actor type mapping (expanded)
# A2D: 1=adult, 2=baby, 3=ball, 4=bird, 5=car, 6=cat, 7=dog
# COCO: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck,
#        13=kite, 14=bird, 15=cat, 16=dog, 17=horse, 18=sheep, 19=cow
COCO_TO_A2D = {
    0: 1,   # person → adult
    2: 5,   # car → car
    3: 5,   # motorcycle → car (A2D has no motorcycle)
    5: 5,   # bus → car
    7: 5,   # truck → car
    14: 4,  # bird → bird
    15: 6,  # cat → cat
    16: 7,  # dog → dog
    32: 3,  # sports ball → ball
}
NUM_ACTOR_TYPES = 7

YOLO_MODEL = None


def _get_yolo():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        from ultralytics import YOLO
        YOLO_MODEL = YOLO("yolov8n.pt")
    return YOLO_MODEL


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-8) if (aa + ab - inter) > 0 else 0.0


def _track_boxes(boxes_per_frame, types_per_frame, max_actors, iou_thresh=0.3):
    """IoU matching across frames → slot assignments."""
    T = len(boxes_per_frame)
    K = max_actors
    slot_boxes = np.zeros((T, K, 4), dtype=np.float32)
    slot_types = np.full(K, -1, dtype=np.int64)
    slot_valid = np.zeros((T, K), dtype=bool)
    slot_ids = np.arange(K, dtype=np.int64)

    n0 = min(len(boxes_per_frame[0]), K)
    for i in range(n0):
        slot_boxes[0, i] = boxes_per_frame[0][i]
        slot_types[i] = types_per_frame[0][i]
        slot_valid[0, i] = True

    for t in range(1, T):
        curr = boxes_per_frame[t]
        curr_types = types_per_frame[t]
        matched = [False] * len(curr)
        for k in range(K):
            if not slot_valid[t - 1, k]:
                continue
            best_iou, best_j = iou_thresh, -1
            for j in range(len(curr)):
                if matched[j]:
                    continue
                v = _iou(slot_boxes[t - 1, k], curr[j])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                slot_boxes[t, k] = curr[best_j]
                slot_valid[t, k] = True
                slot_types[k] = curr_types[best_j]
                matched[best_j] = True
        for j in range(len(curr)):
            if matched[j]:
                continue
            for k in range(K):
                if not slot_valid[:, k].any():
                    slot_boxes[t, k] = curr[j]
                    slot_valid[t, k] = True
                    slot_types[k] = curr_types[j]
                    break
    return slot_boxes, slot_types, slot_valid, slot_ids


class YOLOBoxDataset(Dataset):
    """YOLO-detected A2D dataset with optional GT action labels on annotated frames."""

    def __init__(
        self,
        video_dir: str,
        release_root: str,
        split: str = "train",
        T: int = 5,
        stride: int = 10,
        max_actors: int = 4,
        img_size: int = 256,
        cache_dir: str = None,
        yolo_conf: float = 0.25,
        max_samples: int = None,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.split = split
        self.T = T
        self.stride = stride
        self.max_actors = max_actors
        self.img_size = img_size
        self.yolo_conf = yolo_conf
        self.cache_dir = cache_dir

        # Load videoset.csv for GT annotation mapping
        self.annot_dir = os.path.join(release_root, "Annotations", "mat")
        self.video_info = {}
        csv_path = os.path.join(release_root, "videoset.csv")
        with open(csv_path, "r") as f:
            for row in csv.reader(f):
                vid = row[0]
                self.video_info[vid] = {
                    "height": int(row[4]), "width": int(row[5]),
                    "usage": int(row[8]),
                }

        # Scan videos
        self.video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

        # Build sample list: (video_idx, start_frame)
        self.samples = []
        for vidx, vf in enumerate(self.video_files):
            vpath = os.path.join(video_dir, vf)
            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total <= 0:
                continue
            max_start = total - 1 - (T - 1) * stride
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, stride):
                self.samples.append((vidx, start))
            if max_samples and len(self.samples) >= max_samples:
                break

        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"  [YOLOBoxDataset] {len(self.samples)} samples from {len(self.video_files)} videos "
              f"(split={split}, T={T}, stride={stride})")

        # Cache setup
        if cache_dir:
            os.makedirs(os.path.join(cache_dir, "dets"), exist_ok=True)

        # Pre-build annotation frame lookup: vid → {frame_num: (bbox, actors, actions)}
        self._annot_cache = {}

    def _get_gt_annot(self, vid: str, frame_num: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get GT annotation for a specific frame if it's an annotated frame."""
        if vid not in self._annot_cache:
            vid_dir = os.path.join(self.annot_dir, vid)
            if not os.path.isdir(vid_dir):
                self._annot_cache[vid] = {}
            else:
                import h5py
                cache = {}
                for mf in os.listdir(vid_dir):
                    if not mf.endswith(".mat"):
                        continue
                    fn = int(mf.replace(".mat", ""))
                    try:
                        with h5py.File(os.path.join(vid_dir, mf), "r") as f:
                            bbox = np.array(f["reBBox"]).T  # (N, 4)
                            ids = np.array(f["id"]).flatten()
                        info = self.video_info.get(vid, {"height": 256, "width": 256})
                        sx, sy = self.img_size / info["width"], self.img_size / info["height"]
                        bbox[:, 0] = np.clip(bbox[:, 0] * sx, 0, self.img_size)
                        bbox[:, 1] = np.clip(bbox[:, 1] * sy, 0, self.img_size)
                        bbox[:, 2] = np.clip(bbox[:, 2] * sx, 0, self.img_size)
                        bbox[:, 3] = np.clip(bbox[:, 3] * sy, 0, self.img_size)
                        actors = np.array([int(i // 10) for i in ids], dtype=np.int64)
                        actions = np.array([int(i % 10) for i in ids], dtype=np.int64)
                        cache[fn] = (bbox.astype(np.float32), actors, actions)
                    except Exception:
                        continue
                self._annot_cache[vid] = cache
        return self._annot_cache[vid].get(frame_num)

    def _detect_frame(self, vpath: str, frame_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """YOLO detect single frame → (boxes (N,4), actor_types (N,))."""
        cache_key = f"{os.path.basename(vpath)}_f{frame_num}.pkl"
        cache_path = os.path.join(self.cache_dir, "dets", cache_key) if self.cache_dir else None
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        cap = cv2.VideoCapture(vpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

        frame = cv2.resize(frame, (self.img_size, self.img_size))
        yolo = _get_yolo()
        results = yolo(frame, verbose=False, conf=self.yolo_conf)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            out = (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64))
        else:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            cls = boxes.cls.cpu().numpy().astype(int)
            # Filter to COCO classes that map to A2D
            keep = []
            for i in range(len(cls)):
                if cls[i] in COCO_TO_A2D:
                    keep.append(i)
            if keep:
                xyxy = xyxy[keep]
                actor_types = np.array([COCO_TO_A2D[cls[i]] for i in keep], dtype=np.int64)
            else:
                xyxy = np.zeros((0, 4), dtype=np.float32)
                actor_types = np.zeros(0, dtype=np.int64)
            out = (xyxy, actor_types)

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(out, f)
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vidx, start_frame = self.samples[idx]
        vpath = os.path.join(self.video_dir, self.video_files[vidx])
        vid = self.video_files[vidx].replace(".mp4", "")

        frames = []
        boxes_per_frame = []
        types_per_frame = []
        frame_nums = []

        for t in range(self.T):
            fn = start_frame + t * self.stride
            frame_nums.append(fn)
            # Load video frame
            cap = cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            # YOLO detect
            det_boxes, det_types = self._detect_frame(vpath, fn)
            boxes_per_frame.append(det_boxes)
            types_per_frame.append(det_types)

        # IoU tracking
        slot_boxes, slot_types, slot_valid, slot_ids = _track_boxes(
            boxes_per_frame, types_per_frame, self.max_actors
        )

        # Get action labels: try GT annotation for each frame
        slot_actions = np.full((self.T, self.max_actors), -1, dtype=np.int64)
        for t in range(self.T):
            gt = self._get_gt_annot(vid, frame_nums[t])
            if gt is None:
                continue
            gt_bbox, gt_actors, gt_actions = gt
            # Match YOLO slots to GT boxes by IoU
            for k in range(self.max_actors):
                if not slot_valid[t, k]:
                    continue
                best_iou, best_j = 0.3, -1
                for j in range(len(gt_bbox)):
                    v = _iou(slot_boxes[t, k], gt_bbox[j])
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_j >= 0:
                    act = gt_actions[best_j]
                    slot_actions[t, k] = act - 1 if 1 <= act <= 8 else -1
                    # Also update actor type from GT (more reliable than COCO mapping)
                    slot_types[k] = gt_actors[best_j]

        # Action for transition t (t→t+1) = action at frame t+1
        actions_out = slot_actions[1:]  # (T-1, K)

        videos = torch.from_numpy(np.stack(frames)).float() / 255.0
        boxes = torch.from_numpy(slot_boxes)
        valid_mask = torch.from_numpy(slot_valid)
        track_ids = torch.from_numpy(slot_ids)
        actor_labels = torch.from_numpy(slot_types)
        actions_tensor = torch.from_numpy(actions_out)
        num_actors = int(slot_valid[0].sum())

        return {
            "videos": videos,
            "boxes": boxes,
            "track_ids": track_ids,
            "actor_labels": actor_labels,
            "valid_mask": valid_mask,
            "actions": actions_tensor,
            "num_actors": num_actors,
            "video_id": vid,
        }
