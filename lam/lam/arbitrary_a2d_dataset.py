"""
Arbitrary Frame A2D Dataset — 用 YOLO+BoT-SORT 对视频任意帧生成 masks。

不依赖 A2D 的 GT 标注，可以从视频的任意位置取帧。
通过帧号缓存 (disk cache) 避免重复 YOLO 推理。

用法:
  dataset = ArbitraryA2DDataset(split="train", T=5, stride=10)
  sample = dataset[0]
"""
import os, pickle, yaml
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader


YOLO_MODEL = None
BOTSORT_CFG = None


def _get_yolo():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO("/home/xiaojy/projects/AdaWorld-improving/yolov8n.pt")
    return YOLO_MODEL


def _get_tracker_cfg():
    global BOTSORT_CFG
    if BOTSORT_CFG is None:
        cfg = {"tracker_type":"botsort","track_high_thresh":0.5,"track_low_thresh":0.3,
               "new_track_thresh":0.6,"track_buffer":15,"match_thresh":0.85,
               "fuse_score":True,"gmc_method":"sparseOptFlow",
               "proximity_thresh":0.5,"appearance_thresh":0.8,"with_reid":False}
        BOTSORT_CFG = "/tmp/botsort_arb.yaml"
        with open(BOTSORT_CFG, "w") as f: yaml.dump(cfg, f)
    return BOTSORT_CFG


class ArbitraryA2DDataset(Dataset):
    """用 YOLO 对视频任意帧生成 masks 的 A2D 数据集。

    Args:
        video_dir: A2D MP4 视频目录
        T: 每样本帧数
        stride: 帧间步长 (帧为单位)
        max_actors: 最大主体数
        img_size: resize 尺寸
        cache_dir: YOLO 推理缓存目录
        start_frame_offset: 从视频的第几帧开始采样
    """

    def __init__(
        self,
        video_dir: str,
        T: int = 5,
        stride: int = 10,
        max_actors: int = 4,
        img_size: int = 256,
        cache_dir: str = None,
        start_frame_offset: int = 0,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.T = T
        self.stride = stride
        self.max_actors = max_actors
        self.img_size = img_size
        self.start_frame_offset = start_frame_offset

        # 扫描视频文件
        self.video_files = sorted([
            f for f in os.listdir(video_dir) if f.endswith(".mp4")
        ])
        print(f"  [ArbitraryA2D] {len(self.video_files)} videos in {video_dir}")

        # 计算每个视频的有效窗口数
        self.samples = []
        for idx, vf in enumerate(self.video_files):
            vpath = os.path.join(video_dir, vf)
            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total is None or total <= 0:
                continue
            # +1 for 0-indexed, then need last frame < total
            max_start = total - 1 - (T - 1) * stride - start_frame_offset
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, max(1, stride)):
                self.samples.append((idx, start))

        print(f"  [ArbitraryA2D] {len(self.samples)} samples (T={T}, stride={stride})")

        # 缓存
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(os.path.join(cache_dir, "dets"), exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def _load_video_frame(self, video_path: str, frame_num: int) -> np.ndarray:
        """从视频加载指定帧."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _detect_frame(self, video_path: str, frame_num: int, prev_masks=None):
        """YOLO 检测单帧, 返回 masks (max_actors, H, W)."""
        cache_key = f"{os.path.basename(video_path)}_f{frame_num}"
        cache_path = None
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, "dets", f"{cache_key}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

        img = self._load_video_frame(video_path, frame_num)
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8) if img.dtype == np.float32 else img

        yolo = _get_yolo()
        results = yolo(img_uint8, verbose=False, conf=0.5)

        masks = torch.zeros(self.max_actors, self.img_size, self.img_size)
        boxes = results[0].boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            if len(xyxy) > 0:
                # Simple IoU matching with previous frame for ID consistency
                if prev_masks is not None and prev_masks.sum() > 0:
                    # Match each detection to the best previous slot by IoU
                    matched = set()
                    for slot in range(self.max_actors):
                        prev = prev_masks[slot]  # (H, W)
                        if prev.sum() < 10:
                            continue
                        best_iou, best_box = 0, None
                        best_idx = -1
                        prev_pts = torch.where(prev > 0.5)
                        if len(prev_pts[0]) == 0:
                            continue
                        prev_ymin, prev_ymax = prev_pts[0].min().item(), prev_pts[0].max().item()
                        prev_xmin, prev_xmax = prev_pts[1].min().item(), prev_pts[1].max().item()

                        for bi, box in enumerate(xyxy):
                            if bi in matched:
                                continue
                            x1, y1, x2, y2 = box
                            # IoU with previous
                            inter_x1 = max(x1, prev_xmin)
                            inter_y1 = max(y1, prev_ymin)
                            inter_x2 = min(x2, prev_xmax)
                            inter_y2 = min(y2, prev_ymax)
                            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                                iou = 0
                            else:
                                inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                                box_area = (x2 - x1) * (y2 - y1)
                                prev_area = (prev_xmax - prev_xmin) * (prev_ymax - prev_ymin)
                                iou = inter / (box_area + prev_area - inter + 1e-6)
                            if iou > best_iou:
                                best_iou = iou
                                best_box = box
                                best_idx = bi
                        if best_iou > 0.3 and best_idx >= 0:
                            self._fill_mask(masks, slot, best_box)
                            matched.add(best_idx)

                    # Unmatched boxes go to first empty slots
                    for bi, box in enumerate(xyxy):
                        if bi not in matched:
                            for slot in range(self.max_actors):
                                if masks[slot].sum() < 10:
                                    self._fill_mask(masks, slot, box)
                                    matched.add(bi)
                                    break
                else:
                    # First frame: assign sequentially
                    for i, box in enumerate(xyxy[:self.max_actors]):
                        self._fill_mask(masks, i, box)

        if cache_path and not os.path.exists(cache_path):
            with open(cache_path, "wb") as f:
                pickle.dump(masks, f)

        return masks

    def _fill_mask(self, masks, slot, box):
        x1, y1, x2, y2 = box.round().astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.img_size, x2), min(self.img_size, y2)
        if x2 > x1 and y2 > y1:
            masks[slot, y1:y2, x1:x2] = 1.0

    def __getitem__(self, idx):
        vidx, start_frame = self.samples[idx]
        vpath = os.path.join(self.video_dir, self.video_files[vidx])

        frames = []
        prev_masks = None

        for t in range(self.T):
            frame_num = start_frame + t * self.stride + self.start_frame_offset
            img = self._load_video_frame(vpath, frame_num)
            frames.append(img)

            # 每帧 YOLO 检测
            cur_masks = self._detect_frame(vpath, frame_num, prev_masks)
            if t == 0:
                all_masks = torch.zeros(self.T, self.max_actors, self.img_size, self.img_size)
            all_masks[t] = cur_masks
            prev_masks = cur_masks

        videos = torch.from_numpy(np.stack(frames)).float() / 255.0
        return {"videos": videos, "masks": all_masks, "num_actors": self.max_actors}
