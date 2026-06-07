"""
A2D 数据集加载器 — 支持 Mask-Guided LAM 训练。

从 A2D 标注中提取：
- 视频帧 (T, H, W, C) float32 [0,1]
- bbox 矩形填充 mask (T, max_actors, H, W)
- actor-action labels (T-1, max_actors) long
- actor IDs (用于跨帧跟踪)

A2D 标注格式 (h5py):
- reBBox: (4, N) — [x_min, y_min, x_max, y_max]
- id: (1, N) — actor-action label ID (两位数: 十位=actor, 个位=action)
- class: (1, N) — 字符串标签如 "adult-jumping"
- reMask: (N, H, W) — 像素级分割 mask (可选)

用法:
  dataset = A2DDataset(data_root, release_root, split="train", num_frames=5)
  sample = dataset[0]
"""

import os
import csv
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset
import h5py

# A2D actor/action 定义
ACTOR_NAMES = {1: "adult", 2: "baby", 3: "ball", 4: "bird", 5: "car", 6: "cat", 7: "dog"}
ACTION_NAMES = {
    1: "climbing", 2: "crawling", 3: "eating", 4: "flying",
    5: "jumping", 6: "rolling", 7: "running", 8: "walking", 9: "none"
}

# 我们只关心 action 部分（1-8），none(9) 视为无效
NUM_ACTIONS = 8  # climbing, crawling, eating, flying, jumping, rolling, running, walking
MAX_ACTORS = 8


def parse_a2d_label(label_id: int) -> Tuple[int, int]:
    """解析 A2D 两位数 label ID 为 (actor_id, action_id)。"""
    actor_id = label_id // 10
    action_id = label_id % 10
    return actor_id, action_id


class A2DDataset(Dataset):
    """
    A2D 数据集加载器，输出格式与 DiskSyntheticDataset 兼容。

    输出:
        "videos":    (T, H, W, C) float32 [0,1]
        "masks":     (T, max_actors, H, W) float32 binary (bbox 矩形填充)
        "actions":   (T-1, max_actors) long (-1 = padding)
        "num_actors": int
        "video_id":  str
    """

    def __init__(
            self,
            data_root: str,
            release_root: str,
            split: str = "train",
            num_frames: int = 5,
            max_actors: int = MAX_ACTORS,
            img_size: int = 256,
            use_pixel_mask: bool = False,
            min_actors: int = 1,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.release_root = release_root
        self.split = split
        self.num_frames = num_frames
        self.max_actors = max_actors
        self.img_size = img_size
        self.use_pixel_mask = use_pixel_mask
        self.min_actors = min_actors

        # 读取 videoset.csv
        csv_path = os.path.join(release_root, "videoset.csv")
        self.video_info = {}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                vid = row[0]
                label = int(row[1])
                height = int(row[4])
                width = int(row[5])
                num_frames_total = int(row[6])
                num_annotated = int(row[7])
                usage = int(row[8])  # 0=train, 1=test
                self.video_info[vid] = {
                    "label": label,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames_total,
                    "num_annotated": num_annotated,
                    "usage": usage,
                }

        # 筛选 split
        split_flag = 0 if split == "train" else 1
        self.video_ids = [
            vid for vid, info in self.video_info.items()
            if info["usage"] == split_flag
        ]

        # 标注目录
        self.annot_dir = os.path.join(release_root, "Annotations", "mat")

        # 视频目录
        self.video_dir = os.path.join(data_root, split)

        # 预处理：找出有足够标注帧的视频
        self.valid_samples = self._find_valid_samples()
        print(f"  [A2DDataset] {len(self.valid_samples)} valid samples from {len(self.video_ids)} videos "
              f"(split={split}, min_actors={min_actors})")

    def _find_valid_samples(self) -> List[Dict]:
        """找出有足够标注帧的样本。"""
        valid = []
        for vid in self.video_ids:
            info = self.video_info[vid]
            annot_vid_dir = os.path.join(self.annot_dir, vid)

            if not os.path.isdir(annot_vid_dir):
                continue

            # 获取标注帧列表
            mat_files = sorted([f for f in os.listdir(annot_vid_dir) if f.endswith(".mat")])
            if len(mat_files) < 2:
                continue

            # 检查视频文件
            video_path = os.path.join(self.video_dir, f"{vid}.mp4")
            if not os.path.exists(video_path):
                continue

            # 读取标注帧的帧号
            frame_nums = [int(f.replace(".mat", "")) for f in mat_files]

            # 滑动窗口生成样本
            for start_idx in range(0, len(mat_files) - self.num_frames + 1, self.num_frames):
                end_idx = start_idx + self.num_frames
                if end_idx > len(mat_files):
                    break

                sample_mats = mat_files[start_idx:end_idx]
                sample_frames = frame_nums[start_idx:end_idx]

                # 检查是否有足够的主体
                try:
                    first_mat = os.path.join(annot_vid_dir, sample_mats[0])
                    with h5py.File(first_mat, "r") as f:
                        n_actors = f["reBBox"].shape[1]
                    if n_actors < self.min_actors:
                        continue
                except Exception:
                    continue

                valid.append({
                    "video_id": vid,
                    "mat_files": sample_mats,
                    "frame_nums": sample_frames,
                    "n_actors_first": n_actors,
                })

        return valid

    def __len__(self) -> int:
        return len(self.valid_samples)

    def _load_frame(self, vid: str, frame_num: int) -> np.ndarray:
        """从视频中加载指定帧。"""
        import cv2
        video_path = os.path.join(self.video_dir, f"{vid}.mp4")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # 0-indexed
        ret, frame = cap.read()
        cap.release()

        if not ret:
            # 返回黑色帧
            frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Resize
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _load_annotation(self, vid: str, mat_file: str) -> Dict:
        """加载单帧标注。"""
        mat_path = os.path.join(self.annot_dir, vid, mat_file)
        with h5py.File(mat_path, "r") as f:
            bbox = np.array(f["reBBox"])  # (4, N) [x_min, y_min, x_max, y_max]
            label_ids = np.array(f["id"]).flatten()  # (N,)
            n_actors = bbox.shape[1]

            # 解析 actor-action labels
            actors = []
            actions = []
            for i in range(n_actors):
                actor_id, action_id = parse_a2d_label(int(label_ids[i]))
                actors.append(actor_id)
                actions.append(action_id)

            # 可选：加载像素级 mask
            masks = None
            if self.use_pixel_mask and "reMask" in f:
                masks = np.array(f["reMask"])  # (N, H, W)

        return {
            "bbox": bbox,  # (4, N)
            "label_ids": label_ids,  # (N,)
            "actors": actors,
            "actions": actions,
            "masks": masks,
            "n_actors": n_actors,
        }

    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.valid_samples[idx]
        vid = sample_info["video_id"]
        mat_files = sample_info["mat_files"]
        frame_nums = sample_info["frame_nums"]

        # 获取原始图像尺寸（用于 bbox 缩放）
        info = self.video_info[vid]
        orig_h, orig_w = info["height"], info["width"]

        T = len(mat_files)
        frames = []
        all_masks = []
        all_actions = []
        all_actors = []
        max_n = 0

        for t in range(T):
            # 加载帧
            frame = self._load_frame(vid, frame_nums[t])
            frames.append(frame)

            # 加载标注
            annot = self._load_annotation(vid, mat_files[t])
            n_actors = annot["n_actors"]
            max_n = max(max_n, n_actors)

            # 生成 bbox 矩形填充 mask
            mask_t = np.zeros((self.max_actors, self.img_size, self.img_size), dtype=np.float32)
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h

            for a in range(min(n_actors, self.max_actors)):
                x1 = int(np.clip(annot["bbox"][0, a] * scale_x, 0, self.img_size))
                y1 = int(np.clip(annot["bbox"][1, a] * scale_y, 0, self.img_size))
                x2 = int(np.clip(annot["bbox"][2, a] * scale_x, 0, self.img_size))
                y2 = int(np.clip(annot["bbox"][3, a] * scale_y, 0, self.img_size))
                mask_t[a, y1:y2, x1:x2] = 1.0

            all_masks.append(mask_t)

            # Action labels (action_id 1-8 → 0-7, none(9) → -1)
            actions_t = np.full(self.max_actors, -1, dtype=np.int64)
            for a in range(min(n_actors, self.max_actors)):
                action_id = annot["actions"][a]
                if 1 <= action_id <= 8:
                    actions_t[a] = action_id - 1  # 0-indexed
                else:
                    actions_t[a] = -1  # none/invalid
            all_actors.append(annot["actors"][:min(n_actors, self.max_actors)])

            if t > 0:
                all_actions.append(actions_t)

        # Pad if needed
        while len(all_actions) < T - 1:
            all_actions.append(np.full(self.max_actors, -1, dtype=np.int64))

        videos = torch.from_numpy(np.stack(frames)).float() / 255.0  # (T, H, W, C)
        masks = torch.from_numpy(np.stack(all_masks))  # (T, max_actors, H, W)
        actions = torch.from_numpy(np.stack(all_actions[:T-1]))  # (T-1, max_actors)

        return {
            "videos": videos,
            "masks": masks,
            "actions": actions,
            "num_actors": min(max_n, self.max_actors),
            "video_id": vid,
        }
