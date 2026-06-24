"""
V8 Stage 2B: A2D Box Dataset.

从 A2D .mat 标注直接读取 bbox (而非 mask),输出 V8 slot table 格式:
    videos:        (T, H, W, 3) float32 [0,1]
    boxes:         (T, K, 4) float32 [x1,y1,x2,y2] in scaled pixel coords
    track_ids:     (K,) long  # IoU 匹配的 pseudo track id
    actor_labels:  (K,) long  # A2D actor type (1-7)
    valid_mask:    (T, K) bool
    actions:       (T-1, K) long  # A2D action (0-7), -1=invalid
    num_actors:    int

与 A2DDataset 的区别:
  - 返回 boxes 而非 masks
  - 返回 actor_labels (A2D actor type 1-7) 用于 actor conditioning
  - 返回 track_ids (IoU 匹配) 和 valid_mask
  - action 空间 8 类 (A2D) 而非 5 类 (合成)
"""
import os
import csv
from typing import Dict, List, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset
import h5py

ACTOR_NAMES = {1: "adult", 2: "baby", 3: "ball", 4: "bird", 5: "car", 6: "cat", 7: "dog"}
ACTION_NAMES = {
    1: "climbing", 2: "crawling", 3: "eating", 4: "flying",
    5: "jumping", 6: "rolling", 7: "running", 8: "walking", 9: "none"
}
NUM_ACTIONS = 8
NUM_ACTOR_TYPES = 7


def parse_a2d_label(label_id: int) -> Tuple[int, int]:
    actor_id = label_id // 10
    action_id = label_id % 10
    return actor_id, action_id


def _iou(box_a, box_b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


def _match_tracks(boxes_per_frame, actor_types_per_frame, actions_per_frame, max_actors, iou_threshold=0.3):
    """Greedy IoU matching across frames to assign pseudo track IDs.

    Args:
        boxes_per_frame: list of (N_t, 4) arrays
        actor_types_per_frame: list of (N_t,) arrays
        actions_per_frame: list of (N_t,) arrays (raw A2D action id)
        max_actors: int — pad output to this many slots
    Returns:
        slot_boxes: (T, max_actors, 4)
        slot_actors: (max_actors,)
        slot_actions: (T, max_actors)
        slot_valid: (T, max_actors)
        slot_track_ids: (max_actors,)
    """
    T = len(boxes_per_frame)
    K = max_actors

    slot_boxes = np.zeros((T, K, 4), dtype=np.float32)
    slot_actors = np.full(K, -1, dtype=np.int64)
    slot_actions = np.full((T, K), -1, dtype=np.int64)
    slot_valid = np.zeros((T, K), dtype=bool)
    slot_track_ids = np.arange(K, dtype=np.int64)

    def _act_idx(a):
        return a - 1 if 1 <= a <= 8 else -1

    # Frame 0: assign sequentially
    n0 = len(boxes_per_frame[0])
    for i in range(n0):
        slot_boxes[0, i] = boxes_per_frame[0][i]
        slot_actors[i] = actor_types_per_frame[0][i]
        slot_actions[0, i] = _act_idx(actions_per_frame[0][i])
        slot_valid[0, i] = True

    for t in range(1, T):
        prev_boxes = slot_boxes[t - 1]
        prev_valid = slot_valid[t - 1]
        curr_boxes = boxes_per_frame[t]
        curr_actors = actor_types_per_frame[t]
        curr_actions = actions_per_frame[t]
        n_curr = len(curr_boxes)

        matched = [False] * n_curr
        for k in range(K):
            if not prev_valid[k]:
                continue
            best_iou = iou_threshold
            best_j = -1
            for j in range(n_curr):
                if matched[j]:
                    continue
                iou = _iou(prev_boxes[k], curr_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                slot_boxes[t, k] = curr_boxes[best_j]
                slot_valid[t, k] = True
                slot_actors[k] = curr_actors[best_j]
                slot_actions[t, k] = _act_idx(curr_actions[best_j])
                matched[best_j] = True

        for j in range(n_curr):
            if matched[j]:
                continue
            for k in range(K):
                if not slot_valid[:, k].any():
                    slot_boxes[t, k] = curr_boxes[j]
                    slot_valid[t, k] = True
                    slot_actors[k] = curr_actors[j]
                    slot_actions[t, k] = _act_idx(curr_actions[j])
                    break

    return slot_boxes, slot_actors, slot_actions, slot_valid, slot_track_ids


class A2DBoxDataset(Dataset):
    """A2D Box Dataset for V8 Stage 2B."""

    def __init__(
        self,
        data_root: str,
        release_root: str,
        split: str = "train",
        num_frames: int = 5,
        max_actors: int = 4,
        img_size: int = 256,
        min_actors: int = 1,
        frame_stride: int = 1,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.release_root = release_root
        self.split = split
        self.num_frames = num_frames
        self.max_actors = max_actors
        self.img_size = img_size
        self.min_actors = min_actors
        self.frame_stride = frame_stride

        csv_path = os.path.join(release_root, "videoset.csv")
        self.video_info = {}
        with open(csv_path, "r") as f:
            for row in csv.reader(f):
                vid = row[0]
                self.video_info[vid] = {
                    "label": int(row[1]),
                    "height": int(row[4]),
                    "width": int(row[5]),
                    "num_frames": int(row[6]),
                    "usage": int(row[8]),
                }

        split_flag = 0 if split == "train" else 1
        self.video_ids = [vid for vid, info in self.video_info.items() if info["usage"] == split_flag]
        self.annot_dir = os.path.join(release_root, "Annotations", "mat")
        self.video_dir = os.path.join(data_root, split)
        self.valid_samples = self._find_valid_samples()
        print(f"  [A2DBoxDataset] {len(self.valid_samples)} valid samples (split={split})")

    def _find_valid_samples(self) -> List[Dict]:
        valid = []
        for vid in self.video_ids:
            info = self.video_info[vid]
            annot_vid_dir = os.path.join(self.annot_dir, vid)
            if not os.path.isdir(annot_vid_dir):
                continue
            mat_files = sorted([f for f in os.listdir(annot_vid_dir) if f.endswith(".mat")])
            if len(mat_files) < self.num_frames:
                continue
            video_path = os.path.join(self.video_dir, f"{vid}.mp4")
            if not os.path.exists(video_path):
                continue
            frame_nums = [int(f.replace(".mat", "")) for f in mat_files]
            num_needed = (self.num_frames - 1) * self.frame_stride + 1
            for start_idx in range(0, len(mat_files) - num_needed + 1):
                selected = [start_idx + i * self.frame_stride for i in range(self.num_frames)]
                if max(selected) >= len(mat_files):
                    break
                sample_mats = [mat_files[i] for i in selected]
                sample_frames = [frame_nums[i] for i in selected]
                try:
                    first_mat = os.path.join(annot_vid_dir, sample_mats[0])
                    with h5py.File(first_mat, "r") as f:
                        n_actors = f["reBBox"].shape[1]
                    if n_actors < self.min_actors:
                        continue
                except Exception:
                    continue
                valid.append({"video_id": vid, "mat_files": sample_mats, "frame_nums": sample_frames})
        return valid

    def __len__(self) -> int:
        return len(self.valid_samples)

    def _load_frame(self, vid: str, frame_num: int) -> np.ndarray:
        import cv2
        video_path = os.path.join(self.video_dir, f"{vid}.mp4")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _load_annot(self, vid: str, mat_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (bbox_scaled (N,4), actor_types (N,), actions (N,))."""
        mat_path = os.path.join(self.annot_dir, vid, mat_file)
        info = self.video_info[vid]
        orig_h, orig_w = info["height"], info["width"]
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        with h5py.File(mat_path, "r") as f:
            bbox = np.array(f["reBBox"]).T  # (N, 4) [x1,y1,x2,y2]
            ids = np.array(f["id"]).flatten()  # (N,)
        bbox[:, 0] = np.clip(bbox[:, 0] * scale_x, 0, self.img_size)
        bbox[:, 1] = np.clip(bbox[:, 1] * scale_y, 0, self.img_size)
        bbox[:, 2] = np.clip(bbox[:, 2] * scale_x, 0, self.img_size)
        bbox[:, 3] = np.clip(bbox[:, 3] * scale_y, 0, self.img_size)
        actors = np.array([int(i // 10) for i in ids], dtype=np.int64)
        actions = np.array([int(i % 10) for i in ids], dtype=np.int64)
        return bbox.astype(np.float32), actors, actions

    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.valid_samples[idx]
        vid = sample_info["video_id"]
        mat_files = sample_info["mat_files"]
        frame_nums = sample_info["frame_nums"]
        T = len(mat_files)

        frames = []
        boxes_per_frame = []
        actors_per_frame = []
        actions_per_frame = []

        for t in range(T):
            frame = self._load_frame(vid, frame_nums[t])
            frames.append(frame)
            bbox, actors, actions = self._load_annot(vid, mat_files[t])
            # Limit to max_actors
            n = min(len(bbox), self.max_actors)
            boxes_per_frame.append(bbox[:n])
            actors_per_frame.append(actors[:n])
            actions_per_frame.append(actions[:n])

        # IoU matching across frames
        slot_boxes, slot_actors, slot_actions, slot_valid, slot_track_ids = _match_tracks(
            boxes_per_frame, actors_per_frame, actions_per_frame, self.max_actors
        )

        K = self.max_actors

        # Action for transition t (t→t+1) = action at frame t+1
        actions_out = slot_actions[1:]  # (T-1, K)

        videos = torch.from_numpy(np.stack(frames)).float() / 255.0  # (T, H, W, 3)
        boxes = torch.from_numpy(slot_boxes)
        valid_mask = torch.from_numpy(slot_valid)
        track_ids = torch.from_numpy(slot_track_ids)
        actor_labels = torch.from_numpy(slot_actors)
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
