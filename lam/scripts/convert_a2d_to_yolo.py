"""
将 A2D 数据集转换为 YOLO 训练格式。

A2D actors: adult(1), baby(2), ball(3), bird(4), car(5), cat(6), dog(7)
A2D 标注中的 reBBox 是 (4, N) 的 [x_min, y_min, x_max, y_max] 格式。

生成 YOLO 格式：
- images/: *.jpg (来自视频帧)
- labels/: *.txt (class x_center y_center width height)

用法:
  python convert_a2d_to_yolo.py [--max_samples_per_class 100]
"""
import os
import sys
import argparse
import cv2
import numpy as np
import h5py
from tqdm import tqdm


def parse_videoset(csv_path):
    """解析 videoset.csv，返回 {vid: info} 字典。"""
    import csv
    info = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vid = row[0]
            usage = int(row[8])  # 0=train, 1=test
            info[vid] = {'usage': usage}
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/xiaojy/projects/AdaWorld-improving/data/a2d')
    parser.add_argument('--release_root', type=str, default='/home/xiaojy/projects/AdaWorld-improving/Release')
    parser.add_argument('--output_root', type=str, default='/home/xiaojy/projects/AdaWorld-improving/data/a2d_yolo')
    parser.add_argument('--max_frames', type=int, default=2000, help='每类最多使用多少帧')
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    # 读取 videoset.csv 获取 train/test split
    csv_path = os.path.join(args.release_root, 'videoset.csv')
    video_info = parse_videoset(csv_path)

    # 标注目录
    annot_dir = os.path.join(args.release_root, 'Annotations', 'mat')

    # 统计 actor 类别的帧计数（用于平衡采样）
    actor_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

    for split_name, split_flag in [('train', 0), ('test', 1)]:
        images_dir = os.path.join(args.output_root, split_name, 'images')
        labels_dir = os.path.join(args.output_root, split_name, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        frame_idx = 0
        total_images = 0
        total_boxes = 0

        for vid in tqdm(os.listdir(annot_dir), desc=f'Converting {split_name}'):
            if vid not in video_info:
                continue
            if video_info[vid]['usage'] != split_flag:
                continue

            vid_annot_dir = os.path.join(annot_dir, vid)
            if not os.path.isdir(vid_annot_dir):
                continue

            video_path = os.path.join(args.data_root, split_name, f'{vid}.mp4')
            if not os.path.exists(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            mat_files = sorted([f for f in os.listdir(vid_annot_dir) if f.endswith('.mat')])

            for mat_file in mat_files:
                frame_num = int(mat_file.replace('.mat', ''))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame = cap.read()
                if not ret:
                    continue

                # 读取标注
                mat_path = os.path.join(vid_annot_dir, mat_file)
                try:
                    with h5py.File(mat_path, 'r') as f:
                        bbox = np.array(f['reBBox'])  # (4, N)
                        label_ids = np.array(f['id']).flatten()  # (N,)
                except Exception:
                    continue

                H, W = frame.shape[:2]

                # 为每个 actor 生成 YOLO 标注
                lines = []
                for i in range(len(label_ids)):
                    label_id = int(label_ids[i])
                    actor_id = label_id // 10
                    action_id = label_id % 10

                    if actor_id not in actor_counts:
                        continue

                    # 检查是否达到每类上限
                    if actor_counts[actor_id] >= args.max_frames:
                        continue

                    # YOLO class: actor_id - 1 (0-indexed)
                    cls = actor_id - 1

                    # reBBox: (4, N) → [x_min, y_min, x_max, y_max]
                    x1, y1, x2, y2 = bbox[0, i], bbox[1, i], bbox[2, i], bbox[3, i]

                    # 裁剪到图像范围内
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W, x2)
                    y2 = min(H, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # 归一化 YOLO 格式
                    x_center = ((x1 + x2) / 2) / W
                    y_center = ((y1 + y2) / 2) / H
                    width = (x2 - x1) / W
                    height = (y2 - y1) / H

                    lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    actor_counts[actor_id] += 1
                    total_boxes += 1

                if not lines:
                    continue

                # 保存图像
                img_name = f"frame_{frame_idx:06d}.jpg"
                img_path = os.path.join(images_dir, img_name)
                cv2.imwrite(img_path, frame)

                # 保存标注
                label_name = img_name.replace('.jpg', '.txt')
                label_path = os.path.join(labels_dir, label_name)
                with open(label_path, 'w') as f:
                    f.write('\n'.join(lines))

                frame_idx += 1
                total_images += 1

            cap.release()

        print(f"  {split_name}: {total_images} images, {total_boxes} boxes")

    # 生成 data.yaml
    # A2D actor 名
    A2D_ACTOR_NAMES = ['adult', 'baby', 'ball', 'bird', 'car', 'cat', 'dog']
    yaml_content = f"""# A2D YOLO dataset
path: {os.path.abspath(args.output_root)}
train: train/images
val: test/images

nc: 7
names: {A2D_ACTOR_NAMES}
"""
    yaml_path = os.path.join(args.output_root, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\n  data.yaml: {yaml_path}")
    print(f"  Actor counts: {actor_counts}")
    print(f"  Done!")


if __name__ == '__main__':
    main()
