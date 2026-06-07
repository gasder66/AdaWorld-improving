"""
测试零样本 YOLO (COCO 预训练) 在 A2D 数据集上的检测效果。

核心问题：A2D 只标注了主要演员，但 YOLO 检测所有物体。
我们评估：
1. Recall：检测框覆盖 GT 框的比例（只要检测到就算成功）
2. 误检分析：YOLO 检测到但 A2D 未标注的物体（可能是真实物体）
3. 人工可视化检查

用法:
  CUDA_VISIBLE_DEVICES=2 python test_yolo_zeroshot_a2d.py --num_samples 100
"""
import os
import sys
import cv2
import numpy as np
import h5py
from tqdm import tqdm
from ultralytics import YOLO
import torch

# COCO 类别映射到 A2D actor
COCO_TO_A2D = {
    0: 1,    # person → adult
    2: 5,    # car → car
    15: 4,   # bird → bird
    16: 6,   # cat → cat
    17: 7,   # dog → dog
    32: 3,   # sports ball → ball
}
COCO_NAMES = {0: 'person', 2: 'car', 15: 'bird', 16: 'cat', 17: 'dog', 32: 'sports_ball'}
VALID_COCO_IDS = list(COCO_TO_A2D.keys())

A2D_ACTOR_NAMES = {1: 'adult', 2: 'baby', 3: 'ball', 4: 'bird', 5: 'car', 6: 'cat', 7: 'dog'}


def compute_iou(box1, box2):
    """box: [x1, y1, x2, y2]"""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def match_detections_to_gt(det_boxes, gt_boxes, iou_threshold=0.3):
    """
    将检测框匹配到 GT 框。
    返回：matched_gt (哪些 GT 被检测到), unmatched_det (哪些检测框未匹配 GT)
    """
    n_det = len(det_boxes)
    n_gt = len(gt_boxes)
    
    if n_det == 0 or n_gt == 0:
        return [], list(range(n_det))
    
    # 计算所有 IoU
    iou_matrix = np.zeros((n_det, n_gt))
    for i, det in enumerate(det_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(det, gt)
    
    # 贪婪匹配
    matched_gt = []
    unmatched_det = []
    
    for j in range(n_gt):
        max_iou = iou_matrix[:, j].max()
        if max_iou >= iou_threshold:
            matched_gt.append(j)
    
    for i in range(n_det):
        max_iou = iou_matrix[i, :].max()
        if max_iou < iou_threshold:
            unmatched_det.append(i)
    
    return matched_gt, unmatched_det


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou_thresh', type=float, default=0.3)
    parser.add_argument('--visualize', type=int, default=20, help='可视化样本数')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"零样本 YOLO (COCO) 在 A2D 上的检测测试")
    print(f"  样本数: {args.num_samples}")
    print(f"  conf: {args.conf}, IoU threshold: {args.iou_thresh}")
    print(f"{'='*60}")
    
    # 加载 COCO 预训练 YOLO
    model = YOLO('yolov8n.pt')
    
    # A2D 数据路径
    release_root = '/home/xiaojy/projects/AdaWorld-improving/Release'
    data_root = '/home/xiaojy/projects/AdaWorld-improving/data/a2d'
    annot_dir = os.path.join(release_root, 'Annotations', 'mat')
    
    # 读取 videoset.csv
    import csv
    video_info = {}
    csv_path = os.path.join(release_root, 'videoset.csv')
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vid = row[0]
            usage = int(row[8])  # 0=train, 1=test
            video_info[vid] = {'usage': usage, 'width': int(row[5]), 'height': int(row[4])}
    
    # 统计
    stats = {
        'total_frames': 0,
        'total_gt_boxes': 0,
        'matched_gt': 0,
        'total_det_boxes': 0,
        'unmatched_det': 0,
        'per_class_recall': {k: {'gt': 0, 'matched': 0} for k in A2D_ACTOR_NAMES.keys()},
    }
    
    # 可视化保存目录
    vis_dir = '/home/xiaojy/projects/AdaWorld-improving/lam/results/yolo_zeroshot_vis'
    os.makedirs(vis_dir, exist_ok=True)
    
    # 随机抽样视频
    train_vids = [vid for vid, info in video_info.items() if info['usage'] == 0]
    sample_vids = np.random.choice(train_vids, min(args.num_samples, len(train_vids)), replace=False)
    
    for vid in tqdm(sample_vids, desc="检测"):
        vid_annot_dir = os.path.join(annot_dir, vid)
        if not os.path.isdir(vid_annot_dir):
            continue
        
        video_path = os.path.join(data_root, 'train', f'{vid}.mp4')
        if not os.path.exists(video_path):
            continue
        
        cap = cv2.VideoCapture(video_path)
        mat_files = sorted([f for f in os.listdir(vid_annot_dir) if f.endswith('.mat')])
        
        # 只检查前 3 帧
        for mat_file in mat_files[:3]:
            frame_num = int(mat_file.replace('.mat', ''))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 加载 GT
            mat_path = os.path.join(vid_annot_dir, mat_file)
            try:
                with h5py.File(mat_path, 'r') as f:
                    gt_bbox = np.array(f['reBBox'])  # (4, N)
                    gt_ids = np.array(f['id']).flatten()
                    n_gt = gt_bbox.shape[1]
            except:
                continue
            
            # YOLO 检测（原始分辨率）
            results = model.predict(
                frame, conf=args.conf, iou=0.45, verbose=False,
                classes=VALID_COCO_IDS, imgsz=640,
            )
            
            det_boxes = []
            det_classes = []
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                for i, cls in enumerate(cls_ids):
                    if cls in COCO_TO_A2D:
                        det_boxes.append(boxes[i])
                        det_classes.append(COCO_TO_A2D[cls])
            
            # 匹配
            gt_boxes_list = [gt_bbox[:, j] for j in range(n_gt)]
            matched_gt_idx, unmatched_det_idx = match_detections_to_gt(
                det_boxes, gt_boxes_list, args.iou_thresh
            )
            
            stats['total_frames'] += 1
            stats['total_gt_boxes'] += n_gt
            stats['matched_gt'] += len(matched_gt_idx)
            stats['total_det_boxes'] += len(det_boxes)
            stats['unmatched_det'] += len(unmatched_det_idx)
            
            # 按类别统计 recall
            for j in range(n_gt):
                actor_id = int(gt_ids[j]) // 10
                if actor_id in stats['per_class_recall']:
                    stats['per_class_recall'][actor_id]['gt'] += 1
                    if j in matched_gt_idx:
                        stats['per_class_recall'][actor_id]['matched'] += 1
        
        cap.release()
    
    # 输出统计
    recall = stats['matched_gt'] / max(stats['total_gt_boxes'], 1)
    precision = stats['matched_gt'] / max(stats['total_det_boxes'], 1)
    extra_det_rate = stats['unmatched_det'] / max(stats['total_det_boxes'], 1)
    
    print(f"\n{'='*60}")
    print(f"检测结果统计")
    print(f"  总帧数: {stats['total_frames']}")
    print(f"  GT 框总数: {stats['total_gt_boxes']}")
    print(f"  检测框总数: {stats['total_det_boxes']}")
    print(f"  匹配 GT 数: {stats['matched_gt']}")
    print(f"  未匹配检测数: {stats['unmatched_det']}")
    print(f"\n  Recall: {recall:.2%} (检测到的 GT 比例)")
    print(f"  Precision: {precision:.2%} (检测框匹配 GT 的比例)")
    print(f"  Extra detections: {extra_det_rate:.2%} (检测到但未标注的物体)")
    
    print(f"\n  按类别 Recall:")
    for actor_id, data in stats['per_class_recall'].items():
        if data['gt'] > 0:
            rec = data['matched'] / data['gt']
            print(f"    {A2D_ACTOR_NAMES[actor_id]}: {rec:.2%} ({data['matched']}/{data['gt']})")
    
    print(f"{'='*60}\n")
    
    # 可视化部分样本
    if args.visualize > 0:
        print(f"\n可视化 {args.visualize} 个样本...")
        vis_vids = np.random.choice(sample_vids, min(args.visualize, len(sample_vids)), replace=False)
        
        for i, vid in enumerate(vis_vids):
            vid_annot_dir = os.path.join(annot_dir, vid)
            video_path = os.path.join(data_root, 'train', f'{vid}.mp4')
            cap = cv2.VideoCapture(video_path)
            
            mat_files = sorted([f for f in os.listdir(vid_annot_dir) if f.endswith('.mat')])
            mat_file = mat_files[0]
            frame_num = int(mat_file.replace('.mat', ''))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                continue
            
            # GT 框（绿色）
            mat_path = os.path.join(vid_annot_dir, mat_file)
            with h5py.File(mat_path, 'r') as f:
                gt_bbox = np.array(f['reBBox'])
                gt_ids = np.array(f['id']).flatten()
            
            for j in range(gt_bbox.shape[1]):
                x1, y1, x2, y2 = gt_bbox[:, j]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                actor_id = int(gt_ids[j]) // 10
                label = A2D_ACTOR_NAMES.get(actor_id, str(actor_id))
                cv2.putText(frame, f"GT:{label}", (int(x1), int(y1)-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 检测框（红色）
            results = model.predict(frame, conf=args.conf, verbose=False, classes=VALID_COCO_IDS)
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                for j, cls in enumerate(cls_ids):
                    if cls in COCO_TO_A2D:
                        x1, y1, x2, y2 = boxes[j]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        label = COCO_NAMES[cls]
                        cv2.putText(frame, f"Det:{label}({confs[j]:.2f})", 
                                   (int(x1), int(y2)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 保存
            vis_path = os.path.join(vis_dir, f"sample_{i:03d}_{vid}.jpg")
            cv2.imwrite(vis_path, frame)
        
        print(f"  可视化保存至: {vis_dir}")


if __name__ == "__main__":
    main()