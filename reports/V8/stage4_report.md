# V8 Stage 4: YOLO Zero-shot MOT — Report

## Summary

Stage 4 用 YOLOv8n 检测 A2D 视频任意帧,不受 GT 标注帧限制。
训练样本量从 248 增至 2000,**Action NMI 提升 4.6 倍** (0.067→0.31)。

## Results

| Metric | Stage 2B (GT bbox, 248) | **Stage 4 (YOLO, 2000)** | Chance |
|---|---|---|---|
| Action NMI | 0.0673 | **0.3118** | ~0 |
| Action Probe | 0.2222 | **0.6667** | 0.125 |
| ARI | 0.0111 | **0.0726** | ~0 |
| Actor leakage | 0.1667 | 0.5556 | 0.200 |
| z_actor variance | 0.3533 | 0.2875 | — |
| z_bg variance | 0.3140 | 0.3304 | — |
| dbbox MSE | 926.10 | 872.91 | — |
| Train samples | 248 | **2000** | — |
| Eval samples (labeled) | 266 | 86 | — |

### 关键发现

1. **Action NMI 0.31** (4.6x over GT bbox): 数据量是最大瓶颈,2000 samples 足以让
   V8 学到有意义的 action 结构。Action probe 66.7% (5.3x chance) 确认 z_actor
   编码了 action 信息。

2. **Actor leakage 0.56**: 高于 GT bbox 的 0.17。原因是 YOLO 的 COCO→A2D 映射
   本身就把 actor type 编码进了 actor_labels (person→adult, dog→dog 等)。
   z_actor 通过 motion pattern 间接保留了 type 信息 (dog 的运动模式不同于 car)。

3. **z_bg variance 0.33**: z_bg 在真实视频上活跃,吸收相机运动。

4. **Eval 样本少 (86)**: 因为 GT action labels 只在 A2D 标注帧上,而 YOLO 检测
   的帧大部分不是标注帧。增加 eval 样本需要更多标注帧覆盖。

## Architecture

### YOLOBoxDataset
- `lam/lam/yolo_box_dataset.py`
- YOLOv8n 检测任意帧 → boxes (xyxy) + COCO cls
- COCO→A2D 映射: person→adult, car/truck/bus/motorcycle→car, bird→bird, cat→cat, dog→dog, sports ball→ball
- IoU matching 跨帧 → track_ids (greedy, threshold=0.3)
- GT action labels: 在标注帧上,通过 IoU 匹配 YOLO box → GT box 获取 action
- 非标注帧 action=-1 (训练用 motion_loss,不用于 NMI eval)
- Disk cache: `result/v8_mot_lam/yolo_cache/dets/`

### 预计算 + 训练流程
1. 预计算 YOLO 检测 (101s for 2000 train + 14s for 200 test)
2. 训练: num_workers=4, batch=16, 5000 steps, bbox_scale=48
3. 训练时间: 3288s (~55 min)

## Files

- `lam/lam/yolo_box_dataset.py` — YOLO box dataset with IoU tracking + GT matching
- `lam/scripts/run_v8_yolo.py` — Stage 4 training script
- `result/v8_mot_lam/yolo_cache/` — YOLO detection cache (disk)

## 对比汇总

| Model | Data | Samples | Action NMI | Action Probe | Actor Leakage |
|---|---|---|---|---|---|
| V6c | synthetic | 4000 | 0.0525 | N/A | 1.0000 |
| V7v3 | synthetic | 4000 | 0.0047 | N/A | 0.8970 |
| V8 Stage 1 | synthetic | 4000 | **0.7723** | 0.8741 | 0.3350 |
| V8 Stage 2A | synthetic+cam | 4000 | 0.7589 | 0.8724 | 0.3374 |
| V8 Stage 2B | A2D GT bbox | 248 | 0.0673 | 0.2222 | 0.1667 |
| **V8 Stage 4** | A2D YOLO | 2000 | **0.3118** | **0.6667** | 0.5556 |

## 下一步

1. **增加样本量**: 2000→28000 (全部 train videos),预期 NMI 进一步提升
2. **降低 actor leakage**: 在 YOLO 数据上做 actor conditioning (Stage 2C 的 FiLM)
3. **Stage 3**: MOT noise robustness 测试 (bbox jitter, miss detection, ID switch)
