# V8 Stage 2B: Real A2D Data — Report

## Summary

Stage 2B 将 V8 从合成数据迁移到真实 A2D 视频数据。V8 成功在 A2D 上训练,
z_actor 编码了弱但可检测的 action 信息 (probe acc 0.22 vs chance 0.125),
但 action NMI (0.067) 远低于合成数据 (0.77),反映了真实数据的固有挑战。

## Results

| Metric | Synthetic (Stage 1) | **A2D (Stage 2B)** | Chance |
|---|---|---|---|
| Action NMI | 0.7723 | **0.0673** | ~0 |
| Action Probe Acc | 0.8741 | **0.2222** | 0.125 |
| Actor Leakage | 0.3350 | **0.1667** | 0.143 |
| ARI | 0.6944 | **0.0111** | ~0 |
| z_actor variance | 0.5023 | **0.3533** | — |
| z_bg variance | 0.0754 | **0.3140** | — |
| Active dims | 16/16 | **16/16** | — |
| dbbox MSE (px²) | 10.47 | **926.10** | — |

### 关键发现

1. **Actor type leakage = 0.1667 (chance=0.143)**: z_actor 几乎不泄漏 A2D actor type
   (adult/baby/ball/bird/car/cat/dog)。这是好消息 — V8 的 motion-only 编码在真实
   数据上仍然避免了 actor 信号污染。

2. **Action probe = 0.2222 (chance=0.125)**: z_actor 编码了弱但统计显著的 action 信息。
   线性分类器可以从 z_actor 解码 ~22% 的 action,1.8x chance。

3. **z_bg variance = 0.314**: z_bg 在真实视频上活跃 (合成无扰动时仅 0.075),
   因为 A2D 视频自带相机运动,z_bg 吸收了全局变化。

4. **Action NMI = 0.067**: KMeans 聚类与 GT action 的对齐很弱。
   这不意味着 V8 完全失败,而是反映了真实数据的挑战。

## 性能差距分析

### 1. 训练数据量 (最主要原因)
- 合成: 4000 train samples, 500 eval
- A2D: 248 train samples, 67 eval (266 latent samples)
- A2D 训练样本仅为合成的 6%,严重不足

### 2. 标注帧间隔大
- A2D 标注帧平均间隔 25 帧 (范围 5-50)
- 每个 transition 跳 ~25 帧视频时间,bbox delta 可能很大且方向多变
- 合成数据每帧都有标注,transition 是单步动作 (32px)

### 3. Action 空间复杂度
- 合成: 5 个离散动作 (stay/up/down/left/right),每步固定 32px
- A2D: 8 个语义动作 (climbing/crawling/eating/flying/jumping/rolling/running/walking)
  - 同一 action 的 bbox delta 方向和大小变化极大
  - "running" 可能向任意方向,速度也不同
  - "climbing" 的运动模式与 "jumping" 可能有重叠

### 4. 真实视频复杂度
- 复杂背景、光照变化、遮挡、形变
- bbox 标注噪声
- Actor 跨帧 IoU 匹配不稳定 (A2D 标注顺序可能变)

## Architecture

### 新增文件
- `lam/lam/a2d_box_dataset.py` — A2D box 数据集 (从 .mat 读 bbox + IoU tracking)
- `lam/scripts/run_v8_a2d.py` — A2D 训练脚本
- `lam/scripts/eval_v8_a2d.py` — A2D 评估脚本

### A2DBoxDataset 设计
- 直接从 A2D .mat 标注读取 `reBBox` (4, N) → boxes (T, K, 4)
- `parse_a2d_label`: 两位数 ID → (actor_type 1-7, action 0-7)
- `_match_tracks`: 贪心 IoU 匹配 (threshold=0.3) 分配 pseudo track IDs
- bbox 缩放到 img_size=256,归一化坐标 [0, 256]
- actor_labels = A2D actor type (用于 Phase 2C conditioning)

### 训练配置
- bbox_scale = 48 (A2D bbox delta 中位数 ~25, 90% ~47)
- batch_size = 8 (样本少)
- steps = 3000 (~97 epochs)
- num_workers = 4 (cv2.VideoCapture I/O 瓶颈)
- 训练时间: 1112s (~18.5 min)

## 下一步 (Phase 2C: Actor Conditioning)

Stage 2B 确认了 V8 在真实数据上的基本可行性,但 action NMI 低。
Phase 2C 的 actor conditioning 可能帮助:
- 不同 actor type 有不同运动先验 (ball 自然 rolling, bird 自然 flying)
- 条件化 actor type 让 z_actor 编码 "相对于该 actor 类型的残差 action"
- 可能提升 action clustering 质量

但也需要更多训练数据和可能的架构改进来显著提升 NMI。
