# Mask-Guided Latent Action Model 实验报告

## 一、YOLO 与 ByteTrack 的分工与关系

### 1.1 Tracking-by-Detection 架构

多目标追踪（MOT）的主流范式是 **Tracking-by-Detection**：

```
视频帧序列
    ↓
[检测器 YOLO] 每帧独立检测 → bbox + class + confidence
    ↓
[跟踪器 ByteTrack] 帧间关联 → track ID
    ↓
输出：每帧 [(bbox, track_id, class), ...]
```

### 1.2 YOLO（检测器）的职责

| 项目 | 说明 |
|------|------|
| **输入** | 单帧图像 |
| **输出** | 检测框列表 [(x1,y1,x2,y2), class, confidence] |
| **特点** | 每帧独立处理，不知道前一帧检测到了什么 |
| **参数量** | YOLOv8n = 3.2M |
| **推理速度** | ~30 FPS (0.2ms/帧) |

**核心问题**：YOLO 无法保证帧间 ID 一致性。帧 t 的 bbox1 和帧 t+1 的 bbox3 可能是同一个物体，但 YOLO 不知道。

### 1.3 ByteTrack（跟踪器）的职责

| 项目 | 说明 |
|------|------|
| **输入** | YOLO 每帧的检测结果 + 前一帧的轨迹状态 |
| **输出** | 带 track ID 的检测结果 |
| **核心算法** | IoU 匹配 + 卡尔曼滤波 |
| **参数量** | **几乎为 0**（纯算法，无神经网络） |
| **计算量** | **几乎为 0**（只需计算 IoU 矩阵） |

**核心原理**：
1. 相邻帧中，同一物体的 bbox 位置变化很小 → IoU 很高
2. 用 IoU 匹配当前帧检测与上一帧轨迹 → 保持 ID 一致性
3. 卡尔曼滤波预测下一帧位置 → 处理短暂遮挡

### 1.4 为什么 ByteTrack 最合适？

| 跟踪算法 | 参数量 | 推理速度 | 特点 |
|---------|--------|---------|------|
| **ByteTrack** | ~0 | 30+ FPS | 纯 IoU 匹配，极轻量 |
| DeepSORT | ~几MB | 15-20 FPS | 需要外观特征提取网络（ReID） |
| SORT | ~0 | 30+ FPS | ByteTrack 的前身，更简单 |

**ByteTrack vs DeepSORT**：
- DeepSORT 需要额外的 ReID 网络提取外观特征，更重
- ByteTrack 只用 IoU + 卡尔曼滤波，几乎零额外计算
- 对于 LAM 场景（物体外观变化不大），ByteTrack 足够

### 1.5 Ultralytics 一行代码集成

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# 一行代码完成检测 + 跟踪
results = model.track(frame, tracker='bytetrack.yaml', persist=True)

# 提取 track ID
track_ids = results[0].boxes.id.cpu().numpy()  # [1, 2, 3, ...]
boxes = results[0].boxes.xyxy.cpu().numpy()    # [(x1,y1,x2,y2), ...]
```

---

## 二、实验进度总结

### 2.1 合成数据实验（完整验证）

合成数据特点：5 帧、4 个主体、5 类动作（stay/up/down/left/right）

| 方法 | Mask 来源 | 检测率 | 动作准确率 | PSNR | Slot 余弦相似度 |
|------|----------|--------|-----------|------|----------------|
| GT mask (基线) | 像素级 GT mask | 100% | **88.80%** | 21.85 dB | 低（独立） |
| GT bbox 矩形填充 | GT mask → bbox → 填充 | 100% | **89.20%** | 21.0 dB | 低（独立） |
| YOLO+ByteTrack (1000步) | 检测 bbox → 填充 | 99.7% | 87.09% | 18.3 dB | 中等 |
| **YOLO+ByteTrack (2000步)** | 检测 bbox → 填充 | 99.7% | **87.27%** | 23.06 dB | **低（独立）** |
| YOLO+ByteTrack 无交互 (2000步) | 检测 bbox → 填充 | 99.7% | **87.70%** | 23.81 dB | **高（相关）** |

**关键发现**：

1. **bbox 矩形填充完全可行**：与 GT mask 几乎无差异（89.20% vs 88.80%）
2. **SAM2 不必要**：MaskedPool 在 patch 级别工作，像素级 mask 精度是过度设计
3. **Interaction Module 的关键作用**：
   - 有交互：动作准确率 87.27%，Slot 余弦相似度 ≈ 0（主体独立）
   - 无交互：动作准确率 87.70%，Slot 余弦相似度 0.5-0.7（主体混淆）
   - **结论**：交互模块不提升动作准确率，但让隐动作空间结构化

### 2.2 YOLOv8 检测性能

#### 合成数据（彩色方块）

| 指标 | 值 |
|------|-----|
| mAP50 | **0.995** |
| mAP50-95 | **0.995** |
| 推理速度 | 0.2ms/帧 |
| 参数量 | 3.2M |

#### A2D 数据集（7 类 actor）

| 指标 | 零样本 (COCO) | 微调后 |
|------|--------------|--------|
| mAP50 | ~0.1 | **0.66** |
| Recall | 0.4 | **0.59** |
| 检测率 | 0.8 actor/帧 | **1.3 actor/帧** |

**各类别表现**：
- baby: mAP50=0.85（最好）
- ball: mAP50=0.38（最差，小物体难检测）
- bird/cat/dog/car: mAP50=0.65-0.75

### 2.3 A2D 数据集 LAM 训练（初步）

A2D 特点：2 帧（仅 1 个时间步）、8 类动作、标注帧稀疏（每视频最多 5 帧）

| 方法 | 动作准确率 | 随机基线 |
|------|-----------|---------|
| A2D bbox mask (500步) | 27.42% | 12.5% |
| **A2D bbox mask (2000步)** | **30.05%** | 12.5% |

**各类别表现**：
- walking: 47.30%（最好）
- running/rolling/climbing/eating: 30-33%
- jumping/flying: 16-20%
- crawling: 7.50%（最差）

---

## 三、迁移到 A2D 的完整训练方案

### 3.1 核心挑战

1. **标注帧稀疏**：A2D 每视频最多 5 个标注帧，大部分只有 3 帧
2. **时间信息不足**：2 帧 = 1 个时间步，难以学习时序动态
3. **类别不平衡**：walking 占主导，crawling/climbing 样本少
4. **YOLO 检测不完美**：ball 类别 mAP 只有 0.38

### 3.2 推荐训练流程

#### Phase 1: 预训练（合成数据）

```bash
# 用合成数据预训练 LAM，学习基本的时序动态建模能力
CUDA_VISIBLE_DEVICES=3 python run_yolo_closed_loop.py \
    --name pretrain_synthetic \
    --steps 2000 \
    --batch_size 32 \
    --action_weight 1.0 \
    --kl_beta 0.0002
```

**预期效果**：动作准确率 ~87%，隐动作空间结构化

#### Phase 2: YOLO 微调（A2D 标注）

```bash
# 在 A2D 标注上微调 YOLOv8n，学习检测所有 7 类 actor
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data/a2d_yolo/data.yaml', epochs=30, imgsz=320)
"
```

**已完成**：mAP50=0.66, Recall=0.59

#### Phase 3: LAM 微调（A2D 数据）

```bash
# 用微调后的 YOLO+ByteTrack 检测 A2D 视频，生成 mask，微调 LAM
CUDA_VISIBLE_DEVICES=3 python run_a2d_closed_loop.py \
    --name a2d_finetune \
    --yolo_weights results/a2d_yolo/yolov8n_a2d/weights/best.pt \
    --steps 1000 \
    --pretrained results/yolo_closed_loop/model_pretrain_synthetic.pt \
    --action_weight 2.0 \
    --kl_beta 0.0001
```

**关键参数调整**：
- `action_weight=2.0`：增强动作监督（A2D 动作类别多）
- `kl_beta=0.0001`：降低 KL 惩罚（避免 VAE 过拟合）
- `pretrained`：加载合成数据预训练权重

#### Phase 4: 评估与迭代

```bash
# 评估动作准确率
python evaluate_a2d.py --model results/a2d_closed_loop/model_a2d_finetune.pt

# 分析隐动作空间
python analyze_latent_a2d.py --model results/a2d_closed_loop/model_a2d_finetune.pt
```

### 3.3 数据增强策略

1. **帧采样增强**：从视频中采样多个 2 帧片段（而非固定帧）
2. **类别平衡**：对 crawling/climbing 等少数类别过采样
3. **时序插值**：对只有 2 帧的样本，用帧间插值生成伪中间帧

---

## 四、下一步改进方向

### 4.1 短期改进（1-2 周）

| 方向 | 具体措施 | 预期效果 |
|------|---------|---------|
| **更长训练** | A2D 训练 5000+ 步 | 动作准确率 +5-10% |
| **预训练迁移** | 加载合成数据预训练权重 | 加速收敛，+3-5% |
| **类别平衡** | 过采样少数类别 | crawling/climbing +10% |
| **YOLO 检测优化** | 降低 ball 类别 conf 阈值 | 检测率 +10% |

### 4.2 中期改进（1-2 月）

| 方向 | 具体措施 | 预期效果 |
|------|---------|---------|
| **更多时间步** | 修改 A2D 数据加载，采样 3-5 帧 | 时序建模能力提升 |
| **外观特征** | 引入 DeepSORT 的 ReID 特征 | ID 切换减少 50% |
| **多尺度检测** | YOLOv8 多尺度训练 | 小物体（ball）检测 +20% |
| **动作解码器** | 从隐动作预测未来帧 | 验证隐动作语义 |

### 4.3 长期改进（3-6 月）

| 方向 | 具体措施 | 预期效果 |
|------|---------|---------|
| **端到端训练** | YOLO+ByteTrack+LAM 联合训练 | 检测与隐动作协同优化 |
| **真实场景验证** | 在更多数据集验证（AVA, Charades） | 泛化能力验证 |
| **轻量化部署** | 模型压缩、量化 | 实时推理部署 |
| **交互预测** | 从隐动作预测主体间交互 | 扩展到交互建模 |

---

## 五、核心结论

1. **YOLO+ByteTrack 方案验证成功**：
   - YOLO 负责：每帧检测物体 bbox
   - ByteTrack 负责：帧间关联，保持 ID 一致性
   - 合成数据闭环：87.27%（仅比 GT mask 基线低 1.5%）

2. **bbox 矩形填充 mask 完全可行**：
   - MaskedPool 在 patch 级别工作，像素级 mask 精度是过度设计
   - SAM2 不必要，推理慢且效果差

3. **Interaction Module 关键作用**：
   - 不提升动作准确率
   - 但让隐动作空间结构化（主体间独立性）

4. **A2D 迁移挑战**：
   - 标注帧稀疏（最多 5 帧）
   - 时间信息不足（2 帧 = 1 个时间步）
   - YOLO 检测不完美（ball 类别难检测）
   - 需要：预训练 + 微调 + 类别平衡

5. **推荐训练流程**：
   - Phase 1: 合成数据预训练（学习基本时序动态）
   - Phase 2: A2D YOLO 微调（学习检测所有 actor）
   - Phase 3: A2D LAM 微调（迁移到真实场景）
   - Phase 4: 评估迭代

---

## 六、文件清单

### 核心代码

| 文件 | 说明 |
|------|------|
| `lam/modules/blocks.py` | MaskedPool, ObjectInteractionModule |
| `lam/modules/lam.py` | Mask-Guided LAM 主模型 |
| `lam/a2d_dataset.py` | A2D 数据集加载器 |
| `scripts/run_yolo_closed_loop.py` | YOLO+ByteTrack 闭环训练 |
| `scripts/run_a2d.py` | A2D 训练脚本 |
| `scripts/convert_a2d_to_yolo.py` | A2D 标注 → YOLO 格式 |

### 实验结果

| 目录 | 说明 |
|------|------|
| `results/yolo_closed_loop/` | 合成数据 YOLO+ByteTrack 实验 |
| `results/a2d_exp/` | A2D LAM 实验 |
| `results/a2d_yolo/` | A2D YOLO 微调 |
| `results/bbox_mask_exp/` | bbox 填充验证实验 |

---

---

## 七、最新实验进展（2025-06-05）

### 7.1 隐动作聚类分析结果

**合成数据（YOLO+ByteTrack 闭环训练）**

| 模型 | ARI (action) | ARI (actor) | Action cls | Actor cls | stay cos_sim |
|------|-------------|-------------|-----------|----------|--------------|
| **有交互模块** | 0.6810 | -0.0006 | 82.74% ± 4.18% | **99.87% ± 0.16%** | **0.023 ± 0.172** |
| 无交互模块 | 0.7045 | -0.0007 | 86.10% ± 2.78% | 99.87% ± 0.26% | 0.702 ± 0.151 |

**关键发现**：
1. **Interaction Module 的核心作用**：让不同主体的 stay 隐动作变得独立（cos_sim 从 0.702 → 0.023）
2. **Actor classification 都极高**（99.87%）：主体信息被很好地编码
3. **Action classification**：无交互版本略高（86.10% vs 82.74%），与动作准确率一致

### 7.2 零样本 YOLO (COCO) 在 A2D 上的检测效果

| 指标 | 零样本 YOLO | 微调后 YOLO |
|------|------------|------------|
| **Recall** | **67.79%** | 59% |
| **Precision** | 60.28% | - |
| **Extra detections** | 37.26% | - |

**按类别 Recall**：
- adult/baby: **90%+**（COCO person 类训练数据丰富）
- ball/bird: **20-30%**（COCO sports ball/bird 类训练数据较少）
- car/cat/dog: **50-70%**（中等）

**关键发现**：
1. **零样本 YOLO 的 Recall 甚至比微调后更高**（67.79% vs 59%）
2. **37% 的检测框是"额外检测"**— YOLO 检测到但 A2D 未标注的物体，可能是真实物体
3. **印证了分析**：A2D 微调让 YOLO 学会了"只检测 1-2 个物体"的偏见，零样本检测更全面

### 7.3 下一步计划

1. **AVA 数据集**：暂缓下载（30GB+），先在合成数据上验证完 LAM
2. **推荐方案**：用零样本 YOLO（COCO）替代微调方案，检测更全面
3. **可视化检查**：[yolo_zeroshot_vis](file:///home/xiaojy/projects/AdaWorld-improving/lam/results/yolo_zeroshot_vis) 中有 20 个样本可视化

---

**报告日期**：2025-06-05
**实验环境**：RTX 4090, CUDA 12.0, PyTorch 2.12