# LocateAnything × Mask-Guided LAM 融合计划

## 一、当前状态

### 1.1 已验证的成果

**Mask-Guided LAM**（在合成数据上）：
- 动作预测准确率：88.8%（vs 随机基线 20%）
- 隐动作空间：不同主体的 z_mu 高度分离（余弦相似度≈0，有交互模块时）
- 重建 PSNR：21.9 dB
- 核心架构：GT Mask → MaskedPool → Per-Object VAE → Interaction Module → Decoder

**关键结论**：只要给定了正确的物体分割 mask，模型就能学到区分不同主体的隐动作。

### 1.2 待解决问题

合成数据集提供了完美的 GT mask，但真实场景中：
- A2D 数据集只有 actor-level bounding box 标注，无像素级 mask
- 需要一个自动分割模块来替代 GT mask
- **LocateAnything** 就是这个分割模块

### 1.3 环境检查结果

| 项目 | 状态 |
|------|------|
| conda 环境 `locateanything` | ✅ 可用 |
| PyTorch | ✅ 2.12.0+cu130, CUDA 可用 |
| LocateAnythingWorker | ✅ 可导入 |
| LocateAnythingForConditionalGeneration | ✅ 可导入 |
| detectron2 | ❌ 未安装（暂不需要） |
| SAM/SAM2 | ❌ 未安装（需要安装） |
| LocateAnything-3B 模型权重 | ❌ 未下载 |

---

## 二、LocateAnything 输出分析

### 2.1 输出格式

LocateAnything 输出的是**边界框 (bounding box)**，不是像素级 mask：
```
<ref>object_name</ref><box><x1><y1><x2><y2></box>
```
坐标为 [0, 1000] 范围的整数，需转换为像素坐标。

### 2.2 关键 API

```python
worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
result = worker.detect(img, ["object1", "object2"])
boxes = LocateAnythingWorker.parse_boxes(result["answer"], img.width, img.height)
# boxes: {"object1": [[x1,y1,x2,y2], ...], "object2": [[x1,y1,x2,y2], ...]}
```

### 2.3 从 Bounding Box 到 Mask

**方案 A（推荐）**：Box → SAM/SAM2 → 精确像素级 mask
- LocateAnything 提供 box prompt → SAM 基于 box 生成精确 mask
- 质量最高，但需要安装 SAM

**方案 B（简化）**：Box → 矩形填充 mask
- 直接将 bounding box 填充为二值 mask
- 无需额外模型，但精度低（矩形区域包含背景）

**方案 C（折中）**：Box → ROI Pooling
- 不生成 mask，直接在 box 区域内池化 CNN 特征
- 绕过 MaskedPool，改用 ROI Align
- 精度中等，但需修改 LAM 架构

---

## 三、分步实施计划

### 阶段一：环境准备与基础验证

#### 1.1 下载 LocateAnything-3B 模型权重
```bash
# 在 locateanything 环境中
huggingface-cli download nvidia/LocateAnything-3B
```
预计大小：~6GB（3B 参数模型）

#### 1.2 安装 SAM2（用于 box→mask 转换）
```bash
pip install sam2
```
SAM2 支持从 box prompt 生成精确 mask，是 LocateAnything 的自然补充。

#### 1.3 验证 LocateAnything 在合成数据上的检测能力

**目标**：用 LocateAnything 检测合成数据集中的彩色方块，验证检测质量。

```python
# 伪代码
for sample in val_dataset:
    frame = sample["videos"][0]  # 第一帧
    # 用 LocateAnything 检测所有移动方块
    result = worker.detect(frame, ["colored square", "colored block", "moving object"])
    boxes = parse_boxes(result)
    # 对比 GT positions/masks
```

**验证指标**：
- 检测召回率：LocateAnything 能否找到所有 2-4 个彩色方块？
- 定位精度：预测 box 与 GT box 的 IoU
- 如果检测不到，考虑使用更具体的 prompt 或 fine-tune

#### 1.4 验证 SAM2 的 box→mask 质量

```python
# 伪代码
for sample in val_dataset:
    frame = sample["videos"][0]
    gt_masks = sample["masks"][0]  # (A, H, W)
    # 从 GT bounding box 生成 mask
    gt_boxes = extract_boxes_from_masks(gt_masks)  # 从 GT mask 提取 box
    pred_masks = sam2.predict(frame, gt_boxes)  # 用 GT box 作为 prompt
    # 对比 pred_masks vs gt_masks
    iou = compute_mask_iou(pred_masks, gt_masks)
```

**预期**：SAM2 从精确 box prompt 生成的 mask 与 GT mask IoU 应 > 0.85。

---

### 阶段二：合成数据闭环验证

**目标**：用 LocateAnything + SAM2 替代 GT mask，验证 Mask-Guided LAM 是否仍然有效。

#### 2.1 构建分割管线

```
输入视频帧 → LocateAnything (检测) → SAM2 (box→mask) → 预测 masks
```

具体实现：创建 `SegmentationPipeline` 类

```python
class SegmentationPipeline:
    """用 LocateAnything + SAM2 从视频帧中提取多主体分割 mask。"""

    def __init__(self, locateanything_model, sam2_model):
        self.la_worker = locateanything_model
        self.sam2 = sam2_model

    def segment_frame(self, frame, prompts=None):
        """
        Args:
            frame: (H, W, 3) uint8 RGB 图像
            prompts: 检测提示词列表，如 ["colored block", "object"]
        Returns:
            masks: (A, H, W) binary mask
            boxes: (A, 4) bounding boxes
            labels: (A,) 检测标签
        """
        # Step 1: LocateAnything 检测
        result = self.la_worker.detect(frame, prompts)
        boxes = parse_boxes(result)

        # Step 2: SAM2 从 box prompt 生成 mask
        masks = self.sam2.predict_masks(frame, boxes)

        return masks, boxes, labels

    def segment_video(self, video_frames):
        """
        对视频的所有帧进行分割，保持时序一致性。

        Args:
            video_frames: (T, H, W, 3) uint8
        Returns:
            masks: (T, A, H, W) binary mask
        """
        # 第一帧用 LocateAnything 检测
        masks_0, boxes_0, labels = self.segment_frame(video_frames[0])

        # 后续帧用 SAM2 的 tracking 模式（如果可用）
        # 或每帧独立检测
        ...
```

#### 2.2 生成 "预测 mask" 版本的合成数据集

```python
# 对合成数据集的每个样本：
for sample in train_dataset:
    video = sample["videos"]  # (T, H, W, C)
    gt_masks = sample["masks"]  # (T, A, H, W)

    # 用分割管线生成预测 mask
    pred_masks = pipeline.segment_video(video)  # (T, A', H, W)

    # 保存：video + pred_masks（替换 gt_masks）
    save_sample(video, pred_masks, ...)
```

#### 2.3 对比实验

| 实验配置 | Mask 来源 | 预期效果 |
|----------|-----------|---------|
| **GT Mask** | 合成数据集真值 | 动作准确率 ~89% |
| **Pred Mask (LA+SAM2)** | LocateAnything + SAM2 预测 | 动作准确率应接近 GT |
| **Pred Mask (LA only, box fill)** | 仅 LocateAnything box 矩形填充 | 动作准确率可能下降 |
| **No Mask** | 不使用 mask | 动作准确率大幅下降 |

**关键验证**：如果 Pred Mask 的动作准确率接近 GT Mask（如 >80%），则说明分割管线可以替代 GT mask。

---

### 阶段三：A2D 真实数据验证

#### 3.1 A2D 数据集适配

A2D 数据集格式：
- 视频：RGB 帧序列
- 标注：每帧每个 actor 的 (bounding box, action label, actor ID)
- 动作类别：7 类

适配步骤：
1. 用 LocateAnything 检测 A2D 视频中的所有 actor
2. 将检测结果与 A2D 的 GT bbox 匹配（IoU 阈值 0.5）
3. 用 SAM2 从匹配的 bbox 生成 mask
4. 构建 A2D 版本的训练数据

#### 3.2 训练与评估

```python
# A2D 数据集配置
model = LatentActionModel(
    num_actions=7,  # A2D 有 7 类动作
    max_actors=8,   # A2D 视频中可能更多主体
    ...
)
```

评估指标：
- 动作预测准确率（7 类分类）
- 隐动作聚类质量（ARI）
- 与原始 LAM（单 action prompt）的对比

#### 3.3 消融实验

| 配置 | Mask 来源 | 说明 |
|------|-----------|------|
| A2D + GT bbox → box fill | A2D 标注的 bbox 直接填充 | 最简单，无额外模型 |
| A2D + GT bbox → SAM2 | A2D 标注的 bbox + SAM2 生成 mask | 中等质量 |
| A2D + LA → SAM2 | LocateAnything 检测 + SAM2 | 全自动管线 |
| A2D 原始 LAM | 无 mask，单 action prompt | 基线对比 |

---

## 四、代码修改清单

### 阶段一

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/verify_locateanything.py` | 新建 | 验证 LA 在合成数据上的检测能力 |
| `scripts/verify_sam2.py` | 新建 | 验证 SAM2 的 box→mask 质量 |

### 阶段二

| 文件 | 操作 | 说明 |
|------|------|------|
| `lam/lam/segmentation_pipeline.py` | 新建 | LocateAnything + SAM2 分割管线 |
| `scripts/generate_pred_masks.py` | 新建 | 用分割管线生成预测 mask 版数据集 |
| `scripts/run_single.py` | 修改 | 支持 `--mask_source gt/pred/box_fill` 参数 |
| `lam/lam/modules/lam.py` | 不变 | MaskedPool 接受任意 mask，无需修改 |

### 阶段三

| 文件 | 操作 | 说明 |
|------|------|------|
| `lam/lam/dataset.py` | 修改 | A2D 数据集适配，支持从 bbox 生成 mask |
| `scripts/run_single.py` | 修改 | A2D 模式，num_actions=7 |
| `lam/config/lam_a2d_multislot.yaml` | 新建 | A2D 训练配置 |

---

## 五、风险与对策

| 风险 | 对策 |
|------|------|
| LocateAnything 检测不到合成数据的彩色方块 | 使用更具体的 prompt；或直接用 GT bbox 作为 LA 的输入验证后续管线 |
| SAM2 未安装或不可用 | 先用方案 B（box fill）作为 fallback，验证整体管线可行后再装 SAM2 |
| A2D 视频中 actor 数量变化大 | 设置 max_actors=8，padding 处理不足的帧 |
| LocateAnything 推理速度慢 | 预生成 mask 到磁盘，训练时不在线推理 |
| 预测 mask 与 GT mask 的差异导致性能下降 | 这是预期内的；量化 mask 质量对性能的影响是核心实验目标 |

---

## 六、实施优先级

1. **阶段一**（立即）：下载模型 + 安装 SAM2 + 验证检测能力
2. **阶段二**（核心）：合成数据闭环验证 — 证明分割管线可以替代 GT mask
3. **阶段三**（最终）：A2D 真实数据验证

阶段二是整个方案的验证关键。如果 LocateAnything + SAM2 在合成数据上的预测 mask 能让 LAM 达到接近 GT mask 的动作准确率，就证明了这个管线在真实数据上也应该有效。
