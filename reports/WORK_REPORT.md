# Mask-Guided Latent Action Model 工作报告

**项目目标**：改进 Latent Action Model (LAM)，使其能够处理多物体运动场景，学习不同主体之间的交互关系。

**报告日期**：2025-06-05

---

## 一、方向纠正：从 Slot Attention 到 Mask-Guided

### 1.1 初始误区

最初我们尝试用 **Slot Attention** 无监督地学习出不同的 slot（主体），但这偏离了项目核心目标。用户指出：

> "我们并非要无监督地学习出slot，而是在已经分割出不同主体的情况下学习这些主体之间的交互关系。"

### 1.2 核心洞察

- **Slot Attention 的问题**：试图无监督发现主体，但这是检测/分割的任务，不是 LAM 的核心职责
- **正确方向**：假设分割已经完成（由外部模块提供），LAM 应专注于学习主体间的交互关系
- **简化策略**：先在合成数据集上验证，因为合成数据有完整的 GT mask，暂时不需要 MOT 模块

---

## 二、合成数据集

### 2.1 数据集设计

为了验证 Mask-Guided LAM 的有效性，我们生成了一个自定义合成数据集。

**数据集特点**：

| 属性 | 值 |
|------|-----|
| **视频数** | 1000（train=500, val=500） |
| **帧数** | 5 帧/视频 |
| **分辨率** | 256×256 |
| **主体数** | 4 个彩色方块 |
| **动作类别** | 5 类（stay, up, down, left, right） |
| **标注** | 像素级 mask + bbox + 动作标签 |

**生成逻辑**：
- 每个主体独立运动，随机选择动作
- 每帧所有主体都有完整的像素级 mask
- 动作标签对应主体的运动方向

### 2.2 为什么用合成数据

1. **完整标注**：像素级 mask 完整，无遮挡、无漏标
2. **可控实验**：可以精确控制主体数量、动作类型
3. **快速验证**：无需等待 MOT 模块开发，先验证 LAM 核心架构
4. **定量评估**：可以计算动作准确率、隐动作聚类质量等指标

---

## 三、Mask-Guided LAM 架构改进

### 3.1 核心架构变化

从 Slot Attention 架构重写为 **Mask-Guided 架构**：

```
视频帧序列 (T, H, W, C)
    ↓
[SpatioTemporalTransformer] 编码 patch 特征
    ↓
[MaskedPool] 用 GT mask 将 patch 特征按主体池化
    ↓
主体特征 (B, T, A, D)
    ↓
[Temporal Differencing] 每主体帧间特征差分
    ↓
[Per-Object VAE] 每主体独立 VAE 编码
    ↓
隐动作 z_mu, z_var (B, T-1, A, latent_dim)
    ↓
[ObjectInteractionModule] 主体间交互（可选）
    ↓
[Action Head] 从隐动作预测动作类别
    ↓
[Decoder] Cross-Attention 融合 + SpatioTransformer 重建
```

### 3.2 关键模块

#### MaskedPool

将 patch 级特征按 mask 区域加权平均池化，得到每个主体的特征向量：

```python
class MaskedPool(nn.Module):
    def forward(self, patches, masks):
        # masks: (B, T, A, H, W) → 下采样到 patch 级别
        # patches: (B, T, N, D) → 按 mask 加权池化
        # 返回: obj_feats (B, T, A, D), valid_mask (B, T, A)
        masks_down = F.adaptive_avg_pool2d(masks, (grid_h, grid_w))
        weights = masks_down.unsqueeze(-1)
        feat = patches.unsqueeze(2) * weights
        obj_feats = feat.sum(dim=-2) / mask_sum
        return obj_feats, valid_mask
```

#### ObjectInteractionModule

让不同主体的隐动作互相交互，使用 Transformer Self-Attention：

```python
class ObjectInteractionModule(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x, padding_mask=None):
        return self.transformer(x, src_key_padding_mask=padding_mask)
```

### 3.3 训练损失

```python
loss = recon_weight * MSE(recon, gt) 
       + kl_beta * KL(z_mu, z_var) 
       + action_weight * CrossEntropy(action_logits, gt_actions)
```

---

## 四、合成数据实验效果

### 4.1 动作识别准确率

| 方法 | Mask 来源 | 检测率 | 动作准确率 | PSNR |
|------|----------|--------|-----------|------|
| GT mask (基线) | 像素级 GT mask | 100% | **88.80%** | 21.85 dB |
| GT bbox 矩形填充 | GT mask → bbox → 填充 | 100% | **89.20%** | 21.0 dB |
| YOLO+ByteTrack | 检测 bbox → 填充 | 99.7% | **87.27%** | 23.06 dB |

**关键发现**：
- bbox 矩形填充 mask 与 GT mask 几乎无差异（89.20% vs 88.80%）
- YOLO+ByteTrack 闭环训练仅比基线低 1.5%（87.27% vs 88.80%）

### 4.2 Interaction Module 消融实验

| 模型 | 动作准确率 | Slot 余弦相似度 | Actor 线性探针 |
|------|-----------|----------------|---------------|
| **有交互模块** | 87.27% | **≈0（独立）** | **99.93%** |
| 无交互模块 | 87.70% | 0.5-0.7（混淆） | 82.61% |

**关键发现**：
- Interaction Module **不提升动作准确率**（甚至略降）
- 但**关键作用是让隐动作空间结构化**：
  - 有交互：不同主体的隐动作独立（cos_sim ≈ 0）
  - 无交互：不同主体的隐动作混淆（cos_sim 0.5-0.7）

### 4.3 隐动作聚类分析

**UMAP 可视化 + 聚类质量评估**：

| 模型 | ARI (action) | ARI (actor) | Action cls | Actor cls | stay cos_sim |
|------|-------------|-------------|-----------|----------|--------------|
| **有交互模块** | 0.6810 | -0.0006 | 82.74% ± 4.18% | **99.87% ± 0.16%** | **0.023 ± 0.172** |
| 无交互模块 | 0.7045 | -0.0007 | 86.10% ± 2.78% | 99.87% ± 0.26% | 0.702 ± 0.151 |

**关键发现**：
- **stay 动作的 cross-actor cos_sim**：有交互 = 0.023（几乎独立），无交互 = 0.702（高度相似）
- 说明交互模块让不同主体的 stay 隐动作变得独立，避免混淆

---

## 五、加入 MOT 模块

### 5.1 选择 MOT 模块的依据

**核心需求**：
- 轻量化、实时推理（MOT 是辅助模块，不应喧宾夺主）
- 输出 bounding box（而非像素级 mask）
- 能检测所有物体（不受特定类目限制）

**方案对比**：

| 方案 | 参数量 | 推理速度 | 特点 |
|------|--------|---------|------|
| **YOLOv8n + ByteTrack** | 3.2M + ~0 | **30+ FPS** | 极轻量，纯 IoU 跟踪 |
| DeepSORT | 几MB + ReID 网络 | 15-20 FPS | 需要外观特征提取 |
| LocateAnything | 3B | ~1 FPS | 太重，不适合实时 |

**选择 YOLOv8n + ByteTrack 的原因**：
1. **YOLOv8n 极轻量**：3.2M 参数，0.2ms/帧
2. **ByteTrack 几乎零计算**：纯 IoU 匹配 + 卡尔曼滤波，参数量≈0
3. **Tracking-by-Detection 架构**：YOLO 检测 + ByteTrack 跟踪，分工明确
4. **Ultralytics 一行代码集成**：`model.track(frame, tracker='bytetrack.yaml')`

### 5.2 YOLO 与 ByteTrack 的分工

**YOLO（检测器）**：
- 输入：单帧图像
- 输出：检测框列表 + 类别 + 置信度
- 每帧独立处理，不知道前一帧检测到了什么

**ByteTrack（跟踪器）**：
- 输入：YOLO 每帧的检测结果 + 前一帧的轨迹状态
- 输出：带 track ID 的检测结果
- 核心算法：IoU 匹配 + 卡尔曼滤波
- 参数量：≈0（纯算法，无神经网络）

**Tracking-by-Detection 流程**：

```
帧 t → YOLO → [bbox1, bbox2, bbox3]
         ↓
帧 t → ByteTrack → [(bbox1, ID=1), (bbox2, ID=2), (bbox3, ID=3)]
         ↓
帧 t+1 → YOLO → [bbox4, bbox5, bbox6]
         ↓
帧 t+1 → ByteTrack → [(bbox4, ID=1), (bbox5, ID=2), (bbox6, ID=3)]
                        ↑ 同一物体，ID 保持一致
```

### 5.3 MOT 模块如何融合到 LAM

**融合流程**：

```
视频 → YOLO+ByteTrack → 每帧 [(track_id, bbox), ...]
                         ↓
                    按 track_id 排序，bbox 矩形填充为 mask
                         ↓
                    mask (T, max_actors, H, W)
                         ↓
                    MaskedPool → 每个主体的特征
                         ↓
                    LAM → 每个主体的隐动作
```

**关键点**：
- track ID 保证同一主体在不同帧的特征被正确池化到同一个 slot
- bbox 矩形填充为 mask（无需 SAM2，因为 MaskedPool 在 patch 级别工作）

### 5.4 YOLO+ByteTrack 实验结果

**合成数据（彩色方块）**：

| 指标 | 值 |
|------|-----|
| YOLO mAP50 | **0.995** |
| YOLO 推理速度 | 0.2ms/帧 |
| 检测率 | 99.7% |
| LAM 动作准确率 | **87.27%** |
| LAM PSNR | 23.06 dB |

**结论**：YOLO+ByteTrack 在合成数据上表现完美，闭环训练仅比 GT mask 基线低 1.5%。

---

## 六、迁移到 A2D 数据集时遇到的问题

### 6.1 A2D 数据集分析

**A2D 标注统计（抽样 11926 帧）**：

| 每帧主体数 | 占比 | 帧数 |
|-----------|------|------|
| 1 个主体 | **62.2%** | 7419 |
| 2 个主体 | 23.6% | 2814 |
| 3+ 个主体 | 14.2% | 1693 |

**核心问题**：A2D 本质上是一个**单主体动作识别数据集**，62% 的帧只标注了 1 个 actor。

### 6.2 具体问题

#### 问题 1：标注不完整

A2D 只标注了"主要演员"，但画面中可能有多个物体/人物。

**实际案例**：
- 视频 `-d8EYyveK_E`：A2D 标注了 1 个 dog（动作=walking）
- 但 YOLO 检测到了：1 car + 2 dog + 4 person
- A2D 未标注的 7 个物体被当作 false positive 惩罚

#### 问题 2：YOLO 微调后的偏见

我们在 A2D 标注上微调 YOLO，但：
- 训练数据平均每帧只有 1.62 个 bbox
- YOLO 学会了"每帧只检测 1-2 个物体"
- 在多目标场景变得保守，宁缺毋滥

**对比**：

| 方案 | Recall | 说明 |
|------|--------|------|
| 微调后 YOLO (A2D) | 59% | 学会了"只检测 1-2 个" |
| 零样本 YOLO (COCO) | **67.79%** | 检测更全面 |

#### 问题 3：评估指标不匹配

LAM 处理多主体，但 A2D 标注不完整：
- LAM 正确预测了 4 个主体的动作
- A2D 只给 1 个 GT
- 评估时算错（其他 3 个被当作误检）

#### 问题 4：时间信息不足

A2D 标注帧稀疏：
- 每视频最多 5 个标注帧
- 大部分只有 3 帧
- 2 帧 = 1 个时间步，难以学习时序动态

---

## 七、尝试的解决方案

### 7.1 方案 1：YOLO 零样本检测

**思路**：用 COCO 预训练 YOLO（不在 A2D 上微调），做类映射。

**类映射**：

| A2D Actor | COCO 类 |
|-----------|---------|
| adult / baby | person (cls 0) |
| ball | sports ball (cls 32) |
| bird | bird (cls 14) |
| car | car (cls 2) |
| cat | cat (cls 15) |
| dog | dog (cls 16) |

**实验结果**：

| 指标 | 零样本 YOLO | 微调后 YOLO |
|------|------------|------------|
| **Recall** | **67.79%** | 59% |
| **Precision** | 60.28% | - |
| **Extra detections** | 37.26% | - |

**按类别 Recall**：
- adult/baby: **90%+**
- ball/bird: **20-30%**
- car/cat/dog: **50-70%**

**结论**：零样本 YOLO 的 Recall 比微调后更高，检测更全面。

### 7.2 方案 2：寻找更适合的数据集

**推荐 AVA (Atomic Visual Actions)**：

| 维度 | A2D | AVA |
|------|-----|-----|
| **多主体标注** | 62% 只标 1 个 | **每帧标注所有人物** |
| **动作类别** | 8 类 | **80 类原子动作** |
| **标注密度** | 每视频 3-5 帧 | **1Hz 采样，15 分钟片段** |
| **跨帧追踪** | 稀疏，帧间断裂 | **人物跨帧链接** |
| **标注总量** | ~1.5 万帧 | **158 万个动作标签** |

**AVA 最适合的原因**：
- 标注策略是"每帧的所有人物都标注"
- 标注了人物之间的交互
- 人物跨帧链接，天然支持 track ID

**暂缓原因**：AVA 约 30GB+，先在合成数据上验证完 LAM。

---

## 八、当前瓶颈

### 8.1 技术瓶颈

1. **ball/bird 检测率低**（20-30%）
   - COCO sports ball/bird 类训练数据较少
   - A2D 中这些物体可能较小
   - 可能需要专门的小物体检测模型

2. **A2D 不适合多主体评估**
   - 标注不完整（62% 只标 1 个）
   - 无法正确评估 LAM 的多主体能力

3. **时间信息不足**
   - A2D 标注帧稀疏（最多 5 帧）
   - 2 帧 = 1 个时间步，难以学习时序动态

### 8.2 方向瓶颈

1. **缺少真实多主体数据集验证**
   - 合成数据验证成功，但真实场景更复杂
   - AVA 适合但数据量大（30GB+）

2. **MOT 模块在真实场景的表现未知**
   - YOLO+ByteTrack 在合成数据上完美（mAP=0.995）
   - 但在真实场景（遮挡、运动模糊、小物体）可能下降

---

## 九、距离初始目标的差距

### 9.1 已完成

| 目标 | 状态 | 说明 |
|------|------|------|
| Mask-Guided LAM 架构 | ✅ 完成 | 从 Slot Attention 重写为 Mask-Guided |
| 合成数据验证 | ✅ 完成 | 动作准确率 87-89%，隐动作空间结构化 |
| Interaction Module | ✅ 完成 | 让不同主体的隐动作独立 |
| YOLO+ByteTrack 融合 | ✅ 完成 | 合成数据闭环训练成功 |
| bbox 矩形填充验证 | ✅ 完成 | 与 GT mask 几乎无差异 |
| 隐动作聚类分析 | ✅ 完成 | UMAP + ARI + 线性探针 |

### 9.2 待完成

| 目标 | 状态 | 说明 |
|------|------|------|
| 真实多主体数据集验证 | ⏳ 待做 | AVA 适合但数据量大 |
| 小物体检测优化 | ⏳ 待做 | ball/bird 检测率低（20-30%） |
| 长时序建模 | ⏳ 待做 | A2D 标注帧稀疏，需要更多时间步 |
| 端到端训练 | ⏳ 待做 | YOLO+ByteTrack+LAM 联合训练 |
| 实时推理部署 | ⏳ 待做 | 模型压缩、量化 |

### 9.3 核心差距

**初始目标**：
> "改进 LAM，使其能够处理多物体运动场景，学习不同主体之间的交互关系"

**当前进展**：
- ✅ 架构改进完成（Mask-Guided + Interaction Module）
- ✅ 合成数据验证成功（动作准确率 87-89%）
- ✅ MOT 模块融合成功（YOLO+ByteTrack）
- ⏳ 真实场景验证待做（A2D 不适合，AVA 待下载）

**差距**：
1. **缺少真实多主体场景的定量验证**
   - 合成数据是理想场景（无遮挡、无漏检）
   - 真实场景更复杂，需要 AVA 等数据集验证

2. **小物体检测是薄弱环节**
   - ball/bird 检测率低（20-30%）
   - 可能需要多尺度检测或专门的小物体模型

3. **长时序建模未充分验证**
   - 合成数据只有 5 帧
   - A2D 标注帧稀疏
   - 需要更长的视频验证时序建模能力

---

## 十、下一步建议

### 10.1 短期（1-2 周）

1. **优化小物体检测**
   - 尝试 YOLOv8 多尺度训练
   - 或使用专门的小物体检测模型（如 YOLOv8-p2）

2. **人工检查零样本 YOLO 检测结果**
   - 可视化已保存至 `results/yolo_zeroshot_vis`
   - 判断"额外检测"是否是真实物体

3. **在合成数据上增加难度**
   - 增加遮挡、运动模糊
   - 增加主体数量（从 4 → 8）
   - 验证 LAM 在更复杂场景的表现

### 10.2 中期（1-2 月）

1. **下载 AVA 数据集**
   - 430 个 15 分钟视频
   - 每帧标注所有人物
   - 真正的多主体场景验证

2. **端到端训练**
   - YOLO+ByteTrack+LAM 联合训练
   - 检测与隐动作协同优化

3. **长时序建模**
   - 修改数据加载，采样 10-20 帧
   - 验证 LAM 的时序建模能力

### 10.3 长期（3-6 月）

1. **实时推理部署**
   - 模型压缩、量化
   - 在边缘设备上验证

2. **扩展到交互预测**
   - 从隐动作预测主体间交互
   - 扩展到交互建模

---

## 十一、总结

### 核心成果

1. **Mask-Guided LAM 架构**：从 Slot Attention 重写，用 GT mask 直接池化主体特征
2. **Interaction Module**：让不同主体的隐动作独立，隐动作空间结构化
3. **YOLO+ByteTrack 融合**：轻量化 MOT 方案，合成数据闭环训练成功
4. **bbox 矩形填充验证**：与 GT mask 几乎无差异，SAM2 不必要

### 核心发现

1. **Interaction Module 不提升动作准确率，但让隐动作空间结构化**
2. **零样本 YOLO 检测比微调后更全面**（Recall 67.79% vs 59%）
3. **A2D 不适合多主体评估**（62% 只标 1 个主体）
4. **AVA 是最适合的数据集**（每帧标注所有人物）

### 当前瓶颈

1. 缺少真实多主体数据集验证
2. 小物体检测率低（ball/bird 20-30%）
3. 长时序建模未充分验证

### 距离目标差距

架构改进和合成数据验证已完成，但真实场景验证待做。建议下载 AVA 数据集进行真实多主体场景验证。

---

**报告完成日期**：2025-06-05