# Mask-Guided Latent Action Model 工作报告（修正版）

**项目目标**：改进 Latent Action Model (LAM)，使其能够处理多物体运动场景，**无监督学习**不同主体之间的交互关系。

**报告日期**：2025-06-05

---

## 重要修正（2025-06-05）

### 核心问题发现

之前的实验中，在合成数据上训练 LAM 时使用了动作标签作为监督：

```python
loss = recon_weight * MSE + kl_beta * KL + action_weight * CrossEntropy(action_logits, gt_actions)
```

这违背了 LAM 的核心目标——**无监督学习隐动作**。

### 修正措施

1. **移除动作监督**，改为纯无监督训练（重建 + KL）
2. **动作标签只能用于事后评估**（线性探针、聚类质量），不能用于训练
3. **只使用合成数据的位置信息**（mask/bbox），不使用动作信息

### 修正后的实验结果

| 模型 | PSNR | Slot cos_sim | Linear probe |
|------|------|-------------|--------------|
| **有交互模块** | 17.73 dB | 0.02-0.35 | **47.64%** |
| 无交互模块 | 18.96 dB | 0.57-0.80 | 47.69% |

**关键发现**：
- 线性探针准确率 **47%**，远高于随机基线（20%）
- 说明隐动作确实编码了动作信息，验证了无监督学习的有效性
- 交互模块让主体间隐动作更独立（cos_sim 0.02-0.35 vs 0.57-0.80）

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

**重要说明**：
- **训练时只使用位置信息**（mask/bbox），不使用动作标签
- **动作标签只用于事后评估**（线性探针、聚类质量）

### 2.2 为什么用合成数据

1. **完整标注**：像素级 mask 完整，无遮挡、无漏标
2. **可控实验**：可以精确控制主体数量、动作类型
3. **快速验证**：无需等待 MOT 模块开发，先验证 LAM 核心架构
4. **定量评估**：可以事后用动作标签评估隐动作质量

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
[Decoder] Cross-Attention 融合 + SpatioTransformer 重建
```

**重要**：无动作预测 head，纯无监督训练。

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

### 3.3 训练损失（无监督）

```python
loss = recon_weight * MSE(recon, gt) + kl_beta * KL(z_mu, z_var)
# 无动作监督！
```

---

## 四、无监督实验效果

### 4.1 重建质量

| 模型 | PSNR | Recon MSE |
|------|------|-----------|
| **有交互模块** | 17.73 dB | 0.0169 |
| 无交互模块 | **18.96 dB** | 0.0127 |

### 4.2 隐动作空间分析

**Slot cosine similarity（主体间独立性）**：

| 模型 | Slot cos_sim | 说明 |
|------|-------------|------|
| **有交互模块** | **0.02-0.35** | 主体间较独立 |
| 无交互模块 | 0.57-0.80 | 主体间混淆 |

**Slot variances（主体活跃度）**：

| 模型 | Slot 0 | Slot 1 | Slot 2 | Slot 3 |
|------|--------|--------|--------|--------|
| 有交互模块 | 0.079 | 0.081 | 0.047 | 0.035 |
| 无交互模块 | 0.131 | 0.118 | 0.057 | 0.034 |

### 4.3 事后评估：线性探针

用动作标签训练一个简单的线性分类器，验证隐动作是否编码了动作信息：

| 模型 | Linear probe accuracy | 随机基线 |
|------|----------------------|---------|
| **有交互模块** | **47.64%** | 20% |
| 无交互模块 | 47.69% | 20% |

**关键发现**：
- 线性探针准确率 **47%**，是随机基线的 **2.4 倍**
- 说明隐动作确实编码了动作信息
- 验证了无监督学习的有效性

---

## 五、Interaction Module 的作用

### 5.1 消融实验对比

| 维度 | 有交互模块 | 无交互模块 |
|------|-----------|-----------|
| **PSNR** | 17.73 dB | **18.96 dB** |
| **Slot cos_sim** | **0.02-0.35** | 0.57-0.80 |
| **Linear probe** | 47.64% | 47.69% |

### 5.2 核心结论

**Interaction Module 不提升动作识别准确率**（线性探针几乎相同），但**让隐动作空间结构化**：
- 有交互：不同主体的隐动作较独立（cos_sim 0.02-0.35）
- 无交互：不同主体的隐动作混淆（cos_sim 0.57-0.80）

---

## 六、下一步方向

### 6.1 当前验证完成

- ✅ 无监督训练成功（线性探针 47%）
- ✅ Interaction Module 作用明确（让隐动作空间结构化）
- ✅ 合成数据验证通过

### 6.2 待验证

1. **更复杂的合成数据**
   - 增加主体数量（从 4 → 8）
   - 增加遮挡、运动模糊
   - 增加帧数（从 5 → 10）

2. **真实数据验证**
   - AVA 数据集（每帧标注所有人物）
   - 需要下载 30GB+ 数据

3. **MOT 模块融合**
   - YOLO+ByteTrack（轻量化）
   - 验证在真实场景的表现

---

## 七、总结

### 核心成果

1. **修正了训练方式**：从监督训练改为无监督训练
2. **验证了无监督学习有效性**：线性探针 47%（远高于随机 20%）
3. **明确了 Interaction Module 作用**：让隐动作空间结构化，不提升准确率

### 核心发现

1. **隐动作确实编码了动作信息**（线性探针 47%）
2. **Interaction Module 让主体间隐动作独立**（cos_sim 从 0.80 → 0.35）
3. **无监督训练可行**：不需要动作标签，只用位置信息

### 当前状态

架构改进和合成数据无监督验证已完成，下一步需要：
- 更复杂的合成数据验证
- 真实数据验证（AVA）
- MOT 模块融合

---

**报告完成日期**：2025-06-05