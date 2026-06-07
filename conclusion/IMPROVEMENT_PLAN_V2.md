# AdaWorld 隐动作学习改进规划 v2：基于已知分割的多主体交互建模

## 一、根本性方向纠正

### 1.1 之前的错误方向

我们之前的核心假设是错误的：

> **错误假设**：需要用 Slot Attention 无监督地"发现"并分割画面中的不同主体，让 K 个可学习 slot token 通过注意力竞争来追踪不同的运动物体。

**具体表现**：
- `object_slots` 作为可学习 token，通过 Transformer 注意力机制与 patch 交互
- 引入 Slot Attention 竞争机制（双重 softmax），让 slot 在空间维度竞争
- 添加空间掩码解码器（`SlotSpatialDecoder`），让每个 slot 预测一个空间区域
- 用 ARI 评估 slot 是否"发现"了不同主体
- 添加掩码熵损失、slot 多样性损失等辅助目标

**为什么失败**：
- 合成数据集已经提供了完美的 ground-truth 分割（`masks`、`positions`），但模型完全没有利用这些信息
- Slot Attention 的设计初衷是**无监督 object discovery**（从原始图像中发现物体），而我们的场景中物体已经被分割好了
- 强行用重建损失驱动 slot 去学习分割，信号太弱且不稳定——纯视觉重建不包含足够的语义信息来区分"哪个 patch 属于哪个主体"
- 这就像已经有了 GPS 定位，却还在训练模型去学怎么看地图找路

### 1.2 正确的方向

> **正确理解**：我们不是在做无监督的物体发现，而是在**已知分割的情况下建模多主体之间的交互关系**。

原计划中的 MOT（Multi-Object Tracking）模块的作用就是提供分割和跟踪信息。合成数据集的价值在于它直接提供了这些 ground-truth 信息，让我们可以跳过 MOT 模块，专注于验证核心问题：**能否学到有意义的、区分不同主体的隐动作表示**。

**新的架构思路**：

```
之前（错误）:
  图像 → [patchify] → [K个可学习slot + N个patch] → [Transformer] → K个隐动作
  （slot 不知道自己代表哪个物体，全靠注意力去"猜"）

现在（正确）:
  图像 + GT Masks → [patchify] → [按mask池化每个主体的特征] → K个主体特征
       → [Per-object VAE] → K个隐动作 z_k
       → [交互建模] → 建模主体之间的关系
       → [预测/重建] → 预测下一帧或每个主体的未来状态
```

---

## 二、新架构设计

### 2.1 核心思想：Mask-Guided Per-Object Feature Extraction

不再使用可学习的 slot token，而是**直接利用 ground-truth mask 来提取每个主体的特征**。

```python
# 伪代码：per-object feature extraction
def extract_object_features(patches, masks, num_objects):
    """
    patches: (B, T, N_patches, D_patch)  — 图像的 patch 特征
    masks:   (B, T, max_actors, H, W)     — 每个主体的二值 mask
    num_objects: (B,)                     — 实际主体数量

    返回: object_features: (B, T, max_actors, D_feature) — 每个主体的特征
    """
    # 将 mask 下采样到 patch 级别
    mask_patches = downsample_mask_to_patches(masks)  # (B, T, A, N_patches)

    # 对每个主体，在其 mask 区域内池化 patch 特征
    features = []
    for a in range(max_actors):
        mask_a = mask_patches[:, :, a, :]  # (B, T, N_patches)
        feat_a = masked_pool(patches, mask_a)  # (B, T, D_feature)
        features.append(feat_a)

    return stack(features, dim=2)  # (B, T, A, D_feature)
```

### 2.2 完整架构

```
输入:
  videos:  (B, T, C, H, W)          — 视频帧
  masks:   (B, T, A, H, W)         — GT 分割 mask（A = max_actors）
  actions: (B, T-1, A)             — GT 动作标签

步骤 1: 视觉编码
  videos → [CNN/ViT Encoder] → visual_features: (B, T, N, D)

步骤 2: Mask-Guided 主体特征提取
  visual_features + masks → [Mask Pooling] → obj_features: (B, T, A, D)
  - 对每个主体 a，在 mask_a 覆盖的 patch 上做加权平均池化
  - 未被任何 mask 覆盖的 patch 归入 "background" 主体

步骤 3: 时序变化编码（LAM 核心）
  obj_features_t0 = obj_features[:, 0, :, :]   # (B, A, D)  第 0 帧
  obj_features_t1 = obj_features[:, 1:, :, :]  # (B, T-1, A, D)

  对每个主体 a:
    delta_a = obj_features_t1[:, :, a, :] - obj_features_t0[:, a, :]  # (B, T-1, D)
    z_mu_a, z_var_a = VAE_fc(delta_a)                                # (B, T-1, latent_dim)
    z_rep_a = reparameterize(z_mu_a, z_var_a)

  所有主体的隐动作: z_rep: (B, T-1, A, latent_dim)

步骤 4: 交互建模（新增模块）
  z_rep → [Object Interaction Module] → interacted_z: (B, T-1, A, latent_dim)
  - 使用 Self-Attention 让不同主体的隐动作互相"看到"
  - 可选：加入相对位置编码（来自 positions 标签）

步骤 5: 解码 / 预测

  选项 A — 帧重建（保持与原 LAM 兼容）:
    interacted_z → [Object-to-Patch Projection] → action_tokens
    action_tokens + visual_features → [Decoder] → recon_frames

  选项 B — 动作预测（更直接的验证）:
    interacted_z → [Action Classifier] → predicted_actions: (B, T-1, A, num_actions)
    loss = CrossEntropy(predicted_actions, gt_actions)

  选项 C — 位置预测（利用 position 标签）:
    interacted_z → [Position Predictor] → pred_positions: (B, T-1, A, 2)
    loss = MSE(pred_positions, gt_positions[:, 1:, :])
```

### 2.3 与原 LAM 的关键区别

| 维度 | 原 LAM | 旧方案（错误） | 新方案（正确） |
|------|--------|----------------|---------------|
| 主体发现方式 | 不区分主体 | Slot Attention 无监督发现 | **GT Mask 直接提供** |
| 输入 | 仅视频 | 视频 + 可学习 slot | **视频 + GT masks + positions** |
| 特征提取 | 全局平均/单 prompt | Slot→Patch 注意力 | **Mask 内池化** |
| 隐动作维度 | (B, T-1, 1, D) | (B, T-1, K, D) | **(B, T-1, A, D)** |
| 主体间交互 | 无 | 隐式（通过共享 decoder） | **显式（Interaction Module）** |
| 评估方式 | 聚类质量 | Slot-物体 IoU / ARI | **动作预测准确率 / 位置预测误差** |

### 2.4 为什么这个方向更合理

1. **信息充分利用**：合成数据集提供了 masks、positions、actions 三种标注，旧方案只用到了 videos，新方案全部利用
2. **信号清晰明确**：不需要通过间接的重建损失去"猜测"哪个 slot 应该关注哪个物体，mask 直接告诉了答案
3. **可解释性强**：每个隐动作向量对应一个真实存在的主体，可以直接验证其是否编码了该主体的动作
4. **与最终系统一致**：在真实场景中，MOT 模块提供的就是这种分割+跟踪信息；合成数据只是模拟了 MOT 的输出
5. **评估更直接**：不需要 ARI 这种间接指标，可以直接用动作分类准确率或位置预测误差

---

## 三、实施计划

### 阶段一：Mask-Guided Feature Extraction（最高优先级）

**目标**：实现基于 GT mask 的逐主体特征提取，替代可学习 slot token。

#### 3.1.1 修改文件

**`lam/lam/modules/lam.py`** — 重写 `LatentActionModel`：

```python
class LatentActionModelV2(nn.Module):
    """
    基于 GT Mask 的多主体隐动作模型。

    与 v1 的关键区别：
    - 不再使用可学习的 object_slots
    - 直接利用 GT masks 提取每个主体的特征
    - 每个主体独立经过 VAE 编码
    - 新增 Object Interaction Module 建模主体间交互
    """

    def __init__(self, ..., max_actors=4, use_interaction=True):
        # 视觉编码器（保持不变）
        self.encoder = SpatioTemporalTransformer(...)

        # Mask 池化层：将 patch 特征按 mask 区域聚合为主体特征
        self.mask_pool = MaskedPool(model_dim, model_dim)

        # Per-object VAE（每个主体独立的 fc 层）
        self.obj_vae = nn.ModuleList([
            nn.Linear(model_dim, latent_dim * 2)
            for _ in range(max_actors)
        ])

        # 交互建模模块（可选）
        if use_interaction:
            self.interaction = ObjectInteractionModule(
                latent_dim, num_heads=4, num_layers=2
            )

        # 解码器
        self.decoder = ...
```

**`lam/lam/modules/blocks.py`** — 新增模块：

```python
class MaskedPool(nn.Module):
    """在给定 mask 区域内对 patch 特征做加权平均池化。"""

class ObjectInteractionModule(nn.Module):
    """建模多个主体之间的交互关系。
    
    输入: (B, A, D) — A 个主体的隐动作
    输出: (B, A, D) — 经过交互后的隐动作
    
    使用 Self-Attention + 相对位置编码。
    """
```

#### 3.1.2 数据流适配

**`lam/lam/disk_synthetic_dataset.py`** — 保持不变，已提供所需数据。

**`scripts/run_single.py`** — 重写训练循环：
- 将 `masks` 和 `positions` 送入模型
- 损失函数改为动作预测损失（而非纯重建损失）

### 阶段二：多任务联合训练

**目标**：同时优化重建质量和动作预测能力。

#### 3.2.1 损失函数设计

```python
# 总损失 = 重建损失 + 动作预测损失 + KL 正则化
loss = (
    λ_recon * MSE(recon, gt_frame)      # 帧级重建
    + λ_action * CE(pred_actions, gt_actions)  # 动作分类（每主体独立）
    + λ_pos * MSE(pred_pos, gt_position)       # 位置回归（可选）
    + β * KL(q(z|x) || p(z))                   # VAE 正则化
)

# 超参数建议:
λ_recon = 1.0
λ_action = 1.0   # 主要监督信号
λ_pos = 0.1      # 辅助监督
β = 0.0002       # 保持与原 LAM 一致
```

#### 3.2.2 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **Action Accuracy** | 每主体动作分类准确率 | > 80%（5 类随机=20%） |
| **Position Error** | 预测位置 vs GT 位置的 MSE | < 1.0 grid unit |
| **Latent Action Clustering** | 同动作类别的 z_mu 是否聚集 | ARI > 0.5 |
| **Cross-Actor Separation** | 不同主体的 z_mu 是否可分 | 类间距离 >> 类内距离 |
| **Reconstruction PSNR** | 重建质量（兼容性检查） | 不显著下降 |

### 阶段三：交互建模消融实验

**目标**：验证交互建模模块的有效性。

#### 3.3.1 消融配置

| 配置 | 交互模块 | 位置编码 | 说明 |
|------|----------|----------|------|
| Baseline | 无 | 无 | 各主体独立编码，无交互 |
| + Self-Attn | Self-Attention | 无 | 主体间全局交互 |
| + Rel-Pos | Self-Attention | 相对位置 | 加入空间关系先验 |
| + Temporal | Self-Attention + Temporal Attn | 相对位置 | 加入时序依赖 |

#### 3.3.2 关键验证点

1. **交互是否帮助动作预测？**：+Self-Attn vs Baseline 的动作准确率差异
2. **位置信息是否有用？**：+Rel-Pos vs +Self-Attn 的提升
3. **是否存在"群体行为"？**：例如两个主体相向运动时，交互模块是否能捕捉到这种协调

### 阶段四：真实数据集迁移

**目标**：将验证通过的架构迁移到 A2D 数据集。

#### 3.4.1 A2D 适配

A2D 数据集提供的是 actor-level 标注（bounding box + action label），而非像素级 mask。适配方案：

1. **Bounding Box → Mask**：将 actor bounding box 转换为近似的 binary mask
2. **或 ROI Pooling**：直接在 bounding box 区域内池化 CNN 特征（ Faster R-CNN 的 ROI Align 思路）
3. **MOT 集成**：使用预训练的 MOT 模型（如 ByteTrack）提供时序一致的跟踪结果

#### 3.4.2 与世界模型的接口

改进后 LAM 的输出格式：
```python
z_rep: (B, T-1, A, latent_dim)  # A 个主体的隐动作
```

世界模型端 `LAMEmbedder` 需要：
- 方案 A：将 A 个 slot flatten 为 (B, T-1, A*latent_dim)，作为条件向量
- 方案 B：用 Attention 聚合 A 个 slot 为单个条件向量（保留交互信息）
- 方案 C：分别处理每个 slot，生成 A 条条件路径（最灵活）

---

## 四、代码修改清单

### 阶段一（核心架构改造）

| 文件 | 操作 | 说明 |
|------|------|------|
| `lam/lam/modules/lam.py` | **重写** | 移除 object_slots / SlotSpatialDecoder，新增 MaskedPool / Per-object VAE / Interaction Module |
| `lam/lam/modules/blocks.py` | **新增** | `MaskedPool`、`ObjectInteractionModule` 类 |
| `scripts/run_single.py` | **重写** | 适配新接口，使用 masks/positions/actions 作为输入，动作预测损失 |
| `lam/lam/disk_synthetic_dataset.py` | 不变 | 已提供所需数据 |

### 阶段二（多任务训练）

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/run_single.py` | 修改 | 添加动作分类头、位置回归头、多任务损失 |

### 阶段三（消融实验）

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/run_single.py` | 修改 | 支持 --interaction_type 参数切换消融配置 |

### 阶段四（A2D 迁移）

| 文件 | 操作 | 说明 |
|------|------|------|
| `lam/lam/dataset.py` | 修改 | 从 A2D bbox 标注生成 mask 或 ROI 特征 |
| `lam/lam/model.py` | 修改 | Lightning 模型适配新接口 |

---

## 五、风险与对策

| 风险 | 对策 |
|------|------|
| Mask 池化丢失细节信息 | 使用多头池化（类似 multi-head attention 的思路），或保留局部特征 |
| 不同样本的主体数量不同 | 固定 max_actors=4，不足的用 padding + mask 标记无效 |
| 交互模块过拟合 | 使用层归一化 + Dropout，限制交互层数 ≤ 2 |
| 合成数据过于简单（格子世界） | 逐步增加复杂度：更多主体类型、更复杂的背景、更长的轨迹 |
| 从合成到真实的 gap | 合成数据仅用于验证"mask-guided 架构是否合理"，不在合成数据上调参过度 |

---

## 六、实施优先级

1. **阶段一**（立即执行）：重写核心架构，实现 Mask-Guided Feature Extraction
2. **阶段二**（紧随其后）：添加动作预测损失，验证隐动作质量
3. **阶段三**（并行）：交互建模消融实验
4. **阶段四**（后续）：A2D 真实数据迁移 + 世界模型集成

---

## 七、预期效果

相比旧方案，新方案应能实现：

1. **明确的主体-隐动作对应**：每个 z 向量直接对应一个真实主体，无需匈牙利匹配
2. **更高的动作预测准确率**：因为监督信号直接、明确（cross-entropy on actions）
3. **可解释的交互模式**：Interaction Module 的注意力权重可视化能展示哪些主体的动作互相影响
4. **更稳定的训练**：不需要 Slot Attention 的脆弱竞争机制，mask 提供了硬约束
