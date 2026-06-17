# AdaWorld 隐动作学习改进规划：多主体感知的隐动作模型

## 一、问题分析

### 1.1 当前架构的核心缺陷

当前 LAM (Latent Action Model) 使用**单个** `action_prompt` token 聚合帧间变化信息。其工作流程为：

```
输入: (B, T, H, W, C) 视频
  → patchify → (B, T, N, patch_token_dim)    # N=256 个 patch
  → 拼接 action_prompt → (B, T, 1+N, patch_token_dim)  # 仅 1 个 prompt
  → SpatioTemporalTransformer 编码
  → 取 prompt token 输出 → (B, T-1, 1, model_dim)  # 单个全局隐动作
  → VAE → z_rep: (B, T-1, 1, latent_dim=32)
```

**核心问题**：当画面中存在多个主体时，不同主体的运动变化被压缩到同一个隐动作向量中，导致：
- **运动抵消**：主体A向左移动、主体B向右移动 → 全局隐动作可能接近零向量
- **无法区分主体与动作**：隐动作编码了"什么发生了变化"，但无法编码"是谁发生了变化"
- **聚类结构缺失**：A2D 数据集上的聚类验证表明，学习到的隐动作没有按 actor/action 形成有意义的聚类结构

### 1.2 简单分割的局限性

若对多主体分别进行分割提取，会丧失主体之间的**交互信息**。主体间的空间关系、因果关系对于世界模型的预测至关重要。

### 1.3 需求

我们需要一种机制，能够：
1. **显式建模多个主体**：每个主体拥有独立的隐动作表示
2. **保留主体间交互信息**：主体之间通过注意力机制交互
3. **保持端到端可学习**：不依赖外部分割模块的梯度截断

---

## 二、改进方案设计

### 2.1 核心思想：从单 Action Prompt 到多 Object Slot

将单一的 `action_prompt` 替换为 **K 个 Object Slot**，每个 Slot 竞争性地关注画面中的不同主体，从而实现主体-动作的解耦。

灵感来源：
- **Slot Attention** (Locatello et al., 2020)：通过竞争性注意力将场景分解为对象级别的表示
- **Multi-Agent Motion Forecasting as Language Modeling**：用 attention 建模多智能体之间的交互关系

### 2.2 架构改动概览

```
当前架构:
  action_prompt: (1, 1, 1, patch_token_dim)     → 单个全局隐动作
  padded_patches: (B, T, 1+N, patch_token_dim)  → 1个prompt + N个patch

改进架构:
  object_slots: (1, 1, K, patch_token_dim)       → K个对象槽
  padded_patches: (B, T, K+N, patch_token_dim)   → K个slot + N个patch
  → 编码后取 K 个 slot 的输出 → (B, T-1, K, model_dim)  → K个独立隐动作
```

关键变化：
- `action_prompt` → `object_slots`：从 1 个变为 K 个可学习 token
- 编码后提取**所有 K 个 slot** 的输出，而非仅第 0 个
- 每个 slot 独立经过 per-slot VAE（见 2.5 节），产生 K 个隐动作表示
- 解码时通过 Cross-Attention 一次前融合将 K 个隐动作注入 patch 空间（见 2.4 节）

**位置编码适配**：当前 `PositionalEncoding` 使用 sinusoidal 编码（`max_len=5000`），支持任意序列长度，因此从 1+N 扩展到 K+N 无需修改位置编码实现。但需注意：slot token 与 patch token 共享同一套位置编码，slot 占据前 K 个位置（索引 0~K-1），patch 占据索引 K~K+N-1。这意味着 slot 的位置编码不对应任何实际空间位置，而是作为"虚拟位置"参与注意力计算——这与 DETR 的 object query 设计一致。

### 2.3 Slot Attention 竞争机制（可选增强）

为了让不同 slot 关注不同主体，可在 SpatioTemporalTransformer 的每一层中引入 Slot Attention 竞争机制：

```python
# 标准 Self-Attention：每个 token 独立计算 attention
# Slot Attention：slot 之间通过 softmax 竞争，确保不同 slot 关注不同区域

# 具体实现：对 slot → patch 的 attention，在 patch 维度做 softmax（标准），
# 但额外在 slot 维度做 softmax，使得每个 patch 最多被一个 slot 强烈关注
```

这是 Slot Attention 的核心思想——通过**双重 softmax** 实现软聚类。

**初步方案**：先不引入 Slot Attention 竞争机制，仅使用多个可学习 slot token。原因：
- 标准 Transformer 的自注意力本身具有一定的"竞争"特性（不同 query 可以关注不同 key）
- 先验证最简单的多 slot 方案是否有效，再逐步增强

### 2.4 解码器的融合策略

当前解码器将隐动作加性调制到视频 patch 上：
```python
video_action_patches = video_patches + action_patches  # (B, T-1, N, model_dim)
```

改进后，K 个隐动作需要融合。可选策略：

**策略 A：Cross-Attention 融合（推荐）**
```python
# video_patches 作为 Query，K个 action_slots 作为 Key/Value
# 每个 patch 通过 cross-attention 从 K 个 slot 中选择性地获取动作信息
video_action_patches = CrossAttention(q=video_patches, kv=action_slots)
video_action_patches = video_action_patches + video_patches  # 残差连接
```
- 优势：patch 可以自适应地关注相关的 slot，天然实现"哪个主体影响了哪个区域"
- 与世界模型中的 cross-attention 机制一致

**策略 B：加权求和融合**
```python
# 学习一个 (N, K) 的分配矩阵，将 K 个 slot 的动作加权分配给 N 个 patch
weights = softmax(Linear(video_patches) @ Linear(action_slots).T)  # (B, T-1, N, K)
action_patches = weights @ action_slots  # (B, T-1, N, model_dim)
```

**策略 C：简单相加**
```python
# K 个 slot 的动作全部加到每个 patch 上
action_patches = action_slots.sum(dim=2, keepdim=True).expand(-1, -1, N, -1)
video_action_patches = video_patches + action_patches
```

**优先选择策略 A**，因为它最符合"主体-区域"的自然对应关系。

### 2.5 损失函数调整

当前损失：
```python
loss = MSE(recon, gt) + beta * KL(q(z|x) || p(z))
```

改进后，K 个 slot 各自有独立的 KL 项：
```python
# 每个 slot 独立的 KL 散度
kl_loss = sum over k: (-0.5 * sum(1 + z_var_k - z_mu_k^2 - exp(z_var_k)))

# 可选：添加 slot 分离损失，鼓励不同 slot 关注不同区域
# 例如：slot attention map 的熵正则化

loss = MSE(recon, gt) + beta * kl_loss
```

### 2.6 完整改进架构

```
输入: videos (B, T, H, W, C)
  |
  v
[patchify] → patches: (B, T, N, patch_token_dim)
  |
  v
[拼接 object_slots] → padded_patches: (B, T, K+N, patch_token_dim)
  |                        K 个可学习 slot token
  v
[SpatioTemporalTransformer 编码器] (16层)
  |  空间注意力: 让 slot 与 patch 交互，slot 学习关注不同空间区域
  |  时间注意力: 让 slot 在时间维度上捕捉运动变化
  v
编码输出: z: (B, T, K+N, model_dim)
  |
  v
[提取 K 个 slot] → z_slots: (B, T-1, K, model_dim)  # 取 t>=1 帧的 slot 输出
  |
  v
[独立 VAE] → z_rep: (B, T-1, K, latent_dim)  # K 个独立隐动作
  |
  v
解码:
  video_patches = patch_up(patches[:, :-1])        # (B, T-1, N, model_dim)
  action_slots = action_up(z_rep)                   # (B, T-1, K, model_dim)
  video_action_patches = CrossAttn(q=video_patches, kv=action_slots) + video_patches
  |
  v
[SpatioTransformer 解码器] (16层)
  |
  v
[unpatchify + sigmoid] → recon: (B, T-1, H, W, C)
```

---

## 三、分阶段实施计划

### 阶段一：合成数据生成与基线验证

**目标**：构建 8×8 网格合成数据集，验证当前 LAM 在多主体场景下的缺陷。

#### 3.1.1 合成数据集设计

- **网格规格**：8×8 网格，每个格子为一个基本单元
- **图像分辨率**：256×256（与原 LAM 一致），每个网格单元 32×32 像素
- **主体类型**：
  - 正方形（占 1 个格子，32×32 像素）
  - 圆形（占 1 个格子，内切圆）
  - 三角形（占 1 个格子）
  - 不同颜色区分不同主体
- **主体数量**：每帧 2-5 个主体（随机）
- **动作空间**：每个主体每帧独立执行 5 种动作之一：上/下/左/右/静止
- **视频长度**：2 帧（与当前 LAM 训练设置一致），也可生成长视频用于扩展实验
- **背景**：纯黑或纯灰背景
- **标注**：每个样本记录每个主体的 (ID, 位置, 动作)

#### 3.1.2 数据生成实现

- 创建 `lam/lam/synthetic_dataset.py`，实现 PyTorch Dataset
- 在线生成（无需预存到磁盘），支持随机种子复现
- 输出格式与现有 `dataset.py` 一致：`{"videos": (T, H, W, C)}`
- 额外输出标注信息：`{"actor_positions": (T, num_actors, 2), "actor_actions": (T, num_actors), "actor_ids": (num_actors,)}`

#### 3.1.3 基线验证实验

1. **用当前 LAM 在合成数据上训练**：观察重建质量
2. **隐动作聚类分析**：提取 z_mu，按 ground truth 动作标签聚类
   - 若隐动作不能区分不同主体的动作 → 验证了问题假设
3. **隐动作与动作的相关性分析**：
   - 同一主体相同动作 → 隐动作应相似
   - 不同主体相同动作 → 隐动作是否混淆
   - 多主体同时反向运动 → 隐动作是否抵消

### 阶段二：多 Object Slot 架构实现

**目标**：实现多 slot 架构，在合成数据上验证主体-动作解耦能力。

#### 3.2.1 核心模块修改

**文件 `lam/lam/modules/lam.py` — `LatentActionModel` 类**：

1. 将 `action_prompt` (1,1,1,D) 替换为 `object_slots` (1,1,K,D)：
   ```python
   self.object_slots = nn.Parameter(torch.empty(1, 1, num_slots, patch_token_dim))
   nn.init.uniform_(self.object_slots, a=-1, b=1)
   ```

2. 修改 `encode` 方法：
   ```python
   slot_pad = self.object_slots.expand(B, T, -1, -1)  # (B, T, K, D)
   padded_patches = torch.cat([slot_pad, patches], dim=2)  # (B, T, K+N, D)
   z = self.encoder(padded_patches)  # (B, T, K+N, E)
   z = z[:, 1:, :K]  # (B, T-1, K, E) 取 K 个 slot
   ```

3. 修改 VAE 层：对 K 个 slot 独立进行 VAE 编码
   ```python
   z = z.reshape(B * (T - 1) * K, self.model_dim)
   moments = self.fc(z)  # 或 per-slot fc
   ```

4. 修改 `forward` 方法的解码部分：用 Cross-Attention 融合替代简单相加

**文件 `lam/lam/modules/blocks.py`**：

1. 新增 `CrossAttention` 模块（用于解码器的 slot-patch 融合）
2. 修改 `SpatioBlock` 或新增 `SpatioCrossAttnBlock` 以支持 cross-attention 解码
3. 位置编码需适配 K+N 的序列长度

#### 3.2.2 配置文件

新建 `lam/config/lam_synthetic.yaml`：
- 继承 `lam.yaml` 的基本配置
- 添加 `num_slots` 超参数（默认 K=4）
- 数据源设为 `synthetic`
- 降低模型规模以便快速迭代（如 enc_blocks=8, dec_blocks=8, model_dim=512）

#### 3.2.3 验证实验

1. **重建质量**：多 slot 模型 vs 单 prompt 模型
2. **隐动作聚类分析**：
   - 按 ground truth 动作对每个 slot 的 z_mu 聚类
   - 期望：每个 slot 的隐动作应与该 slot 关注的主体的动作强相关
3. **Slot-主体对应分析**：
   - 可视化每个 slot 对空间位置的 attention weight
   - 期望：不同 slot 关注不同的空间区域/主体
4. **多主体运动不抵消**：验证反向运动的主体的隐动作不再抵消

### 阶段三：Slot Attention 竞争机制（可选增强）

**目标**：引入 Slot Attention 竞争机制，强化 slot 的主体分离能力。

#### 3.3.1 实现

在 SpatioTemporalTransformer 的空间注意力中，对 slot-to-patch 的注意力添加竞争约束：

```python
# 标准注意力: softmax over key dim (每个 query 独立)
# Slot 竞争: 额外 softmax over slot dim (slot 之间竞争)

attn = Q @ K.T / sqrt(d)  # (B, K+N, K+N)
# 对 slot 部分，在 slot 维度做额外 softmax
slot_attn = attn[:, :K, K:]  # slot → patch 的注意力
slot_attn = softmax(slot_attn, dim=1)  # 在 slot 维度竞争：每个 patch 最多被一个 slot 强关注
```

#### 3.3.2 验证

- 对比有无竞争机制的 slot-主体对应质量
- 分析 slot collapse（多个 slot 关注同一主体）的问题是否缓解

### 阶段四：真实数据集验证

**目标**：在 A2D 等真实数据集上验证改进效果。

#### 3.4.1 A2D 数据集实验

1. 在 A2D 上训练多 slot LAM
2. 利用 A2D 的 actor/action 标注进行评估：
   - 每个 slot 的隐动作是否按 actor 聚类
   - 每个 slot 的隐动作是否按 action 聚类
   - 与原版 LAM 的聚类结果对比
3. 可视化 slot attention map 与真实 actor 边界框的对应关系

#### 3.4.2 集成 MOT 模块

当合成数据验证通过后，引入真实的 MOT（Multi-Object Tracking）模块：
- 使用现有 MOT 工具（如 ByteTrack、BoT-SORT）从视频提取多主体轨迹
- MOT 提供的主体信息作为 slot 的初始化或辅助监督
- 评估 MOT 误差对整体性能的影响

### 阶段五：世界模型集成

**目标**：将改进后的多 slot LAM 集成到世界模型中。

#### 3.5.1 条件编码器适配

当前世界模型通过 `LAMEmbedder` 使用 LAM 的隐动作作为条件：
```python
z_rep: (B, T-1, 1, latent_dim)  # 单个隐动作
```

改进后：
```python
z_rep: (B, T-1, K, latent_dim)  # K 个隐动作
```

需要修改 `worldmodel/vwm/modules/encoders/modules.py` 中的 `LAMEmbedder`，处理 K 个 slot 的嵌入：
- 方案 A：将 K 个 slot flatten 后整体编码
- 方案 B：对 K 个 slot 分别编码后拼接
- 方案 C：用 attention 聚合 K 个 slot

#### 3.5.2 适配训练适配

适配阶段（ActionBook / ActionMLP）也需要相应调整，以映射到 K-slot 隐动作空间。

---

## 四、代码修改清单

### 阶段一（合成数据 + 基线验证）

| 文件 | 操作 | 说明 |
|------|------|------|
| `lam/lam/synthetic_dataset.py` | 新建 | 8×8 网格合成数据集 |
| `lam/lam/dataset.py` | 修改 | 添加 `synthetic` 数据源支持 |
| `lam/config/lam_synthetic.yaml` | 新建 | 合成数据训练配置 |
| `lam/lam/model.py` | 修改 | 添加聚类验证相关代码 |

### 阶段二（多 Object Slot 架构）

| 文件 | 操作 | 说明 |
|------|------|------|
| `lam/lam/modules/lam.py` | 修改 | 核心：action_prompt → object_slots，编码/解码改造 |
| `lam/lam/modules/blocks.py` | 修改 | 新增 CrossAttention，修改解码器 Block |
| `lam/lam/model.py` | 修改 | 适配新的 LatentActionModel 接口 |
| `lam/config/lam_synthetic.yaml` | 修改 | 添加 num_slots 参数 |

### 阶段三（Slot Attention 竞争机制）

| 文件 | 操作 | 说明 |
|------|------|------|
| `lam/lam/modules/blocks.py` | 修改 | 在空间注意力中添加 slot 竞争机制 |

### 阶段四（真实数据集验证）

| 文件 | 操作 | 说明 |
|------|------|------|
| `lam/config/lam_a2d_multislot.yaml` | 新建 | A2D 多 slot 训练配置 |
| `lam/lam/model.py` | 修改 | 添加 A2D 评估指标 |

### 阶段五（世界模型集成）

| 文件 | 操作 | 说明 |
|------|------|------|
| `worldmodel/vwm/modules/encoders/modules.py` | 修改 | LAMEmbedder 适配多 slot |
| `worldmodel/external/lam/` | 同步 | 与 lam/ 的修改保持一致 |
| `worldmodel/configs/training/adaworld.yaml` | 修改 | 添加 num_slots 配置 |

---

## 五、关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_slots` (K) | 4 | Object Slot 数量，对应最大主体数 |
| `slot_init` | uniform(-1, 1) | Slot 初始化方式 |
| `cross_attn_heads` | 16 | 解码器 cross-attention 头数 |
| `beta` | 0.0002 | KL 散度权重（保持不变） |
| `latent_dim` | 32 | 每个 slot 的隐动作维度 |

---

## 六、评估指标

1. **重建质量**：PSNR, SSIM（与原版对比，不应显著下降）
2. **隐动作聚类质量**：
   - Adjusted Rand Index (ARI)：衡量聚类与 ground truth 的一致性
   - 聚类纯度 (Purity)：按 action 标签评估
3. **Slot-主体对应质量**：
   - Slot Attention Map 与 Ground Truth Bounding Box 的 IoU
   - Slot 分配的一致性（同一主体在不同帧是否被同一 slot 关注）
4. **动作识别准确率**：从隐动作预测 ground truth 动作的准确率
5. **世界模型下游性能**：适配后的动作可控性、预测质量

---

## 七、风险与对策

| 风险 | 对策 |
|------|------|
| Slot Collapse（多个 slot 关注同一主体） | 引入 Slot Attention 竞争机制；添加 slot 分离正则化损失 |
| 重建质量下降（模型复杂度增加） | 先在合成数据上充分调参，再迁移到真实数据 |
| Slot 数量 K 难以确定 | 设为可配置参数；实验验证不同 K 的影响；可尝试自适应 K |
| 合成数据与真实数据 gap | 合成数据仅用于验证架构改进，最终以真实数据结果为准 |
| 世界模型集成复杂度 | 分阶段进行，先验证 LAM 独立效果 |

---

## 八、实施优先级

1. **阶段一**（最高优先）：合成数据生成 + 基线验证 → 确认问题存在
2. **阶段二**（高优先）：多 Object Slot 架构 → 核心改进
3. **阶段三**（中优先）：Slot Attention 竞争 → 可选增强
4. **阶段四**（高优先）：A2D 真实数据验证 → 论文核心实验
5. **阶段五**（中优先）：世界模型集成 → 完整系统验证
