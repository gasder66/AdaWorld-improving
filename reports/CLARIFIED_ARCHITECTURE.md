# 多主体隐动作模型 — 澄清后的架构设计

**日期**：2025-06-08
**目的**：梳理改进思路，区分当前代码中正确与错误的部分，明确下一步架构方向。

---

## 一、核心流程（澄清后）

```
  相邻两帧 Iₜ, Iₜ₊₁
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ 1. 多目标追踪 + 主体分割                      │
  │    YOLO / SAM2 / LocateAnything               │
  │    → 每帧 bbox 或 像素 mask                    │
  │    → 跨帧追踪，每个 slot 对应固定主体           │
  │    → 输出: (B, K) 个 bbox/mask per frame       │
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ 2. 每主体独立特征提取                          │
  │    对每个主体裁剪其 bbox 区域                   │
  │    → 编码器（DINOv2 / ViT）提取特征            │
  │    → 每个主体得到一个 d 维特征向量              │
  │    → 输出: (B, T, K, D)                       │
  │    关键：不再对全图做 patchify + 全局编码       │
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ 3. 对象级时空注意力（唯一的一次时空建模）        │
  │    输入 K 个 token（K 个主体）× T 帧           │
  │    → Transformer Self-Attention               │
  │    → 同时建模：主体间交互(空间) + 时序动态(时间) │
  │    → 输出: (B, T, K, D)  交互后的特征           │
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ 4. Per-Object VAE：每主体独立编码隐动作        │
  │    每个主体的特征向量独立过 VAE                │
  │    → z_mu, z_var (B, T-1, K, latent_dim)     │
  │    → z_rep = z_mu + noise (重参数化)           │
  │    输出: (B, T-1, K, 32)  K个隐动作向量        │
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ 5. Decoder + 对象级重建                       │
  │    a) 全局解码：用 K 个隐动作 + CrossAttn      │
  │       重建下一帧 Iₜ₊₁                         │
  │    b) 对象级重建：每个主体隐动作 → MLP         │
  │       预测下一帧该主体的特征                   │
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ 6. 损失函数（纯无监督）                        │
  │    L = L_recon + β·KL + λ·L_obj_recon        │
  │    - L_recon: 场景级帧重建 (MSE)              │
  │    - KL: 每主体 VAE 的 KL 散度                │
  │    - L_obj_recon: 每主体特征预测 (MSE)        │
  └─────────────────────────────────────────────┘
```

---

## 二、当前代码 vs 澄清后架构 — 对比分析

### 2.1 架构差异总览

| 环节 | 当前代码（需修正） | 澄清后架构（正确方向） | 问题级别 |
|------|------------------|---------------------|---------|
| **特征提取** | 全图 patchify → SpatioTemporalTransformer 编码全部 patch | 每主体裁剪 bbox → DINOv2 等提取独立特征 | ❌ 错误 |
| **主体特征获取** | Mask Pooling 在 patch 特征上用 GT mask 聚合 | 直接由 bbox crop 编码得到，无需额外池化 | ❌ 冗余 |
| **算力开销** | 编码器处理全部 256 个 patch token（含大量背景） | 只处理 K 个主体（通常 ≤ 8） | ❌ 浪费 |
| **背景建模** | 显式背景槽（A+1 模式） | 隐式包含在重建损失中（需讨论） | ❓ 待定 |
| **时空注意力** | 对象级 ObjectSpatioTemporalAttention ✅ | 保留 | ✅ 正确 |
| **Per-Object VAE** | 每主体独立编码 ✅ | 保留 | ✅ 正确 |
| **对象级重建** | 从隐动作预测下一帧主体特征 ✅ | 保留 | ✅ 正确 |
| **全局解码器** | CrossAttn(patches, slots) + SpatioTransformer ✅ | 保留但需适配新特征空间 | ⚠️ 需适配 |
| **损失函数** | L_recon + KL + L_obj_recon ✅ | 保留 | ✅ 正确 |
| **无动作监督** | 已移除 Action Head ✅ | 无监督，仅用重建信号 | ✅ 正确 |
| **无差分** | 已移除 delta 逻辑 ✅ | 无差分 | ✅ 正确 |

### 2.2 当前代码中"做对"的部分

以下模块设计正确，应当保留：

1. **ObjectSpatioTemporalAttention** ([blocks.py#L627](file:///home/xiaojy/projects/AdaWorld-improving/lam/lam/modules/blocks.py#L627))
   - 对 (B, T, K, D) 做 Transformer Self-Attention
   - 同时建模主体间交互和时序动态
   - 这是整个架构的核心创新

2. **Per-Object VAE** ([lam.py#L150](file:///home/xiaojy/projects/AdaWorld-improving/lam/lam/modules/lam.py#L150))
   - 每个主体独立编码为 32 维隐动作
   - 输出多向量 (B, T-1, A+1, 32)

3. **ObjectReconHead** ([blocks.py#L692](file:///home/xiaojy/projects/AdaWorld-improving/lam/lam/modules/blocks.py#L692))
   - 从每主体隐动作预测下一帧特征
   - 提供无监督对象级学习信号，缓解后验坍缩

4. **多 slot Cross-Attention Decoder** ([lam.py#L246](file:///home/xiaojy/projects/AdaWorld-improving/lam/lam/modules/lam.py#L246))
   - 用 K 个 slot 作为 Cross-Attention 的 key/value
   - Patch 作为 query，fuse 后重建帧

5. **Loss 设计** ([run_single.py#L124](file:///home/xiaojy/projects/AdaWorld-improving/lam/scripts/run_single.py#L124))
   - 纯无监督：L_recon + β·KL + λ·L_obj_recon
   - 无动作监督，无差分

### 2.3 当前代码中"做错"或需要修正的部分

#### ❌ 问题 1：全图 Patchify + 全局编码是冗余的

**现状**：
```python
patches = patchify(videos, self.patch_size)  # (B, T, 256, 768)
encoded = self.encoder(patches)              # (B, T, 256, 256)  ← SpatioTemporalTransformer
```

**问题分析**：
- 全图切成 16×16 patch 后做 SpatioTemporal 编码，计算量巨大
- 256 个 patch 中大量是背景，只有少数包含主体
- 后续 Mask Pooling 还要从 256 个 patch 中聚合主体特征——说明 patch 级编码做了大量无用功
- 实际上我们只需要 K 个主体各自的特征向量

**正确做法**：
```
对每个主体：
    1. 裁剪 bbox 区域
    2. 输入 DINOv2 编码器（可 frozen，仅作为特征提取器）
    3. 取 [CLS] token 或平均池化 → d 维特征向量
→ 直接得到 (B, T, K, D)
```

#### ❌ 问题 2：Mask Pooling 是中间产物，不再需要

**现状**：`MaskPool` 把 patch 特征按 mask 区域加权平均。

**问题**：
- 这是 patch 级编码的"配套产物"——因为用了 patch，所以需要 mask pooling 来提取主体特征
- 如果用 bbox crop + DINOv2，直接从源头上就得到了每主体的特征向量
- Mask Pooling 还依赖 GT mask 的精确对齐，对 mask 质量敏感

#### ❌ 问题 3：Decoder 的 patch_up 输入需要重新设计

**现状**：Decoder 接收 `patches[:, :-1]`（patchified 视频），然后做 `patch_up` 升维。
**问题**：如果编码器改为 DINOv2（输出特征维度 768 或 1024），不再有 patch 特征，Decoder 的输入需要改变。
**方案**：Decoder 仍然需要 patch 级信息来重建像素——可以用一个轻量级的 patch 编码器（如简单的 Conv/PatchEmbed），
和 DINOv2 特征提取**解耦**。

---

## 三、修正后的架构设计

### 3.1 双流架构（建议）

```
输入帧 Iₜ
    │
    ├── 流 A：主体级特征提取（DINOv2）
    │       │
    │       对每个主体 bbox crop
    │       → DINOv2 → 主体特征向量 (B, K, D₁)
    │
    ├── 流 B：像素级 patch 编码（轻量级）
    │       │
    │       Patchify + 轻量 SpatioEncoder
    │       → patch 特征 (B, N, D₂) ← 仅用于 Decoder
    │
    ▼
    ┌─────────────────────────────────────┐
    │ 对象级时空注意力                      │
    │ (B, T, K, D₁) → Transformer         │
    └─────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────┐
    │ Per-Object VAE → K 个隐动作          │
    │ (B, T-1, K, latent_dim)             │
    └─────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────┐
    │ Decoder：CrossAttn(patch, slots)    │
    │ → 重建下一帧 Iₜ₊₁                    │
    └─────────────────────────────────────┘
```

**为什么保留 patch 流？**
- Decoder 需要从 patch 重建像素，这是不可避免的
- 用一个轻量级的 patch 编码器（1-2 层 SpatioBlock，只做空间注意）即可
- DINOv2 流负责"语义级"的主体特征提取，patch 流负责"像素级"的重建
- 两者在 Decoder 的 Cross-Attention 中融合

### 3.2 等价简化方案

如果希望更简洁，也可以：
1. **用 DINOv2 冻结编码器** → 提取 K 个主体特征 (B, T, K, D)
2. **完全去掉 patch 级编码器**（不保留 patch 流）
3. **Decoder 改为直接像素生成**（如 CNN decoder 或 ViT decoder）+ CrossAttn with slot features

但这需要验证 DINOv2 特征是否包含足够信息来重建像素。更稳妥的方案是上述双流架构。

---

## 四、下一步行动建议

### Phase 1：架构重构（核心改动）

1. **新增 `SubjectFeatureExtractor` 模块**
   - 输入：帧图像 + 每个主体的 bbox
   - 过程：crop bbox → resize → DINOv2 → 特征向量
   - 输出：(B, T, K, D)

2. **修改 `LatentActionModel` 的 `encode()`**
   - 移除 `patchify()` 和 `encoder`（或改为轻量级 patch 编码）
   - 改为调用 `SubjectFeatureExtractor`
   - 保留 `ObjectSpatioTemporalAttention`（核心组件）
   - 保留 `Per-Object VAE`

3. **调整 Decoder**
   - 如果保留 patch 流：轻量级 patch 编码器 + CrossAttn
   - 如果不保留 patch 流：直接像素生成 Decoder

### Phase 2：数据流适配

4. **修改 `DiskSyntheticDataset` 或新建数据管道**
   - 支持 bbox 格式输入（当前是 mask）
   - 或者从 mask 计算 bbox

5. **训练脚本更新**
   - 适配新的数据/模型接口

### Phase 3：验证

6. **在合成数据集上验证**
   - 确认 K 个隐动作是否有效区分不同主体
   - 对比新旧架构的 ARI、线性探针准确率

7. **迁移到真实数据**
   - 集成 YOLO/SAM2 检测结果

---

## 五、常见误区澄清

| 误区 | 事实 |
|------|------|
| "必须先无监督分割出主体" | ❌ 用现成的检测/分割工具（YOLO/SAM2）即可，不需要无监督分割 |
| "每帧主体数量必须固定" | ❌ 使用 valid_mask 处理不同帧的主体数量变化 |
| "背景槽是必须的" | ❓ 可以保留（让背景有独立隐动作），也可以去掉（背景信息由重建损失隐式处理） |
| "DINOv2 需要微调" | ❓ 可以先冻结作为特征提取器，后期可以微调 |
| "patch 级编码和 DINOv2 是二选一" | ❌ 可以共存（双流架构），也可以只用 DINOv2 + 轻量 decoder |
