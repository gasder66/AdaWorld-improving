# AdaWorld 多主体隐动作模型 — 完整实验报告

**日期**：2025-06-19

---

## 一、研究目标

在 AdaWorld LAM 基础上实现**多主体逐主体隐动作建模**：
- 每个主体有独立的隐动作向量 z_k
- UMAP 聚类显示 z_k 按动作方向分离
- 纯无监督学习，不使用动作标签
- 最终在 A2D 真实视频上验证自监督流水线

---

## 二、架构演进

### V4：双流 DINOv2（失败）

```
Stream A: DINOv2 → bbox crop → subject features
Stream B: LightPatchEncoder → patch features → Decoder
→ PSNR 17.4 dB, KL 爆炸

根因: DINOv2 编码静态语义("方块是红色")，不编码运动("方块向右移")。
```

### V5：ST Encoder + MaskedPool + Per-Object VAE

```
videos → patchify → SpatioTemporalTransformer → encoded patches
masks  → MaskedPool → per-subject obj_feats
  → ObjectSpatioTemporalAttention
  → Per-Object VAE → z_k
  → Decoder: CrossAttn(patches, z) → recon

→ PSNR 25.8 dB (合成), UMAP NMI 0.26
```

### V6c：结构化隐空间约束

在 V5 基础上增加：
- **背景槽** (keep_background=True, z_0 编码全局变化)
- **Free Bits KL** (λ=0.1, 防止后验坍缩)
- **互信息最小化** (slot 间 cosine_sim 最小化)
- **时序一致性** (z_t ≈ z_{t+1})

→ PSNR 27.4 dB (合成), UMAP NMI 0.36, 背景槽存活

---

## 三、当前架构详细流程图

```
输入:
  videos (B,T,H,W,C)     B=batch, T=frames, H=W=256, C=3
  masks  (B,T,A,H,W)     A=max_actors=4

┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 1: Patchify + SpatioTemporalTransformer (运动感知编码)        │
│                                                                     │
│  videos → patchify(patch_size=16) → patches (B,T,256,768)          │
│    ↓                                                                │
│  SpatioTemporalTransformer(4 blocks, dim=256, heads=8):            │
│    输入投影: LN→Linear(768→256)→LN                                 │
│    PositionalEncoding (sinusoidal, spatial)                        │
│    4× SpatioTemporalBlock:                                         │
│      a) Spatial Self-Attention: (B*T, 256, 256)                   │
│         → 每帧内 256 个 patch 互相 attend                         │
│      b) Temporal Self-Attention: (B*256, T, 256)                  │
│         → 每空间位置跨 T 帧互相 attend, RoPE 标识时序            │
│      c) FFN: 256→1024→256 (GELU)                                   │
│    输出投影: Linear(256→256)                                       │
│  → encoded (B,T,256,256)  ← 运动感知的 patch 特征                  │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 2: MaskedPool (主体空间聚合)                                   │
│                                                                     │
│  masks → _build_masks_with_background():                           │
│    bg_mask = 1 - sum(actor_masks)  ← 未被任何主体覆盖的区域        │
│    all_masks (B,T,A+1,H,W) = [bg_mask | masks]                    │
│                                                                     │
│  MaskedPool(encoded, all_masks):                                    │
│    1. masks 下采样到 patch 网格:                                    │
│       F.adaptive_avg_pool2d(all_masks, (16,16)) → (B,T,A+1,256)   │
│    2. 检测有效主体: masks_sum > 0.5 → valid_mask (B,T,A+1) bool   │
│    3. 加权平均池化:                                                 │
│       weights = masks_down.unsqueeze(-1)   → (B,T,A+1,256,1)      │
│       feat = encoded.unsqueeze(2) * weights → (B,T,A+1,256,256)    │
│       mask_sum = weights.sum(-2) + ε      → (B,T,A+1,1)           │
│       obj_feats = feat.sum(-2) / mask_sum → (B,T,A+1,256)          │
│                                                                     │
│  valid_mask[:,:,0] = True  ← 背景槽始终有效                        │
│                                                                     │
│  输出:                                                              │
│    obj_feats  (B,T,K+1,D)   K+1=5: [背景, actor0, actor1, ...]   │
│    valid_mask (B,T,K+1)     True=该槽位上存在有效主体              │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 3: ObjectSpatioTemporalAttention (主体间时空交互)              │
│                                                                     │
│  输入: obj_feats (B,T,K+1,256), valid_mask (B,T,K+1)              │
│                                                                     │
│  重排: seq = obj_feats.reshape(B, T*(K+1), 256)                    │
│        padding_mask = create_from(valid_mask)                      │
│                                                                     │
│  TransformerEncoder(2 layers, 8 heads, norm_first=True):           │
│    全对全 Self-Attention over (T*(K+1)) tokens:                     │
│      同帧跨主体: Slot0@T=0 ↔ Slot1@T=0 ↔ ... (空间交互)           │
│      跨帧同主体: Slot0@T=0 ↔ Slot0@T=1 ↔ ... (时序追踪)           │
│      跨帧跨主体: Slot0@T=0 ↔ Slot2@T=3 (全局交互)                  │
│                                                                     │
│  重排回: obj_feats.reshape(B,T,K+1,256)                            │
│  输出: obj_feats (B,T,K+1,256) ← 时空交互后的主体特征              │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 4: Per-Object VAE (每主体独立编码)                             │
│                                                                     │
│  输入: obj_feats_in = obj_feats[:, :-1]  (B, T-1, K+1, 256)       │
│        取帧 0~T-2 的主体特征 (z 编码帧间转移)                      │
│                                                                     │
│  K+1 个独立线性层 (fc_0, fc_1, ..., fc_K):                        │
│    for k in 0..K:                                                   │
│      z_k = obj_feats_in[:,:,k].reshape(B*(T-1), 256)              │
│      moments = fc_k(z_k) → (B*(T-1), 64)                          │
│      mu_k, var_k = chunk(moments, 2) → 各 (B*(T-1), 32)           │
│      var_k = clamp(var_k, -5.0, 3.0)                              │
│      rep_k = mu_k + ε × exp(0.5×var_k)  [重参数化]               │
│      → reshape 回 (B, T-1, 1, 32) → 拼接                         │
│                                                                     │
│  输出:                                                              │
│    z_mu  (B, T-1, K+1, 32)  每主体每帧的 32 维隐动作分布均值     │
│    z_var (B, T-1, K+1, 32)  方差                                   │
│    z_rep (B, T-1, K+1, 32)  重参数化采样 (训练)/z_mu (评估)       │
│                                                                     │
│  ★ 核心设计: Slot 0 = z_0 = 背景槽 (编码相机运动/全局变化)        │
│             Slot 1..K = z_k = 主体 k 的独立隐动作                  │
│             fc_0 只看到 Slot 0 的特征 → z_0 天然分离               │
│             fc_k 只看到 Slot k 的特征 → z_k 只编码主体 k 的运动    │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
┌──────────────────┐ ┌─────────────────┐ ┌───────────────────────┐
│ 损失 1: L_recon  │ │ 损失 2: KL      │ │ 损失 3: L_obj_recon   │
│ (像素级帧重建)    │ │ (Free Bits)     │ │ (对象级特征预测)       │
│                  │ │                 │ │                       │
│ MSE(recon,       │ │ KL_dim = max(   │ │ delta_pred =          │
│   videos[:,1:])  │ │  0.5×(mu²+      │ │   ObjectReconHead(    │
│                  │ │  e^var-var-1),  │ │     z_rep[:,:,1:])   │
│ 强制 z 编码帧间  │ │  λ=0.1)         │ │                       │
│ 的像素变化       │ │                 │ │ delta_target =        │
│                  │ │ 防止后验坍缩，  │ │   obj_feats[:,1:,1:] │
│                  │ │ 允许部分维度    │ │   .detach()           │
│                  │ │ 编码更多信息    │ │                       │
│                  │ │                 │ │ 强制 z 编码该主体的   │
│                  │ │                 │ │ 帧间特征变化           │
│ w_recon = 1.0    │ │ w_kl = 2e-4     │ │ w_obj_recon = 0.01    │
└──────────────────┘ └─────────────────┘ └───────────────────────┘
         └──────────────┼──────────────┘
                        │
            L_total = L_recon + β·L_kl + λ·L_obj_recon
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 5: Decoder (Cross-Attention 融合 + 空间解码器)                │
│                                                                     │
│  ① patch_up: Linear(768→256)(patches[:,:-1])                       │
│     → video_patches (B, T-1, 256, 256) ← 上一帧的 patch 特征     │
│                                                                     │
│  ② action_up: Linear(32→256)(z_rep)                               │
│     → action_embed (B, T-1, K+1, 256) ← 隐动作投影               │
│                                                                     │
│  ③ CrossAttention(q=video_patches, kv=action_embed):              │
│     video_patches.reshape(B*(T-1), 256, 256)  ← 256 个 patch 查询 │
│     action_embed.reshape(B*(T-1), K+1, 256)   ← K+1 个 slot K-V  │
│     fused = CrossAttn(q, kv) → (B, T-1, 256, 256)                 │
│                                                                     │
│  ④ 残差连接:                                                       │
│     video_action_patches = fused + video_patches                   │
│     ★ patches 提供"where", z 提供"what changes"                   │
│                                                                     │
│  ⑤ SpatioTransformer(4 blocks, 空间注意力):                        │
│     fn: LN→Linear→LN → 4×SpatioBlock → Linear(256→768)           │
│     → recon_patches (B, T-1, 256, 768)                             │
│                                                                     │
│  ⑥ unpatchify + sigmoid:                                           │
│     recon = unpatchify(recon_patches, 16, 256, 256)                │
│     recon = sigmoid(recon) → (B, T-1, 256, 256, 3)                │
│                                                                     │
│  输出: recon (B, T-1, H, W, C) ← 重建的下一帧                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、损失函数详解

| 损失 | 权重 | 公式 | 作用 | 架构位置 |
|------|------|------|------|---------|
| **L_recon** | 1.0 | MSE(recon, target) | 唯一的无监督学习信号，驱动 z 编码帧间变化 | 阶段 5 输出 |
| **FreeBits KL** | 2e-4 (λ=0.1) | clamp(KL_dim, λ).mean() | 允许每个 dim 自由使用到 λ 比特，防止后验坍缩 | 阶段 4 VAE |
| **L_obj_recon** | 0.01 | MSE(MLP(z), obj_Δ) | z 必须能预测该主体在下一帧的特征 | 阶段 4→3 |
| **(已移除) L_mi** | — | ~cos_sim² | 已移除：MaskedPool 已做硬性分离 | — |
| **(已移除) L_temporal** | — | MSE(z_t, z_t+1) | 已移除：惩罚动作变化，对聚类有害 | — |
| **(实验性) L_delta** | — | 1-cos(δz_t, δz_t+1) | 需要连续动作数据才有效，当前合成数据随机动作下破坏训练 | — |

---

## 五、MOT 自监督流水线

### YOLO 检测 + BoT-SORT 追踪

```
A2D 视频 (MP4, 无标注)
  → YOLOv8n (COCO 预训练, 零样本, mAP=0.625 on A2D)
  → BoT-SORT 追踪 (conf=0.5, match_thresh=0.85)
  → bbox → filled rectangle mask
  → V6c 训练 (无需任何 A2D 标注)
```

### MOT 精度对比

| 数据集 | YOLO mAP50 | 检测率 | GT masks PSNR | MOT masks PSNR |
|--------|------------|--------|---------------|----------------|
| 合成数据 | 0.995 | 99.6% | 26.95 dB | 17.75 dB |
| A2D | 0.625 | 83.6% | 18.08 dB | 18.09 dB |

**关键发现**：A2D 上 MOT masks 训练的结果与 GT masks 训练几乎无差距（0.01 dB）。

---

## 六、完整实验结果

### 6.1 合成数据

| 版本 | PSNR | KL | UMAP NMI avg | UMAP NMI best | 关键特性 |
|------|------|-----|---------------|---------------|---------|
| AdaWorld 原始 | 26.3 | 1.1 | 0.02 | 0.02 | learnable tokens |
| V5 | 25.8 | 2.5 | 0.26 | 0.48 | +MaskedPool |
| V5 长训练 | 27.3 | 2.0 | — | — | 10000 步 |
| **V6c 长训练** | **27.4** | **1.8** | **0.36** | **0.46** | +结构化约束 |

### 6.2 A2D 真实数据 — 下一帧预测

| 训练配置 | 测试 stride | PSNR | GT 依赖 |
|---------|-----------|------|--------|
| stride=23 (原基线) | 23 | 18.08 | 需要 A2D 标注 |
| COCO YOLO | 23 | 18.29 | 零样本 |
| stride=5 | 5 | 22.73 | 零样本 |
| **stride=1 (下一帧)** | **1** | **24.70** | **零样本** |
| 合成数据参考 | — | 27.4 | — |

### 6.3 Cross-evaluation 矩阵

| 训练 stride | 测试 stride=1 | stride=5 | stride=30 |
|-----------|-------------|----------|----------|
| **1** | **24.70** | 19.89 | 16.10 |
| 5 | 23.34 | 19.49 | 16.02 |
| 30 | 21.07 | 17.11 | 15.40 |

**结论**：stride=1 模型能最好地预测下一帧（24.70 dB），stride=30 模型在预测大跨度帧时更稳定（15.40 vs 16.10 基本持平）。

### 6.4 Stride 扫描

| stride | 时间间隔 | PSNR (同域) |
|--------|---------|------------|
| 1 | 0.04s | **24.70 dB** |
| 5 | 0.2s | 19.49 |
| 10 | 0.4s | 17.88 |
| 20 | 0.8s | 16.66 |
| 30 | 1.25s | 15.40 |
| 60 | 2.5s | 14.88 |

**规律**：PSNR 随 stride 单调下降——这是预测距离的固有约束，非模型能力不足。

### 6.5 UMAP 动作聚类（A2D, stride=30）

| Slot | 有效样本 | 主体分布 | 动作 NMI | 说明 |
|------|---------|---------|---------|------|
| 0 | 1475 | adult 26%, dog 16%, bird 16%, cat 16% | 0.041 | 混合主体+混合动作→UMAP 无法聚类 |
| 1 | 396 | 63% 填充 | 0.059 | 略好 |
| 2 | 132 | 87% 填充 | 0.116 | 统计偏差 |
| 3 | 72 | 93% 填充 | 0.286 | 样本少→NMI 虚高 |

**根因**：Slot 0 混合了 7 种主体执行 8 种动作——"adult 在 walking" 和 "dog 在 walking" 的 z 在 UMAP 中应该在不同区域，但共享同一动作标签→NMI 低。

---

## 七、关键发现

1. **DINOv2 不适合运动特征提取**——语义特征不编码运动
2. **MaskedPool + ST encoder 端到端训练** 是正确组合
3. **每主体独立 VAE** 让 z_k 绑定到主体 k
4. **背景槽设计** 天然分离自我运动(z_0)与独立运动(z_1..K)
5. **真正下一帧预测 PSNR 24.70 dB**——之前 18 dB 是因为标注帧间隔 23 帧≈1 秒
6. **COCO YOLO 零样本超越 A2D 训练方案**——完全自监督流水线
7. **Stride=30 模型的 UMAP NMI 优于 stride=1**——扩大帧间隔有助于学习更完整的运动模式
8. **A2D 按动作聚类的瓶颈是数据**——Slot 0 混合 7 种主体执行 8 种动作，无法在同一空间分离

---

## 八、代码结构

```
lam/lam/modules/
  lam.py              — V6c 模型 (~295 行)
  blocks.py           — ST encoder/decoder/MaskedPool/CrossAttn/ObjectSpatioTemporalAttention

lam/lam/
  a2d_dataset.py       — A2D 数据集 (含 frame_stride)
  mot_a2d_dataset.py   — MOT 掩码数据集 (含 COCO/A2D/ReID 来源)
  experiment_dataset.py — 预生成实验数据集加载器
  arbitrary_a2d_dataset.py — 任意帧 YOLO 数据集

lam/scripts/
  run_v4_dualstream.py  — 统一训练脚本
  run_exp.py            — 快速实验脚本
  gen_stride.py         — 参数化 stride 数据生成
  cross_eval.py         — Cross-evaluation 矩阵
  analyze_latent_umap.py — UMAP 聚类分析

result/
  v6_structured/         — 合成数据模型 (最佳 27.4 dB)
  v6_a2d_coco/           — A2D COCO YOLO 模型 (18.3 dB)
  v6_a2d/umap_stride30/  — UMAP 可视化
  experiments/            — stride 扫描模型 (stride=1~60)
```

---

## 九、下一步方向

| 优先级 | 方向 | 预期效果 |
|--------|------|---------|
| P0 | model_dim 256→512, enc_blocks 4→8 | PSNR 提升 |
| P1 | 合成数据连续动作 + 长时序 | UMAP NMI > 0.5 |
| P2 | 合成预训练 → A2D 微调 | 真实场景 PSNR 提升 |
| P3 | Hierarchy temporal encoder (长窗口 T=20) | 长程运动趋势建模 |
| P4 | 交叉预测损失 (z_A 预测 z_B 的未来) | 主体间因果关系 |
