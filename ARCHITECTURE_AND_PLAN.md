# 多主体隐动作模型：架构图与改进方案

## 一、原始 LAM 架构（当前代码 lam.py）

### 1.1 完整数据流

```
输入:
  videos: (B, T, H, W, C)     # B=batch, T=4帧, H=W=256, C=3
  masks:  (B, T, A, H, W)     # A=4 (max_actors), 二值 mask
  ─────────────────────────────────────────────────────────────

Step 1: Patchify
  videos → patchify(patch_size=16)
  patches: (B, T, N, 768)     # N = (256/16)² = 256 patches, 768 = 3×16×16

Step 2: Encoder — SpatioTemporalTransformer
  patches → LayerNorm(768) → Linear(768, 256) → LayerNorm(256)
         → PositionalEncoding(256)
         → ×4 [Spatial SelfAttn → Temporal SelfAttn(RotaryEmb) → FFN]
         → Linear(256, 256)
  encoded: (B, T, N, 256)     # 每个 patch 的时空融合特征

  内部细节 (每个 block):
    Spatial:  (B×T, N, 256) → SelfAttn → 残差 → FFN → 残差
    Temporal: (B×N, T, 256) → SelfAttn(RotaryEmb) → 残差 → FFN → 残差

Step 3: Mask Pooling — MaskedPool
  输入: encoded (B, T, N, 256), masks (B, T, A, H, W)
  
  3a. masks 下采样到 patch 级别:
    masks → adaptive_avg_pool2d → (B, T, A, N)  # 每个 patch 属于哪个主体
  
  3b. 加权平均池化:
    obj_feats = Σ(encoded × mask_weights) / Σ(mask_weights)
    obj_feats: (B, T, A, 256)    # 每个主体的聚合特征
    valid_mask: (B, T, A)         # 主体是否存在

Step 4: Temporal Differencing
  delta = obj_feats[:, 1:] - obj_feats[:, :-1]
  delta: (B, T-1, A, 256)       # 帧间特征差分 = 运动编码

Step 5: Per-Object VAE
  对每个主体 a ∈ {0,1,2,3}:
    delta_a = delta[:, :, a]           # (B, T-1, 256)
    delta_a_flat = reshape(-1, 256)    # (B×(T-1), 256)
    moments = obj_vae[a](delta_a_flat) # Linear(256, 64), 每个主体独立
    z_mu_a, z_var_a = chunk(moments)   # 各 (B×(T-1), 32)
    z_rep_a = z_mu + noise × exp(0.5×z_var)  # 重参数化
  
  合并:
    z_mu:  (B×(T-1), A, 32)    # 后验均值
    z_var: (B×(T-1), A, 32)    # 后验 log 方差
    z_rep: (B, T-1, A, 32)     # 隐动作表示 (含噪声)

Step 6: Interaction Module (可选)
  z_rep → reshape(B×(T-1), A, 32)
        → TransformerEncoder(2层, 4头, dim=32)
        → reshape(B, T-1, A, 32)
  z_rep: (B, T-1, A, 32)       # 交互后的隐动作

Step 7: Action Prediction Head (监督训练时使用)
  z_rep → LayerNorm(32) → Linear(32,64) → GELU → Linear(64,5)
  action_logits: (B, T-1, A, 5)

Step 8: Reconstruction Decoder
  8a. 特征映射:
    patches[:, :-1] → patch_up(Linear 768→256) → video_patches: (B, T-1, N, 256)
    z_rep → action_up(Linear 32→256) → action_slots: (B, T-1, A, 256)
  
  8b. Cross-Attention 融合:
    Q = video_patches  (B×(T-1), N, 256)   ← 当前帧 patch 特征
    K,V = action_slots (B×(T-1), A, 256)   ← 隐动作
    fused = CrossAttn(Q, K, V)              # (B×(T-1), N, 256)
    fused = fused + video_patches            # 残差连接
  
  8c. SpatioTransformer 解码:
    fused → LayerNorm(256) → Linear(256,256) → LayerNorm(256)
          → PositionalEncoding(256)
          → ×4 [Spatial SelfAttn → FFN]
          → Linear(256, 768)
    recon_patches: (B, T-1, N, 768)
  
  8d. Unpatchify:
    recon_patches → unpatchify → sigmoid
    recon: (B, T-1, H, W, C)     # 重建的下一帧

Step 9: 损失计算
  L_recon = MSE(recon, videos[:, 1:])                    # 场景级重建
  L_kl    = -0.5 × Σ(1 + z_var - z_mu² - exp(z_var))    # KL 散度
  L_action = CrossEntropy(action_logits, action_labels)   # 动作分类 (监督)
  L_total = L_recon + β×L_kl + γ×L_action
```

### 1.2 原始 LAM 架构图

```
videos (B,T,H,W,C) ──→ [patchify] ──→ patches (B,T,N,768)
                                              │
                                    ┌─────────▼─────────┐
                                    │   Encoder          │
                                    │   SpatioTemporal   │
                                    │   Transformer      │
                                    │   ×4 blocks        │
                                    └─────────┬─────────┘
                                              │
                              encoded (B,T,N,256)
                              ┌────────┴────────┐
                              │                 │
                    masks (B,T,A,H,W)           │
                              │                 │
                    ┌─────────▼─────────┐       │
                    │   Mask Pooling    │       │
                    │   加权平均池化     │       │
                    └─────────┬─────────┘       │
                              │                 │
                    obj_feats (B,T,A,256)       │
                              │                 │
                    ┌─────────▼─────────┐       │
                    │   Temporal Diff   │       │
                    │   delta = f[t+1]  │       │
                    │         - f[t]    │       │
                    └─────────┬─────────┘       │
                              │                 │
                    delta (B,T-1,A,256)          │
                              │                 │
                    ┌─────────▼─────────┐       │
                    │  Per-Object VAE   │       │
                    │  ×A 个独立 FC 层   │       │
                    │  Linear(256, 64)  │       │
                    └─────────┬─────────┘       │
                              │                 │
                    z_rep (B,T-1,A,32)           │
                              │                 │
                    ┌─────────▼─────────┐       │
                    │  Interaction Mod  │       │
                    │  (可选, 2层SA)     │       │
                    └─────────┬─────────┘       │
                              │                 │
              ┌───────────────┴───────────────┐ │
              │                               │ │
    ┌─────────▼──────────┐        ┌──────────▼▼────────┐
    │  Action Head       │        │  Reconstruction     │
    │  LN→FC→GELU→FC    │        │  Decoder            │
    │  (监督训练用)       │        │                     │
    └─────────┬──────────┘        │  patches[:,-1] ─→ patch_up ─→ Q  │
              │                   │  z_rep ─→ action_up ─→ K,V       │
    action_logits (B,T-1,A,5)    │  CrossAttn(Q, K, V) + 残差       │
              │                   │  SpatioTransformer ×4            │
              │                   │  unpatchify + sigmoid             │
              │                   └──────────┬──────────┘
              │                              │
              │                   recon (B,T-1,H,W,C)
              │                              │
              │              ┌───────────────▼───────────────┐
              │              │  Loss = MSE(recon, gt)         │
              │              │       + β × KL(z_mu, z_var)    │
              │              │       + γ × CE(action, label)  │
              │              └───────────────────────────────┘
```

### 1.3 原始 LAM 的问题

```
问题: 无监督训练时 (γ=0), z_rep 坍塌到接近 0

原因链:
  1. 解码器的 CrossAttn 用 video_patches 作为 Q
     → Q 已包含当前帧的完整像素信息
  2. z_rep 通过 action_up 映射为 K,V
     → CrossAttn 的输出 fused = Attn(Q, K, V)
     → 即使 K,V=0, Attn 也会返回 V 的加权平均 (=0)
     → fused ≈ 0
  3. fused + video_patches (残差连接)
     → 结果 ≈ video_patches
  4. decoder(video_patches) 可以直接从当前帧重建下一帧
     → 帧间变化极小, 复制当前帧就能得到低 MSE
  5. z_rep 不被需要 → KL 把 z_rep 推向 0 → 后验坍塌
```

---

## 二、改进架构：多主体隐动作模型

### 2.1 改进思路

核心问题：原始 LAM 的场景级重建无法驱动每个 slot 独立编码有意义的动作信息。

改进方向：
1. **对象级重建**：每个 slot 只负责重建自己 mask 区域内的像素
2. **背景槽**：背景单独占一个 slot，避免背景信息混入主体
3. **Spatio-Temporal Transformer**：替代简单差分，建模主体间交互和时序动态
4. **时序预测损失**：从当前帧 slot 特征和隐动作预测下一帧 slot 特征

### 2.2 完整数据流

```
输入:
  videos: (B, T, H, W, C)     # T=4帧, H=W=256, C=3
  masks:  (B, T, A, H, W)     # A=4, 二值 mask
  ─────────────────────────────────────────────────────────────

Step 1: Patchify
  同原始 LAM
  patches: (B, T, N, 768)     # N=256

Step 2: Encoder — SpatioTemporalTransformer
  同原始 LAM (非因果, 每帧独立编码)
  encoded: (B, T, N, 256)

Step 3: Mask Pooling (含背景)
  3a. 主体特征 (同原始 LAM):
    obj_feats: (B, T, A, 256)
    actor_valid: (B, T, A)
  
  3b. 背景特征 (新增):
    bg_mask = 1 - Σ(actor_masks)     # 背景区域 = 非任何主体的区域
    bg_feats = 加权平均池化(encoded, bg_mask)
    bg_feats: (B, T, 1, 256)
    bg_valid: (B, T, 1) = True
  
  3c. 合并:
    slot_feats = cat([bg_feats, obj_feats], dim=2)
    slot_feats: (B, T, S, 256)       # S = A + 1 = 5 (含背景)
    valid_mask: (B, T, S)

Step 4: Spatio-Temporal Transformer (替代简单差分)
  输入: slot_feats (B, T, S, 256) → permute → (B, S, T, 256)
  
  4a. 位置编码:
    slot_pe = nn.Embedding(S, 256)    # 可学习的槽位置编码
    time_pe = nn.Embedding(T, 256)    # 可学习的时间位置编码
    pe = slot_pe[:, None] + time_pe[None, :]
  
  4b. 展平为 2D 序列:
    x_flat = reshape(B, S×T, 256)
  
  4c. Transformer Encoder (因果/非因果可选):
    x_flat → TransformerEncoder(2层, 4头)
           → LayerNorm(256)
    slot_feats: (B, S, T, 256)
  
  因果注意力: (s,t) 只能 attend to (s',t') where t' ≤ t
  → 防止未来信息泄漏

Step 5: Temporal Differencing (在 ST-Transformer 输出上)
  delta = slot_feats[:,:,1:] - slot_feats[:,:,:-1]
  delta: (B, S, T-1, 256)

Step 6: VAE 编码 (共享编码器)
  delta_flat = reshape(delta, (-1, 256))
  moments = Linear(256, 2×latent_dim)(delta_flat)
  z_mu, z_var = chunk(moments)
  z_rep = z_mu + noise × exp(0.5 × z_var)
  
  z_mu:  (B, T-1, S, latent_dim)     # latent_dim=32
  z_var: (B, T-1, S, latent_dim)
  z_rep: (B, T-1, S, latent_dim)

Step 7: 对象级像素重建 (解码器)
  7a. 特征映射:
    z_rep → action_up(Linear 32→256) → action_slots: (B, T-1, S, 256)
  
  7b. Cross-Attention 融合:
    Q = pos_queries (1, N, 256) → expand → (B, T-1, N, 256)  # 可学习位置编码
    K,V = action_slots (B, T-1, S, 256)
    fused = CrossAttn(Q, K, V)     # (B, T-1, N, 256)
  
  7c. SpatioTransformer 解码:
    fused → SpatioTransformer(4层) → Linear(256, 768)
    recon: (B, T-1, H, W, C)

Step 8: 损失计算
  8a. 对象级重建损失 (核心改进):
    对每个 slot s ∈ {0,...,S-1}:
      mask_s = masks_with_bg[:, 1:, s]     # (B, T-1, H, W)
      L_s = MSE(recon × mask_s, gt × mask_s) / Σ(mask_s)
    L_obj = mean(L_s for s in valid_slots)
  
  8b. 场景级重建损失 (辅助):
    L_scene = MSE(recon, gt)
  
  8c. KL 损失 (含 Free Bits):
    kl_per_dim = 0.5 × (exp(z_var) - 1 - z_var + z_mu²)
    kl_per_dim = clamp(kl_per_dim, min=free_bits)  # 防止后验坍塌
    L_kl = mean(kl_per_dim)
  
  8d. 时序预测损失:
    raw_slot = Step 3 的输出 (未经 ST-Transformer)
    current = raw_slot[:, :, :-1]   # (B, S, T-1, 256)
    target  = raw_slot[:, :, 1:]    # (B, S, T-1, 256)
    z_rep_st = z_rep.permute(0,2,1,3)  # (B, S, T-1, 32)
    z_proj = Linear(32, 256)(z_rep_st)  # (B, S, T-1, 256)
    pred_next = MLP(current + z_proj)   # 加法式预测
    L_temporal = MSE(pred_next, target)
  
  L_total = λ_obj × L_obj + λ_scene × L_scene + β × L_kl + λ_temp × L_temporal
```

### 2.3 改进架构图

```
videos (B,T,H,W,C) ──→ [patchify] ──→ patches (B,T,N,768)
                                              │
                                    ┌─────────▼─────────┐
                                    │   Encoder          │
                                    │   SpatioTemporal   │
                                    │   Transformer      │
                                    │   (非因果, ×4)      │
                                    └─────────┬─────────┘
                                              │
                              encoded (B,T,N,256)
                                              │
                              ┌────────────────┤
                              │                │
                    masks (B,T,A,H,W)          │
                              │                │
                   ┌──────────▼──────────┐     │
                   │   Mask Pooling      │     │
                   │   + 背景槽           │     │
                   │                     │     │
                   │  主体: 加权平均池化   │     │
                   │  背景: 反向区域池化   │     │
                   └──────────┬──────────┘     │
                              │                │
              slot_feats (B,T,S,256)  S=A+1=5  │
              ┌──────────────┤                │
              │              │                │
              │    ┌─────────▼─────────────┐  │
              │    │  Spatio-Temporal      │  │
              │    │  Transformer          │  │
              │    │                       │  │
              │    │  输入: (B, S, T, 256) │  │
              │    │  展平: (B, S×T, 256)  │  │
              │    │  位置编码: slot_pe    │  │
              │    │            + time_pe  │  │
              │    │  SelfAttn(因果/非因果) │  │
              │    │  ×2 层                │  │
              │    └─────────┬─────────────┘  │
              │              │                │
              │    slot_feats (B,S,T,256)     │
              │              │                │
              │    ┌─────────▼─────────────┐  │
              │    │  Temporal Diff        │  │
              │    │  delta = f[t+1]-f[t]  │  │
              │    └─────────┬─────────────┘  │
              │              │                │
              │    delta (B,S,T-1,256)        │
              │              │                │
              │    ┌─────────▼─────────────┐  │
              │    │  VAE Encoder          │  │
              │    │  Linear(256, 64)      │  │
              │    │  共享编码器            │  │
              │    └─────────┬─────────────┘  │
              │              │                │
              │    z_rep (B,T-1,S,32)         │
              │              │                │
              │    ┌─────────┴─────────────┐  │
              │    │                       │  │
    ┌─────────▼──┐ │              ┌────────▼──▼──────────┐
    │ 时序预测损失 │ │              │  对象级重建解码器      │
    │             │ │              │                      │
    │ raw_slot[t] │ │              │  Q = pos_queries     │
    │   (B,S,T-1, │ │              │    (可学习, N×256)    │
    │    256)     │ │              │  K,V = action_slots  │
    │     │       │ │              │    (z_rep→Linear32→256)│
    │     + z_proj│ │              │                      │
    │     │       │ │              │  CrossAttn(Q,K,V)    │
    │ z_rep→W_up  │ │              │  SpatioTransformer×4 │
    │  (32→256)   │ │              │  unpatchify+sigmoid  │
    │     │       │ │              └──────────┬───────────┘
    │     ▼       │ │                         │
    │ MLP(cur+z)  │ │              recon (B,T-1,H,W,C)    │
    │  (256→256   │ │                         │
    │   →256→256) │ │                         │
    │     │       │ │                         │
    │ pred_next   │ │                         │
    │ (B,S,T-1,256)│ │                        │
    │     │       │ │                         │
    │ MSE(pred,   │ │                         │
    │   raw_slot  │ │                         │
    │   [t+1])    │ │                         │
    └──────┬──────┘ │                         │
           │        │                         │
           │   ┌────▼─────────────────────────▼────┐
           │   │  损失计算                           │
           │   │                                    │
           │   │  L_obj = Σ MSE(recon×mask_s,       │
           │   │              gt×mask_s) / Σmask_s   │
           │   │        ← 每个 slot 只重建自己区域    │
           │   │                                    │
           │   │  L_scene = MSE(recon, gt)           │
           │   │        ← 辅助, 低权重               │
           │   │                                    │
           │   │  L_kl = mean(clamp(kl, min=fb))     │
           │   │        ← Free Bits 防坍塌           │
           │   │                                    │
           │   │  L_temporal = MSE(pred_next, target) │
           │   │        ← 时序预测, 高权重           │
           │   │                                    │
           │   │  L = λ_obj×L_obj + λ_s×L_scene     │
           │   │    + β×L_kl + λ_t×L_temporal       │
           │   └────────────────────────────────────┘
```

---

## 三、原始 LAM vs 改进架构 对比

```
┌──────────────────────────────────────────────────────────────────────┐
│                        原始 LAM (lam.py)                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  videos → patchify → Encoder(ST-Transformer) → encoded              │
│                                                              │      │
│  masks ──→ MaskPool ──→ obj_feats(B,T,A,256) ──→ Diff ──→ VAE     │
│                                                              │      │
│  z_rep ──→ action_up ──→ K,V                                 │      │
│  patches ──→ patch_up ──→ Q ──→ CrossAttn ──→ +残差 ──→ Decoder  │
│                                                              │      │
│  Loss = MSE(recon, gt) + KL                                  │      │
│                                                                      │
│  问题: Q=video_patches 包含当前帧完整信息, z_rep 不被需要             │
│  结果: z_rep 坍塌到 0, ARI≈0.03                                     │
└──────────────────────────────────────────────────────────────────────┘

                                    ↓ 改进

┌──────────────────────────────────────────────────────────────────────┐
│                     改进架构 (lam_v3.py)                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  videos → patchify → Encoder(ST-Transformer) → encoded              │
│                                                              │      │
│  masks ──→ MaskPool+背景 ──→ slot_feats(B,T,S,256) S=A+1           │
│                                    │                                 │
│                    ┌───────────────┤                                 │
│                    │               │                                 │
│                    ▼               ▼                                 │
│          ST-Transformer      raw_slot (保存副本)                     │
│          (跨槽+跨时间)              │                                 │
│                    │               │                                 │
│                    ▼               │                                 │
│          Diff → VAE               │                                 │
│                    │               │                                 │
│          z_rep(B,T-1,S,32)        │                                 │
│                    │               │                                 │
│          ┌─────────┴───────┐      │                                 │
│          │                 │      │                                 │
│    ┌─────▼──────┐   ┌─────▼──────▼───┐                            │
│    │ 时序预测    │   │ 对象级重建      │                            │
│    │            │   │                │                            │
│    │ z_proj =   │   │ Q=pos_queries  │                            │
│    │  W_up(z)   │   │ K,V=z_rep_up   │                            │
│    │            │   │ CrossAttn      │                            │
│    │ pred = MLP │   │ Decoder        │                            │
│    │  (cur+z)   │   │ → recon        │                            │
│    │            │   │                │                            │
│    │ MSE(pred,  │   │ L_obj = Σ_s    │                            │
│    │   target)  │   │  MSE(recon×m_s,│                            │
│    │            │   │   gt×m_s)/Σm_s │                            │
│    └─────┬──────┘   └──────┬─────────┘                            │
│          │                 │                                        │
│    L = λ_t×L_temp + λ_obj×L_obj + λ_s×L_scene + β×L_kl           │
│                                                                      │
│  改进:                                                               │
│  1. Q=pos_queries (非 video_patches), z_rep 是唯一信息源            │
│  2. 对象级重建: 每个 slot 只重建自己 mask 区域                       │
│  3. 背景槽: 背景信息不混入主体                                       │
│  4. ST-Transformer: 替代简单差分, 建模交互+时序                     │
│  5. 时序预测: z_proj 加法式, 迫使 z 非零                            │
│  6. Free Bits: KL 下限, 防止后验坍塌                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 四、关键改进点详解

### 4.1 对象级重建 vs 场景级重建

```
场景级重建 (原始 LAM):
─────────────────────
  recon: (B, T-1, H, W, C)  ← 整张图
  gt:    (B, T-1, H, W, C)  ← 整张图
  Loss = MSE(recon, gt)      ← 所有像素等权

  问题: 背景占 80%+ 像素, 背景是静态的
  → 模型只需重建好背景就能得到低 Loss
  → 主体的动作信息不重要

对象级重建 (改进):
─────────────────────
  对每个 slot s:
    mask_s: (B, T-1, H, W)  ← 第 s 个主体的二值 mask
    Loss_s = MSE(recon × mask_s, gt × mask_s) / Σ(mask_s)

  L_obj = mean(Loss_0, Loss_1, ..., Loss_S-1)

  优势: 每个 slot 必须在自己 mask 区域内重建准确
  → slot 必须编码自己主体的外观和动作信息
  → 不同 slot 编码不同主体的信息 → 自然分离
```

### 4.2 时序预测损失：加法式 vs 拼接式

```
拼接式 (之前尝试, 失败):
─────────────────────
  input = concat([current_slot, z_rep])  # (B, S, T-1, 256+32)
  pred = MLP(input)                       # MLP 可以忽略 z_rep 维度

  问题: MLP 有 288→256 的入口层
  → 可以学会权重使得 z_rep 的 32 维对输出无影响
  → z_rep 仍然不被需要

加法式 (改进):
─────────────────────
  z_proj = Linear(32→256)(z_rep)          # 投影到与 current 相同空间
  input = current_slot + z_proj            # 加法, 不是拼接!
  pred = MLP(input)                        # MLP 只接受 256 维

  优势: z_proj 直接修改 current_slot 的每一维
  → MLP 的每一层都能感知到 z_proj 的影响
  → z_proj=0 时 pred = MLP(current), 对动态场景预测不准
  → 预测损失迫使 z_proj ≠ 0 → z_rep ≠ 0
```

### 4.3 Free Bits 防止后验坍塌

```
标准 KL:
  kl = 0.5 × (exp(z_var) - 1 - z_var + z_mu²)
  → 当 z_mu→0, z_var→0 时 kl→0
  → KL 损失鼓励 z_mu=0, z_var=0

Free Bits:
  kl = clamp(kl, min=free_bits)  # 每维度最少 free_bits nats
  → 即使 z_mu→0, kl 仍 ≥ free_bits
  → 梯度不再把 z_mu 推向 0 (已经满足下限)
  → 预测损失的梯度可以让 z_mu 增大

free_bits=0.1, latent_dim=32:
  → 每个维度至少编码 0.1 nats 信息
  → 总信息量 ≥ 32 × 0.1 = 3.2 nats
```

---

## 五、实验记录

### 5.1 已完成的实验

| # | 名称 | 编码器 | 解码器 | KL | z_mu 范数 | ARI | Linear Probe | 关键发现 |
|---|------|--------|--------|-----|----------|-----|-------------|---------|
| A | causal_delta | 非因果 | 残差连接 | 常量 | 0.013 | 0.0012 | 20.42% | 非因果编码器泄漏未来信息 |
| B | causal_enc | 因果 | 残差连接 | 常量 | 0.010 | 0.0012 | 20.42% | 残差连接允许复制当前帧 |
| C | residual_klann | 因果 | 残差解码 | 退火 | 0.151 | 0.0005 | 20.42% | KL退火有效但最终仍坍塌 |
| D | slot_cond | 因果 | pos_q+[z;cur] | 退火 | 0.005 | 0.0013 | 20.42% | current_slot_feats 信息太丰富 |
| E | z_only_pred | 因果 | pos_q+[z;cur] | 退火 | 0.12-0.19 | 0.0020 | 23.86% | 首次超过随机基线! |
| F | aggressive | 因果 | 无重建 | 退火 | 0.001 | 0.0042 | 20.42% | 无重建时 slot 方向分化但范数坍塌 |
| G | freebits | 因果 | pos_q+[z;cur] | FB=0.1 | 0.185 | 0.0003 | 25.07% | Free Bits 防坍塌, 但 z 不区分动作 |
| H | freebits_large | 因果 | 无重建 | FB=0.5 | 0.823 | 0.0014 | 23.32% | 大 free_bits, z 编码 slot 身份 |

### 5.2 核心发现

1. **场景级重建无法驱动隐动作分离**：所有实验 ARI < 0.01
2. **Free Bits 可以防止后验坍塌**：z_mu 范数从 0.001 提升到 0.823
3. **但 z_mu 编码的是 slot 身份而非动作语义**：所有动作的 mean_norm 完全相同
4. **拼接式时序预测无法迫使 z 非零**：MLP 可以忽略 z 维度
5. **下一步**：尝试加法式时序预测 + 对象级重建的组合

---

**报告日期**：2025-06-05（更新）
