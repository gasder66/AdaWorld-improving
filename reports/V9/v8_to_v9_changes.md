# V9 相对 V8 的具体改动

## 概述

V9 = V8 + 重建 Decoder。V8 代码**完全未修改**, V9 通过组合 (composition) 方式
包裹 V8 模型并添加重建路径。新增 4 个文件, 不改动任何 V8 文件。

---

## 1. 新增文件清单

| 文件 | 行数 | 说明 |
|---|---|---|
| `lam/lam/modules/v9_decoder.py` | 132 | ReconDecoder + FiLMLayer + PSNR/SSIM 工具函数 |
| `lam/lam/modules/v9_model.py` | 296 | LatentActionModelV9 (组合 V8 + decoder) + RGBMotionTokenEncoder (V9-B) |
| `lam/scripts/run_v9.py` | ~250 | V9 训练脚本 (合成数据) |
| `lam/scripts/eval_v9.py` | ~260 | 统一评估 (聚类 + 4-way PSNR + GO/NO-GO) |

**未修改的 V8 文件:**
- `lam/lam/modules/slot_time_lam.py` — V8 模型, 原封不动
- `lam/lam/modules/motion_token_encoder.py` — V8 编码器, 原封不动
- `lam/scripts/run_v8_mot_lam.py` — V8 训练脚本, 原封不动
- `lam/scripts/eval_v8_action_cluster.py` — V8 评估脚本, 原封不动

---

## 2. 架构改动: V8 → V9

### 2.1 数据流对比

```
V8 数据流 (无重建):
  video → MotionTokenEncoder → motion_tokens (B,T,K+1,D)
       → TransitionTokenBuilder → trans_tokens (B,T-1,K+1,D)
       → TemporalTransformer (2层) → SlotTransformer (1层)
       → SharedActorActionHead → z_actor (B,T-1,K,16)
       → BackgroundMotionHead → z_bg (B,T-1,16)
       → ActorMotionPredictor → Δbbox_res
       → CameraMotionPredictor → Δbbox_bg
       Loss: L_motion + β·L_KL + γ·L_bg

V9-A 数据流 (V8 + 重建):
  [V8 路径, 完全相同] → z_actor, z_bg, L_motion, L_KL, L_bg
  [V9 新增重建路径]:
       crop_t = _crop_resize(video[:, :-1], boxes[:, :-1])  → (B,T-1,K,3,32,32)
       crop_tp1 = _crop_resize(video[:, 1:], boxes[:, 1:])  → target
       z_actor_flat = z_actor.reshape(B*(T-1)*K, 16)
       z_bg_flat = z_bg.expand → (B*(T-1)*K, 16)
       recon = ReconDecoder(crop_t, z_actor, z_bg)  → (B,T-1,K,3,32,32)
       Loss: L_motion + β·L_KL + γ·L_bg + δ·L_recon
       L_recon = L1(recon, crop_tp1) + (1 - SSIM(recon, crop_tp1))
```

### 2.2 关键区别: V8 probe vs V9-A decoder

V8 的 probe decoder 是**独立的事后验证工具** (冻结 z_actor, 单独训练 decoder)。
V9-A 的 decoder 是**模型的一部分** (z_actor 与 decoder 联合训练)。

| 属性 | V8 Probe Decoder | V9-A ReconDecoder |
|---|---|---|
| 训练方式 | 冻结 z_actor, 单独训练 decoder | z_actor 与 decoder 联合训练 |
| z_actor 来源 | 从 V8 checkpoint 提取 (mu) | 实时计算 (含 recon 梯度回流) |
| FiLM 层数 | 1 层 (仅 4×4 bottleneck) | **3 层** (16×16, 8×8, 4×4 全尺度) |
| Skip connection | 无 | **U-Net** (encoder→decoder 3 个 skip) |
| z_bg 输入 | 无 | **有** (z_bg 与 z_actor concat) |
| 参数量 | 190,723 | 220,451 |
| 结果 | NO-GO (z 被忽略) | **GO** (ΔPSNR(z=0)=+3.32) |

### 2.3 ReconDecoder 架构细节

```python
# v9_decoder.py — ReconDecoder

# Encoder: 3 层 stride-2 conv (32×32 → 4×4)
enc1: Conv2d(3, 32, 3, 2, 1)    # 32→16,  896 params
enc2: Conv2d(32, 64, 3, 2, 1)   # 16→8,  18,496 params
enc3: Conv2d(64, 128, 3, 2, 1)  # 8→4,   73,856 params

# Multi-scale FiLM (z_cond = z_actor(16) + z_bg(16) = 32 dim)
film1: FiLMLayer(32, 32)   # z→(γ,β) for 32ch,  2,112 params
film2: FiLMLayer(32, 64)   # z→(γ,β) for 64ch,  4,224 params
film3: FiLMLayer(32, 128)  # z→(γ,β) for 128ch, 8,448 params

# Decoder: 3 层 ConvTranspose + U-Net skip
dec3: ConvTranspose2d(128, 64, 3, 2, 1)      # 4→8,           73,792 params
dec2: ConvTranspose2d(128, 32, 3, 2, 1)      # 8→16 (64+64 skip), 36,896 params
dec1: ConvTranspose2d(64, 3, 3, 2, 1)        # 16→32 (32+32 skip), 1,731 params

# 总计: 220,451 params
```

**FiLM 机制:** `z_cond (32) → Linear(32, 2C) → (gamma, beta) → feat = gamma * feat + beta`
在每个 encoder level 后应用, 使 z 条件化影响所有尺度的特征。

**U-Net skip:** decoder 每层接收 encoder 对应层的特征 (concat), 保留空间细节。

### 2.4 LatentActionModelV9 组合方式

V9 不继承 V8, 而是**组合** (wrapping):

```python
class LatentActionModelV9(nn.Module):
    def __init__(self, ..., variant="A"):
        # V8 backbone — 完整实例化, 参数不共享
        self.v8 = LatentActionModelV8(...)
        
        # V9-B: 替换 V8 的 motion_encoder
        if variant == "B":
            self.v8.motion_encoder = RGBMotionTokenEncoder(...)
        
        # V9-C: 额外的外观路径
        if variant == "C":
            self.app_encoder = ActorCropEncoder(...)
            self.app_vae = nn.Sequential(...)
        
        # 重建 decoder
        self.recon_decoder = ReconDecoder(...)
    
    def forward(self, batch):
        # 1. V8 前向 (计算 motion_loss, kl_loss, bg_loss, z_actor, z_bg)
        out = self.v8(batch)
        
        # 2. V9 重建 (计算 recon_loss)
        z_actor = out["z_actor"]
        crop_t = _crop_resize(video[:, :-1], boxes[:, :-1])
        crop_tp1 = _crop_resize(video[:, 1:], boxes[:, 1:])
        recon = self.recon_decoder(crop_t, z_actor, z_bg)
        recon_loss = L1(recon, crop_tp1) + (1 - SSIM(recon, crop_tp1))
        out["recon_loss"] = recon_loss
        return out
```

关键: `self.v8(batch)` 返回的 dict 中包含 `z_actor`, `z_bg`, `motion_loss`, `kl_loss`, `bg_loss`
等所有 V8 的输出。V9 只需在此基础上添加 `recon_loss`。

---

## 3. 损失函数改动

### V8 损失
```
L_v8 = L_motion + β · L_KL + γ · L_bg
```

### V9 损失
```
L_v9 = L_motion + β · L_KL + γ · L_bg + δ · L_recon
```

| 损失项 | 公式 | V8 | V9 |
|---|---|---|---|
| L_motion | MSE(Δbbox_pred, Δbbox_obs) / bbox_scale² | ✓ | ✓ (不变) |
| L_KL | Free Bits KL(z_actor) + Free Bits KL(z_bg) | ✓ | ✓ (不变) |
| L_bg | MSE(cam_pred, cam_target) / cam_scale² | ✓ (Stage 2A) | ✓ (不变) |
| **L_recon** | **L1(recon, crop_tp1) + (1 - SSIM(recon, crop_tp1))** | **无** | **新增** |

**L_recon 细节:**
- L1: `|recon - crop_tp1|` 在 32×32 actor crops 上, masked by valid_mask
- SSIM: 简化版 (per-channel global mean/var/cov), 范围 [0, 1]
- L_recon = L1 + (1 - SSIM), 典型值 ≈ 0.2 (收敛后)
- δ = 0.1 (recon_weight), 使 L_recon 贡献 ≈ 0.02, 远小于 L_motion (≈0.01) + L_KL (≈1.0)

---

## 4. 训练脚本改动 (run_v9.py vs run_v8_mot_lam.py)

### 新增参数
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--variant` | A | 选择 A/B/C 变体 |
| `--recon_weight` | 0.1 | L_recon 权重 (δ) |
| `--z_app_dim` | 16 | V9-C 的 z_appearance 维度 |
| `--detach_z_actor` | False | V9-C: 是否 detach z_actor from recon |

### 训练循环改动
```python
# V8:
loss = motion_loss + args.kl_beta * kl_loss + args.bg_loss_weight * bg_loss

# V9:
loss = (motion_loss + args.kl_beta * kl_loss
        + args.bg_loss_weight * bg_loss
        + args.recon_weight * recon_loss)  # 新增
```

### 日志新增
- `recon` (recon_loss 总值)
- `recon_l1` (L1 部分)
- `recon_ssim` (SSIM 值, 越高越好)
- 每 50 步打印: `ssim={float(outputs['recon_ssim']):.3f}`

### 评估新增
训练后自动计算并保存:
- `psnr_recon`: PSNR(recon, target)
- `psnr_copy`: PSNR(crop_t, target) — baseline
- `ssim_recon` / `ssim_copy`
- `delta_psnr_recon_copy`: ΔPSNR

### 输出目录
- V8: `result/v8_mot_lam/`
- V9: `result/v9/` (新目录, 不覆盖 V8)

---

## 5. 评估脚本改动 (eval_v9.py vs eval_v8_action_cluster.py)

V8 评估只有聚类指标。V9 评估统一聚类 + 重建。

### 聚类指标 (与 V8 eval 相同)
- Overall NMI / ARI: KMeans(5) on all z_actor → vs GT actions
- Per-Slot NMI: 每个 actor 内部 KMeans(5)
- Actor Leakage: LogisticRegression(z_actor → actor_id)
- Action Probe: LogisticRegression(z_actor → action)

### 重建指标 (V9 新增)
- **4-way PSNR**: recon / copy / z=0 / z_shuffle
  - `PSNR(recon)`: 模型正常输出
  - `PSNR(copy)`: crop_t 直接作为预测 (identity baseline)
  - `PSNR(z=0)`: decoder 输入 z_actor=0 (测试 z_actor 是否被使用)
  - `PSNR(z_shuffle)`: decoder 输入 shuffle 后的 z_actor (测试 z_actor 语义正确性)
- **SSIM**: recon vs copy
- **GO/NO-GO 判据**:
  - ΔPSNR(recon - z=0) ≥ 1.0 dB
  - ΔPSNR(recon - z_shuffle) ≥ 0.5 dB
  - ΔPSNR(recon - copy) ≥ 0.5 dB

### 可视化
V9 评估生成 5 列对比图: `[crop_t, GT, recon, z=0, z_shuffle]`

---

## 6. 参数量对比

| 模块 | V8 | V9-A | V9-B | V9-C |
|---|---|---|---|---|
| motion_encoder | 402,768 | 402,768 | 402,768* | 402,768 |
| transition_builder | 264,704 | 264,704 | 264,704 | 264,704 |
| temporal_blocks | 1,578,048 | 1,578,048 | 1,578,048 | 1,578,048 |
| slot_blocks | 788,992 | 788,992 | 788,992 | 788,992 |
| actor_action_head | 75,040 | 75,040 | 75,040 | 75,040 |
| bg_motion_head | 75,040 | 75,040 | 75,040 | 75,040 |
| camera_motion_pred | 1,644 | 1,644 | 1,644 | 1,644 |
| bg_supervision_head | 1,380 | 1,380 | 1,380 | 1,380 |
| actor_motion_pred | 1,380 | 1,380 | 1,380 | 1,380 |
| **recon_decoder** | — | **220,451** | **220,451** | **227,619** |
| **app_encoder** | — | — | — | **126,528** |
| **app_vae** | — | — | — | **75,040** |
| **总计** | **3,188,996** | **3,409,447** | **3,409,447** | **3,618,183** |
| **增幅** | — | +220,451 (6.9%) | +220,451 (6.9%) | +429,187 (13.5%) |

*V9-B 的 motion_encoder 结构相同但行为不同 (RGB crops 替代帧差)

---

## 7. V9 变体差异

### V9-A (推荐): V8 + ReconDecoder
- 编码器: V8 MotionTokenEncoder (帧差 + 几何, t=0 RGB)
- z_actor: 含 recon 梯度 (联合训练)
- decoder 输入: crop_t + z_actor(16) + z_bg(16)
- 新增参数: +220,451 (仅 decoder)

### V9-B: RGB 编码器 + ReconDecoder
- 编码器: RGBMotionTokenEncoder (所有 t 用 RGB crop)
- z_actor: 含 recon 梯度
- decoder 输入: 同 V9-A
- 新增参数: +220,451 (decoder, 编码器参数量不变)
- 与 V9-A 区别: `self.v8.motion_encoder = RGBMotionTokenEncoder(...)` (行 141)

### V9-C: 双路径
- 编码器: V8 MotionTokenEncoder (不变)
- z_actor: **detached** from recon (recon 梯度不流入 z_actor)
- z_appearance: 独立的 RGB crop → ActorCropEncoder → VAE (16 dim)
- decoder 输入: crop_t + z_actor(16,detached) + z_app(16) + z_bg(16)
- 新增参数: +429,187 (decoder + app_encoder + app_vae)
- 额外损失: L_KL_app (Free Bits KL on z_appearance)

---

## 8. 为什么 V8 probe NO-GO 但 V9-A GO?

### Probe decoder 失败原因 (V8)
1. **z_actor 冻结**: 从 V8 checkpoint 提取 mu, 不再训练 → z_actor 只含运动信息
2. **单层 FiLM 太弱**: 仅在 4×4 bottleneck 注入 z, gamma=1/beta=0 是容易到达的极小值
3. **无 skip connection**: decoder 无法利用 encoder 的空间细节
4. **结果**: `PSNR(probe) = PSNR(z=0) = PSNR(z_shuffle)` → FiLM 退化到恒等映射

### V9-A 成功原因
1. **联合训练**: recon loss 梯度通过 z_actor → VAE → transformer → MotionTokenEncoder 回流
   → encoder 被推动从 t=0 RGB crop 保留外观信息到 z_actor
2. **多尺度 FiLM (3层)**: 在 16×16, 8×8, 4×4 三个尺度注入 z
   → 三层 FiLM 无法同时退化到恒等映射, 至少一层会使用 z
3. **U-Net skip**: decoder 可以结合 encoder 空间细节和 z 的条件信息
4. **结果**: `ΔPSNR(recon - z=0) = +3.32 dB` → z_actor 被有效使用

### 信息流路径 (V9-A)
```
t=0: RGB crop(I_0, bbox_0) → ActorCropEncoder → motion_token
    ↓ (通过 TransitionTokenBuilder, TemporalTransformer, SlotTransformer)
    ↓ SharedActorActionHead → z_actor (16 dim)
    ↓ ReconDecoder → recon crop(I_1, bbox_1)
    ↓ L_recon 梯度回流
    ← ← ← ← ← ← ← ← ← ← ← ← ← ←
    encoder 被推动保留 t=0 RGB 中的外观变化信息
```

z_actor 在 16 维 VAE bottleneck 中同时编码:
- 动作聚类信号 (由 L_motion 监督) → NMI 0.77
- 重建所需的外观变化 (由 L_recon 监督) → PSNR 27.40
两者在 16 维空间中不冲突 (leakage 不变 = 0.3350)。

---

## 9. 实验结果对比

| 指标 | V6c | V7v3 | V8 | V9-A | V9-B | V9-C |
|---|---|---|---|---|---|---|
| Overall NMI | 0.0525 | 0.0047 | 0.7723 | **0.7723** | 0.7723 | 0.7723 |
| Per-Slot NMI | 0.3885 | N/A | 0.7684 | 0.7684 | 0.7684 | 0.7684 |
| Actor Leakage | 1.0000 | 0.8970 | 0.3350 | **0.3350** | 0.3374 | 0.3457 |
| Action Probe | N/A | N/A | 0.8741 | 0.8741 | 0.8741 | 0.8741 |
| PSNR(recon) dB | 27.35 | N/A | NO-GO | **27.40** | 27.34 | **27.70** |
| SSIM(recon) | N/A | N/A | 0.094 | **0.796** | 0.794 | **0.813** |
| ΔPSNR(z=0) | N/A | N/A | 0.00 | **+3.32** | +3.12 | **+3.61** |
| ΔPSNR(z_shuffle) | N/A | N/A | 0.00 | **+4.57** | +4.47 | **+4.85** |
| 判定 | — | — | NO-GO | **GO** | GO | GO |
| 参数量 | ~20M | ~20M | 3.19M | 3.41M | 3.41M | 3.62M |

**V9-A 是最佳选择**: 聚类完全不变 (NMI/leakage/action_probe 与 V8 完全一致),
重建达到 V6c 水平, 仅增加 220K 参数 (6.9%)。

---

## 10. 代码复用关系

```
V9-A 代码复用:
  lam.modules.slot_time_lam.LatentActionModelV8     ← 完整复用 (组合)
  lam.modules.motion_token_encoder._crop_resize      ← 复用 (生成 crop_t/crop_tp1)
  lam.modules.v9_decoder.ReconDecoder                ← 新增
  lam.modules.v9_decoder.FiLMLayer                   ← 新增 (改进版, 多尺度)

V9-B 额外复用:
  lam.modules.motion_token_encoder.ActorCropEncoder  ← 复用 (RGB crop 编码)
  lam.modules.motion_token_encoder.BoxGeometryEncoder ← 复用
  lam.modules.motion_token_encoder.BackgroundTokenEncoder ← 复用

V9-C 额外复用:
  lam.modules.motion_token_encoder.ActorCropEncoder  ← 复用 (外观编码器)
  lam.modules.slot_time_lam (Free Bits KL 逻辑)      ← 复用模式
```
