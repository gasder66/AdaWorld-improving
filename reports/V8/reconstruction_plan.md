# V8 重建验证计划: Probe-First Reconstruction Evaluation

## 背景与问题

V8 为了避免 V7 的失败 (SharedEncoder 泄漏 actor identity → NMI 0.0047), 刻意去掉了
像素重建, 只保留 Δbbox 回归。这使 V8 的 NMI 达到 0.77, 但也意味着:

> **V8 不再是传统意义上的 "latent action model"** — 它没有重建目标, z_actor 只是
> clustering embedding, 无法证明它具有 forward-prediction 信息。

本计划通过 **probe decoder** 验证 z_actor 是否包含足够的重建信息, 在不修改主模型的
前提下回答这个核心问题。

## 核心路线

```
Phase 1: Probe Decoder (不改主模型, 验证 z_actor/z_bg 信息量)
  ├── Actor Probe: z_actor + crop(I_t) → crop(I_{t+1})
  ├── 在 v8_stage1 (合成) 上验证
  ├── 在 v8_yolo (A2D) 上验证
  └── 判断: z_actor 对重建是否有显著贡献?

Phase 2 (条件触发): 改主模型, 加入 recon loss
  └── 仅当 Phase 1 证明 z_actor 贡献显著时执行
```

## Actor Probe Decoder 设计

### 输入输出

```
mode A (默认):
  输入: crop_t = crop(I_t, bbox_t)  (3, 32, 32)  +  z_actor (16,)
  输出: crop_pred (3, 32, 32)  — 预测 crop(I_{t+1}, bbox_{t+1})

mode B (可选对比):
  输入: crop_t + z_actor + Δbbox_pred (4,)  +  actor_type (1,)
  输出: crop_pred (3, 32, 32)
  Δbbox_pred 来自 V8 ActorMotionPred (非 GT, 不泄漏标签)
```

### 架构

```
crop_t → CNN encoder (3 conv blocks: 32→64→128, stride 2) → feat (128, 4, 4)
z_actor → FiLM: Linear(16, 128*2) → (gamma_128, beta_128)
feat = gamma * feat + beta           ← FiLM 调制
feat → CNN decoder (3 deconv: 128→64→32→3) → sigmoid → crop_pred
```

- 无 Spatial Transformer: decoder 自己从 z_actor 学位移
- FiLM conditioning: z_actor 作为条件注入, 不是外观来源

### 训练协议

- 冻结 z_actor, z_bg (requires_grad=False)
- Video-grouped 80/20 split (同一视频的所有 slot/time 只在 train 或 test)
- Loss: L1 + 0.5 * (1 - SSIM)
- Adam lr=1e-3, max 3000 steps, batch=64, 早停 (test PSNR 5 epochs 不涨)

## 评估指标

### 三重 Baseline (关键设计)

| Baseline | 说明 | 验证什么 |
|---|---|---|
| **Copy** | crop(I_t, bbox_t) 直接作为预测 | probe 是否比 "什么都不做" 好 |
| **z=0** | z_actor 置零, probe decoder 输出 | decoder 是否依赖非零 z |
| **z_shuffle** | batch 内打乱 z_actor 顺序 | decoder 是否依赖 **正确的** z (非仅分布) |

### GO/NO-GO 判据

```
GO (→ Phase 2):
  PSNR(probe) - PSNR(z=0)       ≥ 1.0 dB   ← z_actor 有信息
  PSNR(probe) - PSNR(z_shuffle) ≥ 0.5 dB   ← z_actor 语义正确 (硬性)
  PSNR(probe) - PSNR(copy)      ≥ 0.5 dB   ← 比 copy 好

NO-GO:
  PSNR(probe) - PSNR(z_shuffle) < 0.5 dB
  → z_actor 被忽略或只利用了幅值, 需重新设计
```

### 双 Crop Target

```
crop_tp1_gtbox:   crop(I_{t+1}, bbox_{t+1})           ← 理想对齐 (训练用)
crop_tp1_predbox: crop(I_{t+1}, bbox_t + Δbbox_pred)   ← V8 完整 forward rollout (评估补充)
```

### 可视化

```
每个 test 样本一行:
  [crop_t] [GT] [probe] [copy] [z=0] [z_shuffle] [diff_map]
```

## 数据保存格式

`latent_pairs_{name}.npz`:

```
z_actor:           (N, 16)
z_bg:              (N, 16)
crop_t:            (N, 3, 32, 32)
crop_tp1_gtbox:    (N, 3, 32, 32)
crop_tp1_predbox:  (N, 3, 32, 32)
dbbox_pred:        (N, 4)
dbbox_obs:         (N, 4)
actions:           (N,)
actor_types:       (N,)
video_id:          (N,)     ← 用于 video-grouped split
frame_idx:         (N,)     ← 用于可视化定位
slot_id:           (N,)     ← 用于 per-actor 分析
```

## PSNR 比较注意

V8 actor-crop PSNR (32×32) **不能** 与 V6c full-frame PSNR (256×256) 直接横比。

| Model | Full-frame PSNR | Actor-crop PSNR | NMI |
|---|---|---|---|
| V6c | 27.35 (合成) | 需重新 crop | 0.05/0.36 |
| V8 S1 | — | probe 结果 | 0.77 |
| V8 S4 | — | probe 结果 | 0.31 |

## 执行顺序

| 优先级 | 文件 | 说明 |
|---|---|---|
| P1 | `lam/lam/modules/probe_decoder.py` | ActorProbeDecoder (FiLM, 无 STN) |
| P1 | `lam/scripts/extract_latents_v8.py` | 从 checkpoint 提取 z + crop pairs |
| P1 | `lam/scripts/train_probe_decoder.py` | 训练 probe (冻结 z, video-grouped split) |
| P2 | `lam/scripts/eval_v8_reconstruction.py` | 4 个 PSNR + 可视化 |
| P2 | 在 v8_stage1 上跑完整流程 | 合成数据验证 |
| P2 | 在 v8_yolo 上跑完整流程 | A2D 真实数据验证 |
| P3 | `lam/scripts/eval_v8.py` | 统一评估入口 (聚类 + 重建) |
| P4 | BackgroundProbeDecoder | 条件触发, 暂缓 |

## 不修改现有文件

Phase 1 完全 additive — 不改 slot_time_lam.py, 不改 run_v8_*.py, 不改数据集。
所有新代码在新建文件中。
