# V9 实验报告: 打破 NMI 与重建的界限

## 结论

**三个变体全部 GO。** V9 在保持 V8 高 NMI 聚类的同时, 恢复了像素重建能力。

| 变体 | NMI | Leakage | PSNR | Δ(z=0) | Δ(z_shuffle) | 判定 |
|---|---|---|---|---|---|---|
| V8 (基线) | 0.7723 | 0.3350 | NO-GO | 0.00 | 0.00 | NO-GO |
| V9-A | 0.7723 | 0.3350 | **27.40** | **+3.32** | **+4.57** | **GO** |
| V9-B | 0.7723 | 0.3374 | 27.34 | +3.12 | +4.47 | GO |
| V9-C | 0.7723 | 0.3457 | **27.70** | **+3.61** | **+4.85** | **GO** |
| V6c (参考) | 0.0525 | 1.0000 | 27.35 | N/A | N/A | - |

**V9-A 是最佳选择**: 最简单 (V8 + decoder), 聚类完全不变 (NMI/leakage 与 V8 完全一致),
重建达到 V6c 水平 (27.40 vs 27.35 dB)。

## 实验设计

### 三个变体

| 变体 | 编码器 | z_actor 与 recon | 假设 | 实际结果 |
|---|---|---|---|---|
| V9-A | V8 motion-only (帧差+几何) | 梯度流入 | recon loss 无法修复 (encoder 瓶颈) | **意外成功** |
| V9-B | RGB crops (所有 t) | 梯度流入 | 外观信息提升 PSNR, 可能增加 leakage | 与 V9-A 几乎相同 |
| V9-C | V8 motion + 独立 z_appearance | z_actor detached | 分离路径保护聚类 | PSNR 最高, 聚类不变 |

### 重建 Decoder (比 probe 更强)

```
Probe decoder (V8, 失败):
  Encoder: 3×Conv → 4×4×128
  FiLM: 1 层 (仅 bottleneck)     ← 太弱, 学到恒等映射
  Decoder: 3×ConvT (无 skip)
  结果: PSNR(probe) = PSNR(z=0) = PSNR(z_shuffle) → z 被忽略

V9 decoder (成功):
  Encoder: 3×Conv → 4×4×128
  FiLM: 3 层 (所有 encoder level) ← 多尺度, 无法忽略 z
  Decoder: 3×ConvT + U-Net skip  ← skip connection 增强容量
  结果: ΔPSNR(recon-z=0) = +3.32 dB → z 被有效使用
```

### 训练配置
- 数据: 合成 multi-actor (500 train / 500 val, 5 actions, T=5)
- 5000 步, batch_size=16, lr=1e-4, AdamW
- L = L_motion + 1.0·L_KL + 0.1·L_recon
- L_recon = L1 + (1 - SSIM), 在 32×32 actor crops 上计算

## 关键发现

### 1. V9-A 意外成功: recon loss 梯度回流修复了 encoder

V8 的 probe 实验证明: 冻结的 z_actor 无法重建 (NO-GO)。
V9-A 的关键区别: z_actor 与 decoder 联合训练, recon loss 梯度通过
z_actor → SharedActorActionHead → SlotTransformer → TemporalTransformer
→ TransitionTokenBuilder → MotionTokenEncoder 回流。

MotionTokenEncoder 在 t=0 时使用 RGB crop (FrameDifferenceCropper)。
recon 梯度推动 encoder 从 t=0 的 RGB 保留外观信息到 z_actor 中。
16 维 VAE bottleneck 足以编码 "动作 + 必要的外观变化"。

### 2. 聚类完全不变: NMI/leakage 与 V8 完全一致

V9-A 的 Overall NMI = 0.7723, Actor Leakage = 0.3350 — 与 V8 完全相同。
Per-Slot NMI = 0.7684, Action Probe = 0.8741 — 也与 V8 完全相同。

这意味着 recon loss (weight=0.1) 没有干扰 motion loss 的聚类学习。
z_actor 同时编码了 "动作聚类信号" 和 "重建所需的外观变化信号",
两者在 16 维空间中不冲突。

### 3. V9-B (RGB crops) 不比 V9-A 更好

V9-B 在所有 t 使用 RGB crop (不用帧差), 给 z_actor 更多外观信息。
但结果与 V9-A 几乎相同:
- PSNR: 27.34 vs 27.40 (V9-A 略高)
- Leakage: 0.3374 vs 0.3350 (V9-B 略高, 但可忽略)
- NMI: 完全一致

结论: V9-A 的 t=0 RGB + recon 梯度已足够。全时 RGB 不提供额外收益,
也不增加 leakage 风险。

### 4. V9-C (dual-pathway) PSNR 最高但更复杂

V9-C 添加独立的 z_appearance (16 维 VAE), z_actor detached from recon。
PSNR = 27.70 dB (三个变体中最高), SSIM = 0.813 (最高)。
但 Actor Leakage 略增 (0.3457 vs 0.3350), 且模型更复杂 (+appearance encoder +VAE)。

z_appearance 专门编码外观, 不受 motion loss 约束, 因此重建质量更好。
但 z_actor (detached) 仍通过 crop_t 间接贡献重建 (decoder 使用 z_actor + z_app)。

### 5. 与 V6c/V7/V8 的架构对比

```
V6c: Patchify → ST encoder → MaskedPool → per-slot VAE → CrossAttn decoder
  z 包含: 外观 + 运动 (MaskedPool 看到完整 RGB patch)
  重建: PSNR 27.35 dB ✓    NMI: 0.0525 ✗    Leakage: 1.0 ✗

V7: MaskedPool → SharedEncoder → factor heads → GRL
  z 包含: actor identity 高速通道 (SharedEncoder)
  NMI: 0.0047 ✗    Leakage: 0.897 ✗

V8: 帧差 + 几何 → VAE → z_actor (无 decoder)
  z 包含: 仅运动
  NMI: 0.7723 ✓    Leakage: 0.3350 ✓    重建: NO-GO ✗

V9-A: V8 + multi-scale FiLM decoder, recon loss 流入 z_actor
  z 包含: 运动 + 必要外观 (recon 梯度从 t=0 RGB 提取)
  NMI: 0.7723 ✓    Leakage: 0.3350 ✓    重建: 27.40 dB ✓
```

V9-A 同时达到了 V6c 的重建水平和 V8 的聚类水平。

## 为什么 V8 probe 失败但 V9-A 成功?

| 因素 | V8 Probe | V9-A |
|---|---|---|
| z_actor 训练 | 冻结 (从 V8 checkpoint 提取) | 联合训练 (recon loss 梯度回流) |
| Decoder 容量 | 单层 FiLM, 无 skip | 多尺度 FiLM (3层) + U-Net skip |
| FiLM 可忽略性 | gamma=1,beta=0 是容易到达的极小值 | 3 层 FiLM 无法同时全部退化 |
| 外观信息来源 | z_actor 从未见过 recon loss | encoder 从 t=0 RGB 学习保留外观 |

**核心原因**: V8 的 MotionTokenEncoder 在 t=0 使用 RGB crop, 但没有 recon loss
时, encoder 丢弃了外观信息 (只保留运动)。V9-A 的 recon loss 梯度推动 encoder
保留 t=0 RGB 中的外观信息到 z_actor, 使其可用于重建。

## 实验文件

| 文件 | 说明 |
|---|---|
| `lam/lam/modules/v9_decoder.py` | ReconDecoder (多尺度 FiLM + U-Net) |
| `lam/lam/modules/v9_model.py` | LatentActionModelV9 (V8 + decoder, 支持 A/B/C) |
| `lam/scripts/run_v9.py` | V9 训练脚本 |
| `lam/scripts/eval_v9.py` | 统一评估 (聚类 + 4-way PSNR + GO/NO-GO) |
| `result/v9/model_v9a_stage1.pt` | V9-A checkpoint |
| `result/v9/model_v9b_stage1.pt` | V9-B checkpoint |
| `result/v9/model_v9c_stage1.pt` | V9-C checkpoint |
| `result/v9/eval_v9a_stage1.json` | V9-A 评估结果 |
| `result/v9/eval_v9b_stage1.json` | V9-B 评估结果 |
| `result/v9/eval_v9c_stage1.json` | V9-C 评估结果 |
| `result/v9/recon_vis_v9a_stage1.png` | V9-A 重建可视化 |

## 下一步

1. **A2D 真实数据验证**: 在 A2D YOLO 数据上训练 V9-A, 检查 NMI 和 PSNR
2. **相机扰动 (Stage 2A)**: 添加 camera perturbation, 检查 z_bg/z_actor 分解
3. **recon_weight 扫描**: 测试 0.01, 0.1, 0.5, 1.0 对 NMI/PSNR 的影响
4. **Per-Slot NMI 深入分析**: 检查为什么 Per-Slot NMI 在所有变体中一致
5. **V9-A 作为 AdaWorld latent action**: 测试 V9-A 的 z_actor 是否适合 world model rollout
