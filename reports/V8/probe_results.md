# V8 Probe Decoder 实验报告: z_actor 重建能力验证

## 结论

**NO-GO: z_actor 不包含足够的像素重建信息。**

Probe decoder 在所有 4 个实验中均未通过 GO 判据。z_actor 在 mode A 下被完全忽略,
在 mode B 下虽有贡献但整体重建质量不优于 copy baseline。

## 实验结果

### 完整对比表

| 实验 | PSNR(probe) | PSNR(copy) | PSNR(z=0) | PSNR(z_shuffle) | Δ(copy) | Δ(z=0) | Δ(z_shuffle) | SSIM(probe) | SSIM(copy) | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| stage1 mode A | 23.52 | 22.86 | 23.52 | 23.52 | +0.66 | **+0.00** | +0.00 | 0.094 | 0.478 | NO-GO |
| stage1 mode B | 23.52 | 22.86 | 23.46 | 19.14 | +0.66 | +0.05 | **+4.37** | 0.094 | 0.478 | NO-GO |
| yolo mode A | 14.15 | **16.92** | 14.15 | 14.15 | **-2.76** | +0.00 | +0.00 | 0.041 | 0.547 | NO-GO |
| yolo mode B | 14.15 | **16.92** | 14.19 | 11.76 | **-2.76** | +0.04 | **+2.39** | 0.041 | 0.547 | NO-GO |

### GO/NO-GO 判据

```
GO 需要 (全部满足):
  ΔPSNR(probe - z=0)       ≥ 1.0 dB   ← z_actor 有信息
  ΔPSNR(probe - z_shuffle) ≥ 0.5 dB   ← z_actor 语义正确
  ΔPSNR(probe - copy)      ≥ 0.5 dB   ← 比 copy 好
```

| 判据 | stage1 A | stage1 B | yolo A | yolo B |
|---|---|---|---|---|
| z=0 ≥ 1.0 dB | ✗ (0.00) | ✗ (0.05) | ✗ (0.00) | ✗ (0.04) |
| z_shuffle ≥ 0.5 dB | ✗ (0.00) | ✓ (4.37) | ✗ (0.00) | ✓ (2.39) |
| copy ≥ 0.5 dB | ✓ (0.66) | ✓ (0.66) | ✗ (-2.76) | ✗ (-2.76) |
| **总判定** | **NO-GO** | **NO-GO** | **NO-GO** | **NO-GO** |

## 关键发现

### 1. Mode A: z_actor 完全被忽略

Mode A (crop_t + z_actor) 中, z=0 和 z_shuffle 的 PSNR 与 probe **完全相同** (到小数点后两位)。
这意味着 FiLM 层学到了恒等映射 (gamma≈1, beta≈0), decoder 完全不依赖 z_actor。

原因: decoder 发现仅从 crop_t 做平均预测就能达到 23.52 dB (合成) / 14.15 dB (A2D),
使用 z_actor 反而增加优化难度。FiLM 架构不够强, 无法强制 decoder 使用 z_actor。

### 2. Mode B: z_actor 被使用, 但信息与 dbbox_pred 冗余

Mode B (crop_t + z_actor + dbbox_pred) 中:
- z_shuffle 导致 PSNR 下降 2-4 dB → **z_actor 确实被 decoder 使用了**
- 但 z=0 不影响 PSNR → decoder 可以用 dbbox_pred 补偿 z_actor=0
- 整体 PSNR 不优于 mode A → z_actor 的额外信息没有提升重建质量

结论: z_actor 包含的运动信息与 dbbox_pred 高度重叠。dbbox_pred 是 V8 ActorMotionPred
从 z_actor 直接预测的 Δbbox, 两者携带相似的运动信号。

### 3. A2D 真实数据: probe 比 copy 更差

在 v8_yolo 上, PSNR(probe)=14.15 < PSNR(copy)=16.92, 差距 -2.76 dB。
probe decoder 在真实数据上学到的"平均预测"比直接 copy 还差。
SSIM(probe)=0.041 极低, 说明输出是模糊的平均值。

### 4. SSIM 异常低

所有实验中 SSIM(probe) 远低于 SSIM(copy):
- stage1: 0.094 vs 0.478
- yolo: 0.041 vs 0.547

这表明 probe decoder 产生了模糊的、缺乏结构的输出, 而不是清晰的重建。
可能原因: L1 loss 倾向于产生中位数 (平均) 预测, SSIM 的简化实现可能不准确。

## 根因分析

### 为什么 z_actor 不能重建?

```
V8 MotionTokenEncoder:
  t=0: crop(I_0, bbox_0) → RGB crop (含外观)
  t>0: crop(ΔI_t, union_bbox) → 帧差 crop (不含外观, 只有运动)
  + bbox 几何 [x,y,w,h,dx,dy,dw,dh]
  → motion_token → ... → z_actor (16 dim)
```

z_actor 编码的是:
- 帧差模式 (运动方向、速度)
- bbox 几何变化 (位移、缩放)
- 通过 VAE bottleneck 压缩的聚类特征

z_actor **不编码**:
- actor 外观 (颜色、纹理、形状)
- 环境外观
- 运动如何映射到像素变化

Probe decoder 需要: 从 crop_t (当前外观) + z_actor (运动信号) → crop_{t+1} (下一帧外观)。
但 z_actor 只知道"在向左移动", 不知道"左边的像素长什么样"。

### 与 V6c/AdaWorld 的对比

```
V6c: Patchify → ST encoder → MaskedPool → per-object VAE → CrossAttn decoder
  z 包含: 外观 + 运动 (因为 encoder 看到了完整 patch)
  重建: PSNR 27.35 dB (合成)
  NMI: 0.05 (actor leakage 1.0, 外观污染了 z)

V8: 帧差 + 几何 → VAE → z_actor
  z 包含: 仅运动 (帧差不含外观)
  重建: probe NO-GO (z_actor 无法重建)
  NMI: 0.77 (无外观泄漏, 聚类好)

AdaWorld: 两帧 → latent action encoder → VAE → decoder 重建下一帧
  z 包含: 转移信息 (从两帧提取)
  重建: PSNR 26.3 dB (合成)
  NMI: 0.02 (单场景, 无多主体)
```

V8 用 NMI 换了重建能力: 去掉外观编码 → 聚类好 → 但 z_actor 不再是 latent action。

## 下一步建议

### 选项 1: 接受 V8 定位, 不做重建
- V8 是 "motion clustering model", 不是 "latent action model"
- 论文定位: 用 MOT + motion-only encoder 做 action clustering, 不追求重建
- 优点: NMI 0.77 已有说服力
- 缺点: 脱离 AdaWorld latent action 框架

### 选项 2: 双路径架构 (V9)
- 保留 V8 motion pathway (z_actor 用于 clustering)
- 新增 appearance pathway: 独立 encoder → z_appearance → decoder 重建
- z_actor + z_appearance 共同用于重建, 但 z_actor 不受重建 loss 污染
- 优点: 既有 NMI 又有重建
- 缺点: 需要重新设计模型

### 选项 3: 改变 z_actor 编码方式
- 在 MotionTokenEncoder 中加入 t=0 的 RGB crop (目前 t=0 用 RGB, t>0 用帧差)
- 让 z_actor 同时看到外观和运动
- 风险: 重复 V7 的 actor leakage 问题
- 需要严格的 isolation 机制

### 选项 4: 条件化重建 (z_actor 作为 warp 条件)
- 不是从 z_actor 重建像素, 而是用 z_actor 预测光流/warp
- crop_t + flow(z_actor) → warped crop → crop_{t+1}
- z_actor 编码运动场而非外观
- 优点: 更符合 z_actor 的信息内容
- 缺点: 需要光流估计或 spatial transformer

## 实验文件

| 文件 | 说明 |
|---|---|
| `lam/lam/modules/probe_decoder.py` | ActorProbeDecoder (FiLM conditioning) |
| `lam/scripts/extract_latents_v8.py` | 从 checkpoint 提取 z + crop pairs |
| `lam/scripts/train_probe_decoder.py` | 训练 probe (video-grouped split) |
| `lam/scripts/eval_v8_reconstruction.py` | 4 重 PSNR + 可视化 |
| `result/v8_mot_lam/latent_pairs_v8_stage1.npz` | 合成数据 latent pairs (6072 samples) |
| `result/v8_mot_lam/latent_pairs_v8_yolo.npz` | A2D 数据 latent pairs (851 samples) |
| `result/v8_mot_lam/recon_vis_probe_*.png` | 重建可视化 |
