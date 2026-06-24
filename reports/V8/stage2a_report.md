# V8 Stage 2A: Camera Perturbation — Background Slot Validation

## Summary

Stage 2A 在合成数据上加入全局相机扰动 (pan/zoom/brightness),对比 V8-with-bg 和
V8-no-bg 两个变体,验证背景槽 (z_bg) 能否吸收相机运动、保护 z_actor 不受污染。

**结论: z_bg / z_actor 分解设计完全成功。**

## Results

### 主对比表

| Metric | V8-no-bg | **V8-with-bg** | 说明 |
|---|---|---|---|
| Overall NMI | 0.7268 | **0.7589** | with-bg 更高 (z_actor 更纯净) |
| Per-Actor NMI | 0.7217 | **0.7569** | with-bg 更高 |
| Actor Leakage | 0.3407 | **0.3374** | 两者都低 (~chance 0.25) |
| Action Probe | 0.8658 | **0.8724** | with-bg 略高 |
| dbbox MSE (px²) | 26.50 | **21.56** | with-bg 低 19% |
| dbbox RMSE (px) | 5.15 | **4.64** | — |

### Camera Probe R² (核心验证)

| Probe | V8-no-bg | V8-with-bg | 预期 |
|---|---|---|---|
| z_bg → camera params | -0.004 | **0.901** | with-bg 高 ✓ |
| z_actor → camera params | 0.381 | **0.223** | with-bg 低 ✓ |

- **z_bg → camera R² = 0.901**: z_bg 几乎完美编码相机参数 (pan/zoom/brightness)
- **z_actor → camera R² = 0.223**: z_actor 的 camera leakage 显著低于 no-bg (0.381)
- no-bg 的 z_bg R² ≈ 0 (z_bg 不存在,全零),z_actor 被迫吸收相机信息 (R²=0.381)

### Zero-out Ablation

| Variant | normal | z_bg=0 | z_actor=0 |
|---|---|---|---|
| **with-bg** | 21.48 | 37.63 (Δ+16.15) | 344.61 (Δ+323.13) |
| no-bg | 26.38 | 26.38 (Δ+0.00) | 370.07 (Δ+343.69) |

- **with-bg z_bg=0**: MSE 增加 75% (21→38) — 置零 z_bg 使相机运动预测退化
- **with-bg z_actor=0**: MSE 增加 15x (21→345) — 置零 z_actor 使 actor 运动预测退化
- 两个 latent 有明确的、互补的职责

### Latent Stats

| Metric | V8-no-bg | V8-with-bg |
|---|---|---|
| z_actor variance | 0.4275 | 0.4462 |
| z_bg variance | 0.0000 | **0.3429** |

- with-bg 的 z_bg 方差 0.343 (Stage 1 无扰动时仅 0.075) — z_bg 在有相机运动时活跃

## Architecture Changes (Stage 2A)

### 1. Camera Perturbation (on-the-fly, vectorized)
- `lam/lam/camera_perturbation.py`: 全向量化 affine_grid/grid_sample,无 Python 循环
- 每帧随机 pan (±8px) + zoom (±10%) + brightness (±0.05),累积变换
- 返回 `camera_params` (T-1, 4) 作为 L_bg 伪标签

### 2. CameraMotionPredictor (bbox-conditioned)
- 旧版: `z_bg → Δbbox_bg` (广播到所有 actor,无法建模 zoom)
- 新版: `z_bg + bbox_t → Δbbox_bg` (条件化 bbox 位置,边缘 bbox 在 zoom 时移动更多)

### 3. BgSupervisionHead + L_bg
- `z_bg → camera_params_pred` (4 维)
- `L_bg = MSE(camera_params_pred, camera_params / scale)`
- 每个 camera param 独立归一化到 O(1)

### 4. use_bg_slot flag (ablation)
- `use_bg_slot=False`: 禁用 bg slot,z_bg 不存在,dbbox_pred = dbbox_res only

## Training

- **Steps**: 5000 (each variant)
- **Loss (with-bg)**: L_motion + 1.0×L_KL + 1.0×L_bg
- **Loss (no-bg)**: L_motion + 1.0×L_KL
- **Time**: ~15 min each (879s / 847s)
- **Memory**: 0.4 GB each

## Files

- `lam/lam/camera_perturbation.py` — 向量化相机扰动
- `lam/scripts/eval_v8_stage2a.py` — Stage 2A 评估 (camera probe + zero-out)
- `result/v8_mot_lam/model_v8_s2a_bg.pt` — with-bg checkpoint
- `result/v8_mot_lam/model_v8_s2a_nobg.pt` — no-bg checkpoint
- `result/v8_mot_lam/ablation_zero_out.json` — zero-out ablation data

## Success Criteria (from V8_architecture.md)

| Criterion | Result | Status |
|---|---|---|
| with-bg actor NMI 不下降 | 0.7589 > no-bg 0.7268 | PASS ✓ |
| with-bg z_actor camera leakage 明显低 | 0.223 < no-bg 0.381 | PASS ✓ |
| z_bg camera probe R² 明显高 | 0.901 >> no-bg -0.004 | PASS ✓ |

所有 Stage 2A 成功标准均满足。背景槽不是装饰品 — 它有效吸收相机运动,
保护 z_actor 的 action 编码纯度。
