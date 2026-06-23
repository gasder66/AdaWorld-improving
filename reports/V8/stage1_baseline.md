# V8 Stage 1 — V6c 基线指标（Phase 0 产出）

**日期**: 2026-06-23
**V6c checkpoint**: `result/old_result/v6_structured/model_v6c_long.pt` (10000 步训练, PSNR=27.35 dB)
**评估数据**: 合成数据 val 集 15 batch (7680 样本, 5832 有效)

## 指标

| 指标 | V6c 实测 | 说明 |
|---|---|---|
| **Overall NMI** | **0.0525** | 所有 slot 的 z 混合后 KMeans(5) vs GT action |
| Overall ARI | 0.0239 | 同上 |
| **Per-slot NMI avg** | **0.3885** | 每个 slot 内部 KMeans(5) vs GT action |
| Per-slot NMI (Slot 0/1/2/3) | 0.3945 / 0.3029 / 0.3417 / 0.5150 | — |
| **Actor Leakage** | **1.0000** | z_actor → slot_index 分类准确率 (chance=0.25) |

## 关键解读

1. **V6c 的 "NMI=0.36" 是 per-slot NMI**，不是 overall NMI。Per-slot NMI 衡量每个 slot 内部按动作聚类，但跨 slot 完全不可比（z_actor 被 actor 身份完全主导）。
2. **Overall NMI = 0.0525** 接近 chance，说明 V6c 的 z 不能跨 actor 比较 action — 这正是 V8 要解决的核心问题。
3. **Actor leakage = 1.0000** 证实 V6c 的 z 完全编码 actor 身份。V8 目标是让 actor_leakage 接近 chance (0.25)。

## V8 Stage 1 成功标准修正

| 指标 | V6c 基线 | V8 Stage 1 目标 |
|---|---|---|
| Overall NMI | 0.0525 | **≥ 0.20** (远超 V6c) |
| Per-slot NMI avg | 0.3885 | ≥ 0.30 (接近 V6c) |
| Actor Leakage | 1.0000 | **≤ 0.50** (远低于 V6c) |
| bbox motion MSE | — | < 常数预测 baseline |

V8 的核心价值不是 per-slot NMI 提升，而是 **overall NMI 提升 + actor leakage 下降** — 即真正实现"所有 actor 共享 action space"。
