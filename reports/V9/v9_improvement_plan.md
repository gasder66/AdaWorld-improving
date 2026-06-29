# V9 线改进计划: Motion-Only Latent Action (聚类优先)

## 定位

**这是 V8/V9 线的延续。** V8 用帧差+几何编码实现高 NMI 聚类 (0.77), 但无重建。
V9 加了 recon decoder, 合成数据 GO (NMI 0.77 + PSNR 27.4), 真实数据 NO-GO
(z_actor motion-only 无法重建真实视频的 appearance change)。

V9 线的目标: **动作聚类理解**, 重建是诊断工具而非核心目标。改进方向是
更强的运动特征 (超越 bbox 位移) 和更公平的评估。

## V9 线 vs V10 线 (明确区分)

| | V9 线 (本计划) | V10 线 (并行) |
|---|---|---|
| **编码器** | MotionTokenEncoder (帧差+几何) | ST encoder + MaskedPool (完整RGB) |
| **z 设计** | motion-only, 刻意排除外观 | 外观+运动, 不排除 |
| **重建角色** | 诊断工具 (验证 z 是否携带 transition 信息) | 核心目标 (world model) |
| **聚类** | 天然成功 (shared head + 无外观) | 需要修复 (shared VAE) |
| **核心问题** | bbox 位移太简陋, 真实数据失效 | per-slot VAE 导致聚类失败 |
| **改进方向** | 更精细的运动特征 (光流, SAM) | shared VAE + 重建保留 |
| **论文定位** | "不需要外观也能理解动作" | "object-level world model" |

## 当前 V9 线的问题 (基于实验结果)

### 问题 1: bbox 运动特征在真实数据上失效

V8 MotionTokenEncoder:
- t>0 用帧差 crop → 只看到 "哪里变了"
- 几何只有 `[cx,cy,w,h,dx,dy,dw,dh]` (8 维)
- 合成数据: motion = appearance change (方块位移) → 有效
- 真实数据: motion ≠ appearance change (姿态/光照/形变) → z_actor 对重建无贡献

**V9-C v3 实验确认**: ΔPSNR(z=0)=-0.16, z_actor 被decoder 忽略。

### 问题 2: Leakage 指标定义不适用于 V9 线

V9 线的 z_actor 是 motion-only, leakage 应该很低。但 V9-C v3 (no FiLM)
leakage=0.5556 = V8 baseline。这说明:

- leakage 不来自 z_actor 内容 (motion-only), 而来自 **slot 位置本身**
- 不同 slot 有不同的 bbox 位置分布 → slot ID 可从 z 的几何分量恢复
- 这不是真正的 "actor identity 泄漏", 而是 "位置信息泄漏"

**对 V9 线**: leakage 指标意义不大 (z_actor 确实不含外观), 应关注 NMI 和
Action Probe。

### 问题 3: 合成数据 PSNR 虚高

代码确认根因 (非黑色 padding):
1. **静态背景**: checkerboard 所有帧相同 (seed 被忽略), 占 69% 像素
2. **平坦 actor**: 纯色实心, 内部重建误差≈0
3. **copy baseline 已 22.68 dB**: actor 常不动或移一格

**对 V9 线的影响**: V9-A 的 PSNR 27.40 Δ(copy)=+4.72 看起来好, 但
masked PSNR 可能大幅下降。需要 actor-masked PSNR 验证。

### 问题 4: FiLM 条件化导致 leakage 上升 (V9-C YOLO)

V8 Stage 2C 和 V9 YOLO 都确认: FiLM (num_actor_types=7) 使 leakage
从 0.56 升到 0.78-0.89。

**对 V9 线**: 如果不用 FiLM, leakage 保持 V8 baseline。FiLM 对 NMI
帮助有限 (V8 Stage 2C: NMI 0.07→0.07), 可以不用。

## 改进方案

### Phase 1: 更精细的运动特征 (V9-D)

**目标**: 超越 bbox 位移, 让 z_actor 在真实数据上也能捕获有意义的 transition

**方案 A: 光流条件化重建**
- z_actor → 预测光流 → warp(crop_t) → crop_{t+1}
- z_actor 编码运动场而非像素变化
- 更符合 motion-only 的信息内容
- 不需要外观信息, 只需要 "怎么动"

**方案 B: 密集光流作为编码器输入**
- 用 RAFT 预计算 actor region 内的光流
- 光流 (H×W×2) 替代帧差 crop (H×W×3) 作为 MotionTokenEncoder 输入
- 光流包含方向+速度, 比帧差更丰富

**方案 C: SAM mask + 帧差**
- SAM 精确分割 actor → mask 内的帧差 (排除背景干扰)
- 比 union bbox 的帧差更干净
- 可以捕获非刚性形变的运动模式

**推荐**: 方案 A (光流条件化重建) 最符合 V9 线的 motion-only 定位。
z_actor 做它擅长的事 (预测运动), 不做它做不了的事 (生成像素)。

### Phase 2: 评估改进

1. **Actor-masked PSNR**: 排除背景, 真正衡量 actor 重建
2. **Action Probe (不用 leakage)**: V9 线的 z_actor 不含外观, leakage 指标
   衡量的是位置信息, 不是 actor identity。用 Action Probe 作为主要指标。
3. **动态背景合成数据**: 使用 noise 背景 (每帧不同) 或给 checkerboard 加
   帧间亮度抖动, 使 PSNR 不被静态背景主导

### Phase 3: 真实数据 (A2D)

1. 用 RAFT 预计算 A2D 光流
2. 训练 V9-D (光流条件化重建)
3. 评估: NMI + Action Probe + 光流重建质量

## 实施优先级

| 优先级 | 任务 | 工时 | 依赖 |
|---|---|---|---|
| P1 | Actor-masked PSNR (V9 现有模型) | 1h | 无 |
| P1 | Action Probe 作为 V9 主指标 | 0.5h | 无 |
| P1 | 动态背景合成数据 | 1h | 无 |
| P2 | V9-D: 光流条件化重建 | 4h | 无 |
| P2 | RAFT 预计算 A2D 光流 | 3h | 无 |
| P3 | SAM mask + 帧差 (备选) | 3h | SAM |

## V9 线的论文定位

"我们证明 motion-only 编码足以实现动作聚类 (NMI 0.77), 不需要外观信息。
进一步, 通过光流条件化重建, z_actor 可以预测运动场而不需要生成像素,
在真实数据上也能工作。

V9 线与 V10 线互补:
- V9: '不需要外观也能理解动作' (motion-only, 聚类优先)
- V10: 'object-level world model' (外观+运动, 重建+聚类)
两条线分别探索了 latent action 的不同侧面。"
