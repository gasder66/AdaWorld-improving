# V9 YOLO A2D 真实数据实验报告

## 结论

**合成数据 GO, 真实数据 NO-GO。** 根因: 合成数据中 motion = appearance change
(简单形状位移), z_actor 够用; 真实数据中 motion ≠ appearance change
(姿态/光照/纹理变化), z_actor (motion-only) 无法贡献重建。

## 完整对比表

### 合成数据 (Stage 1)

| 指标 | V8 | V9-A | V9-B | V9-C |
|---|---|---|---|---|
| NMI | 0.7723 | **0.7723** | 0.7723 | 0.7723 |
| Leakage | 0.3350 | **0.3350** | 0.3374 | 0.3457 |
| PSNR(recon) | NO-GO | **27.40** | 27.34 | 27.70 |
| Δ(z=0) | 0.00 | **+3.32** | +3.12 | +3.61 |
| Verdict | NO-GO | **GO** | GO | GO |

### 真实数据 (YOLO A2D)

| 指标 | V8 | V9-A | V9-C (FiLM) | V9-C v2 (FiLM+z_bg det) | V9-C v3 (no FiLM, z_bg det) |
|---|---|---|---|---|---|
| NMI | 0.3118 | 0.2682 | 0.3374 | 0.3374 | 0.2579 |
| Leakage | 0.5556 | 0.7778 | 0.8889 | 0.7778 | **0.5556** |
| PSNR(recon) | NO-GO | 17.57 | 17.24 | 17.67 | 17.78 |
| Δ(z=0) | 0.00 | +0.26 | -0.77 | -0.17 | -0.16 |
| Δ(copy) | - | +0.67 | +0.34 | +0.77 | +0.88 |
| Verdict | NO-GO | NO-GO | NO-GO | NO-GO | NO-GO |

## 诊断过程

### 1. 训练终止根因

| 问题 | 根因 | 修复 |
|---|---|---|
| `num_workers=4` 挂起 | `persistent_workers=True` | 移除 persistent_workers |
| nohup 进程被杀 | session 结束发 SIGHUP | 使用 `setsid` 脱离 session |
| 信号处理器误杀 | 捕获了 SIGCHLD (无害信号) | 只捕获致命信号 |
| 管道输出无响应 | bash timeout 杀管道 → SIGPIPE | 输出到文件, 不用管道 |

### 2. Leakage 根因: FiLM 而非 recon loss

| 实验 | FiLM | z_bg detach | Leakage | 结论 |
|---|---|---|---|---|
| V8 YOLO | 无 | - | 0.5556 | 基线 |
| V9-A YOLO | **有** | 否 | 0.7778 | FiLM + recon → leakage↑ |
| V9-C YOLO | **有** | 否 | 0.8889 | FiLM + z_bg recon → leakage↑↑ |
| V9-C v2 | **有** | **是** | 0.7778 | z_bg detach 降 0.89→0.78 |
| V9-C v3 | **无** | **是** | **0.5556** | 无 FiLM + z_bg detach = V8 基线 |

**结论**: FiLM 条件化 (num_actor_types=7) 是 leakage 上升的主因。
V8 Stage 2C 已验证: FiLM 使 leakage 从 0.17 升到 0.56。
V9-C v3 (无 FiLM, z_bg detached) 的 leakage = 0.5556 = V8 基线, 确认 V8 完全隔离。

### 3. z_actor 对真实视频重建无贡献

V9-C v3 的 4-way PSNR:
- PSNR(recon) = 17.78 dB (z_app 重建)
- PSNR(z=0) = 17.94 dB (z_actor=0, 仅 z_app)
- **Δ(z=0) = -0.16 dB** → z_actor 实际有害, decoder 最好忽略它
- Δ(z_shuffle) = +0.05 → z_actor 信号可忽略

z_app 承担了所有重建工作 (SSIM=0.63 vs copy=0.61, PSNR Δ(copy)=+0.88)。
z_actor (motion-only, detached) 对真实视频重建无贡献。

### 4. 合成 vs 真实数据的根本差异

```
合成数据:
  actor = 纯色形状 (正方形/圆形/三角形, 5 种颜色)
  action = 网格位移 (上/下/左/右/不动, 32px 步长)
  appearance change = 形状位移 → motion 完全描述 appearance change
  → z_actor (motion) 足以重建 → GO

真实视频 (A2D):
  actor = 人/车/鸟/猫/狗 (复杂纹理, 非刚性形变)
  action = climbing/crawling/eating/flying/jumping/rolling/running/walking
  appearance change = 姿态变化 + 光照变化 + 视角变化 + 纹理变化
  motion (bbox 位移) 只是 appearance change 的一小部分
  → z_actor (motion) 无法重建 → NO-GO
```

## 下一步建议

### 短期 (保持当前架构)
1. **接受合成数据 V9-A 的成功**: NMI 0.77 + PSNR 27.40, 作为 proof-of-concept
2. **真实数据用 V9-C v3**: z_app 负责重建 (PSNR 17.78), z_actor 负责聚类 (NMI 0.26)
   - 不追求 z_actor 对重建的贡献
   - 将 z_app 作为独立的 appearance code, z_actor 作为 action code

### 中期 (改变 z_actor 编码方式)
3. **光流条件化重建**: z_actor → 光流 → warp(crop_t) ≈ crop_{t+1}
   - z_actor 编码运动场而非像素变化
   - 更符合 z_actor 的 motion-only 信息内容
4. **两帧输入编码器**: 像 AdaWorld 一样, encoder 接收 (I_t, I_{t+1}) 联合编码
   - z_actor 从两帧差值中提取 transition, 而非从帧差 crop
   - 可能比 MotionTokenEncoder 携带更多 transition 信息

### 长期 (架构革新)
5. **V10: 解耦 appearance 和 action 的 world model**
   - z_app (appearance) + z_actor (action) → decoder → next frame
   - z_actor 不做像素重建, 而是预测 z_app 的变化 (latent dynamics)
   - 类似 AdaWorld 的 world model rollout, 但在 slot-level
