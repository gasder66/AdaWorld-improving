# V10 改进计划: 回到 V6c, 统一动作空间, 完整故事线

## 核心思路

**退一步, 海阔天空。** 不再试图从 z_actor 中去除 actor 信息 (V8/V9 的路线),
而是接受 actor 类型作为 action 理解的正向条件, 同时修复 V6c 的 per-slot VAE
问题使聚类成为可能。这样我们同时拥有重建 (V6c 的 CrossAttn decoder) 和聚类
(shared VAE), 完成整个故事线。

## 问题诊断 (基于代码分析)

### 问题 1: V6c 的 per-slot VAE 是聚类失败的唯一根因

V6c 的 `self.fcs = nn.ModuleList([nn.Linear(model_dim, latent_dim*2) for _ in range(K)])`
(lam.py:80-82) — 5 个独立的线性层, 每个 slot 一个。

**后果**: slot 0 的 z 和 slot 1 的 z 由不同参数产生 → 在不同子空间 → 混在一起
聚类没有意义 (NMI=0.05)。但 per-slot NMI=0.39 (每个 slot 内部可以聚类)。

**修复**: 替换为单个 shared `nn.Linear(model_dim, latent_dim*2)` — 所有 slot
共享参数 → z 在同一空间 → 可以跨 slot 聚类。decoder 已经是 slot-agnostic 的,
不需要修改。

### 问题 2: Actor leakage 指标的定义需要修正

当前指标: `LogisticRegression(z_actor → actor_id)` 准确率, 目标 ≤ 0.50。

**问题**: actor 类型 (人/车/猫/狗) 本身对理解 action 有正向意义。人的 "running"
和车的 "running" 是不同的 action 语义。把 actor 类型当作 "应该去除的泄漏" 是
错误的预设。

**修正方案**:
- **保留 leakage 指标但重新解读**: leakage 高不一定坏, 关键看 NMI 是否也高
- **新增 "Conditional NMI"**: 给定 actor 类型后的 action NMI (条件互信息)
  - 如果 `NMI(z, action | actor_type) > NMI(z, action)` → z 编码了超越
    actor 类型的 action 信息
  - 如果 `NMI(z, action | actor_type) ≈ NMI(z, action)` → z 只编码了
    actor 类型, 没有额外的 action 信息
- **新增 "Action Probe given Type"**: 在每个 actor 类型内部分别训练
  action 分类器, 取平均准确率

### 问题 3: bbox 追踪太简陋

当前 V8 的 MotionTokenEncoder:
- t>0 用帧差 crop (`I_t - I_{t-1}`) → 只能看到 "哪里变了"
- 几何信息只有 `[cx, cy, w, h, dcx, dcy, dw, dh]` (8 维)
- 在合成数据上 (纯色方块移动) 有效, 在真实视频上 (姿态变化、非刚性形变) 失效

**改进方向**:
- **SAM 分割**: 用 SAM 替代 bbox, 获取精确的 object mask
  - mask-pooled 特征比 bbox-pooled 特征更干净 (不含背景)
  - 可以捕获非刚性形变 (人的姿态变化)
- **光流**: 在 object region 内计算光流, 作为密集运动特征
  - 光流比帧差包含更多信息 (方向、速度、形变模式)
- **预训练特征**: DINOv2 / CLIP visual encoder 提取语义特征
  - 不依赖像素级重建, 而是在特征空间操作

### 问题 4: 合成数据 PSNR 虚高

**根因分析** (代码已确认):
- **不是黑色 padding**: camera_perturbation 用 `padding_mode="border"` (边缘复制),
  不是 zeros; bbox 被 clamp 到图像范围内, crop 不会采样到界外
- **真正原因**: 
  1. **静态背景**: checkerboard 背景在所有帧中完全相同 (`synthetic_dataset.py:76-89`,
     seed 参数被忽略), 占 ~69% 像素, 重建误差≈0
  2. **平坦 actor 内部**: 纯色实心形状, 内部像素重建误差≈0
  3. **copy baseline 已经 22.68 dB**: actor 在网格上经常不动或只移一格,
     `crop_t ≈ crop_{t+1}`
  4. **V6c 的 27.35 dB 是全帧 PSNR**: 被 69% 的易重建背景主导

**修复方案**:
- **Actor-masked PSNR**: 只在 actor mask 区域内计算 PSNR
  - `DiskSyntheticDataset` 已经返回 `masks (T, K, H, W)`, 直接用
  - 这排除了背景, 真正衡量 actor 重建质量
- **报告 copy baseline**: 每次 eval 必须同时报告 PSNR(copy) 作为参考
  - `ΔPSNR(recon - copy)` 才是真正的重建增益
- **动态背景**: 修改合成数据生成器, 使背景在帧间变化 (使用 `noise` 模式
  而非 `checkerboard`, 或给 checkerboard 加帧间亮度抖动)

## 实施方案

### Phase 1: V6c + Shared VAE (V10-A)

**目标**: 证明 shared VAE 修复聚类, 保留重建

**代码改动**:

1. **新建 `lam/lam/modules/v10_model.py`**:
   - 继承 `LatentActionModel` (V6c)
   - 重写 `__init__`: 用单个 `self.fc = nn.Linear(model_dim, latent_dim*2)` 替代
     `self.fcs = nn.ModuleList([...])`
   - 可选: 加入 FiLM 条件化 (actor type → gamma, beta)
   - 重写 `encode()`: 用 shared fc 处理所有 slot
   - 保留 decoder, CrossAttention, SpatioTransformer 不变

2. **新建 `lam/scripts/run_v10.py`**:
   - 修复 `run_single.py` 的兼容性问题 (移除不存在的 kwargs)
   - 使用 model 内部的 `outputs["kl_loss"]` (Free Bits KL)
   - 添加 `outputs["delta_loss"]` 和 `outputs["contrast_loss"]` 到总损失
   - Loss = `recon_loss + β·kl_loss + λ·obj_recon_loss + μ·delta_loss + ν·contrast_loss`

3. **新建 `lam/scripts/eval_v10.py`**:
   - Overall NMI / Per-Slot NMI / Actor Leakage (同 V8 eval)
   - **新增 Conditional NMI**: `NMI(z, action | actor_type)`
   - **新增 Action Probe given Type**: 每个 actor 类型内部 action 分类
   - **Actor-masked PSNR**: 只在 mask 区域计算
   - **Copy baseline PSNR**: `PSNR(crop_t, crop_tp1)` 作为参考
   - UMAP 可视化 (4 图: action / actor_type / KMeans / conditional)

**实验**:
- 合成数据: 5000 步, 与 V6c/V8/V9 相同配置
- 评估: NMI (Overall + Per-Slot + Conditional), Leakage, PSNR (masked + copy baseline)
- 对比: V6c (per-slot VAE) vs V10-A (shared VAE) vs V8 (motion-only)

**预期结果**:
- NMI: 0.05 → >0.20 (shared VAE 使跨 slot 聚类成为可能)
- PSNR: 保持 ~27 dB (decoder 不变)
- Leakage: 可能仍高 (MaskedPool 仍泄漏外观), 但配合 Conditional NMI 重新解读

### Phase 2: 改进评估指标 (V10-B)

**目标**: 建立更公平的评估体系

1. **Conditional NMI 实现**:
   ```python
   # 给定 actor_type 后, z 还能提供多少 action 信息?
   for actor_type in unique_types:
       idx = actor_types == actor_type
       if idx.sum() < 50: continue
       kmeans = KMeans(n_clusters=n_actions)
       pred = kmeans.fit_predict(z[idx])
       nmi_conditional.append(normalized_mutual_info_score(actions[idx], pred))
   nmi_conditional_avg = mean(nmi_conditional)
   ```

2. **Actor-masked PSNR**:
   ```python
   # 只在 actor mask 区域计算 MSE
   mask = masks[:, 1:]  # (B, T-1, K, H, W)
   masked_pred = recon * mask
   masked_gt = gt * mask
   mse = ((masked_pred - masked_gt) ** 2).sum() / (mask.sum() * 3 + 1e-8)
   psnr = -10 * log10(mse)
   ```

3. **动态背景合成数据** (可选):
   - 修改 `synthetic_dataset.py` 使 checkerboard 帧间变化
   - 或使用 `noise` 背景 (已有, 每帧不同种子)

### Phase 3: 更精细的特征提取 (V10-C)

**目标**: 超越 bbox, 用更丰富的运动特征

**方案 A: SAM 分割替代 bbox**
- 用 SAM (Segment Anything Model) 预计算每个 actor 的精确 mask
- 用 mask-pooled 特征替代 bbox-pooled 特征
- 保留 V6c 的 MaskedPool (它已经支持 mask 输入), 只需要更好的 mask

**方案 B: 光流特征**
- 在 actor region 内计算光流 (RAFT / FlowNet)
- 光流作为额外的 motion token 输入
- 光流包含密集运动场 (方向 + 速度), 比 bbox 位移丰富得多

**方案 C: 预训练视觉特征**
- 用 DINOv2 提取 patch 特征, 替代 V6c 的 from-scratch ST encoder
- 预训练特征包含语义信息, 可能帮助 action 理解
- 风险: 预训练特征可能携带更多 actor identity

**推荐**: 先做方案 A (SAM), 因为:
- SAM 可以离线预计算, 不增加训练开销
- mask 比 bbox 精确得多, 特别是对于非刚性物体 (人、动物)
- V6c 的 MaskedPool 已经支持 mask 输入, 改动最小

### Phase 4: 真实数据验证

**目标**: 在 A2D 上验证 V10

1. 用 SAM 对 A2D 视频预计算 mask (替代 YOLO bbox)
2. 训练 V10-A + SAM mask
3. 评估:
   - Overall NMI (跨所有 actor 类型)
   - Conditional NMI (给定 actor 类型后)
   - Action Probe given Type
   - Actor-masked PSNR
   - Leakage (重新解读: 高 leakage + 高 conditional NMI = actor 类型信息被正确编码)

## 论文故事线

### 当前版本 (V8/V9)
"我们去除 actor 外观信息, 实现高 NMI 聚类, 但牺牲了重建能力。"
→ 问题是真实数据上 z_actor 无法重建 (motion ≠ appearance change)

### 改进版本 (V10)
"我们学习 object-level latent action, 同时实现:
  (1) 像素重建 (CrossAttn decoder)
  (2) 跨 actor 聚类 (shared VAE)
  (3) actor 类型条件化 (FiLM)
Actor 类型是 action 理解的正向条件, 不是需要去除的噪声。
Conditional NMI 证明 z 编码了超越 actor 类型的 action 信息。"

### 关键贡献
1. **Shared VAE**: 解决 V6c per-slot VAE 导致的聚类失败
2. **Conditional NMI**: 新的评估指标, 区分 "actor 类型信息" 和 "action 信息"
3. **Actor-masked PSNR**: 公平的重建评估, 不被背景主导
4. **SAM 分割**: 替代 bbox, 捕获非刚性形变

## 实施优先级

| 优先级 | 任务 | 预计工时 | 依赖 |
|---|---|---|---|
| P0 | V10-A: shared VAE + 训练 + 评估 | 4h | 无 |
| P0 | Actor-masked PSNR + copy baseline | 1h | V10-A |
| P1 | Conditional NMI + Action Probe given Type | 2h | V10-A |
| P1 | 动态背景合成数据 | 1h | 无 |
| P2 | SAM 预计算 A2D mask | 3h | 无 |
| P2 | V10-C: SAM mask 替代 bbox | 2h | SAM 预计算 |
| P3 | 光流特征 (如果 SAM 不够) | 4h | 无 |
| P3 | DINOv2 特征 (如果需要) | 3h | 无 |

## 文件清单

| 文件 | 说明 |
|---|---|
| `lam/lam/modules/v10_model.py` | V6c + shared VAE + optional FiLM |
| `lam/scripts/run_v10.py` | V10 训练脚本 (修复 run_single.py 问题) |
| `lam/scripts/eval_v10.py` | 统一评估 (NMI + Conditional NMI + masked PSNR) |
| `lam/scripts/precompute_sam.py` | SAM 预计算 mask (Phase 3) |
| `reports/V10/v10_results.md` | 实验报告 |
