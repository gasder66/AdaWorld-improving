# V10 计划: V6c 线 — Object-Level Latent Action (重建 + 聚类)

## 定位

**这是 V6c 线的延续。** V6c 有像素重建 (PSNR 27 dB) 但聚类失败 (NMI 0.05)。
V8/V9 走了另一条路 (motion-only, 无重建), 是独立的探索线。

V10 的目标: 在 V6c 的架构上修复聚类, 同时保留重建 → 完整的 object-level latent action model。

## V6c 线 vs V9 线 (明确区分)

| | V6c → V10 (本线) | V8 → V9 (并行线) |
|---|---|---|
| **编码器输入** | 完整 RGB patch (含外观) | 帧差 + 几何 (不含外观) |
| **z 内容** | 外观 + 运动 | 仅运动 |
| **重建** | CrossAttn decoder, 全帧 256×256 | FiLM decoder, crop 32×32 |
| **聚类机制** | per-slot VAE 失败 → **shared VAE 修复** | shared head 天然成功 |
| **leakage** | 高 (外观在 z 中), 需重新解读 | 低 (无外观), 但重建弱 |
| **核心贡献** | 重建 + 聚类同时实现 | motion-only 聚类, 重建是诊断 |
| **论文定位** | 完整 world model (能预测下一帧) | 动作理解 (能聚类, 不追求重建) |

## V6c 失败根因 (代码确认)

**唯一根因: per-slot VAE** (`lam.py:80-82`):
```python
self.fcs = nn.ModuleList([nn.Linear(model_dim, latent_dim*2) for _ in range(K)])
```
5 个独立 Linear → 每个 slot 的 z 在不同子空间 → 混合聚类无意义 (NMI=0.05)。
但 per-slot NMI=0.39 (slot 内部可以聚类)。

**decoder 不是问题**: CrossAttention + SpatioTransformer 已经是 slot-agnostic 的
(Q=patches, KV=所有 slot 共享的 z)。换 shared VAE 后 decoder 无需修改。

## Phase 1: V10-A — Shared VAE

### 代码改动

**新建 `lam/lam/modules/v10_model.py`**:
- 继承 `LatentActionModel` (V6c)
- `self.fcs` (5 个 Linear) → `self.fc` (1 个 shared Linear)
- 重写 `encode()`: 所有 slot 共享一个 VAE head
- 可选 FiLM: actor type → (gamma, beta) 条件化 z
- 保留 decoder 完全不变

**新建 `lam/scripts/run_v10.py`**:
- 修复 `run_single.py` 的不兼容 kwargs
- 使用 model 内部 `outputs["kl_loss"]` (Free Bits KL, λ=0.1)
- Loss = `recon_loss + β·kl + λ·obj_recon + μ·delta + ν·contrast`
- 合成数据: 5000 步, batch=16, lr=2.5e-4 (与 V6c 一致)

**新建 `lam/scripts/eval_v10.py`**:
- Overall NMI / Per-Slot NMI / Actor Leakage
- **Actor-masked PSNR**: 只在 mask 区域计算 (不被背景主导)
- **Copy baseline**: `PSNR(crop_t, crop_{t+1})` 作为参考
- UMAP 可视化

### 预期结果
- NMI: 0.05 → >0.20 (shared VAE)
- PSNR: 保持 ~27 dB (decoder 不变)
- Leakage: 可能仍高, 但用 Conditional NMI 重新解读

## Phase 2: 评估指标改进

### Conditional NMI
给定 actor 类型后, z 还能提供多少 action 信息:
```python
for actor_type in unique_types:
    idx = actor_types == actor_type
    kmeans = KMeans(n_clusters=n_actions).fit(z[idx])
    nmi_cond = NMI(actions[idx], kmeans.labels_)
```
- `NMI_conditional > NMI_overall` → z 编码了超越类型的 action 信息
- `NMI_conditional ≈ NMI_overall` → z 只编码了类型

### Actor-masked PSNR
只在 actor mask 内计算 (DiskSyntheticDataset 已返回 masks):
```python
mse = ((recon * mask - gt * mask) ** 2).sum() / (mask.sum() * 3 + 1e-8)
```

## Phase 3: 更精细的特征提取

### SAM 分割替代 bbox
- SAM 预计算每个 actor 的精确 mask
- V6c 的 MaskedPool 已支持 mask → 改动最小
- mask 比 bbox 精确, 捕获非刚性形变 (人的姿态)

### 光流 (备选)
- RAFT 在 actor region 内计算密集光流
- 光流比帧差包含更多信息 (方向, 速度, 形变模式)

## Phase 4: 真实数据 (A2D)

1. SAM 预计算 A2D mask
2. 训练 V10 + SAM mask
3. 评估 Conditional NMI + masked PSNR

## 实施优先级

| 优先级 | 任务 | 工时 |
|---|---|---|
| P0 | V10-A: shared VAE + 训练 + 评估 | 4h |
| P0 | Actor-masked PSNR + copy baseline | 1h |
| P1 | Conditional NMI | 2h |
| P2 | SAM 预计算 mask | 3h |
| P2 | V10 + SAM mask | 2h |
| P3 | 光流特征 | 4h |
