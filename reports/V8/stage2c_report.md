# V8 Stage 2C: Actor Conditioning — Report

## Summary

Stage 2C 在 SharedActorActionHead 中加入 FiLM actor type 条件化,
让 z_actor 编码 "相对于该 actor 类型的残差 action"。

**结论: 在当前 A2D 数据量下 (248 train samples),actor conditioning 效果有限。**

## Results

| Metric | No conditioning | **With conditioning** | Chance |
|---|---|---|---|
| Action NMI | 0.0673 | **0.0717** (+6.5%) | ~0 |
| Action Probe | 0.2222 | **0.2222** (same) | 0.125 |
| Actor type leakage | 0.1667 | **0.5556** | 0.143 |
| z_actor variance | 0.3533 | **0.2867** | — |
| z_bg variance | 0.3140 | **0.2165** | — |
| dbbox MSE | 926.10 | **855.65** | — |
| Params | 3,188,996 | **3,194,116** | — |

### 分析

1. **Action NMI 略提升 (0.067→0.072)**: Conditioning 帮助有限,
   在 248 样本的噪声范围内可能不显著。

2. **Actor type leakage 大幅增加 (0.17→0.56)**: FiLM conditioning 把
   actor type 信息注入了 z_actor。这本身不是坏事 (conditioning 的目的就是
   让 z_actor 知道 actor 类型),但说明 z_actor 现在部分编码了 "WHO" 而非
   纯粹的 "WHAT action"。

3. **dbbox MSE 略降 (926→856)**: Conditioning 帮助模型更好地预测 bbox 运动化,
   因为不同 actor 类型有不同的运动模式。

4. **Action Probe 不变 (0.2222)**: 线性可解码性没有提升,说明 conditioning
   没有让 action 信息在 z_actor 中更清晰。

### 为什么 conditioning 没有显著帮助

1. **数据量太少**: 248 train samples, 7 个 actor types, 8 个 actions。
   每个 (actor_type, action) 组合平均只有 ~4 个样本,FiLM 难以学到有意义的
   per-type 条件化参数。

2. **Actor type 与 action 的相关性弱**: A2D 中 actor type (adult/cat/ball/...)
   与 action (running/jumping/...) 的关联不像 "ball→rolling" 那样强。
   adult 可以 running/jumping/walking,cat 也可以。

3. **FiLM 在 VAE bottleneck 之前施加**: z_actor 仍然需要通过 KL 正则化,
   condition 信息可能被 free bits 吸收而非用于 action 分解。

## Architecture

### FiLM Conditioning

```python
# SharedActorActionHead with FiLM:
h = norm1(x)
h = gelu(fc1(h))
if conditioning:
    gamma, beta = film(actor_label)  # Embedding → (gamma, beta)
    h = h * (1 + gamma) + beta       # FiLM modulation
h = norm2(h)
mu, logvar = fc2(h)
```

- `num_actor_types=7` (A2D: adult/baby/ball/bird/car/cat/dog)
- `nn.Embedding(8, model_dim*2)` (+1 for unknown/padding)
- FiLM 在 fc1 (非线性) 之后、fc2 (VAE bottleneck) 之前施加
- Backward compatible: `num_actor_types=0` 时退化为无 conditioning

## Files

- `lam/lam/modules/slot_time_lam.py` — SharedActorActionHead + FiLM, model forward
- `lam/scripts/run_v8_a2d.py` — `--num_actor_types` 参数

## 下一步建议

Actor conditioning 在当前数据量下效果有限。更有价值的方向:

1. **增加训练数据**: A2D 仅 248 samples 是最大瓶颈。可考虑:
   - 数据增强 (bbox jitter, frame stride 变化)
   - 使用 ArbitraryA2DDataset (YOLO 检测,不受标注帧限制)
   - frame_stride=2 或 3 扩展样本数

2. **Stage 3 (MOT noise robustness)**: 测试 V8 对 bbox 噪声/ID switch 的鲁棒性

3. **对比 V6c on A2D**: 在相同 A2D 数据上评估 V6c,确认 V8 是否优于 V6c
