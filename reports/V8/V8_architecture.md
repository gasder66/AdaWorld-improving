我的结论先说清楚：

> **不要从 V7 继续，不要回到原版 AdaWorld；应该回到 V6c 的工程基座，但新开一个 V8/MOT-LAM 分支。**

理由是：V7 的方向已经被实验否定，三版 NMI 都接近 chance，远低于 V6c 的 0.36，而且 z_action 仍然严重泄漏 actor 信息。报告也明确建议“保留 V6c 作为生产基线，V7 代码归档，不再投入训练资源”。 原版 AdaWorld 可以作为思想基线，因为它的 latent action autoencoder 本质是用信息瓶颈从两帧中抽取转移信息，但它是单场景 latent action，不解决多主体 slot 和 track id 问题。 V6c 虽然有视觉特征纠缠问题，但它的代码、数据管线、MOT 流水线、评估脚本和背景槽经验都已经可用，并且合成数据 PSNR 27.4、UMAP NMI 0.36，是当前最可靠的实验基座。

---

# 一、项目重新定位

建议把新分支叫：

```text
V8: MOT-Guided Slot-Time Latent Action Model
```

不要叫 V7 continuation。

核心主张：

```text
YOLO + BoT-SORT 负责 actor identity 和 temporal tracking；
模型只从 frame difference 和 bbox motion 中学习 per-object latent action。
```

这和 V6c 的区别是：

```text
V6c:
video patch → ST encoder → masked pooling → object feature → VAE → z

V8:
MOT → tracked slot table → frame difference / bbox motion → slot-time transformer → shared latent action
```

V8 的目标不是先追求像素重建，而是先证明：

```text
1. z_actor 主要编码 object residual motion
2. z_bg 主要编码 global camera / illumination change
3. 所有 actor slot 共享一个 latent action space
4. 这个 latent space 比 V6c 更适合按 action 聚类
```

---

# 二、应该从哪里开始

## 结论：从 V6c 开始，而不是 V7 或 AdaWorld

### 不从 V7 继续

V7 的问题不是小 bug，而是结构性失败。它把 `MaskedPool` 得到的 actor-dominant `obj_feats` 输入 SharedEncoder，导致 actor 信息成为所有 factor head 的高速通道；报告里明确指出 z_action、z_actor、z_motion、z_bg 都会继承 actor 主导表征。 同时，GRL 没能把 actor 信息擦掉，原型正则也只能保证离散化，不能保证按 action 语义离散化。

所以 V7 只适合作为“失败对照”和“不要重蹈覆辙”的代码参考。

### 不回到原版 AdaWorld

AdaWorld 的核心是两帧输入、latent action encoder、VAE bottleneck、decoder 预测下一帧；它适合做单一 latent action 学习。 但你现在的问题是多主体、track id、background slot、actor/action 解耦。直接回 AdaWorld 会把已经解决的多主体结构全部丢掉。

### 回到 V6c 工程基座

V6c 至少已经证明了三件事：第一，ST encoder + MaskedPool + Per-Object VAE 能跑通；第二，背景槽和 Free Bits KL 有正向效果；第三，MOT 自监督流水线在 A2D 上可以工作，COCO YOLO 的零样本方案甚至在 A2D stride=23 上达到 18.29 dB，和 GT masks 基本没有差距。

但 V8 不应该保留 V6c 的 MaskedPool 主路径，而应该复用它的：

```text
数据集读取
MOT 结果加载
训练脚本框架
评估脚本
UMAP / NMI 评估
Free Bits KL 实现
背景槽经验
```

---

# 三、完整分步骤改进计划

## Phase 0：冻结基线，建立可比较实验表

目标：先保证之后每一步都能和 V6c / V7 对齐。

保留三个 baseline：

```text
B0: AdaWorld original
B1: V6c
B2: V7 v3 failure checkpoint
```

记录指标：

```text
合成数据:
  PSNR
  bbox motion MSE
  action NMI / ARI / Clustering Acc
  actor leakage
  background leakage

A2D:
  PSNR, if reconstruction retained
  action NMI by A2D label
  actor NMI
  slot validity statistics
  track continuity statistics
```

成功标准不是一开始超过 V6c，而是：

```text
V8-minimal 能稳定训练；
z_actor 不明显坍缩；
z_bg 有可验证功能；
motion MSE 低于简单常数预测 baseline。
```

---

## Phase 1：MOT Slot Table 构建

输入：

```text
video: (B, T, H, W, 3)
```

运行或加载：

```text
YOLO:
  bbox_{t,k}
  actor_label_k
  confidence_{t,k}

BoT-SORT:
  track_id_k
  valid_{t,k}
```

输出：

```text
actor_boxes:  (B, T, K, 4)
actor_labels: (B, K)
track_ids:    (B, K)
valid_mask:   (B, T, K)
```

然后加入背景槽：

```text
slot 0 = background
slot 1..K = tracked actors
```

得到：

```text
slot_table: (B, T, K+1)
```

这一阶段要做两个检查：

```text
track continuity:
  同一 track_id 是否跨帧稳定

slot coverage:
  actor valid ratio
  missing detection ratio
  ID switch ratio
```

如果这一步不稳定，后续 latent action 全都会被污染。

---

## Phase 2：Motion-only Slot Token Encoder

这是替代 V6c 的 `Patchify + ST Encoder + MaskedPool` 的部分。

对每个 actor slot：

```text
t = 0:
  X_{0,k} = crop(I_0, bbox_{0,k})

t > 0:
  ΔI_t = I_t - I_{t-1}
  X_{t,k} = crop(ΔI_t, union_bbox_{t-1,k,t,k})
```

同时构造几何输入：

```text
g_{t,k} = [x, y, w, h, dx, dy, dlogw, dlogh]
```

actor label 输入：

```text
e_actor,k = Embedding(actor_label_k)
```

背景槽输入：

```text
X_{t,0} = ΔI_t × background_mask_t
g_{t,0} = global statistics
```

用轻量 CNN + MLP 得到：

```text
h_{t,k} = CNN(X_{t,k}) + MLP(g_{t,k}) + label_condition(actor_label_k)
```

输出：

```text
motion_tokens: (B, T, K+1, D)
```

这一阶段的关键原则是：

```text
不要再提取 object visual appearance feature；
不要再对 patch feature 做 masked pooling；
尽量让输入只包含运动和几何。
```

---

## Phase 3：Shared Latent Action Transformer

这是你问的“之前第五部分现在具体怎么做”。

### Step 3.1：Transition Token Construction

latent action 应该表示转移，而不是状态。

输入：

```text
motion_tokens: H = (B, T, K+1, D)
```

构造：

```text
r_{t,k} = MLP([h_{t,k}, h_{t+1,k}, h_{t+1,k} - h_{t,k}])
```

输出：

```text
transition_tokens: R = (B, T-1, K+1, D)
```

这一步相当于 AdaWorld 中“从连续帧提取 latent action”的思想，但对象从整帧变成了 tracked slot。AdaWorld 中 latent action encoder 也是从连续帧中抽取转移信息，并通过紧凑 latent 迫使其捕获关键变化。

### Step 3.2：Temporal Attention

对每个 slot 沿时间做 attention：

```text
R[:, :, k, :] → Temporal Transformer
```

作用：

```text
识别持续运动、停止、转向、跳跃等时间模式
```

输出仍为：

```text
R_temporal: (B, T-1, K+1, D)
```

建议第一版：

```text
2 layers
4 heads
dim=128 or 256
```

不要一上来做很深。

### Step 3.3：Slot Attention / Interaction Attention

对同一时间步的 slot 做轻量 attention：

```text
R_temporal[:, t, :, :] → Slot Transformer
```

作用：

```text
actor 感知 background motion
actor 感知同帧其他主体
```

但注意不要太强，建议：

```text
1 layer only
```

否则又可能出现 V6c/V7 中的 token 污染。

输出：

```text
contextual_tokens: U = (B, T-1, K+1, D)
```

### Step 3.4：Shared Actor Action Head

对 actor slot 使用共享 head：

```text
z_{t,k} = SharedActorActionHead(U_{t,k}, actor_label_k), k > 0
```

这里最重要的是：

```text
所有 actor slot 共享一个 action latent space
```

不要用 V6c 那种每个 slot 一个独立 fc，也不要用 V7 那种 SharedEncoder + 多 factor head。可以用：

```text
shared MLP + actor-conditioned LayerNorm / FiLM
```

其中 actor label 是条件，不是要被编码进 z。

输出：

```text
z_actor: (B, T-1, K, d_z)
```

### Step 3.5：Background Motion Head

背景槽单独处理：

```text
z_bg,t = BackgroundHead(U_{t,0})
```

输出：

```text
z_bg: (B, T-1, d_bg)
```

最终：

```text
Z = [z_bg, z_actor]
shape: (B, T-1, K+1, d_z)
```

---

# 四、简化损失函数设计

我建议第一版只用三个损失：

```text
L_total = L_motion + β L_KL + γ L_bg
```

不要再加 prototype、GRL、MI、contrastive、feature reconstruction。V7 已经证明复杂损失会让问题不可解释。

## 1. Actor residual motion loss

预测 actor bbox 残差运动，而不是像素或 obj_feat delta。

观测运动：

```text
Δbbox_obs,t,k = bbox_{t+1,k} - bbox_{t,k}
```

背景预测公共运动：

```text
Δbbox_bg,t,k = CameraMotion(z_bg,t, bbox_{t,k})
```

actor residual：

```text
Δbbox_res,t,k = ActorMotion(z_actor,t,k, actor_label_k)
```

组合：

```text
Δbbox_pred,t,k = Δbbox_bg,t,k + Δbbox_res,t,k
```

损失：

```text
L_motion = MSE(Δbbox_pred, Δbbox_obs)
```

这个设计的好处是，背景槽不是一个装饰品，它必须解释所有 actor 共同受到的相机运动；actor latent 只需要解释扣除全局运动后的残差。

## 2. KL / Free Bits bottleneck

延续 V6c 的 Free Bits KL：

```text
L_KL = FreeBitsKL(q(z|r) || N(0,I))
```

V6c 中 Free Bits KL 是结构化隐空间约束的一部分，并且 V6c 相比 V5 在合成数据上取得更高 NMI 和 PSNR。

## 3. Background supervision loss

给 `z_bg` 一个明确职责：

```text
g_t = [dx_cam, dy_cam, scale_cam, Δbrightness]
```

预测：

```text
g_pred,t = BgHead(z_bg,t)
```

损失：

```text
L_bg = MSE(g_pred,t, g_t)
```

其中伪标签 `g_t` 可以由 background region 估计：

```text
dx_cam, dy_cam:
  phase correlation / ECC / background optical-flow median

Δbrightness:
  non-actor region mean RGB difference
```

---

# 五、背景槽的验证方案

## 验证 A：合成全局扰动实验

在合成数据里加入三类扰动：

```text
camera pan
camera zoom
global brightness change
```

比较：

```text
V8-no-bg
V8-with-bg
```

指标：

```text
actor action NMI
bbox motion MSE
z_actor → camera motion linear probe R²
z_bg → camera motion linear probe R²
```

预期：

```text
with-bg:
  actor action NMI 更高
  bbox motion MSE 更低
  z_bg camera probe 高
  z_actor camera probe 低
```

这是最重要的背景槽验证。

## 验证 B：zero-out ablation

推理时分别置零：

```text
z_bg = 0
z_actor = 0
```

预期：

```text
z_bg = 0:
  全局相机/亮度变化预测显著变差
  actor 相对运动仍部分保留

z_actor = 0:
  全局运动仍保留
  actor 自身残差运动消失
```

这能直接说明两个 latent 的职责不同。

## 验证 C：background swap

取两个视频：

```text
A: actor action
B: camera / lighting change
```

组合：

```text
z_actor from A
z_bg from B
```

预期：

```text
actor 运动接近 A
global motion / brightness 接近 B
```

这个实验适合做可视化图，也适合在组会上解释“背景槽不是摆设”。

## 验证 D：真实 A2D 统计验证

A2D 没有真实 camera motion 标签，但可以用伪标签：

```text
global flow median
background RGB mean shift
```

做 linear probe：

```text
z_bg → global flow / brightness
z_actor → global flow / brightness
```

如果 `z_bg` 的 R² 明显高于 `z_actor`，就能说明背景槽在真实视频上也吸收了全局变化。

---

# 六、分阶段实验路线

## Stage 1：Synthetic-Motion Minimal

数据：

```text
合成多 actor 数据
5 actions: stay/up/down/left/right
4 actor categories
无相机扰动
```

模型：

```text
MOT slot table 用 GT bbox / GT id
frame difference token
shared actor action head
no background loss
```

目标：

```text
证明新架构基本可训练
```

成功标准：

```text
bbox motion MSE < constant baseline
action NMI ≥ V6c 的 0.36 或至少接近 0.30
actor leakage 不高于 V6c
```

## Stage 2：Synthetic + Camera / Lighting

加入：

```text
global pan
zoom
brightness shift
```

模型比较：

```text
without bg slot
with bg slot
```

成功标准：

```text
with bg 的 actor NMI 不下降或上升
with bg 的 z_actor camera leakage 明显低
z_bg camera probe R² 明显高
```

## Stage 3：MOT Noise Robustness

把 GT bbox/id 换成模拟噪声：

```text
bbox jitter
miss detection
ID switch
false positive
```

目标：

```text
测试对 YOLO / BoT-SORT 错误的鲁棒性
```

成功标准：

```text
轻度噪声下 NMI 不崩
valid mask 机制能处理 missing slot
```

## Stage 4：A2D Zero-shot MOT

使用已有 A2D MOT pipeline：

```text
A2D video
→ YOLOv8n
→ BoT-SORT
→ bbox / actor label / track id
```

训练 V8。

评估：

```text
A2D action NMI
actor leakage
background probe
UMAP
不同 stride 下的表现
```

V6c 报告显示 A2D 上 COCO YOLO 零样本流水线已经可行，MOT masks 与 GT masks 的 PSNR 只有 0.01 dB 差距。 所以这一步的重点不是重新证明 MOT 可用，而是证明新的 motion-only slot token 是否比 MaskedPool 更适合 action clustering。

## Stage 5：和 V6c/V7/AdaWorld 对比

最终表格建议：

| Model    | Input      | Factorization    | Loss         | Synthetic NMI |  A2D NMI | Actor leakage | BG probe |
| -------- | ---------- | ---------------- | ------------ | ------------: | -------: | ------------: | -------: |
| AdaWorld | full frame | no               | recon+KL     |           low |      low |          high |       no |
| V6c      | patch+mask | per-slot         | recon+KL+obj |          0.36 |    ~0.10 |        medium |     weak |
| V7       | obj_feat   | explicit factors | many losses  |        ~0.005 |        — |          high |   failed |
| V8       | MOT+diff   | slot-time        | motion+KL+bg |      target ↑ | target ↑ |      target ↓ | target ↑ |

---

# 七、代码实施计划

## Week 1：数据接口

新建：

```text
lam/lam/mot_slot_dataset.py
```

输出：

```python
{
    "video":        (T,H,W,3),
    "boxes":        (T,K,4),
    "actor_labels": (K,),
    "track_ids":    (K,),
    "valid_mask":   (T,K),
    "bg_mask":      (T,H,W),
}
```

复用：

```text
mot_a2d_dataset.py
arbitrary_a2d_dataset.py
```

## Week 2：Motion Token Encoder

新建：

```text
lam/lam/modules/motion_token_encoder.py
```

包含：

```text
FrameDifferenceCropper
BoxGeometryEncoder
ActorLabelCondition
BackgroundTokenEncoder
```

输出：

```text
motion_tokens: (B,T,K+1,D)
```

## Week 3：Slot-Time Latent Action Transformer

新建：

```text
lam/lam/modules/slot_time_lam.py
```

包含：

```text
TransitionTokenBuilder
TemporalTransformer
SlotTransformer
SharedActorActionHead
BackgroundMotionHead
```

不要复用 V7 的 FactorizedBottleneck。

可以复用 V6c 的 Free Bits KL 代码。

## Week 4：训练脚本

新建：

```text
lam/scripts/run_v8_mot_lam.py
```

第一版只保留：

```text
L_motion
L_KL
L_bg
```

日志必须记录：

```text
motion_mse
kl
bg_loss
z_var
z_norm
valid_actor_ratio
```

## Week 5：评估脚本

扩展或新建：

```text
lam/scripts/eval_v8_action_cluster.py
lam/scripts/eval_background_slot.py
```

指标：

```text
NMI / ARI / ACC
actor leakage
camera leakage
background R²
zero-out ablation
UMAP
```

---

# 八、是否要保留 Decoder

第一版不要保留像素 Decoder。

你们当前目标是 action clustering 和解释性，不是 world model rollout。V7 已经说明，重建或 feature delta 会把模型拉回“像素运动”而不是“动作语义”。

如果师兄希望仍然保留和 AdaWorld 的联系，可以把 decoder 放成第二阶段：

```text
Stage A:
  learn V8 latent actions from motion / bbox

Stage B:
  optional lightweight decoder:
    previous frame + z_bg + z_actor → next bbox / coarse frame
```

也就是说，Decoder 是验证 latent 是否有用的辅助，不是主训练目标。

---

# 九、最终推荐路线

最稳的路线是：

```text
从 V6c 工程基座出发
    ↓
删掉 Patchify-ST-MaskedPool 主路径
    ↓
接入 MOT slot table
    ↓
用 frame difference + bbox geometry 构造 motion tokens
    ↓
实现 Slot-Time Latent Action Transformer
    ↓
只用 L_motion + KL + L_bg
    ↓
先在合成数据验证背景槽和 action NMI
    ↓
再迁移到 A2D
```

一句话版本：

> **V8 应该是“V6c 工程基座 + MOT-guided motion-only 重构”，而不是“V7 继续修补”，也不是“回到原版 AdaWorld”。**

这样做的好处是：你保留了 V6c 已经证明有效的训练和评估基础，避开了 V7 的复杂因子化失败，同时吸收 AdaWorld 的信息瓶颈思想和 FLAM 的“多主体共享 latent action space”思想。
