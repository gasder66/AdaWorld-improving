# AdaWorld 多主体隐动作模型 — 组会报告

**日期**: 2025-06-19  
**汇报人**: xiaojy

---

## 1. 研究目标

在 AdaWorld LAM 基础上实现**多主体逐主体隐动作建模**，纯无监督学习。

核心要求：
- 每个主体有独立的隐动作向量 z_k
- UMAP 聚类显示 z_k 按动作方向（up/down/left/right/stay）分离
- 不使用任何动作标签

---

## 2. 最终架构 (V6c Structured)

```
输入: videos (B,T,256,256,3) + masks (B,T,K,256,256)

1. Patchify (16×16) → SpatioTemporalTransformer (4 blocks, 256 dim)
   → 运动感知的 patch 特征 (B,T,256,256)

2. MaskedPool(GT masks) → 每主体特征 (含背景槽, K+1)
   Slot 0=背景, Slot 1..K=各主体

3. ObjectSpatioTemporalAttention → 主体间时空交互

4. Per-Object VAE (K+1 独立线性层) → z_k (B,T-1,K+1,32)
   ★ 每个 z_k 对应一个具体主体

5. Decoder: CrossAttn(patches, z) → 重建下一帧

损失函数 (纯无监督):
  L_total = L_recon + 2e-4·FreeBits_KL + 0.01·L_obj_recon
           + 0.01·L_mi + 0.01·L_temporal
```

**参数量**: 10.0M (model_dim=256, 4 enc blocks, 4 dec blocks)

---

## 3. 实验设置

| 参数 | 值 |
|------|----|
| model_dim | 256 |
| latent_dim | 32 |
| enc_blocks / dec_blocks | 4 / 4 |
| num_heads | 8 |
| max_actors | 4 |
| keep_background | True |
| kl_beta | 2e-4 |
| free_bits_lambda | 0.1 |
| obj_recon_weight | 0.01 |
| mi_weight / temporal_weight | 0.01 / 0.01 |
| lr | 2.5e-4 |
| optimizer | AdamW |

**合成数据集**:
- 4000 训练 / 500 验证
- 8×8 grid, 5 帧, 2-4 个主体
- 动作: stay/up/down/left/right (随机独立)
- 分辨率: 256×256

**A2D 数据集**:
- 3036 训练视频 / 746 测试视频
- 7 类: adult/baby/ball/bird/car/cat/dog
- 8 种动作: climbing/crawling/eating/flying/jumping/rolling/running/walking
- num_frames=2, 3276 训练样本
- YOLO: COCO 预训练 yolov8n (零样本, 无需 A2D 标注)

---

## 4. 实验结果

### 4.1 合成数据集 (PSNR + UMAP)

| 版本 | PSNR | KL | UMAP NMI avg | UMAP NMI best | 训练步数 |
|------|------|-----|---------------|---------------|---------|
| AdaWorld 原始 | 26.3 dB | 1.1 | 0.02 | 0.02 | 2000 |
| V5 (MaskedPool) | 25.8 dB | 2.5 | 0.26 | 0.48 | 2000 |
| V5 长训练 | 27.3 dB | — | — | — | 10000 |
| **V6c 长训练** | **27.4 dB** | **1.8** | **0.36** | **0.46** | 10000 |

Per-slot NMI (V6c): Slot0=0.28, Slot1=0.29, Slot2=0.34, **Slot3=0.46**

### 4.2 A2D 真实数据集 (PSNR)

| 训练配置 | 评估用 | PSNR | GT 依赖 |
|---------|-------|------|--------|
| GT masks | GT masks | 17.38 dB | 需要 A2D bbox 训练 YOLO |
| COCO YOLO (5k) | GT masks | 18.14 dB | 零样本 |
| COCO+ReID (5k) | GT masks | 18.20 dB | 零样本 |
| **COCO YOLO long (10k)** | GT masks | **18.29 dB** | 零样本 |

### 4.3 MOT 模块

| 数据集 | YOLO mAP50 | 检测率 | MOT PSNR | GT PSNR |
|--------|------------|--------|----------|---------|
| 合成 | 0.995 | 99.6% | 17.75 dB | 26.95 dB |
| A2D | 0.625 | 83.6% | 18.09 dB | 18.08 dB |

### 4.4 完全自监督流水线验证

```
A2D 视频 → COCO-YOLO (零样本) → BoT-SORT → bbox masks → V6c 训练
                                              ↑
                                    PSNR 18.09 ≈ GT PSNR 18.08 (差距 0.01 dB)
                                    ★ 根本不需要人工标注！
```

---

## 5. 可视化结果

| 文件 | 内容 |
|------|------|
| `result/v6_structured/vis/` | V6c 合成数据重建对比 (8 张: GT+Recon+Diff) |
| `result/v6_structured/umap/umap_per_slot_action.png` | UMAP 聚类——每个 Slot 单独按 5 种动作着色 |
| `result/v6_structured/umap/umap_by_slot.png` | UMAP 按 Slot 着色——验证 4 个 slot 在 z 空间分离 |
| `result/v6_structured/umap/umap_by_action.png` | UMAP 按动作着色——全局视角 |
| `result/v6_a2d/vis_compare/` | A2D GT vs MOT 重建对比 (8 张) |
| `result/v6_a2d/vis/` | A2D 重建详细分析 |

---

## 6. 关键发现

1. **DINOv2 不适合运动特征提取** — 语义 vs 运动不匹配，导致 KL 爆炸
2. **MaskedPool + ST encoder 端到端训练**才是正确组合
3. **每主体独立 VAE** 让 slot 0..K 分别对应主体 0..K
4. **背景槽设计**天然分离自我运动 (z_0) 与独立运动 (z_1..K)
5. **FreeBits + MI + Temporal 结构化约束**显著提升 UMAP (avg NMI 0.02→0.36)
6. **COCO YOLO 零样本方案超越 A2D 训练方案** — 18.29 vs 17.38 dB
7. **动作级别聚类需要连续运动数据** — L_delta 在随机动作下彻底失败 (NMI 0.36→0.006)

---

## 7. 技术教训

### 走了哪些弯路
- V4: 用 DINOv2 替换 ST encoder → 语义特征不编码运动, KL 爆炸, 放弃
- V8: L_delta loss 在随机动作数据的联合训练 → PSNR 崩溃, 放弃
- A2D stride=2 大窗口训练 → 无显著提升 (标注稀疏是瓶颈)

### 正确的选择
- MaskedPool (GT masks) + ST encoder → 主体级分离 ✅
- Per-Object VAE (每个 fc_k 独立) → z_k 绑定到主体 k ✅
- COCO YOLO 替代 GT masks → 零样本训练流水线 ✅

---

## 8. 下一步方向

| 优先级 | 方向 | 预期效果 |
|--------|------|---------|
| P0 | model_dim 256→512, 训练 20K 步 | PSNR 提升 |
| P1 | 合成数据连续动作 + L_delta | UMAP NMI > 0.5 |
| P2 | 合成预训练 → A2D 迁移 | 真实场景 PSNR 提升 |
| P3 | 大规模无标注视频验证 | 泛化能力测试 |
| P4 | hierarchy temporal encoder (长窗口 T=20) | 长程运动趋势建模 |

---

## 9. 代码结构

```
lam/lam/modules/
  lam.py       — V6c 模型 (242 行, 包含 6 个损失函数)
  blocks.py    — ST encoder/decoder/CrossAttn/ObjectReconHead/MaskedPool

lam/scripts/
  run_v4_dualstream.py  — V6c 训练脚本 (支持 synthetic/a2d/mot_a2d)
  eval_mot_pipeline.py  — MOT 评估
  eval_a2d_mot.py       — A2D MOT 评估
  analyze_latent_umap.py — UMAP 聚类分析

result/v6_structured/    — 合成数据模型 + 可视化
result/v6_a2d_coco/      — A2D COCO 训练模型
result/yolo_a2d/         — A2D YOLO 模型 (mAP 0.625)
result/yolo_synthetic/   — 合成数据 YOLO 模型 (mAP 0.995)
```

---

## 最快运行命令

```bash
# 合成数据训练
CUDA_VISIBLE_DEVICES=2 python lam/scripts/run_v4_dualstream.py \
    --gpu 0 --name exp --dataset synthetic --steps 5000

# A2D 零样本训练 (COCO YOLO masks)
CUDA_VISIBLE_DEVICES=2 python lam/scripts/run_v4_dualstream.py \
    --gpu 0 --name exp --dataset mot_a2d --yolo_source coco --steps 5000

# UMAP 分析
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python lam/scripts/analyze_latent_umap.py
```
