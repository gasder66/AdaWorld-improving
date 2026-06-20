# AdaWorld 组会最终报告 (2025-06-19)

## 实验结果总表

### 合成数据 (4000 样本, 5 帧, 几何动作)

| 版本 | 改动 | PSNR | UMAP NMI avg | UMAP NMI best |
|------|------|------|-------------|--------------|
| AdaWorld 原始 | learnable tokens | 26.3 dB | 0.02 | 0.02 |
| V5 | MaskedPool + Per-Obj VAE | 27.3 dB | 0.26 | 0.48 |
| V6c | +背景槽+FreeBits+MI+Temporal | **27.4 dB** | **0.36** | 0.46 |

**结论**: 架构验证通过。MaskedPool + ST encoder + Per-Object VAE 能学习逐主体隐动作。

### A2D (真实视频, 8 类抽象动作)

| 实验 | T | 帧间隔 | mask 来源 | PSNR | 备注 |
|------|---|--------|----------|------|------|
| GT掩码 | 2 | 23帧 | GT bbox | 18.08 dB | 需要 A2D 标注 |
| COCO YOLO | 2 | 23帧 | 零样本 YOLO | 18.29 dB | **无任何 A2D 标注** |
| E1 (最佳) | 2 | **30帧(1.25s)** | YOLO | **21.16 dB** | 时序扩增 +3.1 dB |
| E2 | 5 | 10帧(0.4s) | YOLO | 20.29 dB | — |
| E3 | 10 | 5帧(0.2s) | YOLO | 20.59 dB | — |

**结论**: 时序窗口扩大显著提升重建质量 (+3.1 dB)，但仍无法区分 A2D 的 8 类抽象动作。

## 关键发现

1. **DINOv2 不适合运动特征提取** → 放弃
2. **MaskedPool + ST encoder 端到端训练** → 正确组合
3. **每主体独立 VAE** → z_k 成功绑定主体 k
4. **背景槽** → 分离自我运动与独立运动
5. **COCO YOLO 零样本** → 完全自监督流水线 (18.29 dB ≈ GT 18.08 dB)
6. **扩大帧间隔** → PSNR 从 18 → 21 dB (但 UMAP 聚类不提升)
7. **当前极限**: 对抽象动作 (jump vs run vs walk) 无法学习语义级区分

## 可视化路径

```
合成数据:
  result/v6_structured/vis/              — 重建对比图
  result/v6_structured/umap/umap_per_slot_action.png  — UMAP 按动作聚类 (核心图)

A2D:
  result/v6_a2d_coco/analysis/            — UMAP + 重建
  result/v6_a2d/vis_compare/              — GT vs MOT 重建对比
```

## 全部模型路径

```
result/v6_structured/model_v6c_long.pt          — 合成数据最佳 (27.4 dB)
result/v6_a2d_coco/model_v6c_coco_long.pt        — A2D COCO YOLO (18.3 dB)
result/experiments/model_E1_stride30.pt           — A2D stride=30 (21.2 dB)
result/experiments/model_E2_T5_S10.pt             — A2D T=5 (20.3 dB)
result/experiments/model_E3_T10_S5.pt             — A2D T=10 (20.6 dB)
```

## 代码结构

```
lam/lam/modules/lam.py            — V6c 模型 (242 行)
lam/lam/modules/blocks.py         — ST encoder/decoder/MaskedPool
lam/scripts/run_v4_dualstream.py  — 统一训练脚本
lam/scripts/run_exp.py            — 快速实验脚本
lam/lam/a2d_dataset.py            — A2D 数据集 (含 frame_stride)
lam/lam/mot_a2d_dataset.py        — MOT 掩码数据集
lam/lam/experiment_dataset.py     — 预生成数据集
