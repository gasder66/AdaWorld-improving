# AdaWorld 组会最终报告 v2 (2025-06-19)

## Phase 1: Stride 扫描 (T=2)

| stride | 帧间隔 | 时间 | PSNR |
|--------|-------|------|------|
| 5 | 5 帧 | 0.2s | **22.73 dB** |
| 10 | 10 帧 | 0.4s | 21.71 dB |
| 20 | 20 帧 | 0.8s | 21.26 dB |
| 30 | 30 帧 | 1.25s | 21.16 dB |
| 40 | 40 帧 | 1.7s | 21.14 dB |
| 50 | 50 帧 | 2.1s | 20.75 dB |
| 60 | 60 帧 | 2.5s | 20.70 dB |

**结论：stride 越小、PSNR 越高。** VAE 32 维瓶颈在 stride 大时无法编码更多信息。

## Phase 2: T 扫描 (stride=5)

| T | 时间步数 | VAE 每步容量 | PSNR |
|---|---------|-------------|------|
| 2 | 1 步 | 32 维 | **22.73 dB** |
| 3 | 2 步 | 16 维/步 | 21.08 dB |
| 5 | 4 步 | 8 维/步 | 21.07 dB |

**结论：T=2 最优。** 更多时间步不增加信息，只分摊 VAE 容量。

## 核心结论

| 发现 | 修正前假设 | 实际 |
|------|----------|------|
| stride 影响 | 越大→信号越强→更好 | **越小→差异越小→更容易重建→PSNR 更高** |
| T 影响 | 越多帧→时序上下文越多→更好 | **T=2 最优→更多帧分摊 VAE 容量→每步精度下降** |
| VAE 瓶颈紧度 | "可能不够紧" | **紧的！stride 从 5→60 时，PSNR 从 22.7→20.7 (↓2 dB)** |

## 最优配置

```
A2D 自监督 MOT 训练:
  T=2, stride=5 (0.2s 间隔)
  COCO YOLO + BoT-SORT
  V6c 损失: L_recon + KL + L_obj_recon

PSNR: 22.73 dB (200 视频)
```

## 可视化与模型

```
模型文件:
  result/experiments/model_stride5.pt    — 最优模型 (22.73 dB)
  result/experiments/model_stride10.pt   — stride=10 (21.71 dB)
  result/experiments/model_stride20.pt   — stride=20 (21.26 dB)

对比基线:
  result/v6_a2d_coco/model_v6c_coco_long.pt  — 原始自监督 (18.29 dB)
  result/experiments/model_stride5.pt         — 优化后 (22.73 dB)
```

## 代码结构 (快速命令)

```bash
# 生成 stride 数据集
python lam/scripts/gen_stride.py 5

# 训练
python lam/scripts/run_exp.py --name myexp \
    --data_path result/v6_a2d/experiments/exp_stride5.pt \
    --steps 3000 --batch_size 16
```
