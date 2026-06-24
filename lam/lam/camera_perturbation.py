"""
V8 Stage 2A: Camera Perturbation (on-the-fly, vectorized).

对合成数据施加全局相机运动 (pan + zoom + brightness),
验证 V8 的 z_bg / z_actor 分解设计。

变换模型 (forward, 内容移动):
    x_out = s * x_in + (1 - s) * W/2 + tx
    y_out = s * y_in + (1 - s) * H/2 + ty

逆变换 (grid_sample 采样):
    theta = [[1/s, 0, -2*tx/(W*s)],
             [0, 1/s, -2*ty/(H*s)]]

每 transition 的相机参数 (L_bg 伪标签):
    g_t = [dx_t, dy_t, dscale_t, dbright_t]
"""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def apply_camera_perturbation(
    video: Tensor,
    boxes: Tensor,
    valid_mask: Tensor,
    pan_range: float = 5.0,
    zoom_range: float = 0.05,
    brightness_range: float = 0.05,
    seed: int = 0,
) -> Dict[str, Tensor]:
    """对 video + boxes 施加随机相机扰动 (全向量化, 无 Python 循环).

    Args:
        video: (T, H, W, 3) float32 [0,1]
        boxes: (T, K, 4) [x1,y1,x2,y2] pixel coords
        valid_mask: (T, K) bool
        pan_range: 每帧最大平移 (像素)
        zoom_range: 每帧最大缩放变化 (如 0.05 = ±5%)
        brightness_range: 每帧最大亮度变化
        seed: 随机种子

    Returns:
        dict:
            video: (T, H, W, 3) 扰动后
            boxes: (T, K, 4) 扰动后
            camera_params: (T-1, 4) [dx, dy, dscale, dbright] per transition
    """
    T, H, W, C = video.shape
    K = boxes.shape[1]
    rng = torch.Generator().manual_seed(seed)

    # 每帧随机相机参数
    dx = (torch.rand(T - 1, generator=rng) * 2 - 1) * pan_range
    dy = (torch.rand(T - 1, generator=rng) * 2 - 1) * pan_range
    dscale = (torch.rand(T - 1, generator=rng) * 2 - 1) * zoom_range
    dbright = (torch.rand(T - 1, generator=rng) * 2 - 1) * brightness_range

    # 累积变换 (frame 0 = 无扰动) — cumsum/cumprod 向量化
    cum_dx = torch.zeros(T, dtype=video.dtype)
    cum_dy = torch.zeros(T, dtype=video.dtype)
    cum_s = torch.ones(T, dtype=video.dtype)
    cum_b = torch.zeros(T, dtype=video.dtype)
    cum_dx[1:] = torch.cumsum(dx, dim=0)
    cum_dy[1:] = torch.cumsum(dy, dim=0)
    cum_s[1:] = torch.cumprod(1.0 + dscale, dim=0)
    cum_b[1:] = torch.cumsum(dbright, dim=0)

    # === 扰动 video (一次性 grid_sample 所有帧) ===
    video_chw = video.permute(0, 3, 1, 2)  # (T, 3, H, W)
    # theta: (T, 2, 3)
    theta = torch.zeros(T, 2, 3, dtype=video.dtype)
    inv_s = 1.0 / cum_s
    theta[:, 0, 0] = inv_s
    theta[:, 1, 1] = inv_s
    theta[:, 0, 2] = -2.0 * cum_dx / (W * cum_s)
    theta[:, 1, 2] = -2.0 * cum_dy / (H * cum_s)
    grid = F.affine_grid(theta, [T, C, H, W], align_corners=False)
    perturbed_video = F.grid_sample(
        video_chw, grid, align_corners=False, padding_mode="border",
    )  # (T, 3, H, W)
    # brightness (向量化)
    perturbed_video = perturbed_video + cum_b.view(T, 1, 1, 1)
    perturbed_video = perturbed_video.permute(0, 2, 3, 1).clamp(0.0, 1.0)  # (T, H, W, 3)

    # === 扰动 boxes (向量化) ===
    s = cum_s.view(T, 1)          # (T, 1)
    tx = cum_dx.view(T, 1)        # (T, 1)
    ty = cum_dy.view(T, 1)        # (T, 1)
    cx_offset = (1 - s) * W / 2   # (T, 1)
    cy_offset = (1 - s) * H / 2   # (T, 1)
    x1 = s * boxes[..., 0] + cx_offset + tx
    y1 = s * boxes[..., 1] + cy_offset + ty
    x2 = s * boxes[..., 2] + cx_offset + tx
    y2 = s * boxes[..., 3] + cy_offset + ty
    perturbed_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    perturbed_boxes[..., [0, 2]] = perturbed_boxes[..., [0, 2]].clamp(0, W)
    perturbed_boxes[..., [1, 3]] = perturbed_boxes[..., [1, 3]].clamp(0, H)

    # === camera params (L_bg 伪标签) ===
    camera_params = torch.stack([dx, dy, dscale, dbright], dim=-1)  # (T-1, 4)

    return {
        "video": perturbed_video,
        "boxes": perturbed_boxes,
        "camera_params": camera_params,
    }
