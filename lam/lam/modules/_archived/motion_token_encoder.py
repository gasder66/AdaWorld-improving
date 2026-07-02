"""
V8 Motion Token Encoder.

替代 V6c 的 Patchify + ST Encoder + MaskedPool 主路径。
核心原则: 只编码运动和几何, 不编码外观。

子模块:
  1. FrameDifferenceCropper: t=0 用 RGB crop, t>0 用帧差 crop
  2. BoxGeometryEncoder: bbox [x,y,w,h,dx,dy,dw,dh] -> geom token
  3. BackgroundTokenEncoder: 全图帧差 -> bg token
  4. MotionTokenEncoder: 组合输出 (B, T, K+1, D)

输出 motion_tokens (B, T, K+1, D):
  slot 0 = background
  slot 1..K = actor slots
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _crop_resize(video: Tensor, boxes: Tensor, crop_size: int = 32) -> Tensor:
    """从 video 中按 boxes 裁剪并 resize 到 crop_size.

    Args:
        video: (B, T, H, W, 3)
        boxes: (B, T, K, 4) [x1,y1,x2,y2]
        crop_size: 输出大小
    Returns:
        crops: (B, T, K, 3, crop_size, crop_size)
    """
    B, T, H, W, C = video.shape
    K = boxes.shape[2]

    # 展开到 (B*T*K, 4)
    boxes_flat = boxes.reshape(B * T * K, 4)
    # video: (B, T, H, W, 3) -> (B, T, 1, H, W, 3) -> 按 K 广播
    # 用 grid_sample 实现 batch crop
    video_flat = video.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B*T, 3, H, W)
    video_exp = video_flat.unsqueeze(1).expand(-1, K, -1, -1, -1)  # (B*T, K, 3, H, W)
    video_exp = video_exp.reshape(B * T * K, C, H, W)

    # 归一化 boxes 到 [0, 1] 用于 grid_sample
    x1 = boxes_flat[:, 0] / W
    y1 = boxes_flat[:, 1] / H
    x2 = boxes_flat[:, 2] / W
    y2 = boxes_flat[:, 3] / H
    # grid_sample 需要归一化坐标 [-1, 1]
    gx = (x1 + x2) / 2 * 2 - 1  # center x
    gy = (y1 + y2) / 2 * 2 - 1  # center y
    dx = (x2 - x1) / 2 * 2  # half width (normalized)
    dy = (y2 - y1) / 2 * 2

    # 构建 affine grid: 6 个参数 [scale_x, 0, tx, 0, scale_y, ty]
    theta = torch.zeros(B * T * K, 2, 3, device=video.device, dtype=video.dtype)
    theta[:, 0, 0] = dx   # scale: bbox half-width in normalized space
    theta[:, 0, 2] = gx   # translation: bbox center in normalized space
    theta[:, 1, 1] = dy
    theta[:, 1, 2] = gy

    grid = F.affine_grid(theta, [B * T * K, C, crop_size, crop_size], align_corners=False)
    crops = F.grid_sample(video_exp, grid, align_corners=False, padding_mode="zeros")
    crops = crops.reshape(B, T, K, C, crop_size, crop_size)
    return crops


class FrameDifferenceCropper(nn.Module):
    """对每个 actor slot 裁剪帧差区域.

    t=0: crop(I_0, bbox_0) — 首帧 RGB (作为参考)
    t>0: crop(I_t - I_{t-1}, union(bbox_{t-1}, bbox_t)) — 帧差

    输出: actor_crops (B, T, K, 3, crop_size, crop_size)
    """

    def __init__(self, crop_size: int = 32) -> None:
        super().__init__()
        self.crop_size = crop_size

    def forward(self, video: Tensor, boxes: Tensor) -> Tensor:
        """
        Args:
            video: (B, T, H, W, 3) float32 [0,1]
            boxes: (B, T, K, 4) [x1,y1,x2,y2]
        Returns:
            crops: (B, T, K, 3, crop_size, crop_size)
        """
        B, T, H, W, C = video.shape
        K = boxes.shape[2]
        cs = self.crop_size

        # t=0: 直接 crop 首帧
        crops_0 = _crop_resize(video[:, :1], boxes[:, :1], cs)  # (B,1,K,3,cs,cs)

        if T == 1:
            return crops_0

        # t>0: 帧差 + union bbox
        frame_diff = video[:, 1:] - video[:, :-1]  # (B, T-1, H, W, 3)
        # union bbox: 取 min(x1,y1), max(x2,y2) of t-1 and t
        x1 = torch.min(boxes[:, :-1, ..., 0], boxes[:, 1:, ..., 0])
        y1 = torch.min(boxes[:, :-1, ..., 1], boxes[:, 1:, ..., 1])
        x2 = torch.max(boxes[:, :-1, ..., 2], boxes[:, 1:, ..., 2])
        y2 = torch.max(boxes[:, :-1, ..., 3], boxes[:, 1:, ..., 3])
        union_boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, T-1, K, 4)

        crops_rest = _crop_resize(frame_diff, union_boxes, cs)  # (B, T-1, K, 3, cs, cs)

        return torch.cat([crops_0, crops_rest], dim=1)  # (B, T, K, 3, cs, cs)


class ActorCropEncoder(nn.Module):
    """CNN: actor crops (3, cs, cs) -> token (D,)."""

    def __init__(self, in_channels: int = 3, model_dim: int = 256, crop_size: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),  # cs/2
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # cs/4
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # cs/8
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(128),
            nn.Linear(128, model_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., 3, cs, cs) -> (..., D)."""
        return self.net(x)


class BoxGeometryEncoder(nn.Module):
    """bbox 几何 [x,y,w,h,dx,dy,dw,dh] -> token (D,).

    归一化到 [0,1] (相对图像 256x256)。
    """

    def __init__(self, model_dim: int = 256, img_size: int = 256) -> None:
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, model_dim),
        )

    def forward(self, boxes: Tensor) -> Tensor:
        """
        Args:
            boxes: (B, T, K, 4) [x1,y1,x2,y2] pixel
        Returns:
            geom_tokens: (B, T, K, D)
        """
        B, T, K, _ = boxes.shape
        S = self.img_size

        x1 = boxes[..., 0] / S
        y1 = boxes[..., 1] / S
        x2 = boxes[..., 2] / S
        y2 = boxes[..., 3] / S
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1)
        h = (y2 - y1)

        # 帧间 delta (t=0 用 0)
        if T > 1:
            dcx = torch.zeros_like(cx)
            dcx[:, 1:] = cx[:, 1:] - cx[:, :-1]
            dcy = torch.zeros_like(cy)
            dcy[:, 1:] = cy[:, 1:] - cy[:, :-1]
            dw = torch.zeros_like(w)
            dw[:, 1:] = w[:, 1:] - w[:, :-1]
            dh = torch.zeros_like(h)
            dh[:, 1:] = h[:, 1:] - h[:, :-1]
        else:
            dcx = dcy = dw = dh = torch.zeros_like(cx)

        geom = torch.stack([cx, cy, w, h, dcx, dcy, dw, dh], dim=-1)  # (B,T,K,8)
        return self.net(geom)


class BackgroundTokenEncoder(nn.Module):
    """全图帧差 -> bg token.

    t=0: 首帧 RGB downsample
    t>0: (I_t - I_{t-1}) downsample
    """

    def __init__(self, model_dim: int = 256, bg_size: int = 32) -> None:
        super().__init__()
        self.bg_size = bg_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(128),
            nn.Linear(128, model_dim),
        )

    def forward(self, video: Tensor) -> Tensor:
        """
        Args:
            video: (B, T, H, W, 3)
        Returns:
            bg_tokens: (B, T, D)
        """
        B, T, H, W, C = video.shape

        # t=0: 首帧
        v0 = video[:, :1].reshape(B, C, H, W)
        v0_ds = F.interpolate(v0, size=(self.bg_size, self.bg_size), mode="bilinear", align_corners=False)
        token_0 = self.net(v0_ds).unsqueeze(1)  # (B, 1, D)

        if T == 1:
            return token_0

        # t>0: 帧差
        diff = (video[:, 1:] - video[:, :-1]).reshape(B * (T - 1), C, H, W)
        diff_ds = F.interpolate(diff, size=(self.bg_size, self.bg_size), mode="bilinear", align_corners=False)
        token_rest = self.net(diff_ds).reshape(B, T - 1, -1)  # (B, T-1, D)

        return torch.cat([token_0, token_rest], dim=1)  # (B, T, D)


class MotionTokenEncoder(nn.Module):
    """V8 Motion Token Encoder.

    组合 actor crops + geometry + background -> motion_tokens (B, T, K+1, D).
    slot 0 = background
    slot 1..K = actor slots

    关键: t>0 时 actor crop 用帧差 (不含静态外观), 只编码运动。
    """

    def __init__(
        self,
        model_dim: int = 256,
        crop_size: int = 32,
        img_size: int = 256,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.crop_size = crop_size

        self.crop_diff = FrameDifferenceCropper(crop_size=crop_size)
        self.actor_crop_enc = ActorCropEncoder(
            in_channels=3, model_dim=model_dim, crop_size=crop_size,
        )
        self.geom_enc = BoxGeometryEncoder(model_dim=model_dim, img_size=img_size)
        self.bg_enc = BackgroundTokenEncoder(model_dim=model_dim, bg_size=crop_size)

        # 融合: crop + geom -> actor token
        self.fuse_norm = nn.LayerNorm(model_dim * 2)
        self.fuse_proj = nn.Linear(model_dim * 2, model_dim)

    def forward(self, video: Tensor, boxes: Tensor, valid_mask: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            video: (B, T, H, W, 3) float32 [0,1]
            boxes: (B, T, K, 4) [x1,y1,x2,y2]
            valid_mask: (B, T, K) bool, True = valid actor
        Returns:
            motion_tokens: (B, T, K+1, D)  slot 0=bg, 1..K=actors
            actor_tokens: (B, T, K, D)  (for debug/linear probe)
            bg_tokens: (B, T, D)
        """
        B, T, K, _ = boxes.shape
        D = self.model_dim

        # 1. Actor crops (帧差)
        crops = self.crop_diff(video, boxes)  # (B, T, K, 3, cs, cs)
        crop_tokens = self.actor_crop_enc(crops.reshape(B * T * K, 3, self.crop_size, self.crop_size))
        crop_tokens = crop_tokens.reshape(B, T, K, D)

        # 2. Geometry
        geom_tokens = self.geom_enc(boxes)  # (B, T, K, D)

        # 3. 融合 crop + geom
        actor_tokens = self.fuse_proj(
            self.fuse_norm(torch.cat([crop_tokens, geom_tokens], dim=-1))
        )  # (B, T, K, D)

        # 4. Background
        bg_tokens = self.bg_enc(video)  # (B, T, D)

        # 5. 拼接: slot 0=bg, 1..K=actors
        motion_tokens = torch.cat([
            bg_tokens.unsqueeze(2),  # (B, T, 1, D)
            actor_tokens,  # (B, T, K, D)
        ], dim=2)  # (B, T, K+1, D)

        return {
            "motion_tokens": motion_tokens,
            "actor_tokens": actor_tokens,
            "bg_tokens": bg_tokens,
        }
