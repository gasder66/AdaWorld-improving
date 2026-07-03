"""
V12 Content Path: ObjectContentEncoder.

Sees RGB (first frame only) and produces appearance tokens.
c_obj / c_bg NEVER enter IDM / FDM.

Input:  video[:, 0]  (B, H, W, C), masks[:, 0] (B, K, H, W),
        bboxes[:, 0] (B, K, 4) pixel xyxy, valid_mask[:, 0] (B, K)
Output: c_obj (B, K, D_c), c_bg (B, D_c)
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def crop_resize(images: Tensor, boxes: Tensor, size: int, H: int, W: int) -> Tensor:
    """Differentiable crop+resize via affine grid_sample.

    Args:
        images: (B, C, H, W)
        boxes:  (B, K, 4) pixel xyxy
        size:   output crop side length
        H, W:   input image spatial size
    Returns:
        crops: (B, K, C, size, size)
    """
    B, K, _ = boxes.shape
    C = images.shape[1]
    x1, y1, x2, y2 = boxes.unbind(-1)          # (B, K)
    # Normalize box edges to [-1, 1] grid_sample convention.
    nx1 = 2.0 * x1 / W - 1.0
    nx2 = 2.0 * x2 / W - 1.0
    ny1 = 2.0 * y1 / H - 1.0
    ny2 = 2.0 * y2 / H - 1.0
    ncx = (nx1 + nx2) * 0.5
    ncy = (ny1 + ny2) * 0.5
    nw = (nx2 - nx1) * 0.5
    nh = (ny2 - ny1) * 0.5
    zero = torch.zeros_like(nw)
    theta = torch.stack(
        [
            torch.stack([nw, zero, ncx], -1),
            torch.stack([zero, nh, ncy], -1),
        ],
        -2,
    )                                          # (B, K, 2, 3)
    theta = theta.reshape(B * K, 2, 3)
    grid = F.affine_grid(theta, (B * K, C, size, size), align_corners=False)
    imgs_rep = images.unsqueeze(1).expand(B, K, C, H, W).reshape(B * K, C, H, W)
    crops = F.grid_sample(imgs_rep, grid, align_corners=False)
    return crops.reshape(B, K, C, size, size)


class _SmallCNN(nn.Module):
    def __init__(self, in_ch: int, dim: int, channels=(32, 64, 128)):
        super().__init__()
        layers = []
        prev = in_ch
        for ch in channels:
            layers += [
                nn.Conv2d(prev, ch, 3, stride=2, padding=1),
                nn.GroupNorm(8, ch),
                nn.GELU(),
            ]
            prev = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.norm = nn.LayerNorm(prev * 2 * 2)
        self.proj = nn.Linear(prev * 2 * 2, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.norm(x)
        return self.proj(x)


class ObjectContentEncoder(nn.Module):
    """First-frame RGB crops -> appearance tokens. No RGB enters IDM/FDM."""

    def __init__(
        self,
        crop_size: int = 64,
        content_dim: int = 128,
        channels=(32, 64, 128),
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.content_dim = content_dim

        self.obj_cnn = _SmallCNN(in_ch=4, dim=content_dim, channels=channels)  # RGB + mask
        self.bg_cnn = _SmallCNN(in_ch=4, dim=content_dim, channels=channels)   # RGB + bg_mask

    def forward(
        self,
        video0: Tensor,       # (B, H, W, C)
        masks0: Tensor,       # (B, K, H, W)
        boxes0: Tensor,       # (B, K, 4) pixel xyxy
        valid0: Tensor,       # (B, K) bool
    ) -> Tuple[Tensor, Tensor]:
        B, H, W, C = video0.shape
        K = masks0.shape[1]

        img = video0.permute(0, 3, 1, 2).contiguous()             # (B, C, H, W)

        # Object crops (masked). Crop RGB (broadcast across K) with per-object boxes.
        crops = crop_resize(img, boxes0, self.crop_size, H, W)    # (B, K, C, s, s)
        # Crop each object's own mask with its own box.
        masks_flat = masks0.reshape(B * K, 1, H, W).float()       # (B*K, 1, H, W)
        boxes_flat = boxes0.reshape(B * K, 1, 4)                   # (B*K, 1, 4)
        mask_crops = crop_resize(masks_flat, boxes_flat, self.crop_size, H, W)
        mask_crops = mask_crops.reshape(B, K, 1, self.crop_size, self.crop_size)
        masked_crops = crops * mask_crops
        obj_input = torch.cat([masked_crops, mask_crops], dim=2)  # (B, K, C+1, s, s)
        c_obj = self.obj_cnn(obj_input.reshape(B * K, C + 1, self.crop_size, self.crop_size))
        c_obj = c_obj.reshape(B, K, self.content_dim)
        c_obj = c_obj * valid0.unsqueeze(-1).float()

        # Background content (full frame with bg mask).
        bg_mask = (1.0 - masks0.sum(dim=1).clamp(0, 1)).unsqueeze(1)  # (B, 1, H, W)
        bg_input = torch.cat([img, bg_mask], dim=1)                   # (B, C+1, H, W)
        c_bg = self.bg_cnn(bg_input)                                   # (B, D_c)

        return c_obj, c_bg
