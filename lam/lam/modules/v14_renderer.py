"""
V14 Renderer: simple copy baseline + CNN refinement.

Does NOT receive z_action.
First version: background-copy with optional residual CNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LayeredCompositor(nn.Module):

    def __init__(self, image_size: int = 128, hid_dim: int = 32):
        super().__init__()
        self.H = self.W = image_size
        in_c = 3 + 2  # video_0 RGB + bbox_occ + mask_map

        self.cnn = nn.Sequential(
            nn.Conv2d(in_c, hid_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid_dim, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        video_0: Tensor,
        pred_bbox: Tensor,
        pred_mask: Tensor,
        valid: Tensor,
    ) -> Tensor:
        B, T1, K, _, Hp, Wp = pred_mask.shape  # Hp=Wp=mask_grid
        C, H, W = 3, self.H, self.W
        device = video_0.device
        n = B * T1

        # Bbox occupancy map (16x16 grid).
        g = 16
        bb = pred_bbox.reshape(n, K, 4)
        vl = valid.reshape(n, K)
        occ = torch.zeros(n, g * g, device=device)
        for k in range(K):
            vm = vl[:, k]  # (n,)
            if vm.any():
                gx = (bb[:, k, 0] * g).long().clamp(0, g - 1)
                gy = (bb[:, k, 1] * g).long().clamp(0, g - 1)
                row_idx = torch.arange(n, device=device)
                occ[row_idx, gy * g + gx] += vm.float()
        occ = F.interpolate(occ.reshape(n, 1, g, g) / max(K, 1), size=(H, W),
                            mode="bilinear", align_corners=False)

        # Mask map from predicted masks.
        pm = pred_mask.sigmoid().reshape(n, K, 1, Hp, Wp)
        vf = vl.float().reshape(n, K, 1, 1, 1)
        mask_map = (pm * vf).sum(dim=1)  # (n, 1, Hp, Wp)
        mask_map = F.interpolate(mask_map, size=(H, W), mode="bilinear", align_corners=False)

        video_0_exp = video_0.unsqueeze(1).expand(-1, T1, C, H, W).reshape(n, C, H, W)
        x = torch.cat([video_0_exp, occ, mask_map], dim=1)

        residual = self.cnn(x) * 0.05
        recon = (video_0_exp + residual).clamp(0, 1)
        return recon.reshape(B, T1, C, H, W)
