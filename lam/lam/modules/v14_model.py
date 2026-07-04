"""
V14: Semi-Real Mask-Structure Latent Action Model.

Content path: first-frame RGB → c_obj, c_bg (V12 reuse)
Structure path: masks/bboxes → raw_struct (no velocity) → causal s_t
Dynamics: IDM(s_t, s_{t+1}) → z → FDM(s_t, z) → s_hat
StructureHead: s_hat → bbox + mask_low + exist + visible_ratio
Renderer: LayeredCompositor (does NOT receive z)

3 phases: A (renderer oracle), B (structure LAM), C (joint)
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from lam.modules.v12_dynamics import FactorizedIDM, FactorizedFDM
from lam.modules.v14_content import ObjectContentEncoder
from lam.modules.v14_structure import MaskStructureExtractor, CausalMaskStructureEncoder, V14StructureHead
from lam.modules.v14_renderer import LayeredCompositor
from lam.modules.v14_losses import structure_loss_v14, reconstruction_loss_v14, free_bits_kl_v14


class V14Model(nn.Module):

    def __init__(
        self,
        image_size: int = 128,
        max_actors: int = 4,
        # Content
        crop_size: int = 64,
        content_dim: int = 128,
        # Structure
        mask_grid: int = 16,
        mask_feat_dim: int = 32,
        struct_dim: int = 128,
        temporal_layers: int = 2,
        slot_layers: int = 1,
        heads: int = 4,
        # Latent
        latent_dim: int = 16,
        idm_layers: int = 2,
        fdm_layers: int = 2,
        dyn_heads: int = 4,
        free_bits: float = 0.05,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.free_bits = free_bits

        self.content_encoder = ObjectContentEncoder(
            crop_size=crop_size, content_dim=content_dim,
        )
        self.structure_extractor = MaskStructureExtractor(
            mask_grid=mask_grid, mask_feat_dim=mask_feat_dim,
        )
        raw_dim = self.structure_extractor.raw_dim
        self.structure_encoder = CausalMaskStructureEncoder(
            raw_dim=raw_dim, struct_dim=struct_dim,
            temporal_layers=temporal_layers, slot_layers=slot_layers,
            heads=heads, dropout=dropout,
        )
        self.idm = FactorizedIDM(
            struct_dim=struct_dim, latent_dim=latent_dim,
            layers=idm_layers, heads=dyn_heads, dropout=dropout,
        )
        self.fdm = FactorizedFDM(
            struct_dim=struct_dim, latent_dim=latent_dim,
            layers=fdm_layers, heads=dyn_heads, dropout=dropout,
        )
        self.structure_head = V14StructureHead(
            struct_dim=struct_dim, mask_grid=mask_grid,
        )
        self.renderer = LayeredCompositor(image_size=image_size)

    def forward(
        self, batch: Dict, phase: str = "C",
        lambda_box: float = 10.0, lambda_mask: float = 1.0,
        lambda_iou: float = 1.0, lambda_exist: float = 1.0,
        lambda_vis: float = 0.5, lambda_recon: float = 0.1,
        lambda_kl: float = 0.01,
    ) -> Dict:
        video = batch["video"]                           # (B, T, C, H, W)
        masks = batch["masks"]                           # (B, T, K, H, W)
        boxes = batch["boxes"]                           # (B, T, K, 4) cxcywh
        valid = batch["valid"]                           # (B, T, K)
        visible_masks = batch.get("visible_masks", masks)
        B, T, C, H, W = video.shape

        # 1. Content path (first frame only).
        # Convert to V12 format: pixel xyxy bboxes.
        boxes_pix = boxes.clone()
        boxes_pix[..., 0] = boxes[..., 0] * W
        boxes_pix[..., 1] = boxes[..., 1] * H
        boxes_pix[..., 2] = (boxes[..., 0] + boxes[..., 2]) * W
        boxes_pix[..., 3] = (boxes[..., 1] + boxes[..., 3]) * H
        c_obj, c_bg = self.content_encoder(
            video[:, 0].permute(0, 2, 3, 1),  # (B, H, W, C)
            masks[:, 0], boxes_pix[:, 0], valid[:, 0],
        )

        # 2. Structure path.
        raw_struct, struct_targets = self.structure_extractor(
            boxes, masks, visible_masks, valid,
        )
        s = self.structure_encoder(raw_struct, valid)

        s_t = s[:, :-1]; s_tp1 = s[:, 1:]
        valid_t = valid[:, :-1]; valid_tp1 = valid[:, 1:]

        targets_tp1 = {k: v[:, 1:] for k, v in struct_targets.items()}

        outputs: Dict = {"c_obj": c_obj, "c_bg": c_bg, "s": s}

        # 3. IDM.
        z, mu, logvar = self.idm(s_t, s_tp1, valid_t)

        # 4. FDM.
        s_hat = self.fdm(s_t, z, valid_t)

        # 5. StructureHead.
        pred_struct = self.structure_head(s_hat)

        # 6. Structure loss + KL.
        struct_loss = structure_loss_v14(
            pred_struct, targets_tp1, valid_tp1,
            lambda_box, lambda_mask, lambda_iou, lambda_exist, lambda_vis,
        )
        kl_loss = free_bits_kl_v14(mu, logvar, valid_t, self.free_bits)

        outputs.update({"z": z, "mu": mu, "logvar": logvar, "s_hat": s_hat,
                        "pred_struct": pred_struct, "struct_loss": struct_loss, "kl_loss": kl_loss})

        if phase == "B":
            outputs["loss"] = struct_loss + lambda_kl * kl_loss
            return outputs

        # 7. Phase C: renderer.
        recon = self.renderer(video[:, 0], pred_struct["bbox"], pred_struct["mask_low"], valid_tp1)
        recon_loss = reconstruction_loss_v14(
            recon, video[:, 1:], pred_struct["mask_low"], valid_tp1,
        )
        outputs["recon"] = recon
        outputs["recon_loss"] = recon_loss
        outputs["loss"] = struct_loss + lambda_recon * recon_loss + lambda_kl * kl_loss
        return outputs

    @torch.no_grad()
    def forward_ablation(self, batch: Dict, z_mode: str = "normal") -> Dict:
        video = batch["video"]
        masks = batch["masks"]
        boxes = batch["boxes"]
        valid = batch["valid"]
        visible_masks = batch.get("visible_masks", masks)
        B, T, C, H, W = video.shape

        boxes_pix = boxes.clone()
        boxes_pix[..., 0] = boxes[..., 0] * W
        boxes_pix[..., 1] = boxes[..., 1] * H
        boxes_pix[..., 2] = (boxes[..., 0] + boxes[..., 2]) * W
        boxes_pix[..., 3] = (boxes[..., 1] + boxes[..., 3]) * H
        c_obj, c_bg = self.content_encoder(
            video[:, 0].permute(0, 2, 3, 1), masks[:, 0], boxes_pix[:, 0], valid[:, 0],
        )

        raw_struct, struct_targets = self.structure_extractor(boxes, masks, visible_masks, valid)
        s = self.structure_encoder(raw_struct, valid)
        s_t, s_tp1 = s[:, :-1], s[:, 1:]
        valid_t, valid_tp1 = valid[:, :-1], valid[:, 1:]

        z, mu, _ = self.idm(s_t, s_tp1, valid_t)
        if z_mode == "zero":
            z = torch.zeros_like(z)
        elif z_mode == "shuffle":
            z = z[torch.randperm(z.shape[0])]

        s_hat = self.fdm(s_t, z, valid_t)
        pred_struct = self.structure_head(s_hat)
        recon = self.renderer(video[:, 0], pred_struct["bbox"], pred_struct["mask_low"], valid_tp1)

        return {
            "recon": recon, "pred_struct": pred_struct, "z": z, "mu": mu,
            "s_hat": s_hat, "s": s, "targets": struct_targets,
            "boxes_gt": boxes, "valid_gt": valid, "masks_gt": masks,
        }
