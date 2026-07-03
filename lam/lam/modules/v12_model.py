"""
V12: Object-Centric Structure-Action World Model.

Three paths:
  Content:    first-frame RGB -> c_obj, c_bg          (appearance, sees RGB)
  Structure:  masks/boxes -> s_t                      (no RGB)
  Action:     s_t, s_{t+1} -> z_action -> s_hat_{t+1} (structure transition only)

Decoder: content + predicted structure -> RGB (NEVER sees z_action).

Phases:
  A: train ContentEncoder + FusionRenderer with GT structure (verify decoder).
  B: train IDM + FDM + StructureHead + StructureEncoder (structure LAM).
  C: joint: L_struct + small L_recon + L_kl.

Hard constraints (enforced in code):
  1. actor_labels / object_types never enter forward().
  2. z_action never passed to decoder.
  3. content tensors never enter IDM/FDM.
  4. StructureExtractor uses no RGB.
  5. FDM predicts structure residual, not RGB.
  6. Main objective is structure prediction; RGB is auxiliary.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam.modules.v12_content import ObjectContentEncoder
from lam.modules.v12_structure import StructureExtractor, StructureEncoder
from lam.modules.v12_dynamics import FactorizedIDM, FactorizedFDM
from lam.modules.v12_decoder import StructureHead, FusionRenderer


def _dice_bce_loss(pred_logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """pred_logits: (B, T, K, 1, g, g), target: same shape (binary), valid: (B, T, K)."""
    pred_sig = torch.sigmoid(pred_logits)
    smooth = 1.0
    v = valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
    intersection = ((pred_sig * target) * v).sum()
    denom = ((pred_sig + target) * v).sum()
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    bce = (bce * v).sum() / (v.sum() * target.shape[-1] * target.shape[-2] + 1e-6)
    return (1.0 - dice) + bce


class LatentActionModelV12(nn.Module):
    """Object-centric structure-action world model.

    Diagnostic flags (V12.1):
      use_z=False     — No-z FDM baseline (Experiment A)
      use_velocity=False — No-velocity structure (Experiment B)
      encoder_mode="causal" | "per_frame" — Causal StructureEncoder (Experiment C)
    """

    def __init__(
        self,
        image_size: int = 256,
        max_actors: int = 4,
        # Content
        crop_size: int = 64,
        content_dim: int = 128,
        content_channels=(32, 64, 128),
        # Structure
        mask_grid: int = 16,
        mask_feat_dim: int = 32,
        struct_dim: int = 128,
        temporal_layers: int = 2,
        slot_layers: int = 1,
        struct_heads: int = 4,
        # Latent action
        latent_dim: int = 16,
        idm_layers: int = 2,
        fdm_layers: int = 2,
        dyn_heads: int = 4,
        # Decoder
        dec_dim: int = 256,
        patch_size: int = 16,
        dec_blocks: int = 4,
        dec_heads: int = 8,
        # Loss
        free_bits: float = 0.05,
        dropout: float = 0.0,
        # V12.1 diagnostic flags
        use_z: bool = True,
        use_velocity: bool = True,
        encoder_mode: str = "bidirectional",
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.max_actors = max_actors
        self.latent_dim = latent_dim
        self.free_bits = free_bits
        self.use_z = use_z
        self.use_velocity = use_velocity
        self.encoder_mode = encoder_mode

        raw_dim = 4 + (4 if use_velocity else 0) + 6 + mask_feat_dim

        self.content_encoder = ObjectContentEncoder(
            crop_size=crop_size, content_dim=content_dim, channels=content_channels,
        )
        self.structure_extractor = StructureExtractor(
            mask_grid=mask_grid, mask_feat_dim=mask_feat_dim,
            use_velocity=use_velocity,
        )
        self.structure_encoder = StructureEncoder(
            raw_dim=raw_dim, struct_dim=struct_dim,
            temporal_layers=temporal_layers, slot_layers=slot_layers,
            heads=struct_heads, dropout=dropout,
            encoder_mode=encoder_mode,
        )
        if use_z:
            self.idm = FactorizedIDM(
                struct_dim=struct_dim, latent_dim=latent_dim,
                layers=idm_layers, heads=dyn_heads, dropout=dropout,
            )
        else:
            self.idm = None
        self.fdm = FactorizedFDM(
            struct_dim=struct_dim, latent_dim=latent_dim,
            layers=fdm_layers, heads=dyn_heads, dropout=dropout,
        )
        self.structure_head = StructureHead(
            struct_dim=struct_dim, mask_grid=mask_grid,
        )
        self.decoder = FusionRenderer(
            content_dim=content_dim, struct_dim=struct_dim, dec_dim=dec_dim,
            patch_size=patch_size, image_size=image_size,
            num_heads=dec_heads, dec_blocks=dec_blocks, dropout=dropout,
        )

    def _free_bits_kl(self, mu: Tensor, logvar: Tensor, valid: Tensor) -> Tensor:
        kl_dim = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1)
        kl_dim = kl_dim.clamp(min=self.free_bits)
        v = valid.unsqueeze(-1).float()
        n_valid = v.sum().clamp(min=1.0) * mu.shape[-1]
        return (kl_dim * v).sum() / n_valid

    def _structure_loss(
        self,
        pred: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        valid_tp1: Tensor,       # (B, T-1, K) validity at target frames
        lambda_box: float = 10.0,
        lambda_mask: float = 1.0,
        lambda_mom: float = 1.0,
    ) -> Tensor:
        v = valid_tp1.unsqueeze(-1).float()                       # (B, T-1, K, 1)

        # Box: SmoothL1 on normalized cxcywh.
        box_loss = F.smooth_l1_loss(pred["bbox"], targets["bbox"], reduction="none")
        box_loss = (box_loss * v).sum() / (v.sum() * 4 + 1e-6)

        # Mask: Dice + BCE.
        mask_loss = _dice_bce_loss(pred["mask_lowres"], targets["mask_lowres"], valid_tp1)

        # Moments: MSE.
        mom_loss = F.mse_loss(pred["moments"], targets["moments"], reduction="none")
        mom_loss = (mom_loss * v).sum() / (v.sum() * 6 + 1e-6)

        return lambda_box * box_loss + lambda_mask * mask_loss + lambda_mom * mom_loss

    def _recon_loss(
        self,
        recon: Tensor,           # (B, T-1, H, W, C)
        target: Tensor,          # (B, T-1, H, W, C)
        masks_tp1: Tensor,       # (B, T-1, K, H, W)
        lambda_actor_masked: float = 2.0,
    ) -> Tensor:
        l1 = F.l1_loss(recon, target, reduction="mean")
        # Actor-masked L1.
        actor_mask = masks_tp1.sum(dim=2).clamp(0, 1).unsqueeze(-1)  # (B, T-1, H, W, 1)
        masked_err = ((recon - target).abs() * actor_mask).sum()
        denom = actor_mask.sum().clamp(min=1.0) * recon.shape[-1]
        masked_l1 = masked_err / denom
        return l1 + lambda_actor_masked * masked_l1

    def forward(
        self,
        batch: Dict,
        phase: str = "C",
        lambda_box: float = 10.0,
        lambda_mask: float = 1.0,
        lambda_mom: float = 1.0,
        lambda_actor_masked: float = 2.0,
    ) -> Dict:
        """Forward pass. actor_labels/object_types must NOT be in batch for training."""
        videos = batch["videos"]                                   # (B, T, H, W, C)
        masks = batch["masks"]                                     # (B, T, K, H, W)
        boxes = batch["bboxes"]                                    # (B, T, K, 4) pixel xyxy
        valid = batch["valid_mask"]                                # (B, T, K)
        B, T, H, W, C = videos.shape

        # 1. Content path (first frame only).
        c_obj, c_bg = self.content_encoder(
            videos[:, 0], masks[:, 0], boxes[:, 0], valid[:, 0],
        )

        # 2. Structure path (all frames).
        raw_struct, struct_targets = self.structure_extractor(masks, boxes, valid)
        s = self.structure_encoder(raw_struct, valid)              # (B, T, K, D_s)

        s_t = s[:, :-1]                                            # (B, T-1, K, D_s)
        s_tp1 = s[:, 1:]                                           # GT structure at t+1
        valid_t = valid[:, :-1]
        valid_tp1 = valid[:, 1:]

        # Targets at t+1 for structure loss.
        targets_tp1 = {
            "bbox": struct_targets["bbox"][:, 1:],
            "mask_lowres": struct_targets["mask_lowres"][:, 1:],
            "moments": struct_targets["moments"][:, 1:],
        }

        outputs: Dict = {
            "c_obj": c_obj, "c_bg": c_bg, "s": s,
        }

        if phase == "A":
            # Decoder with GT structure — verify renderer.
            recon = self.decoder(c_obj, c_bg, s_tp1, valid_tp1)
            recon_loss = self._recon_loss(
                recon, videos[:, 1:], masks[:, 1:], lambda_actor_masked,
            )
            outputs["recon"] = recon
            outputs["recon_loss"] = recon_loss
            outputs["loss"] = recon_loss
            return outputs

        # 3. Inverse dynamics: s_t, s_{t+1} -> z.
        if self.use_z and self.idm is not None:
            z, mu, logvar = self.idm(s_t, s_tp1, valid_t)
        else:
            # No-z baseline: z is zeros, no IDM, no KL.
            Bz, Tz, Kz = s_t.shape[:3]
            z = torch.zeros(Bz, Tz, Kz, self.latent_dim, device=s_t.device, dtype=s_t.dtype)
            mu = z
            logvar = -5.0 * torch.ones_like(z)

        # 4. Forward dynamics: s_t, z -> s_hat_{t+1}.
        s_hat = self.fdm(s_t, z, valid_t)

        # 5. Structure head: s_hat -> interpretable structure.
        pred_struct = self.structure_head(s_hat)

        # Structure loss + KL.
        struct_loss = self._structure_loss(
            pred_struct, targets_tp1, valid_tp1,
            lambda_box, lambda_mask, lambda_mom,
        )
        kl_loss = self._free_bits_kl(mu, logvar, valid_t)

        outputs["z"] = z
        outputs["mu"] = mu
        outputs["logvar"] = logvar
        outputs["s_hat"] = s_hat
        outputs["pred_struct"] = pred_struct
        outputs["struct_loss"] = struct_loss
        outputs["kl_loss"] = kl_loss

        if phase == "B":
            outputs["loss"] = struct_loss + 0.01 * kl_loss
            return outputs

        # Phase C: joint — add RGB reconstruction with predicted structure.
        recon = self.decoder(c_obj, c_bg, s_hat, valid_tp1)
        recon_loss = self._recon_loss(
            recon, videos[:, 1:], masks[:, 1:], lambda_actor_masked,
        )
        outputs["recon"] = recon
        outputs["recon_loss"] = recon_loss
        outputs["loss"] = struct_loss + 0.1 * recon_loss + 0.01 * kl_loss
        return outputs

    # === Ablation / eval helpers ===

    @torch.no_grad()
    def forward_ablation(
        self,
        batch: Dict,
        z_mode: str = "normal",   # "normal" | "zero" | "shuffle"
        slot_keep: int = -1,       # -1 = all slots; k = keep only slot k
    ) -> Dict:
        """Run full forward with z ablation for eval.

        Args:
            z_mode: "normal" | "zero" | "shuffle"
            slot_keep: if >= 0, zero out z for all slots except slot_keep.
        """
        videos = batch["videos"]
        masks = batch["masks"]
        boxes = batch["bboxes"]
        valid = batch["valid_mask"]
        B, T, H, W, C = videos.shape

        c_obj, c_bg = self.content_encoder(
            videos[:, 0], masks[:, 0], boxes[:, 0], valid[:, 0],
        )
        raw_struct, struct_targets = self.structure_extractor(masks, boxes, valid)
        s = self.structure_encoder(raw_struct, valid)

        s_t, s_tp1 = s[:, :-1], s[:, 1:]
        valid_t, valid_tp1 = valid[:, :-1], valid[:, 1:]

        if self.use_z and self.idm is not None:
            z, mu, logvar = self.idm(s_t, s_tp1, valid_t)
        else:
            Bz, Tz, Kz = s_t.shape[:3]
            z = torch.zeros(Bz, Tz, Kz, self.latent_dim, device=s_t.device, dtype=s_t.dtype)
            mu = z

        if z_mode == "zero":
            z = torch.zeros_like(z)
        elif z_mode == "shuffle":
            z = z[torch.randperm(z.shape[0])]  # shuffle across batch

        if slot_keep >= 0:
            mask = torch.zeros_like(z)
            if z.dim() == 4:
                mask[..., slot_keep, :] = 1.0
            z = z * mask

        s_hat = self.fdm(s_t, z, valid_t)
        pred_struct = self.structure_head(s_hat)
        recon = self.decoder(c_obj, c_bg, s_hat, valid_tp1)

        return {
            "recon": recon,
            "pred_struct": pred_struct,
            "z": z,
            "mu": mu,
            "s_hat": s_hat,
            "struct_targets": struct_targets,
            "s": s,
        }
