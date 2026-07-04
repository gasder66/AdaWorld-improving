"""
V13-A: AtariBBox Object/Group Structure-Action World Model.

Combines:
  SlotBuilder → StructureExtractor (no velocity) → CausalStructureEncoder
  → FactorizedIDM → FactorizedFDM → AtariStructureHead → AtariBBoxRenderer

Hard constraints:
  1. raw_structure: NO velocity / delta / future / flow.
  2. StructureEncoder: CAUSAL temporal attention.
  3. Content path never enters IDM/FDM.
  4. Decoder/renderer never receives z_action directly.
  5. Background not in z_action (via SlotBuilder slot_is_background).
  6. All bbox through SlotBuilder, not raw fixed slots.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from lam.modules.v13_slot_builder import FreewaySlotBuilder
from lam.modules.v13_structure import AtariStructureExtractor, CausalStructureEncoder, AtariStructureHead
from lam.modules.v12_dynamics import FactorizedIDM, FactorizedFDM
from lam.modules.v13_renderer import AtariBBoxRenderer
from lam.modules.v13_losses import structure_loss_v13, reconstruction_loss_v13, free_bits_kl


class V13AtariBBoxModel(nn.Module):
    """Object/group structure-action world model for Atari bbox data."""

    def __init__(
        self,
        slot_builder: FreewaySlotBuilder,
        image_size: int = 84,
        # Structure
        role_emb_dim: int = 32,
        struct_dim: int = 128,
        temporal_layers: int = 2,
        slot_layers: int = 1,
        heads: int = 4,
        group_feat_dim: int = 19,
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

        self.slot_builder = slot_builder
        K_focus = slot_builder.K_focus

        self.structure_extractor = AtariStructureExtractor(
            role_emb_dim=role_emb_dim, group_feat_dim=group_feat_dim,
        )
        raw_dim = self.structure_extractor.raw_dim

        self.structure_encoder = CausalStructureEncoder(
            raw_dim=raw_dim, struct_dim=struct_dim,
            temporal_layers=temporal_layers, slot_layers=slot_layers, heads=heads,
            dropout=dropout,
        )
        self.idm = FactorizedIDM(
            struct_dim=struct_dim, latent_dim=latent_dim,
            layers=idm_layers, heads=dyn_heads, dropout=dropout,
        )
        self.fdm = FactorizedFDM(
            struct_dim=struct_dim, latent_dim=latent_dim,
            layers=fdm_layers, heads=dyn_heads, dropout=dropout,
        )
        self.structure_head = AtariStructureHead(
            struct_dim=struct_dim, group_feat_dim=group_feat_dim,
        )
        self.renderer = AtariBBoxRenderer(
            image_size=image_size,
        )

    def forward(
        self,
        batch: Dict,
        phase: str = "B",
        lambda_box: float = 10.0,
        lambda_exist: float = 1.0,
        lambda_group: float = 1.0,
        lambda_iou: float = 1.0,
        lambda_recon: float = 0.05,
        lambda_kl: float = 0.01,
    ) -> Dict:
        video = batch["video"]                              # (B, T, C, H, W) float
        B, T, C, H, W = video.shape

        # 0. SlotBuilder: raw bbox -> focus slots.
        slots = self.slot_builder(batch)
        slot_bbox = slots["slot_bbox"]
        slot_valid = slots["slot_valid"]
        slot_is_group = slots["slot_is_group"]
        slot_group_feat = slots["slot_group_feat"]

        # 1. Structure path.
        raw_struct, struct_targets = self.structure_extractor(
            slot_bbox, slots["slot_role"], slot_valid, slot_group_feat,
        )
        s = self.structure_encoder(raw_struct, slot_valid)

        s_t = s[:, :-1]
        s_tp1 = s[:, 1:]
        valid_t = slot_valid[:, :-1]
        valid_tp1 = slot_valid[:, 1:]

        targets_tp1 = {
            "bbox": struct_targets["bbox"][:, 1:],
            "valid": struct_targets["valid"][:, 1:],
        }
        if slot_group_feat.shape[-1] > 0:
            targets_tp1["group_feat"] = slot_group_feat[:, 1:]

        outputs: Dict = {"s": s}

        # 2. IDM: s_t, s_{t+1} -> z_action.
        z, mu, logvar = self.idm(s_t, s_tp1, valid_t)

        # 3. FDM: s_t, z -> s_hat.
        s_hat = self.fdm(s_t, z, valid_t)

        # 4. StructureHead.
        pred_struct = self.structure_head(s_hat)

        # 5. Structure loss + KL.
        struct_loss = structure_loss_v13(
            pred_struct, targets_tp1, valid_tp1,
            slot_is_group[:, 1:],
            lambda_box, lambda_exist, lambda_group, lambda_iou,
        )
        kl_loss = free_bits_kl(mu, logvar, valid_t, self.free_bits)

        outputs["z"] = z
        outputs["mu"] = mu
        outputs["logvar"] = logvar
        outputs["s_hat"] = s_hat
        outputs["pred_struct"] = pred_struct
        outputs["struct_loss"] = struct_loss
        outputs["kl_loss"] = kl_loss

        if phase == "B":
            outputs["loss"] = struct_loss + lambda_kl * kl_loss
            return outputs

        # 6. Phase C: renderer.
        recon = self.renderer(video[:, :-1], pred_struct["bbox"], valid_tp1)
        recon_loss = reconstruction_loss_v13(
            recon, video[:, 1:], pred_struct["bbox"], valid_tp1, self.image_size,
        )
        outputs["recon"] = recon
        outputs["recon_loss"] = recon_loss
        outputs["loss"] = struct_loss + lambda_recon * recon_loss + lambda_kl * kl_loss
        return outputs

    @torch.no_grad()
    def forward_ablation(
        self,
        batch: Dict,
        z_mode: str = "normal",
    ) -> Dict:
        video = batch["video"]
        B, T, C, H, W = video.shape

        slots = self.slot_builder(batch)
        slot_bbox = slots["slot_bbox"]
        slot_valid = slots["slot_valid"]
        slot_group_feat = slots["slot_group_feat"]

        raw_struct, struct_targets = self.structure_extractor(
            slot_bbox, slots["slot_role"], slot_valid, slot_group_feat,
        )
        s = self.structure_encoder(raw_struct, slot_valid)

        s_t, s_tp1 = s[:, :-1], s[:, 1:]
        valid_t, valid_tp1 = slot_valid[:, :-1], slot_valid[:, 1:]

        z, mu, logvar = self.idm(s_t, s_tp1, valid_t)

        if z_mode == "zero":
            z = torch.zeros_like(z)
        elif z_mode == "shuffle":
            z = z[torch.randperm(z.shape[0])]

        s_hat = self.fdm(s_t, z, valid_t)
        pred_struct = self.structure_head(s_hat)
        recon = self.renderer(video[:, :-1], pred_struct["bbox"], valid_tp1)

        return {
            "recon": recon,
            "pred_struct": pred_struct,
            "z": z,
            "mu": mu,
            "s_hat": s_hat,
            "s": s,
            "slot_bbox": slot_bbox,
            "slot_valid": slot_valid,
            "slot_is_group": slots["slot_is_group"],
        }
