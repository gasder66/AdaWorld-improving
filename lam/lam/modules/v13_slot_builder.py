"""
V13 SlotBuilders: vectorized Atari bbox → focus slots.

FreewaySlotBuilder: K_focus=16 (1 agent + 10 lanes + 5 nearby cars).
"""
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


ROLE_AGENT = 0
ROLE_LANE_GROUP = 1
ROLE_NEARBY_CAR = 2
ROLE_BULLET = 3
ROLE_FORMATION_GROUP = 4
ROLE_GHOST = 6
ROLE_PELLET_GROUP = 7


class FreewaySlotBuilder(nn.Module):
    """Vectorized: 1 agent + 10 lane groups + 5 nearest cars."""

    def __init__(self, num_lanes=10, num_nearby=5, occ_bins=16, role_emb_dim=32):
        super().__init__()
        self.K_focus = 1 + num_lanes + num_nearby
        self.num_lanes = num_lanes
        self.num_nearby = num_nearby
        self.occ_bins = occ_bins
        self.role_emb_dim = role_emb_dim
        self.role_embed = nn.Embedding(8, role_emb_dim)
        self.agent_type = 0
        self.car_type = 3

    def forward(self, batch: Dict) -> Dict[str, Tensor]:
        bbox = batch["bbox"]       # (B, T, K_raw, 4)
        valid = batch["valid"]     # (B, T, K_raw)
        obj_type = batch["obj_type"]  # (B, K_raw)
        B, T, K_raw, _ = bbox.shape
        K = self.K_focus
        device = bbox.device

        out = {
            "slot_bbox": torch.zeros(B, T, K, 4, device=device),
            "slot_role": torch.full((B, T, K,), -1, dtype=torch.long, device=device),
            "slot_valid": torch.zeros(B, T, K, dtype=torch.bool, device=device),
            "slot_group_feat": torch.zeros(B, T, K, self.occ_bins + 3, device=device),
            "slot_source": torch.full((B, T, K,), -1, dtype=torch.long, device=device),
            "slot_is_group": torch.zeros(B, T, K, dtype=torch.bool, device=device),
            "slot_is_background": torch.zeros(B, T, K, dtype=torch.bool, device=device),
        }

        is_agent = (obj_type.unsqueeze(1) == self.agent_type) & valid  # (B,T,K_raw)
        is_car = (obj_type.unsqueeze(1) == self.car_type) & valid

        # --- Agent slot (0): first valid agent ---
        agent_exists = is_agent.any(dim=-1)  # (B, T)
        agent_idx = is_agent.float().argmax(dim=-1)  # (B, T) — first agent per frame
        agent_valid = agent_exists.unsqueeze(-1)  # (B, T, 1)

        # Gather agent bbox.
        flat_idx = agent_idx.reshape(-1)  # (B*T,)
        flat_bbox = bbox.reshape(B * T, K_raw, 4)
        agent_bbox = flat_bbox[torch.arange(B * T, device=device), flat_idx]  # (B*T, 4)
        agent_bbox = agent_bbox.reshape(B, T, 4)
        agent_cx = agent_bbox[..., 0:1]  # (B, T, 1)
        agent_cy = agent_bbox[..., 1:2]

        out["slot_bbox"][..., 0, :] = agent_bbox
        out["slot_role"][..., 0] = torch.where(agent_exists, ROLE_AGENT, -1)
        out["slot_valid"][..., 0] = agent_exists
        out["slot_source"][..., 0] = torch.where(agent_exists, agent_idx, -1)

        # --- Lane group slots (1-10): bin cars by cy ---
        lane_feats = torch.zeros(B, T, self.num_lanes, self.occ_bins + 3, device=device)
        cy_all = bbox[..., 1]  # (B, T, K_raw)
        cx_all = bbox[..., 0]
        lane_id = (cy_all * self.num_lanes).long().clamp(0, self.num_lanes - 1)  # (B,T,K_raw)

        for li in range(self.num_lanes):
            in_lane = (lane_id == li) & is_car  # (B, T, K_raw)
            n_cars = in_lane.float().sum(dim=-1)  # (B, T)
            slot_idx = 1 + li
            lane_cy = (li + 0.5) / self.num_lanes
            out["slot_bbox"][..., slot_idx, :] = torch.tensor(
                [0.5, lane_cy, 1.0, 1.0 / self.num_lanes], device=device,
            )
            out["slot_role"][..., slot_idx] = ROLE_LANE_GROUP
            out["slot_valid"][..., slot_idx] = True
            out["slot_is_group"][..., slot_idx] = True

            # Occupancy bins — vectorized.
            bin_id = (cx_all * self.occ_bins).long().clamp(0, self.occ_bins - 1)  # (B,T,K_raw)
            occ = torch.zeros(B, T, self.occ_bins, device=device)
            for oi in range(self.occ_bins):
                occ[..., oi] = (in_lane & (bin_id == oi)).float().sum(dim=-1)
            lane_feats[..., li, :self.occ_bins] = occ
            lane_feats[..., li, self.occ_bins] = n_cars / 8.0

            # Nearest car distance to agent.
            cars_cx = torch.where(in_lane, cx_all, torch.full_like(cx_all, 2.0))
            cars_cy = torch.where(in_lane, cy_all, torch.full_like(cy_all, 2.0))
            dx = cars_cx - agent_cx
            dy = cars_cy - agent_cy
            dist = (dx ** 2 + dy ** 2).sqrt()
            dist = torch.where(in_lane, dist, torch.full_like(dist, 2.0))
            nearest = dist.min(dim=-1).values  # (B, T)
            collision = ((dist < 0.08) & in_lane).any(dim=-1).float()  # (B, T)
            lane_feats[..., li, self.occ_bins + 1] = nearest.clamp(max=1.0)
            lane_feats[..., li, self.occ_bins + 2] = collision

        out["slot_group_feat"][:, :, 1:1+self.num_lanes] = lane_feats

        # --- Nearby car slots (11-15) ---
        dx = cx_all - agent_cx  # (B, T, K_raw)
        dy = cy_all - agent_cy
        dist = (dx ** 2 + dy ** 2).sqrt()
        dist = torch.where(is_car, dist, torch.full_like(dist, 2.0))
        _, top_indices = dist.topk(min(self.num_nearby, K_raw), dim=-1, largest=False)  # (B, T, N)

        for ci in range(min(self.num_nearby, K_raw)):
            slot_idx = 1 + self.num_lanes + ci
            idx_ci = top_indices[..., ci]  # (B, T)
            ci_valid = is_car.gather(-1, idx_ci.unsqueeze(-1)).squeeze(-1)  # (B, T)
            flat_ci = idx_ci.reshape(-1)
            car_bbox = flat_bbox[torch.arange(B * T, device=device), flat_ci].reshape(B, T, 4)
            out["slot_bbox"][..., slot_idx, :] = car_bbox
            out["slot_role"][..., slot_idx] = ROLE_NEARBY_CAR
            out["slot_valid"][..., slot_idx] = ci_valid
            out["slot_source"][..., slot_idx] = torch.where(ci_valid, idx_ci, -1)

        return out


def build_slot_builder(game: str, **kwargs) -> nn.Module:
    if game == "freeway":
        return FreewaySlotBuilder(**kwargs)
    raise ValueError(f"Unknown game: {game}")
