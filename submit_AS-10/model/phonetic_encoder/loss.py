"""
Contrastive Loss Functions for Bi-Encoder Training (SPEC-DPR-01).

Implements:
  1. Symmetric / In-Batch InfoNCE Loss with optional Hard Negatives
  2. Triplet Margin Loss with hard negative mining
  3. HybridContrastiveLoss combining both objectives
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Loss (Query-to-Catalog and Catalog-to-Query)
    with support for explicit hard negatives.
    """

    def __init__(self, temperature: float = 0.05, learnable_temp: bool = False):
        super().__init__()
        if learnable_temp:
            self.log_inv_tau = nn.Parameter(torch.tensor([1.0 / temperature]).log())
        else:
            self.register_buffer("log_inv_tau", torch.tensor([1.0 / temperature]).log())

    @property
    def inv_tau(self) -> torch.Tensor:
        inv = self.log_inv_tau.exp()
        # Clamp temperature between [0.02, 0.2] -> inv_tau in [5.0, 50.0]
        return torch.clamp(inv, min=5.0, max=50.0)

    def forward(
        self,
        query_embs: torch.Tensor,       # (B, D) L2-normalized
        pos_catalog_embs: torch.Tensor, # (B, D) L2-normalized
        hard_neg_embs: Optional[torch.Tensor] = None, # (B, M, D) or (B*M, D)
    ) -> torch.Tensor:
        B, D = query_embs.shape
        inv_tau = self.inv_tau

        # In-batch similarity matrix: (B, B)
        sim_q2c = torch.matmul(query_embs, pos_catalog_embs.T) * inv_tau

        if hard_neg_embs is not None:
            if hard_neg_embs.dim() == 3:
                # (B, M, D) -> query-specific hard negatives
                # sim: (B, 1, D) x (B, D, M) -> (B, M)
                sim_hard = torch.bmm(query_embs.unsqueeze(1), hard_neg_embs.transpose(1, 2)).squeeze(1) * inv_tau
            else:
                # (K, D) shared hard negatives
                sim_hard = torch.matmul(query_embs, hard_neg_embs.T) * inv_tau
            # Concatenate along candidate dimension: (B, B + M)
            logits_q2c = torch.cat([sim_q2c, sim_hard], dim=1)
        else:
            logits_q2c = sim_q2c

        labels = torch.arange(B, device=query_embs.device, dtype=torch.long)
        loss_q2c = F.cross_entropy(logits_q2c, labels)

        # Catalog to query symmetric loss
        logits_c2q = sim_q2c.T
        loss_c2q = F.cross_entropy(logits_c2q, labels)

        return (loss_q2c + loss_c2q) / 2.0


class TripletMarginHardLoss(nn.Module):
    """
    Margin loss strictly penalizing hard negative candidates that violate safety margin.
    L_margin = max(0, gamma - pos_sim + neg_sim)
    """

    def __init__(self, margin: float = 0.15):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        query_embs: torch.Tensor,       # (B, D)
        pos_catalog_embs: torch.Tensor, # (B, D)
        hard_neg_embs: torch.Tensor,    # (B, M, D)
    ) -> torch.Tensor:
        # pos_sim: (B,)
        pos_sim = (query_embs * pos_catalog_embs).sum(dim=-1)

        if hard_neg_embs.dim() == 3:
            B, M, D = hard_neg_embs.shape
            # neg_sim: (B, M)
            neg_sim = torch.bmm(hard_neg_embs, query_embs.unsqueeze(-1)).squeeze(-1)
            pos_sim_expanded = pos_sim.unsqueeze(1).expand(-1, M)
            losses = F.relu(self.margin - pos_sim_expanded + neg_sim)
            return losses.mean()
        else:
            # Flattened or shared
            neg_sim = torch.matmul(query_embs, hard_neg_embs.T)
            losses = F.relu(self.margin - pos_sim.unsqueeze(1) + neg_sim)
            return losses.mean()


class HybridContrastiveLoss(nn.Module):
    """
    Combined objective (SPEC-DPR-01 section 3.2):
    L_total = L_InfoNCE + lambda * L_margin
    """

    def __init__(
        self,
        temperature: float = 0.05,
        margin: float = 0.15,
        margin_weight: float = 0.5,
        learnable_temp: bool = False,
    ):
        super().__init__()
        self.infonce = SymmetricInfoNCELoss(temperature=temperature, learnable_temp=learnable_temp)
        self.triplet_margin = TripletMarginHardLoss(margin=margin)
        self.margin_weight = margin_weight

    def forward(
        self,
        query_embs: torch.Tensor,
        pos_catalog_embs: torch.Tensor,
        hard_neg_embs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_infonce = self.infonce(query_embs, pos_catalog_embs, hard_neg_embs)
        if hard_neg_embs is not None and self.margin_weight > 0:
            loss_margin = self.triplet_margin(query_embs, pos_catalog_embs, hard_neg_embs)
            return loss_infonce + self.margin_weight * loss_margin
        return loss_infonce


class SupConCosFaceHardLoss(nn.Module):
    """
    Supervised Contrastive CosFace Loss with ANCE/Sister-Vessel Hard Negatives (SPEC-DPR-02).

    Features:
      1. False Negative Masking (vessel_ids collision prevention)
      2. CosFace Additive Angular Margin (tau=0.05, m=0.08)
      3. Multi-Hard-Negative Margin Loss (gamma=0.20, weight=0.75)
    """

    def __init__(
        self,
        temperature: float = 0.05,
        margin: float = 0.08,
        hard_gamma: float = 0.20,
        hard_weight: float = 0.75,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.hard_gamma = hard_gamma
        self.hard_weight = hard_weight

    def forward(
        self,
        query_embs: torch.Tensor,        # (B, D) L2-normalized
        pos_catalog_embs: torch.Tensor,  # (B, D) L2-normalized
        hard_neg_embs: Optional[torch.Tensor] = None,  # (B, K, D) or (K, D)
        vessel_ids: Optional[List[Union[str, int]]] = None,
    ) -> torch.Tensor:
        B, D = query_embs.shape
        device = query_embs.device
        inv_tau = 1.0 / self.temperature

        # In-batch cosine similarities: (B, B)
        cos_sim = torch.matmul(query_embs, pos_catalog_embs.T)

        # 1. CosFace Margin on positive diagonal: (e_q.T @ e_c+ - m) / tau
        eye = torch.eye(B, device=device)
        cos_sim_cosface = cos_sim - eye * self.margin
        logits_in_batch = cos_sim_cosface * inv_tau

        # 2. False Negative Masking: mask out other rows with identical vessel_id
        if vessel_ids is not None and len(vessel_ids) == B:
            v_arr = np.asarray(vessel_ids)
            same_mask = torch.tensor(
                (v_arr[:, None] == v_arr[None, :]) & ~np.eye(B, dtype=bool),
                device=device,
                dtype=torch.bool,
            )
            logits_in_batch = logits_in_batch.masked_fill(same_mask, -1e4)

        # 3. Concatenate explicit hard negatives if present: (B, K)
        if hard_neg_embs is not None:
            if hard_neg_embs.dim() == 2:
                sim_hard = torch.matmul(query_embs, hard_neg_embs.T)
            else:
                sim_hard = torch.bmm(query_embs.unsqueeze(1), hard_neg_embs.transpose(1, 2)).squeeze(1)

            logits_hard = sim_hard * inv_tau
            logits_all = torch.cat([logits_in_batch, logits_hard], dim=1)
        else:
            logits_all = logits_in_batch

        labels = torch.arange(B, device=device, dtype=torch.long)
        loss_cosface = F.cross_entropy(logits_all, labels)

        # 4. Multi-Hard-Negative Margin Loss (gamma=0.20, weight=0.75)
        if hard_neg_embs is not None and self.hard_weight > 0.0:
            pos_sim = (query_embs * pos_catalog_embs).sum(dim=-1, keepdim=True)  # (B, 1)
            if hard_neg_embs.dim() == 2:
                neg_sim = torch.matmul(query_embs, hard_neg_embs.T)  # (B, K)
            else:
                neg_sim = torch.bmm(query_embs.unsqueeze(1), hard_neg_embs.transpose(1, 2)).squeeze(1)
            hard_losses = F.relu(self.hard_gamma - pos_sim + neg_sim)
            loss_hard = hard_losses.mean()
            return loss_cosface + self.hard_weight * loss_hard

        return loss_cosface
