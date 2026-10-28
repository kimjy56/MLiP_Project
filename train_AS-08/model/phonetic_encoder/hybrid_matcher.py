"""
Hybrid Dense-Lexical Candidate Retrieval & Feature Extractor (SPEC-DPR-01 Section 4.2 & 4.3).

Combines:
  1. Dense Phonetic Retrieval:
     - 78,595 Full-Catalog dot product (< 1ms per query)
     - Top-35 Dense candidate extraction
  2. Dense Feature Engineering for Stage-2 Ranker:
     - dense_sim: Cosine similarity raw score [-1.0, 1.0]
     - dense_rank_recip: 1 / rank_dense
     - dense_zscore: (sim - mean_top35) / (std_top35 + eps)
     - dense_margin_top1: sim - max_{j != i} sim_j
  3. Reciprocal Rank Fusion (RRF):
     - RRF_Score = w_lex / (k0 + rank_lex) + w_dense / (k0 + rank_dense)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .model import CharPhoneticBiEncoder


class HybridDenseLexicalMatcher:
    """
    Production-grade, row-independent hybrid candidate retrieval engine.
    """

    def __init__(
        self,
        model_checkpoint: Union[str, Path],
        catalog_embeds_npy: Union[str, Path],
        catalog_csv: Union[str, Path] = "data/vessel_catalog.csv",
        catalog_df: Optional[pd.DataFrame] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.model = CharPhoneticBiEncoder.load_checkpoint(model_checkpoint, device=device)
        self.model.eval()

        # Load catalog embeddings (78595, 128)
        self.catalog_embeds = np.load(str(catalog_embeds_npy)).astype(np.float32)

        # Load catalog metadata
        if catalog_df is not None:
            df_cat = catalog_df
        else:
            df_cat = pd.read_csv(catalog_csv)
        self.catalog_ids = df_cat["vessel_id"].tolist()
        self.catalog_names = df_cat["vessel_name"].fillna("").astype(str).tolist()
        self.catalog_types = df_cat["vessel_type"].fillna("").astype(str).tolist()
        self.catalog_tonnages = df_cat["gross_tonnage"].fillna(0.0).astype(float).tolist()
        self.num_vessels = len(self.catalog_ids)

        assert self.catalog_embeds.shape[0] == self.num_vessels, (
            f"Catalog embeds shape {self.catalog_embeds.shape[0]} does not match catalog rows {self.num_vessels}"
        )

    def retrieve_dense_candidates(
        self,
        query_name: str,
        top_k: int = 35,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K candidates via dot-product dense phonetic search.
        Latency: < 1ms on CPU / < 0.1ms on GPU.
        """
        if not query_name:
            return []

        # Single row encoding
        q_emb = self.model.encode_queries([query_name], batch_size=1)
        sim_scores = np.dot(self.catalog_embeds, q_emb.squeeze(0))  # (78595,)

        # O(N) extraction of Top-K
        k = min(top_k, self.num_vessels)
        top_indices = np.argpartition(sim_scores, -k)[-k:]
        sorted_indices = top_indices[np.argsort(-sim_scores[top_indices])]

        top_scores = sim_scores[sorted_indices]
        mu = float(np.mean(top_scores))
        sigma = float(np.std(top_scores)) + 1e-6
        max_score = float(top_scores[0])
        second_max_score = float(top_scores[1]) if len(top_scores) > 1 else max_score

        candidates: List[Dict[str, Any]] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            sim = float(sim_scores[idx])
            margin = (sim - second_max_score) if rank == 1 else (sim - max_score)

            candidates.append({
                "vessel_id": self.catalog_ids[idx],
                "vessel_name": self.catalog_names[idx],
                "vessel_type": self.catalog_types[idx],
                "gross_tonnage": self.catalog_tonnages[idx],
                "dense_sim": sim,
                "dense_rank": rank,
                "dense_rank_recip": 1.0 / rank,
                "dense_zscore": (sim - mu) / sigma,
                "dense_margin_top1": margin,
            })

        return candidates

    @staticmethod
    def fuse_rrf(
        lexical_candidates: List[Dict[str, Any]],
        dense_candidates: List[Dict[str, Any]],
        top_k: int = 35,
        w_lex: float = 1.0,
        w_dense: float = 1.0,
        k0: float = 60.0,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion combining lexical/rule candidates and dense phonetic candidates.
        RRF(v) = w_lex / (k0 + rank_lex) + w_dense / (k0 + rank_dense)
        """
        fused_scores: Dict[str, float] = {}
        info_by_id: Dict[str, Dict[str, Any]] = {}

        # 1. Index lexical ranks
        for rank, cand in enumerate(lexical_candidates, start=1):
            vid = cand["vessel_id"]
            rrf_contrib = w_lex / (k0 + rank)
            fused_scores[vid] = fused_scores.get(vid, 0.0) + rrf_contrib
            if vid not in info_by_id:
                info_by_id[vid] = dict(cand)
            info_by_id[vid]["lex_rank"] = rank

        # 2. Index dense ranks
        for cand in dense_candidates:
            vid = cand["vessel_id"]
            rank = cand["dense_rank"]
            rrf_contrib = w_dense / (k0 + rank)
            fused_scores[vid] = fused_scores.get(vid, 0.0) + rrf_contrib
            if vid not in info_by_id:
                info_by_id[vid] = dict(cand)
            else:
                # Merge dense features into existing candidate info
                info_by_id[vid]["dense_sim"] = cand["dense_sim"]
                info_by_id[vid]["dense_rank"] = cand["dense_rank"]
                info_by_id[vid]["dense_rank_recip"] = cand["dense_rank_recip"]
                info_by_id[vid]["dense_zscore"] = cand["dense_zscore"]
                info_by_id[vid]["dense_margin_top1"] = cand["dense_margin_top1"]

        # 3. Sort by fused RRF score
        sorted_vids = sorted(fused_scores.keys(), key=lambda vid: -fused_scores[vid])

        out: List[Dict[str, Any]] = []
        for vid in sorted_vids[:top_k]:
            item = info_by_id[vid]
            item["rrf_score"] = fused_scores[vid]
            # Fill default values for candidates missing dense features
            if "dense_sim" not in item:
                item["dense_sim"] = -1.0
                item["dense_rank"] = 999
                item["dense_rank_recip"] = 0.0
                item["dense_zscore"] = -3.0
                item["dense_margin_top1"] = -1.0
            out.append(item)

        return out
