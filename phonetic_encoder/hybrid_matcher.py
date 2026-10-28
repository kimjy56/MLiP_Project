"""
Hybrid Dense-Lexical Candidate Retrieval & Feature Extractor (SPEC-DPR-01 Section 4.2 & 4.3).

Combines:
  1. Dense Phonetic Retrieval:
     - One full-catalog dot product per query
     - Configurable Top-K Dense candidate extraction
  2. Dense Feature Engineering for Stage-2 Ranker:
     - dense_sim: Cosine similarity raw score [-1.0, 1.0]
     - dense_rank_recip: 1 / rank_dense
      - dense_zscore: (sim - mean_topK) / (std_topK + eps)
     - dense_margin_top1: sim - max_{j != i} sim_j
  3. Reciprocal Rank Fusion (RRF):
     - RRF_Score = w_lex / (k0 + rank_lex) + w_dense / (k0 + rank_dense)
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .model import CharPhoneticBiEncoder
from .retrieval import stable_top_indices


@dataclass(frozen=True)
class DenseRowResult:
    """One query's immutable retrieval context; never retained across test rows."""
    query_name: str
    scores: np.ndarray
    top_indices: np.ndarray
    rank_by_index: Dict[int, int]
    mean_top: float
    std_top: float
    max_score: float
    second_score: float


class HybridDenseLexicalMatcher:
    """
    Row-independent hybrid candidate retrieval and row-local feature reuse.
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
        # CPU half precision accumulates batch-dependent rounding and is slower here.
        if device == "cpu":
            self.model = self.model.float()
        self.model.eval()

        # Load catalog embeddings (78595, 128)
        self.catalog_embeds = np.load(str(catalog_embeds_npy)).astype(np.float32)
        metadata_path = Path(catalog_embeds_npy).with_suffix('.json')
        self.provenance_verified = False
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if 'embeddings_sha256' in metadata:
                def digest(path):
                    with Path(path).open('rb') as stream:
                        return hashlib.file_digest(stream, 'sha256').hexdigest()
                if metadata['embeddings_sha256'] != digest(catalog_embeds_npy) or metadata['checkpoint_sha256'] != digest(model_checkpoint):
                    raise ValueError('Catalog embeddings/checkpoint hash mismatch')
                self.provenance_verified = True

        # Load catalog metadata
        if catalog_df is not None:
            df_cat = catalog_df
        else:
            df_cat = pd.read_csv(catalog_csv)
        self.catalog_ids = df_cat["vessel_id"].astype(str).tolist()
        self.id_to_index = {v: i for i, v in enumerate(self.catalog_ids)}
        if self.provenance_verified:
            actual_order = hashlib.sha256(json.dumps(self.catalog_ids, ensure_ascii=False,
                sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if metadata['catalog_ids_sha256'] != actual_order:
                raise ValueError('Catalog row ordering differs from embedding export')
        self.catalog_names = df_cat["vessel_name"].fillna("").astype(str).tolist()
        self.catalog_types = df_cat["vessel_type"].fillna("").astype(str).tolist()
        self.catalog_tonnages = pd.to_numeric(df_cat["gross_tonnage"], errors='coerce').fillna(0.0).astype(float).tolist()
        self.num_vessels = len(self.catalog_ids)

        assert self.catalog_embeds.shape[0] == self.num_vessels, (
            f"Catalog embeds shape {self.catalog_embeds.shape[0]} does not match catalog rows {self.num_vessels}"
        )
        if self.catalog_embeds.ndim != 2 or self.catalog_embeds.shape[1] != self.model.encoder.embed_dim:
            raise ValueError("Catalog embedding dimension mismatch")
        if len(self.id_to_index) != self.num_vessels or not np.isfinite(self.catalog_embeds).all():
            raise ValueError("Duplicate catalog IDs or invalid embeddings")

    def score_query(self, query_name: str, top_k: int = 60) -> DenseRowResult:
        """Exactly one three-view encoding and one full-catalog matrix-vector product."""
        if not query_name or top_k < 1:
            raise ValueError("Nonempty query and positive top_k required")
        query = self.model.encode_queries([query_name], batch_size=1)[0].astype(np.float32)
        scores = np.dot(self.catalog_embeds, query)
        order = stable_top_indices(scores, top_k, self.catalog_ids)
        top = scores[order]
        scores.setflags(write=False)
        order.setflags(write=False)
        return DenseRowResult(str(query_name), scores, order,
                              {int(i): rank for rank, i in enumerate(order, 1)},
                              float(top.mean()), float(top.std()) + 1e-6,
                              float(top[0]), float(top[1] if len(top) > 1 else top[0]))

    def extract_stage2_features(self, result: DenseRowResult, vessel_ids) -> Dict[str, Dict[str, Any]]:
        """Reuse the same scores for all union candidates, including lexical-only ones.

        Reciprocal ranks refer to the selected Dense top K; outside-K candidates
        receive zero reciprocal rank, rather than an invented exact full rank.
        """
        features = {}
        for vessel_id in vessel_ids:
            idx = self.id_to_index.get(str(vessel_id))
            if idx is None:
                continue
            sim = float(result.scores[idx])
            rank = result.rank_by_index.get(idx)
            features[str(vessel_id)] = {
                'vessel_id': str(vessel_id), 'dense_sim': sim,
                'dense_rank': rank if rank is not None else self.num_vessels + 1,
                'dense_rank_recip': 1. / rank if rank is not None else 0.,
                'dense_zscore': (sim - result.mean_top) / result.std_top,
                'dense_margin_top1': sim - (result.second_score if rank == 1 else result.max_score),
            }
        return features

    def retrieve_dense_candidates(
        self,
        query_name: str,
        top_k: int = 35,
        result: Optional[DenseRowResult] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K candidates via dot-product dense phonetic search.
        Measured latency depends on the model, device and catalog size.
        """
        if not query_name:
            return []

        result = result if result is not None else self.score_query(query_name, top_k)
        if result.query_name != str(query_name) or len(result.top_indices) < min(top_k, self.num_vessels):
            raise ValueError("Retrieval context does not match this query/top_k")
        sorted_indices = result.top_indices[:top_k]
        feature_map = self.extract_stage2_features(result, [self.catalog_ids[i] for i in sorted_indices])
        candidates: List[Dict[str, Any]] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            candidates.append({
                "vessel_id": self.catalog_ids[idx],
                "vessel_name": self.catalog_names[idx],
                "vessel_type": self.catalog_types[idx],
                "gross_tonnage": self.catalog_tonnages[idx],
                **feature_map[self.catalog_ids[idx]],
            })

        # Sister-Vessel Name-Confidence Gating (SPEC-DPR-02 Section 3.2)
        if candidates and candidates[0]["dense_sim"] >= 0.82 and len(candidates) > 1:
            top_name_words = candidates[0]["vessel_name"].upper().split()
            top_prefix = top_name_words[0] if top_name_words else ""
            if len(top_prefix) >= 3:
                has_sister_gap = any(
                    cand["vessel_name"].upper().startswith(top_prefix)
                    and (candidates[0]["dense_sim"] - cand["dense_sim"]) >= 0.10
                    for cand in candidates[1:]
                )
                if has_sister_gap:
                    candidates[0]["sister_protected_bonus"] = 1.5

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
