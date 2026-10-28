# -*- coding: utf-8 -*-
"""Cache and labeling format for KO-FUSION-01."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .as09_adapter import CandidateTrace


@dataclass(frozen=True)
class LabeledQuery:
    sample_id: str
    route: str
    baseline_pred_vessel_id: str
    candidate_ids: Tuple[str, ...]
    legacy_scores: np.ndarray
    features: np.ndarray
    labels: np.ndarray  # int32, shape (N,)
    gold_vessel_id: Optional[str]
    gold_in_candidates: bool
    is_single_candidate: bool


def label_trace(trace: CandidateTrace, gold_vessel_id: Optional[str]) -> LabeledQuery:
    """Join ground truth label to trace strictly after trace generation."""
    n_cands = len(trace.candidate_ids)
    if n_cands == 0 or trace.route != "hangul_candidates":
        labels = np.empty(0, dtype=np.int32)
        gold_in = False
    else:
        labels = np.zeros(n_cands, dtype=np.int32)
        gold_in = False
        if gold_vessel_id is not None:
            for idx, vid in enumerate(trace.candidate_ids):
                if vid == gold_vessel_id:
                    labels[idx] = 1
                    gold_in = True
                    break

    return LabeledQuery(
        sample_id=trace.sample_id,
        route=trace.route,
        baseline_pred_vessel_id=trace.baseline_pred_vessel_id,
        candidate_ids=trace.candidate_ids,
        legacy_scores=trace.legacy_scores,
        features=trace.features,
        labels=labels,
        gold_vessel_id=gold_vessel_id,
        gold_in_candidates=gold_in,
        is_single_candidate=(n_cands == 1),
    )


class CandidateCacheWriter:
    """Writes contiguous queries, candidates, features, labels and group sizes."""

    @staticmethod
    def write_cache(
        out_dir: Path,
        labeled_queries: List[LabeledQuery],
        asset_manifest_sha256: str,
        feature_schema_sha256: str,
        filter_trainable_only: bool = False,
    ) -> Dict[str, Any]:
        """Save queries to cache directory.
        
        If filter_trainable_only is True:
          Includes only hangul_candidates with >= 2 candidates and gold_in_candidates == True.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        queries_meta: List[Dict[str, Any]] = []
        candidate_ids_records: List[List[str]] = []
        features_list: List[np.ndarray] = []
        scores_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        group_sizes: List[int] = []

        total_queries = len(labeled_queries)
        retained_queries = 0
        excluded_non_hangul = 0
        excluded_lookup = 0
        excluded_single_cand = 0
        excluded_no_positive = 0

        for q in labeled_queries:
            if q.route == "non_hangul":
                excluded_non_hangul += 1
                if filter_trainable_only:
                    continue
            elif q.route == "lookup":
                excluded_lookup += 1
                if filter_trainable_only:
                    continue
            elif q.is_single_candidate:
                excluded_single_cand += 1
                if filter_trainable_only:
                    continue
            elif not q.gold_in_candidates and q.gold_vessel_id is not None:
                excluded_no_positive += 1
                if filter_trainable_only:
                    continue

            n_cands = len(q.candidate_ids)
            queries_meta.append({
                "sample_id": q.sample_id,
                "route": q.route,
                "baseline_pred_vessel_id": q.baseline_pred_vessel_id,
                "candidate_count": n_cands,
                "gold_vessel_id": q.gold_vessel_id,
                "gold_in_candidates": q.gold_in_candidates,
            })
            candidate_ids_records.append(list(q.candidate_ids))

            if n_cands > 0:
                features_list.append(q.features)
                scores_list.append(q.legacy_scores)
                labels_list.append(q.labels)
                group_sizes.append(n_cands)

            retained_queries += 1

        if features_list:
            X = np.vstack(features_list).astype(np.float64)
            legacy_scores = np.concatenate(scores_list).astype(np.float64)
            labels = np.concatenate(labels_list).astype(np.int32)
            groups = np.array(group_sizes, dtype=np.int32)
        else:
            X = np.empty((0, 28), dtype=np.float64)
            legacy_scores = np.empty(0, dtype=np.float64)
            labels = np.empty(0, dtype=np.int32)
            groups = np.empty(0, dtype=np.int32)

        assert sum(group_sizes) == len(X), f"group_sizes sum {sum(group_sizes)} != len(X) {len(X)}"

        # Save files
        queries_file = out_dir / "queries.jsonl"
        with open(queries_file, "w", encoding="utf-8") as f:
            for qm in queries_meta:
                f.write(json.dumps(qm, ensure_ascii=False) + "\n")

        cands_file = out_dir / "candidate_ids.jsonl"
        with open(cands_file, "w", encoding="utf-8") as f:
            for c_rec in candidate_ids_records:
                f.write(json.dumps(c_rec, ensure_ascii=False) + "\n")

        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "legacy_scores.npy", legacy_scores)
        np.save(out_dir / "labels.npy", labels)
        np.save(out_dir / "group_sizes.npy", groups)

        manifest = {
            "total_queries_input": total_queries,
            "retained_queries": retained_queries,
            "total_candidates": int(len(X)),
            "filter_trainable_only": filter_trainable_only,
            "exclusion_counts": {
                "non_hangul": excluded_non_hangul,
                "lookup": excluded_lookup,
                "single_candidate": excluded_single_cand,
                "no_positive": excluded_no_positive,
            },
            "file_sha256": {
                "queries.jsonl": hashlib.sha256(queries_file.read_bytes()).hexdigest(),
                "candidate_ids.jsonl": hashlib.sha256(cands_file.read_bytes()).hexdigest(),
                "X.npy": hashlib.sha256((out_dir / "X.npy").read_bytes()).hexdigest(),
                "legacy_scores.npy": hashlib.sha256((out_dir / "legacy_scores.npy").read_bytes()).hexdigest(),
                "labels.npy": hashlib.sha256((out_dir / "labels.npy").read_bytes()).hexdigest(),
                "group_sizes.npy": hashlib.sha256((out_dir / "group_sizes.npy").read_bytes()).hexdigest(),
            },
            "asset_manifest_sha256": asset_manifest_sha256,
            "feature_schema_sha256": feature_schema_sha256,
        }
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest


class CandidateCacheReader:
    """Reads saved contiguous cache arrays and query metadata."""

    @staticmethod
    def load_cache(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Returns X, legacy_scores, labels, group_sizes, queries_meta, manifest."""
        manifest_file = cache_dir / "manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        queries_file = cache_dir / "queries.jsonl"
        queries_meta = []
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries_meta.append(json.loads(line))

        X = np.load(cache_dir / "X.npy")
        legacy_scores = np.load(cache_dir / "legacy_scores.npy")
        labels = np.load(cache_dir / "labels.npy")
        group_sizes = np.load(cache_dir / "group_sizes.npy")

        assert len(group_sizes) == 0 or sum(group_sizes) == len(X), "Corrupted cache: sum(group_sizes) != len(X)"
        return X, legacy_scores, labels, group_sizes, queries_meta, manifest
