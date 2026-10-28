# -*- coding: utf-8 -*-
"""Feature schema v1 for KO-FUSION-01 (28 features)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional
import numpy as np

FEATURE_SCHEMA_VERSION = "ko_fusion_v1"

FEATURE_NAMES: List[str] = [
    # Group 1: Existing scores (6)
    "legacy_total_score",
    "legacy_name_score",
    "legacy_num_score",
    "legacy_ton_score",
    "legacy_meta_score",
    "legacy_dense_score",
    # Group 2: Existing name info (6)
    "sim_alpha",
    "sim_phon",
    "sim_skel",
    "sim_tok",
    "word_cnt_bonus",
    "first_match",
    # Group 3: Number status (3)
    "query_num_present",
    "candidate_num_present",
    "num_compatible",
    # Group 4: Tonnage (5)
    "gt_missing",
    "igt_missing",
    "catalog_gt_missing",
    "rel_gt",
    "rel_igt",
    # Group 5: Meta missing (2)
    "flag_missing",
    "type_missing",
    # Group 6: Existing Dense (5)
    "in_dense_top35",
    "dense_sim",
    "dense_rank_recip",
    "dense_zscore",
    "dense_margin_top1",
    # Group 7: Existing rank (1)
    "legacy_rank_recip",
]

assert len(FEATURE_NAMES) == 28, f"Feature count must be exactly 28, got {len(FEATURE_NAMES)}"


def get_feature_schema_dict() -> Dict[str, Any]:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "dtype": "float64",
        "missing_rules": {
            "rel_gt": "NaN if gt <= 0 or catalog_gt <= 0",
            "rel_igt": "NaN if igt <= 0 or catalog_gt <= 0",
            "dense_feats": "NaN for sim/rank_recip/zscore/margin if in_dense_top35 == 0",
        },
    }


def compute_feature_schema_sha256() -> str:
    schema_dict = get_feature_schema_dict()
    payload = json.dumps(schema_dict, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_candidate_features(
    raw_feats_list: List[Dict[str, Any]],
    legacy_ranks: List[int],
) -> np.ndarray:
    """Convert raw candidate features and legacy rank into a (N, 28) float64 array.
    
    Raises:
        ValueError: If Inf/-Inf is present, or shape mismatch occurs.
    """
    n_cands = len(raw_feats_list)
    if n_cands == 0:
        return np.empty((0, 28), dtype=np.float64)

    arr = np.zeros((n_cands, 28), dtype=np.float64)

    for i, (cand_dict, rank) in enumerate(zip(raw_feats_list, legacy_ranks)):
        # Group 1: Existing scores (6)
        arr[i, 0] = float(cand_dict["legacy_total_score"])
        arr[i, 1] = float(cand_dict["legacy_name_score"])
        arr[i, 2] = float(cand_dict["legacy_num_score"])
        arr[i, 3] = float(cand_dict["legacy_ton_score"])
        arr[i, 4] = float(cand_dict["legacy_meta_score"])
        arr[i, 5] = float(cand_dict["legacy_dense_score"])

        # Group 2: Existing name info (6)
        arr[i, 6] = float(cand_dict["sim_alpha"])
        arr[i, 7] = float(cand_dict["sim_phon"])
        arr[i, 8] = float(cand_dict["sim_skel"])
        arr[i, 9] = float(cand_dict["sim_tok"])
        arr[i, 10] = float(cand_dict["word_cnt_bonus"])
        arr[i, 11] = float(cand_dict["first_match"])

        # Group 3: Number status (3)
        arr[i, 12] = 1.0 if cand_dict["query_num_present"] else 0.0
        arr[i, 13] = 1.0 if cand_dict["candidate_num_present"] else 0.0
        arr[i, 14] = 1.0 if cand_dict["num_compatible"] else 0.0

        # Group 4: Tonnage (5)
        gt = cand_dict["query_gt"]
        igt = cand_dict["query_igt"]
        c_gt = cand_dict["catalog_gt"]

        arr[i, 15] = 1.0 if (gt is None or gt <= 0.0) else 0.0
        arr[i, 16] = 1.0 if (igt is None or igt <= 0.0) else 0.0
        arr[i, 17] = 1.0 if (c_gt is None or c_gt <= 0.0) else 0.0

        if gt is not None and gt > 0.0 and c_gt is not None and c_gt > 0.0:
            arr[i, 18] = abs(gt - c_gt) / max(1.0, c_gt)
        else:
            arr[i, 18] = np.nan

        if igt is not None and igt > 0.0 and c_gt is not None and c_gt > 0.0:
            arr[i, 19] = abs(igt - c_gt) / max(1.0, c_gt)
        else:
            arr[i, 19] = np.nan

        # Group 5: Meta missing (2)
        arr[i, 20] = 1.0 if cand_dict["flag_missing"] else 0.0
        arr[i, 21] = 1.0 if cand_dict["type_missing"] else 0.0

        # Group 6: Existing Dense (5)
        in_dense = bool(cand_dict.get("in_dense_top35", False))
        arr[i, 22] = 1.0 if in_dense else 0.0
        if in_dense:
            dense_dict = cand_dict.get("dense_feats", {})
            arr[i, 23] = float(dense_dict.get("dense_sim", 0.0))
            arr[i, 24] = float(dense_dict.get("dense_rank_recip", 0.0))
            arr[i, 25] = float(dense_dict.get("dense_zscore", 0.0))
            arr[i, 26] = float(dense_dict.get("dense_margin_top1", 0.0))
        else:
            arr[i, 23] = np.nan
            arr[i, 24] = np.nan
            arr[i, 25] = np.nan
            arr[i, 26] = np.nan

        # Group 7: Existing rank (1)
        arr[i, 27] = 1.0 / (float(rank) + 1.0)

    # Check for infinite values (NaN is allowed, Inf is not)
    if np.isinf(arr).any():
        raise ValueError("Detected Inf or -Inf in feature extraction array")

    return arr
