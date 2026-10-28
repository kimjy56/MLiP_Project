# -*- coding: utf-8 -*-
"""Unit tests for KO-FUSION-01 (SPEC-KO-FUSION-01 Section 9.1)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

# Path setup
_repo_root = Path(__file__).resolve().parents[1]
_as09_dir = _repo_root / "train_AS-09"
if str(_as09_dir) not in sys.path:
    sys.path.insert(0, str(_as09_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from experimental_matchers.ko_fusion.as09_adapter import (
    AS09RerankAdapter,
    CandidateTrace,
    as09,
)
from experimental_matchers.ko_fusion.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    compute_feature_schema_sha256,
    extract_candidate_features,
)
from experimental_matchers.ko_fusion.cache import (
    CandidateCacheWriter,
    CandidateCacheReader,
    label_trace,
)
from experimental_matchers.ko_fusion.ranker import (
    LGBMReranker,
    SEARCH_GRID,
)


class TestKOFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = AS09RerankAdapter(device="cpu")
        cls.data_df = pd.read_csv(_repo_root / "data" / "train.csv", nrows=30)

    def test_feature_schema_integrity(self):
        """Verify feature count, ordering, and schema hash determinism."""
        self.assertEqual(len(FEATURE_NAMES), 28)
        self.assertEqual(FEATURE_NAMES[0], "legacy_total_score")
        self.assertEqual(FEATURE_NAMES[5], "legacy_dense_score")
        self.assertEqual(FEATURE_NAMES[6], "sim_alpha")
        self.assertEqual(FEATURE_NAMES[12], "query_num_present")
        self.assertEqual(FEATURE_NAMES[15], "gt_missing")
        self.assertEqual(FEATURE_NAMES[20], "flag_missing")
        self.assertEqual(FEATURE_NAMES[22], "in_dense_top35")
        self.assertEqual(FEATURE_NAMES[27], "legacy_rank_recip")

        h1 = compute_feature_schema_sha256()
        h2 = compute_feature_schema_sha256()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_baseline_parity_exact(self):
        """Verify that when ranker is inactive, adapter produces 100% exact baseline IDs."""
        original_preds, _ = as09.predict_rows(
            self.data_df,
            self.adapter.model,
            self.adapter.cat_idx,
            use_train_lookup=self.adapter.use_train_lookup,
            dense_matcher=self.adapter.dense_matcher,
            dense_mode=self.adapter.dense_mode,
            dense_weight=self.adapter.dense_weight,
        )

        traces = self.adapter.trace_df(self.data_df)
        self.assertEqual(len(traces), len(original_preds))

        adapter_preds = [
            self.adapter.select_with_ranker(t, ranker=None)
            for t in traces
        ]

        for i, (orig, adap) in enumerate(zip(original_preds, adapter_preds)):
            self.assertEqual(
                orig, adap,
                f"Row {i} ({self.data_df.iloc[i]['vessel_name']}) mismatch: orig={orig}, adap={adap}"
            )

    def test_routes_and_trace_shapes(self):
        """Verify route properties: non_hangul & lookup have empty features; hangul has (N, 28)."""
        traces = self.adapter.trace_df(self.data_df)
        for t in traces:
            if t.route in ("non_hangul", "lookup"):
                self.assertEqual(len(t.candidate_ids), 0)
                self.assertEqual(t.features.shape, (0, 28))
                self.assertEqual(len(t.legacy_scores), 0)
            elif t.route == "hangul_candidates":
                n = len(t.candidate_ids)
                self.assertGreater(n, 0)
                self.assertEqual(t.features.shape, (n, 28))
                self.assertEqual(t.legacy_scores.shape, (n,))
                self.assertFalse(np.isinf(t.features).any())

    def test_gold_id_isolation(self):
        """Verify trace content does not depend on whether gold ID is supplied."""
        t1 = self.adapter.trace_df(self.data_df.iloc[[0]])[0]
        t2 = self.adapter.trace_df(self.data_df.iloc[[0]], gold_ids=["VESSEL_NONEXISTENT"])[0]

        self.assertEqual(t1.baseline_pred_vessel_id, t2.baseline_pred_vessel_id)
        self.assertEqual(t1.candidate_ids, t2.candidate_ids)
        self.assertTrue(np.allclose(t1.legacy_scores, t2.legacy_scores, atol=1e-10))
        self.assertEqual(t1.candidate_set_sha256, t2.candidate_set_sha256)

    def test_cache_serialization_roundtrip(self):
        """Verify CandidateCacheWriter and CandidateCacheReader roundtrip and assertions."""
        traces = self.adapter.trace_df(self.data_df)
        golds = self.data_df["target_vessel_id"].tolist() if "target_vessel_id" in self.data_df.columns else [None] * len(traces)
        lqs = [label_trace(t, g) for t, g in zip(traces, golds)]

        out_tmp = _repo_root / "scratch" / "test_cache_tmp"
        manifest = CandidateCacheWriter.write_cache(
            out_tmp,
            lqs,
            asset_manifest_sha256=self.adapter.asset_manifest_sha256,
            feature_schema_sha256=self.adapter.feature_schema_sha256,
            filter_trainable_only=False,
        )

        X, legacy_scores, labels, group_sizes, queries_meta, loaded_manifest = CandidateCacheReader.load_cache(out_tmp)
        self.assertEqual(len(X), sum(group_sizes))
        self.assertEqual(manifest["total_candidates"], len(X))

    def test_ranker_fit_predict(self):
        """Verify LGBMRanker fits and predicts on dummy (N, 28) data without errors."""
        X_dummy = np.random.randn(20, 28).astype(np.float64)
        y_dummy = np.zeros(20, dtype=np.int32)
        y_dummy[0] = 1
        y_dummy[10] = 1
        groups = np.array([10, 10], dtype=np.int32)

        ranker = LGBMReranker(n_estimators=10)
        ranker.fit(X_dummy, y_dummy, groups)

        preds = ranker.predict(X_dummy)
        self.assertEqual(preds.shape, (20,))
        self.assertTrue(np.isfinite(preds).all())


if __name__ == "__main__":
    unittest.main()
