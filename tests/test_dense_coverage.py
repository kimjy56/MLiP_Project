"""Synthetic contracts for the isolated Dense coverage comparison."""
from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from experimental_matchers.dense_coverage.comparison import (
    SIM_COLUMN, TOP35_COLUMN, evaluate_ranker_pair, trace_pair,
    train_ranker_pair, validate_pair,
)
from experimental_matchers.ko_fusion.as09_adapter import CandidateTrace


class FakeEncoder:
    def encode_queries(self, names, batch_size=1):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class FakeDense:
    def __init__(self):
        self.catalog_ids = [f"v{i:03d}" for i in range(40)]
        self.catalog_embeds = np.column_stack((np.arange(40, 0, -1), np.zeros(40))).astype(np.float32)
        self.model = FakeEncoder()

    def retrieve_dense_candidates(self, query_name, top_k=35):
        scores = self.catalog_embeds @ self.model.encode_queries([query_name])[0]
        indices = np.argsort(-scores)[:top_k]
        return [{"vessel_id": self.catalog_ids[i], "dense_sim": float(scores[i])} for i in indices]


class FakeAdapter:
    def __init__(self):
        self.dense_matcher = FakeDense()
        self.dense_mode = "hybrid"

    def trace_df(self, df):
        if "target_vessel_id" in df.columns or "fold" in df.columns:
            raise AssertionError("labels must not reach retrieval")
        row = df.iloc[0]
        if row["vessel_name"] == "조회":
            return [CandidateTrace(str(row.sample_id), "lookup", "v039", (), np.empty(0),
                                   np.empty((0, 28)), "", "old", "asset")]
        top = self.dense_matcher.retrieve_dense_candidates(row.vessel_name, top_k=35)
        top_scores = {r["vessel_id"]: r["dense_sim"] for r in top}
        ids = ("v000", "v034", "v035", "v039")
        features = np.zeros((4, 28), dtype=np.float64)
        for i, vid in enumerate(ids):
            features[i, TOP35_COLUMN] = vid in top_scores
            features[i, SIM_COLUMN] = top_scores.get(vid, np.nan)
        return [CandidateTrace(str(row.sample_id), "hangul_candidates", "v000", ids,
                               np.array([4., 3., 2., 1.]), features, "candidate-hash", "old", "asset")]


class ConstantRanker:
    def predict(self, values):
        return np.zeros(len(values))


class DenseCoverageTests(unittest.TestCase):
    def test_only_non_top35_similarity_changes_and_labels_are_excluded(self):
        adapter = FakeAdapter()
        original_backend = adapter.dense_matcher
        query = pd.DataFrame([{"sample_id": "q1", "vessel_name": "선박", "target_vessel_id": "ignored", "fold": 4}])
        pair = trace_pair(adapter, query)
        validate_pair(pair)
        self.assertIs(adapter.dense_matcher, original_backend)
        self.assertEqual(pair.restricted.candidate_ids, pair.expanded.candidate_ids)
        self.assertTrue(np.isnan(pair.restricted.features[2:, SIM_COLUMN]).all())
        np.testing.assert_array_equal(pair.expanded.features[:, SIM_COLUMN], [40., 6., 5., 1.])
        self.assertNotEqual(pair.expanded.feature_schema_sha256, pair.restricted.feature_schema_sha256)

    def test_no_dense_backend_refuses_silent_fallback(self):
        adapter = FakeAdapter()
        adapter.dense_matcher = None
        with self.assertRaisesRegex(RuntimeError, "required"):
            trace_pair(adapter, pd.DataFrame([{"sample_id": "q", "vessel_name": "선박"}]))

    def test_language_is_input_name_not_lookup_route(self):
        adapter = FakeAdapter()
        lookup = trace_pair(adapter, pd.DataFrame([{"sample_id": "ko", "vessel_name": "조회"}]))
        en_trace = replace(lookup.restricted, sample_id="en", route="non_hangul")
        en_pair = replace(lookup, restricted=en_trace,
                          expanded=replace(lookup.expanded, sample_id="en", route="non_hangul"))
        metrics = evaluate_ranker_pair([lookup, en_pair], {"ko": "v039", "en": "v039"},
                                       {"ko": "조회", "en": "STAR"}, ConstantRanker(), ConstantRanker())
        self.assertEqual(metrics["hangul"]["rows"], 1)
        self.assertEqual(metrics["non_hangul"]["rows"], 1)
        self.assertEqual(metrics["expanded_weighted_accuracy"], 1.0)

    def test_real_rankers_train_on_identical_candidates(self):
        adapter = FakeAdapter()
        pairs = [trace_pair(adapter, pd.DataFrame([{"sample_id": f"q{i}", "vessel_name": "선박"}]))
                 for i in range(8)]
        labels = {f"q{i}": ("v000" if i % 2 else "v039") for i in range(8)}
        a, b, info = train_ranker_pair(pairs, labels, n_estimators=2)
        self.assertEqual(info, {"queries": 8, "pairs": 32, "excluded_queries": 0})
        self.assertEqual(len(a.predict(pairs[0].restricted.features)), 4)
        self.assertEqual(len(b.predict(pairs[0].expanded.features)), 4)


if __name__ == "__main__":
    unittest.main()
