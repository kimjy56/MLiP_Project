"""Behavior contracts for the retrieval refactor; synthetic inputs only."""
import unittest
import numpy as np
from tests.test_team_rules import fixture
from model.matcher import CatalogIndex, make_assets, prepare_row
from phonetic_encoder.retrieval import top_score_indices


class TopScoreContracts(unittest.TestCase):
    def test_native_selection_order_including_ties(self):
        rng = np.random.default_rng(17)
        for dtype in (np.float16, np.float32, np.float64):
            for scores in (rng.normal(size=40).astype(dtype),
                           np.ones(40, dtype=dtype),
                           np.arange(40, dtype=dtype) % 4,
                           np.array([np.inf, -np.inf, np.nan] + [0.] * 37, dtype=dtype)):
                for k in (1, 35, 40, 0, -1):
                    # Literal historical operation is the oracle: adding a stable
                    # sort or ID tie-break would silently change candidate order.
                    indices = np.argpartition(scores, -k)[-k:]
                    expected = indices[np.argsort(-scores[indices])]
                    np.testing.assert_array_equal(top_score_indices(scores, k), expected)

    def test_out_of_range_keeps_native_failure(self):
        with self.assertRaises(ValueError):
            top_score_indices(np.zeros(4), 5)

    def test_scores_not_mutated(self):
        scores = np.array([.2, .9, .4, .1], dtype=np.float32)
        original = scores.copy()
        np.testing.assert_array_equal(top_score_indices(scores, 2), [1, 2])
        np.testing.assert_array_equal(scores, original)


class DenseProbe:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def retrieve_dense_candidates(self, name, top_k):
        self.calls.append((name, top_k))
        return self.values


class DenseMapContracts(unittest.TestCase):
    def setUp(self):
        self.train, self.catalog, _ = fixture()

    def index(self, dense):
        assets = make_assets(self.train, self.catalog, retrieval_channels=[],
                             include_dense=True, use_train_lookup=False)
        return CatalogIndex(self.catalog, assets, dense_matcher=dense)

    def test_duplicate_unknown_ids_and_cached_features(self):
        last = {"vessel_id": "002", "dense_sim": .75, "dense_zscore": 1.}
        dense = DenseProbe([{"vessel_id":"002", "dense_sim":.5},
                            {"vessel_id":"NOT_IN_CATALOG", "dense_sim":1.}, last])
        index = self.index(dense)
        row = prepare_row(self.train.iloc[0].to_dict())
        self.assertEqual(index.candidates(row), [2])
        self.assertIs(row["_dense_map"][2], last)
        values = index.features(row, [2, 0])
        np.testing.assert_array_equal(values[:, -4:],
                                      np.array([[.75, 0., 1., 0.], [0., 0., 0., 0.]], dtype=np.float32))
        self.assertEqual(len(dense.calls), 1)

    def test_empty_retrieval_is_retried_and_fallback_preserved(self):
        dense = DenseProbe([])
        index = self.index(dense)
        row = prepare_row(self.train.iloc[0].to_dict())
        self.assertEqual(index.candidates(row), [0])
        self.assertEqual(row["_dense_map"], {})
        index.features(row, [0])
        self.assertEqual(len(dense.calls), 2)

    def test_blank_and_non_string_names_keep_existing_coercion(self):
        dense = DenseProbe([])
        index = self.index(dense)
        for name in ("", "   "):
            row = prepare_row({**self.train.iloc[0].to_dict(), "vessel_name":name})
            self.assertEqual(index.candidates(row), [0])
            self.assertNotIn("_dense_map", row)
            index.features(row, [0])
        self.assertEqual(dense.calls, [])
        row = prepare_row({**self.train.iloc[0].to_dict(), "vessel_name":None})
        index.candidates(row)
        # This coercion is historical, even though normalization treats None as empty.
        self.assertEqual(dense.calls, [("None", 35)])

    def test_features_without_candidate_call_still_retrieve(self):
        dense = DenseProbe([{"vessel_id":"001", "dense_sim":.25}])
        index = self.index(dense)
        row = prepare_row(self.train.iloc[0].to_dict())
        np.testing.assert_array_equal(index.features(row, [1])[:, -4:],
                                      np.array([[.25, 0., 0., 0.]], dtype=np.float32))
        self.assertEqual(len(dense.calls), 1)


if __name__ == "__main__":
    unittest.main()
