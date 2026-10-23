import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'model') not in sys.path:
    sys.path.insert(0, str(ROOT / 'model'))

from model.matcher import (
    Matcher, CatalogIndex, make_assets,
    _similarity_cdist, _tonnage_gap,
    CatBoostScorer, EnsembleScorer, load_decision_model
)
from tests.test_team_rules import fixture


class FakeScorer:
    def __init__(self, val):
        self.val = val
        self.feature_names = ['ratio']

    def predict(self, x, num_threads=None):
        return np.full(len(x), self.val, dtype=np.float32)


class MatcherRefactorTests(unittest.TestCase):
    def test_tonnage_gap_exact_values(self):
        gt = np.array([100.0, 200.0, 0.0, -5.0, np.nan], dtype=np.float32)
        # Normal positive tonnage
        gap1 = _tonnage_gap(gt, 100.0)
        self.assertEqual(gap1[0], 0.0)
        self.assertAlmostEqual(gap1[1], 100.0 / 200.0)
        # NaN / inf / zero handling
        gap_nan = _tonnage_gap(gt, np.nan)
        self.assertEqual(gap_nan[4], 2.0)
        gap_zero = _tonnage_gap(gt, 0.0)
        self.assertEqual(gap_zero[2], 0.0)

    def test_similarity_cdist_empty_and_valid(self):
        from rapidfuzz import fuzz
        empty_sim = _similarity_cdist('', ['ALPHA', 'BRAVO'], fuzz.ratio)
        np.testing.assert_array_equal(empty_sim, np.zeros(2, dtype=np.float32))

        exact_sim = _similarity_cdist('ALPHA', ['ALPHA', 'ALPHB'], fuzz.ratio)
        self.assertAlmostEqual(exact_sim[0], 1.0)
        self.assertGreater(exact_sim[1], 0.7)

    def test_ensemble_scorer_weighting_and_empty(self):
        s1 = FakeScorer(0.8)
        s2 = FakeScorer(0.4)
        ens = EnsembleScorer([s1, s2], weights=[0.75, 0.25])
        x = np.zeros((10, 1), dtype=np.float32)
        expected = 0.75 * 0.8 + 0.25 * 0.4
        np.testing.assert_allclose(ens.predict(x), np.full(10, expected, dtype=np.float32))

        # Empty input
        empty_x = np.zeros((0, 1), dtype=np.float32)
        self.assertEqual(len(ens.predict(empty_x)), 0)

    def test_catboost_scorer_predict_shape_and_empty(self):
        class MockCatBoost:
            def predict_proba(self, x, thread_count=4):
                # Return [1 - p, p]
                p = np.full(len(x), 0.7, dtype=np.float32)
                return np.column_stack([1.0 - p, p])

        cb = CatBoostScorer(MockCatBoost())
        x = np.zeros((5, 2), dtype=np.float32)
        pred = cb.predict(x, num_threads=2)
        self.assertEqual(len(pred), 5)
        np.testing.assert_allclose(pred, np.full(5, 0.7, dtype=np.float32))

        # Empty input
        self.assertEqual(len(cb.predict(np.zeros((0, 2), dtype=np.float32))), 0)

    def test_matcher_e2e_with_synthetic_fixture(self):
        train, catalog, _ = fixture()
        assets = make_assets(train, catalog, name_mode='baseline')
        index = CatalogIndex(catalog, assets)
        matcher = Matcher.__new__(Matcher)
        matcher.index = index
        matcher.model = FakeScorer(0.9)
        queries = train.iloc[:4]
        preds = matcher.predict(queries)
        self.assertEqual(len(preds), 4)
        for p in preds:
            self.assertIn(p, catalog.vessel_id.values)


if __name__ == '__main__':
    unittest.main()
