"""Synthetic correctness contracts for bounded execution, not accuracy evaluation."""
import os
import unittest
from unittest.mock import patch
import numpy as np
from tests.test_team_rules import fixture
from matcher import CatalogIndex, Matcher, make_assets
from torch_indexer import TorchNgramIndexer


class Performance(unittest.TestCase):
    def test_sparse_ties_and_oversized_k(self):
        index = TorchNgramIndexer(['SAME NAME'] * 8)
        self.assertEqual(index.search('SAME NAME', 3), [0, 1, 2])
        self.assertEqual(index.batch_search(['SAME NAME', '', None], 99), [list(range(8)), [], []])
        self.assertIsNone(index.catalog_tensor)
        self.assertLess(index.memory_info['csr_bytes'], 10000)

    def test_algorithms_explicit_and_no_inactive_index(self):
        train, catalog, _ = fixture()
        assets = make_assets(train, catalog, retrieval_channels=['T'])
        assets['search_algorithm'] = 'tfidf'
        index = CatalogIndex(catalog, assets)
        self.assertIsNone(index.torch_names)
        self.assertIsNone(index.torch_romans)
        assets.pop('search_algorithm')
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            CatalogIndex(catalog, assets)

    def test_real_booster_batch_threads_and_row_independence(self):
        import lightgbm as lgb
        train, catalog, _ = fixture()
        catalog['vessel_name'] += ' STAR'
        train['vessel_name'] += ' STAR'
        rows = train.loc[train.index.repeat(4)].reset_index(drop=True)
        for algorithm in ['rapidfuzz', 'tfidf']:
            with self.subTest(algorithm=algorithm):
                assets = make_assets(train, catalog, name_mode='all', phonetic_search=3)
                assets['search_algorithm'] = algorithm
                index = CatalogIndex(catalog, assets, use_train_lookup=False)
                # Real small booster exercises LightGBM's batch path.
                x = np.random.default_rng(1).normal(size=(60, len(index.feature_names)))
                model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=1, verbosity=-1, n_jobs=1)
                model.fit(x, (x[:, 1] > 0).astype(int), feature_name=index.feature_names)
                matcher = Matcher.__new__(Matcher)
                matcher.index, matcher.model = index, model.booster_
                with patch.dict(os.environ, {'VESSEL_PREDICT_BATCH': '1', 'VESSEL_PREDICT_WORKERS': '1'}):
                    expected = matcher.predict_details(rows)
                with patch.dict(os.environ, {'VESSEL_PREDICT_BATCH': '32', 'VESSEL_PREDICT_WORKERS': '2'}):
                    self.assertEqual(matcher.predict_details(rows), expected)
                    self.assertEqual(matcher.predict(rows.iloc[::-1]), expected[0][::-1])
                    self.assertEqual(matcher.predict(rows.iloc[[3]]), [expected[0][3]])
