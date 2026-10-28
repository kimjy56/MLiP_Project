# -*- coding: utf-8 -*-
"""tests/test_dense_features.py — Unit tests for Stage-1 Dense Union and Stage-2 Dense Features in Matcher."""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from model.matcher import (
    CatalogIndex,
    make_assets,
    prepare_row,
    feature_names,
    FEATURES,
    DENSE_FEATURES,
)
from phonetic_encoder.hybrid_matcher import HybridDenseLexicalMatcher


class TestDenseFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df_cat = pd.read_csv(ROOT / "data/vessel_catalog.csv")
        df_train = pd.read_csv(ROOT / "data/train.csv")
        # Ensure train rows match catalog
        cls.train = df_train.head(16)
        cat_subset = df_cat[df_cat["vessel_id"].isin(cls.train["target_vessel_id"].tolist())]
        cls.catalog = pd.concat([cat_subset, df_cat.head(100)]).drop_duplicates("vessel_id").reset_index(drop=True)
        
        # Load real lightweight Bi-Encoder checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.dense_matcher = HybridDenseLexicalMatcher(
            model_checkpoint=str(ROOT / "artifacts/DPR-SYMPHON-01/fold_0_encoder.pt"),
            catalog_embeds_npy=str(ROOT / "artifacts/DPR-SYMPHON-01/catalog_embeds_fp16.npy"),
            catalog_csv=str(ROOT / "data/vessel_catalog.csv"),
            device=device,
        )

    def test_feature_names(self):
        f_base = feature_names('baseline', include_dense=False)
        self.assertEqual(len(f_base), len(FEATURES))
        self.assertNotIn('dense_sim', f_base)

        f_dense = feature_names('baseline', include_dense=True)
        self.assertEqual(len(f_dense), len(FEATURES) + 4)
        for df in DENSE_FEATURES:
            self.assertIn(df, f_dense)

    def test_catalog_index_with_dense(self):
        assets = make_assets(self.train, self.catalog, name_mode='baseline', include_dense=True)
        self.assertTrue(assets['include_dense'])
        self.assertEqual(assets['features'], feature_names('baseline', include_dense=True))

        index = CatalogIndex(self.catalog, assets, dense_matcher=self.dense_matcher)
        self.assertTrue(index.include_dense)
        self.assertEqual(index.feature_names, feature_names('baseline', include_dense=True))

        row = self.train.iloc[0].to_dict()
        prep_r = prepare_row(row)
        cands = index.candidates(prep_r)
        self.assertGreater(len(cands), 0)

        feats = index.features(prep_r, cands)
        self.assertEqual(feats.shape[0], len(cands))
        self.assertEqual(feats.shape[1], len(index.feature_names))

        # Check dense feature values range
        dense_idx_start = len(FEATURES)
        sim_col = feats[:, dense_idx_start]
        recip_col = feats[:, dense_idx_start + 1]
        zscore_col = feats[:, dense_idx_start + 2]
        margin_col = feats[:, dense_idx_start + 3]

        self.assertTrue(np.all(sim_col >= -1.0) and np.all(sim_col <= 1.0))
        self.assertTrue(np.all(recip_col >= 0.0) and np.all(recip_col <= 1.0))
        print("TestDenseFeatures: All assertions passed successfully!")


if __name__ == "__main__":
    unittest.main()
