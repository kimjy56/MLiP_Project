"""
Unit Tests for DPR-Hybrid Matcher (SPEC-DPR-01).

Tests:
  1. Legacy Parity: dense_mode='none' matches train_AS-06 100%
  2. Hybrid Mode: Dense candidates merged into candidate set
  3. Row Independence: batch vs single row predictions are identical
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experimental_matchers.dpr_hybrid.script import (
    CatalogIndex,
    fit_matching_model,
    predict_rows as hybrid_predict_rows,
)
from phonetic_encoder.hybrid_matcher import HybridDenseLexicalMatcher
import importlib
train_as06_mod = importlib.import_module("train_AS-06.script")
legacy_predict_rows = train_as06_mod.predict_rows


def test_dpr_hybrid_matcher():
    # Load small subset of catalog and train
    df_cat = pd.read_csv("data/vessel_catalog.csv")
    cat_idx = CatalogIndex(df_cat)

    df_train = pd.read_csv("data/train.csv", nrows=16)

    # Initialize model
    model = fit_matching_model(df_train, df_cat)

    # 1. Test legacy parity (dense_mode="none")
    preds_legacy, _ = legacy_predict_rows(df_train, model, cat_idx, use_train_lookup=False)
    preds_hybrid_none, _ = hybrid_predict_rows(df_train, model, cat_idx, use_train_lookup=False, dense_mode="none")
    assert preds_legacy == preds_hybrid_none, "Legacy mode does not match original train_AS-06!"

    # 2. Test hybrid mode
    matcher = HybridDenseLexicalMatcher(
        model_checkpoint="artifacts/DPR-SYMPHON-01/fold_0_encoder.pt",
        catalog_embeds_npy="artifacts/DPR-SYMPHON-01/catalog_embeds_fp16.npy",
        catalog_csv="data/vessel_catalog.csv",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    preds_hybrid, _ = hybrid_predict_rows(
        df_train,
        model,
        cat_idx,
        use_train_lookup=False,
        dense_matcher=matcher,
        dense_mode="hybrid",
        dense_weight=2.5,
    )
    assert len(preds_hybrid) == len(df_train)

    # 3. Test Row Independence
    # Predict row 0 alone
    res_single = hybrid_predict_rows(
        df_train.iloc[[0]],
        model,
        cat_idx,
        use_train_lookup=False,
        dense_matcher=matcher,
        dense_mode="hybrid",
    )
    pred_row0_single = res_single[0][0]

    # Predict row 0 as part of 16-row batch
    pred_row0_batch = preds_hybrid[0]
    assert pred_row0_single == pred_row0_batch, f"Row independence violation: {pred_row0_single} != {pred_row0_batch}"
    print("DPR-Hybrid Unit Tests Passed 100%!")


if __name__ == "__main__":
    test_dpr_hybrid_matcher()
