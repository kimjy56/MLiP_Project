# -*- coding: utf-8 -*-
"""LightGBM 4.6.0 LambdaRank model wrapper for KO-FUSION-01."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import lightgbm as lgb
import numpy as np


BASE_RANKER_PARAMS: Dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "label_gain": [0, 1],
    "lambdarank_truncation_level": 5,
    "lambdarank_norm": True,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 20,
    "max_bin": 63,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 6,
    "verbosity": -1,
}

SEARCH_GRID: List[Dict[str, Any]] = [
    {"reg_lambda": 10.0, "n_estimators": 32},
    {"reg_lambda": 10.0, "n_estimators": 64},
    {"reg_lambda": 10.0, "n_estimators": 128},
    {"reg_lambda": 30.0, "n_estimators": 32},
    {"reg_lambda": 30.0, "n_estimators": 64},
    {"reg_lambda": 30.0, "n_estimators": 128},
]


class LGBMReranker:
    """Wrapper around lightgbm.LGBMRanker for candidate scoring."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, n_estimators: int = 64):
        self.params = dict(BASE_RANKER_PARAMS)
        if params is not None:
            self.params.update(params)
        self.n_estimators = n_estimators
        self.model: Optional[lgb.LGBMRanker] = None
        self.booster: Optional[lgb.Booster] = None

    def fit(self, X: np.ndarray, y: np.ndarray, group_sizes: np.ndarray) -> LGBMReranker:
        """Fit LGBMRanker with group sizes."""
        if len(X) == 0:
            raise ValueError("Cannot fit LGBMRanker on empty X")

        self.model = lgb.LGBMRanker(
            **self.params,
            n_estimators=self.n_estimators,
        )
        self.model.fit(
            X,
            y,
            group=group_sizes,
        )
        self.booster = self.model.booster_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Output raw ranker score for candidate feature matrix (N, 28)."""
        if self.booster is None:
            if self.model is not None and getattr(self.model, "booster_", None) is not None:
                self.booster = self.model.booster_
            else:
                raise RuntimeError("LGBMReranker is not fitted")
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        return self.booster.predict(X).astype(np.float64)

    def save_model(self, model_file: Union[str, Path]) -> None:
        """Save booster model to text file."""
        if self.booster is None:
            if self.model is not None and getattr(self.model, "booster_", None) is not None:
                self.booster = self.model.booster_
            else:
                raise RuntimeError("Cannot save unfitted model")
        self.booster.save_model(str(model_file))

    @classmethod
    def load_model(cls, model_file: Union[str, Path]) -> LGBMReranker:
        """Load booster model from text file."""
        booster = lgb.Booster(model_file=str(model_file))
        instance = cls()
        instance.booster = booster
        return instance
