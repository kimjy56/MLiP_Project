"""Experimental fixed-catalog character n-gram channel for non-Korean names."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from experimental_matchers.as09_candidate_extension import as09, predict_row
from experimental_matchers.lookup_conflict import _lookup


class CatalogNgramIndex:
    def __init__(self, catalog_index):
        self.ids = tuple(str(value) for value in catalog_index.vessel_ids)
        if not self.ids or len(self.ids) != len(set(self.ids)):
            raise ValueError("Catalog IDs must be nonempty and unique")
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                          lowercase=False, sublinear_tf=True,
                                          norm="l2", dtype=np.float32)
        docs = list(catalog_index.compact_names)
        self.transpose = self.vectorizer.fit_transform(docs).T.tocsr() if any(len(s) >= 2 for s in docs) else None

    def search(self, name, count=20):
        if count <= 0 or self.transpose is None:
            return ()
        compact = as09.normalize_vessel_name(name, kr_mode="AR02")[0]
        if len(compact) < 2:
            return ()
        scores = (self.vectorizer.transform([compact]) @ self.transpose).tocsr()
        positive = scores.data > 0
        positions, values = scores.indices[positive], scores.data[positive]
        ids = np.asarray([self.ids[i] for i in positions])
        order = np.lexsort((ids, -values))[:count]
        return tuple(self.ids[positions[i]] for i in order)


@dataclass(frozen=True)
class NgramComparison:
    baseline_prediction: str
    expanded_prediction: str
    ngram_candidates: tuple[str, ...]
    baseline_candidates: tuple[tuple[float, str], ...]
    expanded_candidates: tuple[tuple[float, str], ...]
    route: str


def compare_row(query: pd.DataFrame, model, catalog_index, ngram_index, *, count=20,
                dense_matcher=None, dense_mode=None, dense_weight=2.5):
    if len(query) != 1 or "vessel_name" not in query:
        raise ValueError("Expected exactly one query")
    if tuple(catalog_index.vessel_ids) != ngram_index.ids:
        raise ValueError("N-gram index and scoring catalog order differ")
    kwargs = dict(dense_matcher=dense_matcher, dense_mode=dense_mode,
                  dense_weight=dense_weight, diagnostic=True)
    original, baseline = predict_row(query, model, catalog_index, **kwargs)
    if as09.is_korean_name(query.iloc[0].vessel_name):
        return NgramComparison(original, original, (), baseline, baseline, "korean_bypass")
    lookup_on = bool(model.get("use_train_lookup_default", as09.USE_TRAIN_LOOKUP))
    lookup_id, _ = _lookup(query.iloc[0], model, catalog_index) if lookup_on else (None, "disabled")
    if lookup_id:
        return NgramComparison(original, original, (), baseline, baseline, "lookup_bypass")
    extra = ngram_index.search(query.iloc[0].vessel_name, count)
    extended, candidates = predict_row(query, model, catalog_index, candidates=extra, **kwargs)
    base_ids = {vid for _, vid in baseline}
    extended_ids = {vid for _, vid in candidates}
    if not base_ids <= extended_ids:
        raise RuntimeError("Candidate union dropped an existing candidate")
    return NgramComparison(original, extended, extra, baseline, candidates, "candidates")
