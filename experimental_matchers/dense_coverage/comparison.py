"""Compare Dense coverage on identical AS-09 candidate traces.

This module is experimental and is not imported by the submitted entry point.
It deliberately refuses to replace missing Dense assets with a lexical-only path.
Fold-isolated encoder/map provenance must be established by the caller before
using these functions for a performance experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experimental_matchers.ko_fusion.as09_adapter import AS09RerankAdapter, CandidateTrace
from experimental_matchers.ko_fusion.features import FEATURE_NAMES, get_feature_schema_dict
from experimental_matchers.ko_fusion.ranker import LGBMReranker

TOP35_COLUMN = FEATURE_NAMES.index("in_dense_top35")
SIM_COLUMN = FEATURE_NAMES.index("dense_sim")
QUERY_COLUMNS = ("sample_id", "vessel_name", "call_sign", "gross_tonnage",
                 "international_gross_tonnage", "flag_name", "vessel_type", "arrival_year")


def full_coverage_schema_sha256() -> str:
    schema = get_feature_schema_dict()
    schema["version"] = "dense_coverage_v1"
    schema["missing_rules"] = dict(schema["missing_rules"])
    schema["missing_rules"]["dense_feats"] = (
        "dense_sim from all-catalog dot product for every candidate; "
        "rank_recip/zscore/margin remain NaN outside Top-35"
    )
    return hashlib.sha256(json.dumps(schema, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TracePair:
    restricted: CandidateTrace
    expanded: CandidateTrace


class DenseScoreCapture:
    """Wrap the unchanged retrieval method and retain scores for this row only.

    The original retriever currently does not expose its score vector, so query
    encoding is repeated once. A mismatch with its Top-35 scores is an error.
    """

    def __init__(self, backend: Any):
        self.backend = backend
        ids = tuple(str(value) for value in backend.catalog_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Dense catalog IDs must be nonempty and unique")
        if len(ids) != len(backend.catalog_embeds):
            raise ValueError("Dense catalog ID and embedding counts differ")
        self.ids = ids
        self.position = {vid: i for i, vid in enumerate(ids)}
        self.scores: np.ndarray | None = None

    def reset(self) -> None:
        self.scores = None

    def retrieve_dense_candidates(self, query_name: str, top_k: int = 35):
        self.reset()
        top = self.backend.retrieve_dense_candidates(query_name, top_k=top_k)
        if not query_name:
            return top
        embedding = np.asarray(self.backend.model.encode_queries([query_name], batch_size=1))
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError("Dense query encoder returned an invalid shape")
        scores = np.asarray(np.dot(self.backend.catalog_embeds, embedding[0]), dtype=np.float64)
        if scores.shape != (len(self.ids),) or not np.isfinite(scores).all():
            raise ValueError("Invalid all-catalog Dense score vector")
        for candidate in top:
            vid = str(candidate["vessel_id"])
            if vid not in self.position or not np.isclose(
                scores[self.position[vid]], float(candidate["dense_sim"]), rtol=1e-5, atol=1e-6
            ):
                raise ValueError("Repeated Dense encoding disagrees with the original Top-35")
        self.scores = scores
        return top

    def __getattr__(self, name: str):
        return getattr(self.backend, name)


def expand_trace(trace: CandidateTrace, capture: DenseScoreCapture) -> TracePair:
    """Fill only existing candidates' previously missing Dense cosine values."""
    if trace.route != "hangul_candidates":
        return TracePair(trace, replace(trace, feature_schema_sha256=full_coverage_schema_sha256()))
    if capture.scores is None:
        raise RuntimeError("No Dense scores for this row; refusing a lexical-only comparison")
    if trace.features.shape != (len(trace.candidate_ids), len(FEATURE_NAMES)):
        raise ValueError("Unexpected candidate feature schema")
    expanded = trace.features.copy()
    for i, vid in enumerate(trace.candidate_ids):
        position = capture.position.get(str(vid))
        if position is None:
            raise ValueError("Candidate is absent from the Dense catalog")
        if bool(expanded[i, TOP35_COLUMN]):
            if not np.isclose(expanded[i, SIM_COLUMN], capture.scores[position], rtol=1e-5, atol=1e-6):
                raise ValueError("Trace and original Top-35 score differ")
        else:
            if not np.isnan(expanded[i, SIM_COLUMN]):
                raise ValueError("Non-Top-35 restricted similarity must be missing")
            expanded[i, SIM_COLUMN] = capture.scores[position]
    return TracePair(trace, replace(trace, features=expanded,
                                     feature_schema_sha256=full_coverage_schema_sha256()))


class DenseCoverageTracer:
    """Reuse the fixed catalog ID index; do not share one instance across threads."""

    def __init__(self, adapter: AS09RerankAdapter):
        if adapter.dense_matcher is None or adapter.dense_mode != "hybrid":
            raise RuntimeError("Dense encoder and catalog embeddings are required")
        self.adapter = adapter
        self.backend = adapter.dense_matcher
        self.capture = DenseScoreCapture(self.backend)

    def trace_pair(self, query: pd.DataFrame) -> TracePair:
        """Trace a single label-free record; preserve baseline retrieval and ordering."""
        if len(query) != 1 or "sample_id" not in query or "vessel_name" not in query:
            raise ValueError("Pass exactly one query with sample_id and vessel_name")
        if self.adapter.dense_matcher is not self.backend:
            raise RuntimeError("Adapter Dense backend changed during the experiment")
        query = query.loc[:, [c for c in QUERY_COLUMNS if c in query]].copy()
        self.capture.reset()
        self.adapter.dense_matcher = self.capture
        try:
            trace = self.adapter.trace_df(query)[0]
            return expand_trace(trace, self.capture)
        finally:
            self.adapter.dense_matcher = self.backend
            self.capture.reset()


def trace_pair(adapter: AS09RerankAdapter, query: pd.DataFrame) -> TracePair:
    """Convenience wrapper; use DenseCoverageTracer for multiple rows."""
    return DenseCoverageTracer(adapter).trace_pair(query)


def validate_pair(pair: TracePair) -> None:
    a, b = pair.restricted, pair.expanded
    if (a.sample_id != b.sample_id or a.route != b.route or
        a.candidate_ids != b.candidate_ids or a.candidate_set_sha256 != b.candidate_set_sha256 or
        a.baseline_pred_vessel_id != b.baseline_pred_vessel_id or
        not np.array_equal(a.legacy_scores, b.legacy_scores)):
        raise ValueError("Comparison changed candidates, order, route or baseline scores")
    if a.features.shape != b.features.shape:
        raise ValueError("Comparison changed feature shape")
    others = [i for i in range(len(FEATURE_NAMES)) if i != SIM_COLUMN]
    if not np.array_equal(a.features[:, others], b.features[:, others], equal_nan=True):
        raise ValueError("Comparison changed another feature")
    if a.route == "hangul_candidates":
        top = a.features[:, TOP35_COLUMN].astype(bool)
        if not np.array_equal(a.features[top, SIM_COLUMN], b.features[top, SIM_COLUMN], equal_nan=True):
            raise ValueError("Comparison changed a Top-35 Dense score")
        if not np.isfinite(b.features[:, SIM_COLUMN]).all():
            raise ValueError("Expanded Dense scores must be finite")


def train_ranker_pair(pairs: Sequence[TracePair], truth: Mapping[str, str],
                      *, n_estimators: int = 32, reg_lambda: float = 30.0):
    """Fit two rankers on identical queries; caller must enforce split provenance.

    Truth is joined only after the traces exist. Missing positives are excluded
    from training, but must still be retained by the caller for evaluation.
    """
    rows_a, rows_b, ys, groups = [], [], [], []
    excluded = 0
    for pair in pairs:
        validate_pair(pair)
        t = pair.restricted
        if t.sample_id not in truth:
            raise ValueError("Missing training target")
        if t.route != "hangul_candidates" or len(t.candidate_ids) < 2:
            excluded += 1
            continue
        labels = np.asarray([int(v == truth[t.sample_id]) for v in t.candidate_ids], dtype=np.int32)
        if labels.sum() != 1:
            excluded += 1
            continue
        if len(labels) > 10000:
            # The same pre-existing score-based truncation is used for both models.
            order = np.lexsort((np.asarray(t.candidate_ids), -t.legacy_scores))[:10000]
            positive = int(np.flatnonzero(labels)[0])
            if positive not in order:
                order[-1] = positive
            keep = np.sort(order)
        else:
            keep = np.arange(len(labels))
        rows_a.append(t.features[keep])
        rows_b.append(pair.expanded.features[keep])
        ys.append(labels[keep])
        groups.append(len(keep))
    if not groups:
        raise ValueError("No labeled training queries with a retrieved positive")
    labels = np.concatenate(ys)
    group_sizes = np.asarray(groups, dtype=np.int32)
    first = LGBMReranker(params={"reg_lambda": reg_lambda}, n_estimators=n_estimators)
    second = LGBMReranker(params={"reg_lambda": reg_lambda}, n_estimators=n_estimators)
    first.fit(np.concatenate(rows_a), labels, group_sizes)
    second.fit(np.concatenate(rows_b), labels, group_sizes)
    return first, second, {"queries": len(groups), "pairs": len(labels), "excluded_queries": excluded}


def evaluate_ranker_pair(pairs: Sequence[TracePair], truth: Mapping[str, str],
                         names: Mapping[str, str], restricted_ranker: Any, expanded_ranker: Any):
    """Return aggregate paired metrics; derive language from the original name."""
    counts = {key: {"rows": 0, "restricted_correct": 0, "expanded_correct": 0,
                    "fixes": 0, "breaks": 0} for key in ("hangul", "non_hangul")}
    seen = set()
    for pair in pairs:
        validate_pair(pair)
        t = pair.restricted
        if t.sample_id in seen or t.sample_id not in truth or t.sample_id not in names:
            raise ValueError("Missing or duplicate query metadata")
        seen.add(t.sample_id)
        a = AS09RerankAdapter.select_with_ranker(t, restricted_ranker)
        b = AS09RerankAdapter.select_with_ranker(pair.expanded, expanded_ranker)
        name = names[t.sample_id]
        key = "hangul" if isinstance(name, str) and any("가" <= c <= "힣" for c in name) else "non_hangul"
        group = counts[key]
        correct_a, correct_b = a == truth[t.sample_id], b == truth[t.sample_id]
        group["rows"] += 1
        group["restricted_correct"] += int(correct_a)
        group["expanded_correct"] += int(correct_b)
        group["fixes"] += int(not correct_a and correct_b)
        group["breaks"] += int(correct_a and not correct_b)
    for group in counts.values():
        for name in ("restricted", "expanded"):
            group[name + "_accuracy"] = (group[name + "_correct"] / group["rows"] if group["rows"] else None)
    if all(counts[k]["rows"] for k in counts):
        for name in ("restricted", "expanded"):
            counts[name + "_weighted_accuracy"] = (
                .7 * counts["hangul"][name + "_accuracy"] + .3 * counts["non_hangul"][name + "_accuracy"])
    else:
        counts["restricted_weighted_accuracy"] = counts["expanded_weighted_accuracy"] = None
    return counts
