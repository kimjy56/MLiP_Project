"""Experimental AS-09 lookup conflict reconsideration; not enabled in submission."""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
from rapidfuzz.distance import OSA

from experimental_matchers.as09_candidate_extension import as09, predict_row


@dataclass(frozen=True)
class LookupComparison:
    prediction: str
    lookup_id: str | None
    route: str
    reasons: tuple[str, ...]
    candidates: tuple[tuple[float, str], ...] = ()


def _positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return -1.0
    # The reference predictor treats +inf as a positive numeric value, not as
    # missing; mirror that behavior when deciding whether lookup is taken.
    return value if not math.isnan(value) and value > 0 else -1.0


def _lookup(row, model, index):
    call = str(row.get("call_sign", "")).strip().upper() if pd.notna(row.get("call_sign")) else ""
    gt, igt = _positive(row.get("gross_tonnage")), _positive(row.get("international_gross_tonnage"))
    ton = igt if igt > 0 else gt
    alt = gt if gt > 0 and gt != ton else -1.0

    def compatible(vid, tolerance):
        ci = index.vid_to_idx.get(vid)
        if ci is None:
            raise ValueError("Lookup ID absent from catalog")
        c_gt = index.gt[ci]
        if ton <= 0 or c_gt <= 0:
            return True
        diff = abs(ton - c_gt)
        if alt > 0:
            diff = min(diff, abs(alt - c_gt))
        return diff / max(1.0, c_gt) <= tolerance

    calls = model.get("unique_callsign_lookup") or model.get("callsign_lookup", {})
    if call and call in calls and compatible(calls[call], .12):
        return calls[call], "callsign"
    if as09.is_korean_name(row.get("vessel_name")):
        comp = as09.parse_kr_row_features(row.get("vessel_name"), kr_mode=str(model.get("kr_mode", "AR06")))[0]
    else:
        comp = as09.normalize_vessel_name(row.get("vessel_name"), kr_mode="AR02")[0]
    names = model.get("unique_name_lookup") or model.get("name_lookup", {})
    if comp and comp in names and compatible(names[comp], .06):
        return names[comp], "name"
    return None, "none"


def lookup_conflicts(name, candidate_index, index, name_threshold=.45):
    """Detect explicit number conflict and severe same-script Latin name conflict.

    Cross-script name mismatch alone is not a conflict. Metadata mismatches are
    not used as a trigger because input and catalog categories may use different
    conventions. This fixed threshold is experimental, not a tuned claim.
    """
    if not 0.0 <= name_threshold <= 1.0:
        raise ValueError("name_threshold must be between 0 and 1")
    if as09.is_korean_name(name):
        _, _, nums, ordinal, _ = as09.parse_kr_row_features(name, kr_mode="AR06")
        qnum = ordinal or "|".join(nums)
    else:
        _, words, nums = as09.normalize_vessel_name(name, kr_mode="AR02")
        nums = list(nums)
        if words and words[-1] in as09.EN_NUM_WORD_MAP:
            nums.append(as09.EN_NUM_WORD_MAP[words[-1]])
        qnum = "|".join(nums)
    cnum = index.num_strs_ord[candidate_index] or index.num_strs[candidate_index]
    reasons = []
    if qnum and cnum and qnum != cnum:
        reasons.append("number_conflict")
    if not as09.is_korean_name(name):
        _, qwords, _ = as09.normalize_vessel_name(name, kr_mode="AR02")
        qa, ca = "".join(qwords), index.word_Alpha[candidate_index]
        if (len(qa) >= 4 and len(ca) >= 4 and qa.isascii() and ca.isascii() and
            qa.isalpha() and ca.isalpha() and OSA.normalized_similarity(qa, ca) < name_threshold):
            reasons.append("latin_name_conflict")
    return tuple(reasons)


def predict_lookup_conflict(query: pd.DataFrame, model, index, *, dense_matcher=None,
                            dense_mode=None, dense_weight=2.5, name_threshold=.45,
                            use_train_lookup=None):
    if len(query) != 1 or "vessel_name" not in query:
        raise ValueError("Expected exactly one query")
    row = query.iloc[0]
    if use_train_lookup is None:
        use_train_lookup = bool(model.get("use_train_lookup_default", as09.USE_TRAIN_LOOKUP))
    lookup_id, _ = _lookup(row, model, index) if use_train_lookup else (None, "disabled")
    reasons = lookup_conflicts(row.vessel_name, index.vid_to_idx[lookup_id], index, name_threshold) if lookup_id else ()
    if not reasons:
        pred = predict_row(query, model, index, use_train_lookup=use_train_lookup, dense_matcher=dense_matcher,
                           dense_mode=dense_mode, dense_weight=dense_weight)
        return LookupComparison(pred, lookup_id, "unchanged", reasons)
    pred, candidates = predict_row(query, model, index, candidates=[lookup_id], use_train_lookup=False,
                                   dense_matcher=dense_matcher, dense_mode=dense_mode,
                                   dense_weight=dense_weight, diagnostic=True)
    if lookup_id not in {vid for _, vid in candidates}:
        raise RuntimeError("Original lookup candidate was not included in the comparison")
    return LookupComparison(pred, lookup_id, "reconsidered", reasons, candidates)
