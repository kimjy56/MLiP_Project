"""Experimental single-row EXAONE KO→EN candidate-union diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from rapidfuzz import fuzz, process

from experimental_matchers.as09_candidate_extension import as09, predict_row
from experimental_matchers.lookup_conflict import _lookup
from name_decoder.deterministic_ko_rules import clean_korean


def _compact(value):
    return as09.normalize_vessel_name(value, kr_mode="AR02")[0]


@dataclass(frozen=True)
class DecoderCandidate:
    vessel_id: str
    beam_rank: int
    similarity: float
    exact: bool


class ReverseDecoderCandidates:
    def __init__(self, catalog_index, decoder):
        self.decoder = decoder
        self.ids = tuple(str(v) for v in catalog_index.vessel_ids)
        if not self.ids or len(self.ids) != len(set(self.ids)):
            raise ValueError("Catalog IDs must be nonempty and unique")
        # Sort choices by ID so ties are deterministic even if catalog rows move.
        order = sorted(range(len(self.ids)), key=self.ids.__getitem__)
        self.sorted_ids = [self.ids[i] for i in order]
        self.sorted_names = [catalog_index.compact_names[i] for i in order]
        self.exact = {}
        for vid, name in zip(self.sorted_ids, self.sorted_names):
            if name:
                self.exact.setdefault(name, []).append(vid)

    def search(self, korean_name, fuzzy_limit=40):
        name = clean_korean(korean_name)
        if not name or not as09.is_korean_name(name):
            return ()
        decoded = self.decoder.decode([name], num_beams=4,
                                      num_return_sequences=3, max_new_tokens=48)
        if len(decoded) != 1:
            raise ValueError("Decoder did not return one result for the input row")
        texts = [_compact(item["text"]) for item in decoded[0] if item.get("text")]
        candidates = {}
        for rank, name in enumerate(texts, 1):
            for vid in self.exact.get(name, ()):
                candidates.setdefault(vid, DecoderCandidate(vid, rank, 1.0, True))
        if len(candidates) < 35 and texts and fuzzy_limit > 0:
            for _, score, pos in process.extract(texts[0], self.sorted_names, scorer=fuzz.ratio,
                                                  limit=fuzzy_limit, score_cutoff=1):
                vid = self.sorted_ids[pos]
                candidates.setdefault(vid, DecoderCandidate(vid, 1, float(score) / 100.0, False))
        return tuple(sorted(candidates.values(), key=lambda c: (c.beam_rank, -c.similarity, c.vessel_id)))


@dataclass(frozen=True)
class DecoderUnionComparison:
    baseline_prediction: str
    expanded_prediction: str
    decoder_candidates: tuple[DecoderCandidate, ...]
    baseline_candidates: tuple[tuple[float, str], ...]
    expanded_candidates: tuple[tuple[float, str], ...]
    route: str


def compare_row(query: pd.DataFrame, model, catalog_index, retriever, *, dense_matcher=None,
                dense_mode=None, dense_weight=2.5):
    if len(query) != 1 or "vessel_name" not in query:
        raise ValueError("Expected exactly one query")
    if tuple(catalog_index.vessel_ids) != retriever.ids:
        raise ValueError("Decoder and scoring catalog orders differ")
    kwargs = dict(dense_matcher=dense_matcher, dense_mode=dense_mode,
                  dense_weight=dense_weight, diagnostic=True)
    original, baseline = predict_row(query, model, catalog_index, **kwargs)
    if not as09.is_korean_name(query.iloc[0].vessel_name):
        return DecoderUnionComparison(original, original, (), baseline, baseline, "non_korean_bypass")
    lookup_on = bool(model.get("use_train_lookup_default", as09.USE_TRAIN_LOOKUP))
    lookup_id, _ = _lookup(query.iloc[0], model, catalog_index) if lookup_on else (None, "disabled")
    if lookup_id:
        return DecoderUnionComparison(original, original, (), baseline, baseline, "lookup_bypass")
    if not clean_korean(query.iloc[0].vessel_name):
        return DecoderUnionComparison(original, original, (), baseline, baseline, "invalid_decoder_input")
    extra = retriever.search(query.iloc[0].vessel_name)
    extended, candidates = predict_row(query, model, catalog_index,
                                       candidates=[item.vessel_id for item in extra], **kwargs)
    if not {vid for _, vid in baseline} <= {vid for _, vid in candidates}:
        raise RuntimeError("Candidate union dropped an existing candidate")
    return DecoderUnionComparison(original, extended, extra, baseline, candidates, "candidates")
