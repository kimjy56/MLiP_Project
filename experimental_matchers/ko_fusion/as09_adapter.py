# -*- coding: utf-8 -*-
"""AS-09 Adapter for KO-FUSION-01.

Provides instrumentation to trace unrounded candidate scores, 28 features,
and canonical candidate sorting while guaranteeing exact baseline parity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz.distance import OSA

import importlib.util

# Ensure train_AS-09 is explicitly loaded without root script.py collision
_repo_root = Path(__file__).resolve().parents[2]
_as09_dir = _repo_root / "train_AS-09"
if str(_as09_dir) not in sys.path:
    sys.path.insert(0, str(_as09_dir))

if "as09_script" in sys.modules:
    as09 = sys.modules["as09_script"]
else:
    _as09_file = _as09_dir / "script.py"
    _spec = importlib.util.spec_from_file_location("as09_script", str(_as09_file))
    as09 = importlib.util.module_from_spec(_spec)
    sys.modules["as09_script"] = as09
    _spec.loader.exec_module(as09)
from .features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    compute_feature_schema_sha256,
    extract_candidate_features,
)


@dataclass(frozen=True)
class CandidateTrace:
    sample_id: str
    route: str  # 'non_hangul', 'lookup', or 'hangul_candidates'
    baseline_pred_vessel_id: str
    candidate_ids: Tuple[str, ...]  # Canonical ascending order, no duplicates
    legacy_scores: np.ndarray  # float64 unrounded, matching candidate_ids order
    features: np.ndarray  # shape (N, 28) float64, matching candidate_ids order
    candidate_set_sha256: str
    feature_schema_sha256: str
    asset_manifest_sha256: str


def fit_split_model(
    fit_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    base_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit AS-09/AR-06 lookup, alignment, and lift tables on fit_df strictly."""
    merged = fit_df.merge(
        catalog_df[["vessel_id", "vessel_name", "vessel_type", "flag_name", "gross_tonnage"]],
        left_on="target_vessel_id",
        right_on="vessel_id",
        suffixes=("_tr", "_cat"),
        how="inner",
    )

    callsign_counts: Dict[str, Any] = {}
    name_counts: Dict[str, Any] = {}

    for row in fit_df.itertuples(index=False):
        vid = str(row.target_vessel_id)
        cs = str(row.call_sign).strip().upper() if pd.notna(row.call_sign) else ""
        if cs and cs not in ("NAN", "NONE", ""):
            if cs not in callsign_counts:
                callsign_counts[cs] = {}
            callsign_counts[cs][vid] = callsign_counts[cs].get(vid, 0) + 1
        compact, _, _ = as09.normalize_vessel_name(row.vessel_name, kr_mode="AR06")
        if compact:
            if compact not in name_counts:
                name_counts[compact] = {}
            name_counts[compact][vid] = name_counts[compact].get(vid, 0) + 1

    callsign_lookup = {k: max(v.items(), key=lambda x: x[1])[0] for k, v in callsign_counts.items()}
    name_lookup = {k: max(v.items(), key=lambda x: x[1])[0] for k, v in name_counts.items()}

    word_pair_counts: Dict[str, Any] = {}
    syl_chunk_counts: Dict[str, Any] = {}

    for row in merged.itertuples(index=False):
        tr_name = row.vessel_name_tr
        cat_name = row.vessel_name_cat
        if not as09.is_korean_name(tr_name):
            continue
        _, kr_words, _ = as09.normalize_vessel_name(tr_name, kr_mode="AR02")
        _, en_words, _ = as09.normalize_vessel_name(cat_name, kr_mode="AR02")
        if not kr_words or not en_words:
            continue

        if len(kr_words) == len(en_words):
            pairs = list(zip(kr_words, en_words))
        else:
            pairs = [("".join(kr_words), "".join(en_words))]

        for kw, ew in pairs:
            if not as09.re_hangul.search(kw) or not ew.isalpha():
                continue
            if kw not in word_pair_counts:
                word_pair_counts[kw] = {}
            word_pair_counts[kw][ew] = word_pair_counts[kw].get(ew, 0) + 1
            for s_ch, e_chunk in as09.dp_align_kr_en(kw, ew):
                if "가" <= s_ch <= "힣" and e_chunk:
                    if s_ch not in syl_chunk_counts:
                        syl_chunk_counts[s_ch] = {}
                    syl_chunk_counts[s_ch][e_chunk] = syl_chunk_counts[s_ch].get(e_chunk, 0) + 1

    learned_word_map = {k: max(v.items(), key=lambda x: x[1])[0] for k, v in word_pair_counts.items()}
    learned_word_top2 = {k: [w for w, _ in sorted(v.items(), key=lambda x: -x[1])[:2]] for k, v in word_pair_counts.items()}
    learned_syl_top2: Dict[str, List[str]] = {}
    learned_syl_top3: Dict[str, List[str]] = {}
    for ch, cnt in syl_chunk_counts.items():
        sorted_chunks = [chunk for chunk, _ in sorted(cnt.items(), key=lambda x: -x[1])]
        if sorted_chunks:
            learned_syl_top2[ch] = sorted_chunks[:2]
            learned_syl_top3[ch] = sorted_chunks[:3]

    type_joint: Dict[str, Dict[str, int]] = {}
    from collections import Counter
    cat_type_prior = Counter(catalog_df["vessel_type"].fillna("UNK").astype(str))
    total_cat = max(1, len(catalog_df))

    for row in merged.itertuples(index=False):
        kt = str(row.vessel_type_tr).strip() if pd.notna(row.vessel_type_tr) else "UNK"
        ct = str(row.vessel_type_cat).strip() if pd.notna(row.vessel_type_cat) else "UNK"
        if kt not in type_joint:
            type_joint[kt] = {}
        type_joint[kt][ct] = type_joint[kt].get(ct, 0) + 1

    type_log_lift: Dict[str, Dict[str, float]] = {}
    for kt, cnts in type_joint.items():
        tot_kt = sum(cnts.values())
        type_log_lift[kt] = {}
        for ct, c in cnts.items():
            p_cond = (c + 0.5) / (tot_kt + 5.0)
            p_prior = max(1e-5, cat_type_prior.get(ct, 1) / total_cat)
            type_log_lift[kt][ct] = round(float(np.clip(math.log(p_cond / p_prior), -2.5, 4.0)), 4)

    flag_joint: Dict[str, Dict[str, int]] = {}
    cat_flag_prior = Counter(catalog_df["flag_name"].fillna("UNK").astype(str))

    for row in merged.itertuples(index=False):
        kf = str(row.flag_name_tr).strip() if pd.notna(row.flag_name_tr) else "UNK"
        cf = str(row.flag_name_cat).strip() if pd.notna(row.flag_name_cat) else "UNK"
        yb = as09.year_bucket(row.arrival_year)
        if kf != "UNK":
            k_key = f"{kf}|{yb}"
            k_all = f"{kf}|ALL"
            for k in (k_key, k_all):
                if k not in flag_joint:
                    flag_joint[k] = {}
                flag_joint[k][cf] = flag_joint[k].get(cf, 0) + 1

    flag_log_lift: Dict[str, Dict[str, float]] = {}
    for key, cnts in flag_joint.items():
        tot_k = sum(cnts.values())
        flag_log_lift[key] = {}
        for cf, c in cnts.items():
            p_cond = (c + 0.5) / (tot_k + 5.0)
            p_prior = max(1e-5, cat_flag_prior.get(cf, 1) / total_cat)
            flag_log_lift[key][cf] = round(float(np.clip(math.log(p_cond / p_prior), -2.5, 4.5)), 4)

    type_trans_prob: Dict[str, Dict[str, float]] = {}
    for kt, cnts in type_joint.items():
        tot_kt = sum(cnts.values())
        type_trans_prob[kt] = {ct: round(c / tot_kt, 6) for ct, c in cnts.items()}

    flag_trans_prob: Dict[str, Dict[str, float]] = {}
    for key, cnts in flag_joint.items():
        tot_k = sum(cnts.values())
        flag_trans_prob[key] = {cf: round(c / tot_k, 6) for cf, c in cnts.items()}

    ko_map = {}
    if base_model is not None:
        ko_map = base_model.get("catalog_full_ko_map", {})

    return {
        "version": "1022-AR06-split",
        "kr_mode": "AR06",
        "en_mode": "EN_V3",
        "use_train_lookup_default": True,
        "callsign_lookup": callsign_lookup,
        "name_lookup": name_lookup,
        "learned_word_map": learned_word_map,
        "learned_word_top2": learned_word_top2,
        "learned_syl_top2": learned_syl_top2,
        "learned_syl_top3": learned_syl_top3,
        "type_log_lift": type_log_lift,
        "flag_log_lift": flag_log_lift,
        "type_trans_prob": type_trans_prob,
        "flag_trans_prob": flag_trans_prob,
        "catalog_full_ko_map": ko_map,
        "use_full_ko_map": True,
        "dense_mode": "hybrid",
        "dense_weight": 2.5,
    }


class AS09RerankAdapter:
    """Production adapter instrumenting AS-09 for controlled candidate reranking."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        device: str = "cpu",
        use_train_lookup: bool = True,
        dense_weight: float = 2.5,
        custom_model: Optional[Dict[str, Any]] = None,
    ):
        self.model_dir = model_dir or (_as09_dir / "model")
        self.data_dir = data_dir or (_repo_root / "data")
        self.device = device
        self.use_train_lookup = use_train_lookup
        self.dense_weight = dense_weight
        self.feature_schema_sha256 = compute_feature_schema_sha256()

        # Load model.json or use custom_model
        model_path = self.model_dir / "model.json"
        if custom_model is not None:
            self.model = custom_model
        else:
            with open(model_path, "r", encoding="utf-8") as f:
                self.model = json.load(f)

        # Load catalog
        catalog_path = self.data_dir / "vessel_catalog.csv"
        self.catalog_df = pd.read_csv(catalog_path, encoding="utf-8-sig")

        use_ko_map = bool(self.model.get("use_full_ko_map", as09.USE_FULL_KO_MAP))
        ko_map = self.model.get("catalog_full_ko_map", {}) if use_ko_map else {}
        self.cat_idx = as09.CatalogIndex(self.catalog_df, full_ko_map=ko_map)

        # Dense matcher setup
        enc_path = self.model_dir / "full_encoder.pt"
        npy_path = self.model_dir / "catalog_embeds_full_fp16.npy"
        self.dense_matcher = None
        if enc_path.exists() and npy_path.exists():
            self.dense_matcher = as09.HybridDenseLexicalMatcher(
                model_checkpoint=str(enc_path),
                catalog_embeds_npy=str(npy_path),
                catalog_df=self.catalog_df,
                device=device,
            )
            self.dense_mode = "hybrid"
        else:
            self.dense_mode = "none"

        # Compute asset manifest SHA256
        asset_info = {
            "model_json": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "catalog_csv": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "full_encoder_pt": hashlib.sha256(enc_path.read_bytes()).hexdigest() if enc_path.exists() else None,
            "catalog_embeds_npy": hashlib.sha256(npy_path.read_bytes()).hexdigest() if npy_path.exists() else None,
            "feature_schema_sha256": self.feature_schema_sha256,
        }
        self.asset_manifest_sha256 = hashlib.sha256(
            json.dumps(asset_info, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def trace_df(
        self,
        df: pd.DataFrame,
        gold_ids: Optional[List[str]] = None,
    ) -> List[CandidateTrace]:
        """Generate CandidateTraces for every row in df."""
        traces: List[CandidateTrace] = []
        n_rows = len(df)

        names = df["vessel_name"].tolist()
        sample_ids = df["sample_id"].astype(str).tolist() if "sample_id" in df.columns else [str(i) for i in range(n_rows)]
        callsigns = df["call_sign"].tolist() if "call_sign" in df.columns else [None] * n_rows
        gts = pd.to_numeric(df["gross_tonnage"], errors="coerce").fillna(-1.0).tolist() if "gross_tonnage" in df.columns else [-1.0] * n_rows
        igts = pd.to_numeric(df["international_gross_tonnage"], errors="coerce").fillna(-1.0).tolist() if "international_gross_tonnage" in df.columns else [-1.0] * n_rows
        flags = df["flag_name"].tolist() if "flag_name" in df.columns else [None] * n_rows
        vtypes = df["vessel_type"].tolist() if "vessel_type" in df.columns else [None] * n_rows
        years = df["arrival_year"].tolist() if "arrival_year" in df.columns else [None] * n_rows

        callsign_lookup = self.model.get("unique_callsign_lookup") or self.model.get("callsign_lookup", {})
        name_lookup = self.model.get("unique_name_lookup") or self.model.get("name_lookup", {})
        word_top2 = self.model.get("learned_word_top2", {})
        syl_top2 = self.model.get("learned_syl_top2", {})
        syl_top3 = self.model.get("learned_syl_top3", syl_top2)
        type_lift = self.model.get("type_log_lift", {})
        flag_lift = self.model.get("flag_log_lift", {})
        type_trans_prob = self.model.get("type_trans_prob", {})
        flag_trans_prob = self.model.get("flag_trans_prob", {})

        for row_i in range(n_rows):
            sid = sample_ids[row_i]
            raw_name = names[row_i]
            cs = str(callsigns[row_i]).strip().upper() if pd.notna(callsigns[row_i]) else ""
            gt = gts[row_i]
            igt = igts[row_i]
            ton = igt if igt > 0 else (gt if gt > 0 else -1.0)
            alt_ton = gt if (gt > 0 and gt != ton) else -1.0

            kf = str(flags[row_i]).strip() if pd.notna(flags[row_i]) else "UNK"
            kt = str(vtypes[row_i]).strip() if pd.notna(vtypes[row_i]) else "UNK"
            yb = as09.year_bucket(years[row_i])
            raw_yr = years[row_i]
            try:
                arrival_yr = float(raw_yr) if pd.notna(raw_yr) else None
            except (ValueError, TypeError):
                arrival_yr = None

            kr_flag = as09.is_korean_name(raw_name)

            def _is_tonnage_compatible(cand_vid: str, rel_tol: float) -> bool:
                c_idx = self.cat_idx.vid_to_idx.get(cand_vid)
                if c_idx is None:
                    return True
                c_gt = self.cat_idx.gt[c_idx]
                if ton <= 0 or c_gt <= 0:
                    return True
                diff = abs(ton - c_gt)
                if alt_ton > 0:
                    diff = min(diff, abs(alt_ton - c_gt))
                return (diff / max(1.0, c_gt)) <= rel_tol

            # 1. Lookup check (Callsign)
            if self.use_train_lookup:
                if cs and cs in callsign_lookup and _is_tonnage_compatible(callsign_lookup[cs], 0.12):
                    pred_vid = callsign_lookup[cs]
                    traces.append(CandidateTrace(
                        sample_id=sid,
                        route="lookup",
                        baseline_pred_vessel_id=pred_vid,
                        candidate_ids=(),
                        legacy_scores=np.empty(0, dtype=np.float64),
                        features=np.empty((0, 28), dtype=np.float64),
                        candidate_set_sha256="",
                        feature_schema_sha256=self.feature_schema_sha256,
                        asset_manifest_sha256=self.asset_manifest_sha256,
                    ))
                    continue

            # 2. Non-Korean route
            if not kr_flag:
                comp, wtoks, ntoks = as09.normalize_vessel_name(raw_name, kr_mode="AR02")
                if self.use_train_lookup and comp and comp in name_lookup and _is_tonnage_compatible(name_lookup[comp], 0.06):
                    pred_vid = name_lookup[comp]
                    traces.append(CandidateTrace(
                        sample_id=sid,
                        route="lookup",
                        baseline_pred_vessel_id=pred_vid,
                        candidate_ids=(),
                        legacy_scores=np.empty(0, dtype=np.float64),
                        features=np.empty((0, 28), dtype=np.float64),
                        candidate_set_sha256="",
                        feature_schema_sha256=self.feature_schema_sha256,
                        asset_manifest_sha256=self.asset_manifest_sha256,
                    ))
                    continue

                # Run exact AS-09 non-hangul prediction for this single row
                sub_df = df.iloc[[row_i]]
                pred_vid = as09.predict_rows(
                    sub_df,
                    self.model,
                    self.cat_idx,
                    use_train_lookup=False,  # Already checked above
                    dense_matcher=self.dense_matcher,
                    dense_mode=self.dense_mode,
                    dense_weight=self.dense_weight,
                )[0][0]

                traces.append(CandidateTrace(
                    sample_id=sid,
                    route="non_hangul",
                    baseline_pred_vessel_id=pred_vid,
                    candidate_ids=(),
                    legacy_scores=np.empty(0, dtype=np.float64),
                    features=np.empty((0, 28), dtype=np.float64),
                    candidate_set_sha256="",
                    feature_schema_sha256=self.feature_schema_sha256,
                    asset_manifest_sha256=self.asset_manifest_sha256,
                ))
                continue

            # 3. Korean candidate route
            word_map = self.model.get("learned_word_map", {})
            comp, wtoks, ntoks, kr_ord_nstr, cond_a_2char_ho = as09.parse_kr_row_features(raw_name, kr_mode="AR06")
            if self.use_train_lookup and comp and comp in name_lookup and _is_tonnage_compatible(name_lookup[comp], 0.06):
                pred_vid = name_lookup[comp]
                traces.append(CandidateTrace(
                    sample_id=sid,
                    route="lookup",
                    baseline_pred_vessel_id=pred_vid,
                    candidate_ids=(),
                    legacy_scores=np.empty(0, dtype=np.float64),
                    features=np.empty((0, 28), dtype=np.float64),
                    candidate_set_sha256="",
                    feature_schema_sha256=self.feature_schema_sha256,
                    asset_manifest_sha256=self.asset_manifest_sha256,
                ))
                continue

            # Normal Korean candidate retrieval and scoring
            nstr = "|".join(ntoks)

            cands_pairs = [as09.kr_word_to_en_candidates(w, syl_top2, word_map) for w in wtoks]
            en_wtoks1 = [p[0] for p in cands_pairs]
            en_wtoks2 = [p[1] for p in cands_pairs]
            query_alpha1 = "".join(en_wtoks1)
            query_alpha2 = "".join(en_wtoks2)
            query_phon_words = [
                as09.en_to_phonetic(w_en) if (w in word_map or not as09.re_hangul.search(w_en)) else as09.kr_to_phonetic(w)
                for w, w_en in zip(wtoks, en_wtoks1)
            ]
            query_phon_full = "".join(query_phon_words)
            query_skel1 = as09.consonant_skeleton(query_alpha1)
            query_skel2 = as09.consonant_skeleton(query_alpha2)

            expanded_wtoks: List[str] = []
            for w in wtoks:
                acr = as09.extract_kr_acronym_prefix(w)
                if acr is not None and acr[1]:
                    expanded_wtoks.append(acr[0])
                    expanded_wtoks.append(acr[1])
                elif acr is not None and not acr[1]:
                    expanded_wtoks.append(w)
                else:
                    expanded_wtoks.append(w)

            idea_b_word_cands = [as09.generate_idea_b_word_variants(w, syl_top3, word_top2) for w in expanded_wtoks]
            idea_b_full_variants: List[str] = []
            for v_idx in range(8):
                full_v = "".join(wc[min(v_idx, len(wc) - 1)] for wc in idea_b_word_cands if wc)
                if full_v and full_v not in idea_b_full_variants:
                    idea_b_full_variants.append(full_v)

            full_kr_concat = "".join(wtoks)
            if full_kr_concat in as09.IDEA_B_WORD_VARS:
                for fv_direct in as09.IDEA_B_WORD_VARS[full_kr_concat]:
                    if fv_direct not in idea_b_full_variants:
                        idea_b_full_variants.insert(0, fv_direct)

            wtoks_no_ord_cands: Optional[List[List[str]]] = None
            if kr_ord_nstr and len(expanded_wtoks) >= 2:
                wtoks_no_ord_cands = idea_b_word_cands[:-1]
                for v_idx in range(3):
                    full_v_no_ord = "".join(wc[min(v_idx, len(wc) - 1)] for wc in wtoks_no_ord_cands if wc)
                    if full_v_no_ord and full_v_no_ord not in idea_b_full_variants:
                        idea_b_full_variants.append(full_v_no_ord)

            raw_clean = str(raw_name).strip()
            ends_with_ho_raw = raw_clean.endswith("호") and bool(wtoks) and not ntoks
            ho_preserved_alpha: Optional[str] = (query_alpha1 + "HO") if ends_with_ho_raw else None
            ho_preserved_phon: Optional[str] = (query_phon_full + as09.en_to_phonetic("HO")) if ends_with_ho_raw else None
            if cond_a_2char_ho and ho_preserved_alpha and ho_preserved_alpha not in idea_b_full_variants:
                idea_b_full_variants.append(ho_preserved_alpha)

            q_jamo_wtoks = [as09.hangul_to_jamo(w, phonetic_norm=True) for w in wtoks if w]
            q_jamo_raw = "".join(as09.hangul_to_jamo(w, phonetic_norm=False) for w in wtoks if w)
            q_jamo_norm = "".join(q_jamo_wtoks)

            cand_set = set()
            if q_jamo_norm and self.cat_idx.jamo_ngram3_idx:
                if q_jamo_norm in self.cat_idx.exact_jamo_idx:
                    cand_set.update(self.cat_idx.exact_jamo_idx[q_jamo_norm])
                if len(q_jamo_norm) >= 3:
                    cand_set.update(self.cat_idx.jamo_prefix3_idx.get(q_jamo_norm[:3], []))
                if len(q_jamo_norm) >= 4:
                    cand_set.update(self.cat_idx.jamo_suffix3_idx.get(q_jamo_norm[-3:], []))
                for jw in q_jamo_wtoks:
                    if len(jw) >= 3:
                        cand_set.update(self.cat_idx.jamo_word_idx.get(jw[:4], []))
                cand_set.update(self.cat_idx.query_jamo_top_candidates(q_jamo_norm, top_k=250))

            if query_alpha1 in self.cat_idx.exact_alpha_idx:
                cand_set.update(self.cat_idx.exact_alpha_idx[query_alpha1])
            if ton > 0:
                cand_set.update(self.cat_idx.query_tonnage_window(ton, rel_tol=0.098, max_cands=None))
                cand_set.update(self.cat_idx.query_l_win_top_osa(ton, query_alpha1, "", rel_tol=0.10, top_k=40))
            if alt_ton > 0:
                cand_set.update(self.cat_idx.query_tonnage_window(alt_ton, rel_tol=0.098, max_cands=None))
            if query_alpha1:
                cand_set.update(self.cat_idx.prefix3_idx.get(query_alpha1[:3], []))
            if query_skel1:
                cand_set.update(self.cat_idx.skel3_idx.get(query_skel1[:3], []))
            if query_phon_full:
                cand_set.update(self.cat_idx.phon3_idx.get(query_phon_full[:3], []))
            for w in en_wtoks1:
                if len(w) >= 3:
                    cand_set.update(self.cat_idx.word_idx.get(w[:4], []))
            for fv in idea_b_full_variants[:7]:
                if fv in self.cat_idx.exact_alpha_idx:
                    cand_set.update(self.cat_idx.exact_alpha_idx[fv])
                if len(fv) >= 3:
                    cand_set.update(self.cat_idx.prefix3_idx.get(fv[:3], []))
            for wc in idea_b_word_cands:
                for vw in wc[:5]:
                    if len(vw) >= 3:
                        cand_set.update(self.cat_idx.word_idx.get(vw[:4], []))
            if nstr and nstr in self.cat_idx.num_idx:
                num_cands = self.cat_idx.num_idx[nstr]
                if len(num_cands) <= 500:
                    cand_set.update(num_cands)
            if kr_ord_nstr and kr_ord_nstr in self.cat_idx.num_idx:
                num_cands_ord = self.cat_idx.num_idx[kr_ord_nstr]
                if len(num_cands_ord) <= 500:
                    cand_set.update(num_cands_ord)

            dense_feats_map: Dict[int, Dict[str, float]] = {}
            if self.dense_matcher is not None and self.dense_mode == "hybrid":
                dense_cands = self.dense_matcher.retrieve_dense_candidates(raw_name, top_k=35)
                for dc in dense_cands:
                    c_idx_dense = self.cat_idx.vid_to_idx.get(dc["vessel_id"])
                    if c_idx_dense is not None:
                        cand_set.add(c_idx_dense)
                        dense_feats_map[c_idx_dense] = dc

            if not cand_set:
                if ton > 0:
                    cand_set.update(self.cat_idx.query_tonnage_window(ton, rel_tol=0.30, max_cands=200))
                if not cand_set:
                    cand_set.add(0)

            t_lifts = type_lift.get(kt, {})
            f_lifts = flag_lift.get(f"{kf}|{yb}", flag_lift.get(f"{kf}|ALL", {}))
            n_q_words = len(en_wtoks1)
            n_b_words = len(idea_b_word_cands)

            scored_cands_raw = []
            best_jamo_by_ci: Dict[int, float] = {}

            for c_i in cand_set:
                c_alpha = self.cat_idx.word_Alpha[c_i]
                c_skel = self.cat_idx.skeletons[c_i]
                c_nstr = self.cat_idx.num_strs[c_i]
                c_nstr_ord = self.cat_idx.num_strs_ord[c_i]
                c_gt = self.cat_idx.gt[c_i]
                c_wtoks = self.cat_idx.word_tokens[c_i]
                c_pwords = self.cat_idx.phon_words[c_i]
                c_pfull = self.cat_idx.phon_full[c_i]

                sim_a1 = OSA.normalized_similarity(query_alpha1, c_alpha) if (query_alpha1 and c_alpha) else 0.0
                sim_a2 = OSA.normalized_similarity(query_alpha2, c_alpha) if (query_alpha2 and c_alpha) else 0.0
                sim_alpha = max(sim_a1, 0.97 * sim_a2)

                sim_s1 = OSA.normalized_similarity(query_skel1, c_skel) if (query_skel1 and c_skel) else 0.0
                sim_s2 = OSA.normalized_similarity(query_skel2, c_skel) if (query_skel2 and c_skel) else 0.0
                sim_skel = max(sim_s1, sim_s2)

                sim_phon = OSA.normalized_similarity(query_phon_full, c_pfull) if (query_phon_full and c_pfull) else 0.0

                allow_ho_match = ends_with_ho_raw and (cond_a_2char_ho or (len(c_wtoks) > 0 and c_wtoks[-1] == "HO"))
                if allow_ho_match and ho_preserved_alpha and c_alpha:
                    sim_alpha = max(sim_alpha, OSA.normalized_similarity(ho_preserved_alpha, c_alpha))
                    if ho_preserved_phon and c_pfull:
                        sim_phon = max(sim_phon, OSA.normalized_similarity(ho_preserved_phon, c_pfull))

                if idea_b_full_variants and c_alpha:
                    best_b_full = max(OSA.normalized_similarity(fv, c_alpha) for fv in idea_b_full_variants)
                    sim_alpha = max(sim_alpha, best_b_full)
                    best_b_skel = max(
                        OSA.normalized_similarity(as09.consonant_skeleton(fv), c_skel)
                        for fv in idea_b_full_variants[:5]
                    ) if c_skel else 0.0
                    sim_skel = max(sim_skel, best_b_skel)
                    best_b_phon = max(
                        OSA.normalized_similarity(as09.en_to_phonetic(fv), c_pfull)
                        for fv in idea_b_full_variants[:4]
                    ) if c_pfull else 0.0
                    sim_phon = max(sim_phon, 0.96 * best_b_phon)

                matched_word_count = False
                sim_tok = max(sim_alpha, sim_phon)

                if n_q_words > 0 and n_q_words == len(c_wtoks):
                    w_sims = []
                    for qw1, qw2, qpw, cw, cpw in zip(en_wtoks1, en_wtoks2, query_phon_words, c_wtoks, c_pwords):
                        s_w = max(
                            OSA.normalized_similarity(qw1, cw),
                            0.97 * OSA.normalized_similarity(qw2, cw),
                            0.95 * OSA.normalized_similarity(qpw, cpw),
                        )
                        w_sims.append(s_w)
                    sim_tok = sum(w_sims) / n_q_words
                    matched_word_count = True

                if n_b_words > 0 and n_b_words == len(c_wtoks):
                    bw_sims = []
                    for wc_list, cw, cpw in zip(idea_b_word_cands, c_wtoks, c_pwords):
                        best_w = max(OSA.normalized_similarity(cand_w, cw) for cand_w in wc_list)
                        best_pw = max(
                            OSA.normalized_similarity(as09.en_to_phonetic(cand_w), cpw)
                            for cand_w in wc_list[:4]
                        ) if cpw else 0.0
                        bw_sims.append(max(best_w, 0.95 * best_pw))
                    sim_tok = max(sim_tok, sum(bw_sims) / n_b_words) if matched_word_count else (sum(bw_sims) / n_b_words)
                    matched_word_count = True

                if not matched_word_count and wtoks_no_ord_cands is not None and len(wtoks_no_ord_cands) == len(c_wtoks) and len(c_wtoks) > 0 and c_nstr:
                    bw_sims_no_ord = []
                    for wc_list, cw, cpw in zip(wtoks_no_ord_cands, c_wtoks, c_pwords):
                        best_w = max(OSA.normalized_similarity(cand_w, cw) for cand_w in wc_list)
                        best_pw = max(
                            OSA.normalized_similarity(as09.en_to_phonetic(cand_w), cpw)
                            for cand_w in wc_list[:3]
                        ) if cpw else 0.0
                        bw_sims_no_ord.append(max(best_w, 0.95 * best_pw))
                    sim_tok = max(sim_tok, sum(bw_sims_no_ord) / len(c_wtoks))
                    matched_word_count = True

                if not matched_word_count and allow_ho_match and (n_b_words + 1) == len(c_wtoks):
                    cands_plus_ho = idea_b_word_cands + [["HO"]]
                    bw_sims_ho = []
                    for wc_list, cw, cpw in zip(cands_plus_ho, c_wtoks, c_pwords):
                        best_w = max(OSA.normalized_similarity(cand_w, cw) for cand_w in wc_list)
                        bw_sims_ho.append(best_w)
                    sim_tok = max(sim_tok, sum(bw_sims_ho) / len(c_wtoks))
                    matched_word_count = True

                jamo_first_ok = False
                if q_jamo_norm and self.cat_idx.synth_jamo_norm[c_i]:
                    c_jraws = self.cat_idx.synth_jamo_raw[c_i]
                    c_jnorms = self.cat_idx.synth_jamo_norm[c_i]
                    c_jwtoks_list = self.cat_idx.synth_jamo_wtoks[c_i]
                    sim_j_raw = max(
                        (1.0 if q_jamo_raw == jr else OSA.normalized_similarity(q_jamo_raw, jr))
                        for jr in c_jraws if jr
                    ) if (q_jamo_raw and c_jraws) else 0.0
                    sim_j_norm = max(
                        (1.0 if q_jamo_norm == jn else OSA.normalized_similarity(q_jamo_norm, jn))
                        for jn in c_jnorms if jn
                    )
                    sim_jamo = 0.60 * sim_j_raw + 0.40 * sim_j_norm
                    best_jamo_by_ci[c_i] = max(sim_jamo, sim_j_norm)
                    sim_alpha = max(sim_alpha, sim_jamo)
                    sim_phon = max(sim_phon, sim_j_norm)
                    sim_skel = max(sim_skel, sim_j_norm)
                    jamo_first_ok = any(jn and q_jamo_norm[0] == jn[0] for jn in c_jnorms)
                    for cj_wtoks in c_jwtoks_list:
                        if len(q_jamo_wtoks) > 0 and len(q_jamo_wtoks) == len(cj_wtoks):
                            j_w_sim = sum(
                                1.0 if a == b else OSA.normalized_similarity(a, b)
                                for a, b in zip(q_jamo_wtoks, cj_wtoks)
                            ) / len(q_jamo_wtoks)
                            sim_tok = max(sim_tok, j_w_sim) if matched_word_count else max(sim_jamo, j_w_sim)
                            matched_word_count = True
                        elif sim_jamo > sim_tok:
                            sim_tok = sim_jamo

                if matched_word_count:
                    word_cnt_bonus = 0.45
                else:
                    if sim_alpha >= 0.90:
                        word_cnt_bonus = 0.25
                    elif sim_alpha >= 0.82 or sim_phon >= 0.85:
                        word_cnt_bonus = 0.0
                    else:
                        word_cnt_bonus = -0.35 * min(2, abs(n_q_words - len(c_wtoks)))

                first_match = 0.35 if (jamo_first_ok or (query_phon_full and c_pfull and query_phon_full[0] == c_pfull[0])) else 0.0

                name_score = (
                    3.8 * sim_alpha
                    + 2.8 * sim_phon
                    + 2.2 * sim_skel
                    + 2.7 * sim_tok
                    + word_cnt_bonus
                    + first_match
                )

                num_score = 0.0
                if nstr:
                    if c_nstr_ord:
                        num_score = 3.2 if (nstr == c_nstr or nstr == c_nstr_ord) else -4.0
                    else:
                        num_score = -2.8
                elif kr_ord_nstr:
                    if c_nstr or c_nstr_ord:
                        num_score = 3.2 if (kr_ord_nstr == c_nstr or kr_ord_nstr == c_nstr_ord) else -4.0
                    else:
                        num_score = 0.0
                else:
                    if c_nstr:
                        num_score = -1.6

                ton_score = 0.0
                if ton > 0 and c_gt > 0:
                    diff = abs(ton - c_gt)
                    if alt_ton > 0:
                        diff = min(diff, abs(alt_ton - c_gt))
                    rel = diff / max(1.0, c_gt)
                    if diff <= 1.0:
                        ton_score = 3.6
                    elif rel <= 0.005:
                        ton_score = 3.1
                    elif rel <= 0.02:
                        ton_score = 2.5
                    elif rel <= 0.052:
                        ton_score = 2.1 - 4.0 * (rel - 0.02)
                    elif rel <= 0.095:
                        ton_score = 0.8 - 10.0 * (rel - 0.052)
                    elif (sim_alpha >= 0.88 or sim_phon >= 0.90) and rel <= 0.25:
                        ton_score = -0.8
                    else:
                        ton_score = -3.5 * min(2.5, rel * 6.0)

                ct = self.cat_idx.types[c_i]
                cf = self.cat_idx.flags[c_i]

                if type_trans_prob and flag_trans_prob:
                    if kt == "UNK" or kt not in type_trans_prob:
                        t_score = 0.5
                    else:
                        p_t = type_trans_prob[kt].get(ct, 0.0)
                        if p_t >= 0.20:
                            t_score = 1.0
                        elif p_t >= 0.02:
                            t_score = p_t
                        else:
                            t_score = 0.0

                    if kf == "UNK":
                        f_score = 0.5
                    else:
                        fp_dict = flag_trans_prob.get(f"{kf}|{yb}", flag_trans_prob.get(f"{kf}|ALL", flag_trans_prob.get(kf, {})))
                        p_f = fp_dict.get(cf, 0.0) if fp_dict else 0.0
                        sim_for_flag = max(sim_alpha, sim_phon)
                        is_korea = cf in ("South Korea", "Korea", "Republic of Korea")
                        if is_korea or p_f >= 0.30:
                            f_score = 1.0
                        elif p_f >= 0.02:
                            f_score = p_f
                        elif sim_for_flag >= 0.85:
                            f_score = 0.5
                        elif sim_for_flag >= 0.70 and (arrival_yr is not None and arrival_yr <= 2019):
                            f_score = 0.5
                        else:
                            f_score = 0.0

                    meta_score = 1.6 * (t_score - 0.5) + 1.2 * (f_score - 0.5)
                else:
                    t_val = t_lifts.get(ct, -0.15)
                    if kf == "UNK":
                        f_val = 0.0
                    else:
                        f_val = f_lifts.get(cf, 0.0)
                        if yb == "OLD" and f_val < 0:
                            f_val *= 0.25
                    meta_score = t_val + f_val

                dense_score = 0.0
                in_dense = c_i in dense_feats_map
                df_dense = dense_feats_map.get(c_i, {}) if in_dense else {}
                if in_dense and self.dense_mode == "hybrid":
                    d_sim = df_dense.get("dense_sim", 0.0)
                    d_recip = df_dense.get("dense_rank_recip", 0.0)
                    d_z = df_dense.get("dense_zscore", 0.0)
                    d_marg = df_dense.get("dense_margin_top1", 0.0)
                    dense_score = (
                        self.dense_weight * d_sim
                        + 1.0 * d_recip
                        + 0.3 * max(0.0, d_z)
                        + 0.5 * max(0.0, d_marg)
                    )

                total_score = name_score + num_score + ton_score + meta_score + dense_score
                cand_vid = self.cat_idx.vessel_ids[c_i]

                cand_raw_dict = {
                    "legacy_total_score": total_score,
                    "legacy_name_score": name_score,
                    "legacy_num_score": num_score,
                    "legacy_ton_score": ton_score,
                    "legacy_meta_score": meta_score,
                    "legacy_dense_score": dense_score,
                    "sim_alpha": sim_alpha,
                    "sim_phon": sim_phon,
                    "sim_skel": sim_skel,
                    "sim_tok": sim_tok,
                    "word_cnt_bonus": word_cnt_bonus,
                    "first_match": first_match,
                    "query_num_present": bool(nstr or kr_ord_nstr),
                    "candidate_num_present": bool(c_nstr or c_nstr_ord),
                    "num_compatible": (num_score > 0.0),
                    "query_gt": gt,
                    "query_igt": igt,
                    "catalog_gt": c_gt,
                    "flag_missing": (kf == "UNK"),
                    "type_missing": (kt == "UNK"),
                    "in_dense_top35": in_dense,
                    "dense_feats": df_dense,
                }
                scored_cands_raw.append((total_score, cand_vid, cand_raw_dict))

            # Legacy baseline sorting: (-total_score, vessel_id)
            scored_cands_raw.sort(key=lambda x: (-x[0], x[1]))
            baseline_pred = scored_cands_raw[0][1]

            # Collect rank for each candidate
            cand_ranks: Dict[str, int] = {}
            cand_scores: Dict[str, float] = {}
            cand_feats: Dict[str, Dict[str, Any]] = {}
            for rank_idx, (tot_sc, vid, raw_dict) in enumerate(scored_cands_raw):
                cand_ranks[vid] = rank_idx
                cand_scores[vid] = tot_sc
                cand_feats[vid] = raw_dict

            # Canonical order: ascending by vessel_id string
            canonical_vids = tuple(sorted(cand_ranks.keys()))
            legacy_scores = np.array([cand_scores[v] for v in canonical_vids], dtype=np.float64)
            features = extract_candidate_features(
                [cand_feats[v] for v in canonical_vids],
                [cand_ranks[v] for v in canonical_vids],
            )
            cset_sha256 = hashlib.sha256(
                json.dumps(list(canonical_vids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

            traces.append(CandidateTrace(
                sample_id=sid,
                route="hangul_candidates",
                baseline_pred_vessel_id=baseline_pred,
                candidate_ids=canonical_vids,
                legacy_scores=legacy_scores,
                features=features,
                candidate_set_sha256=cset_sha256,
                feature_schema_sha256=self.feature_schema_sha256,
                asset_manifest_sha256=self.asset_manifest_sha256,
            ))

        return traces

    @staticmethod
    def select_with_ranker(trace: CandidateTrace, ranker: Optional[Any] = None) -> str:
        """Select top-1 prediction for a CandidateTrace.
        
        Tie-break:
          When ranker is None: (-legacy_score, vessel_id)
          When ranker is given: (-ranker_score, -legacy_score, vessel_id)
        """
        if trace.route != "hangul_candidates" or len(trace.candidate_ids) == 0:
            return trace.baseline_pred_vessel_id

        if len(trace.candidate_ids) == 1:
            return trace.candidate_ids[0]

        if ranker is None:
            items = [(-score, vid) for vid, score in zip(trace.candidate_ids, trace.legacy_scores)]
            items.sort()
            return items[0][1]

        ranker_scores = ranker.predict(trace.features)
        items = [
            (-float(r_sc), -float(l_sc), vid)
            for vid, l_sc, r_sc in zip(trace.candidate_ids, trace.legacy_scores, ranker_scores)
        ]
        items.sort()
        return items[0][2]
