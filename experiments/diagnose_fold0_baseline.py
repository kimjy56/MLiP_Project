"""Diagnose Fold 0 Baseline matching performance and candidate retrieval coverage.

Strict isolation conditions:
- Fitted ONLY on train[fold != 0] using AR-06/AS-04 engine (zero data leakage).
- Evaluated on all 2,372 rows of Fold 0 validation set (239 Hangul, 2,133 Non-Hangul).
- Dissects errors into:
  1. Retrieval Miss: Gold vessel ID was NEVER present in the candidate set.
  2. Ranking Error: Gold vessel ID was retrieved in the candidate set, but ranked behind another candidate.
- Records exact baseline predictions to serve as the ground reference for N-channel evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "AR-06") not in sys.path:
    sys.path.insert(0, str(ROOT / "AR-06"))

import numpy as np
import pandas as pd

# Load engine from AR-06/script.py
from script import (
    CatalogIndex,
    fit_matching_model,
    is_korean_name,
    normalize_vessel_name,
    parse_kr_row_features,
    kr_word_to_en_candidates,
    generate_idea_b_word_variants,
    extract_kr_acronym_prefix,
    consonant_skeleton,
    en_to_phonetic,
    kr_to_phonetic,
    c_idx_has_num,
    lattice_word_similarity,
    re_hangul,
    IDEA_B_WORD_VARS,
    USE_TRAIN_LOOKUP,
    KR_MODE,
    EN_MODE,
)
from rapidfuzz.distance import OSA


def run_diagnostics():
    data_dir = ROOT / "data"
    folds_path = ROOT / "eval_kit/folds_group5_seed42.csv"
    train_path = data_dir / "train.csv"
    catalog_path = data_dir / "vessel_catalog.csv"

    print("=== Loading Data and Splitting Fold 0 ===")
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog_df = pd.read_csv(catalog_path, encoding="utf-8-sig")
    folds_df = pd.read_csv(folds_path, dtype={"sample_id": str, "fold": int}, encoding="utf-8-sig")

    merged = train_df.merge(folds_df[["sample_id", "fold"]], on="sample_id", how="inner")
    fit_df = merged[merged["fold"] != 0].reset_index(drop=True)
    val_df = merged[merged["fold"] == 0].reset_index(drop=True)

    print(f"Fit rows (Fold != 0): {len(fit_df):,} rows")
    print(f"Validation rows (Fold == 0): {len(val_df):,} rows")
    hangul_mask = val_df["vessel_name"].apply(is_korean_name)
    n_hangul = int(hangul_mask.sum())
    n_non_hangul = int((~hangul_mask).sum())
    print(f"  - Hangul validation rows: {n_hangul:,}")
    print(f"  - Non-Hangul validation rows: {n_non_hangul:,}")

    print("\n=== [1] Building Catalog Index & Fitting Fold 0 Model ===")
    t0_cat = time.perf_counter()
    cat_idx = CatalogIndex(catalog_df)
    cat_sec = time.perf_counter() - t0_cat
    print(f"Catalog indexed: {len(cat_idx.vessel_ids):,} ships ({cat_sec:.2f}s)")

    t0_fit = time.perf_counter()
    fold0_model = fit_matching_model(
        fit_df,
        catalog_df,
        kr_mode="AR06",
        en_mode="EN_V3",
        use_train_lookup=False,
    )
    fit_sec = time.perf_counter() - t0_fit
    print(f"Fold 0 model fitted strictly on fit_df in {fit_sec:.2f}s (zero leakage)")

    print("\n=== [2] Running Detailed Baseline Diagnostics on Fold 0 ===")
    t0_eval = time.perf_counter()

    callsign_lookup = fold0_model.get("callsign_lookup", {})
    name_lookup = fold0_model.get("name_lookup", {})
    word_map = fold0_model.get("learned_word_map", {})
    word_top2 = fold0_model.get("learned_word_top2", {})
    syl_top2 = fold0_model.get("learned_syl_top2", {})
    syl_top3 = fold0_model.get("learned_syl_top3", syl_top2)
    type_lift = fold0_model.get("type_log_lift", {})
    flag_lift = fold0_model.get("flag_log_lift", {})
    type_trans_prob = fold0_model.get("type_trans_prob", {})
    flag_trans_prob = fold0_model.get("flag_trans_prob", {})

    names = val_df["vessel_name"].tolist()
    callsigns = val_df["call_sign"].tolist() if "call_sign" in val_df.columns else [None] * len(val_df)
    gts = pd.to_numeric(val_df["gross_tonnage"], errors="coerce").fillna(-1.0).tolist()
    igts = pd.to_numeric(val_df["international_gross_tonnage"], errors="coerce").fillna(-1.0).tolist()
    flags = val_df["flag_name"].tolist() if "flag_name" in val_df.columns else [None] * len(val_df)
    vtypes = val_df["vessel_type"].tolist() if "vessel_type" in val_df.columns else [None] * len(val_df)
    years = val_df["arrival_year"].tolist() if "arrival_year" in val_df.columns else [None] * len(val_df)
    golds = val_df["target_vessel_id"].astype(str).tolist()
    sample_ids = val_df["sample_id"].astype(str).tolist()

    rows_detail = []

    for row_i in range(len(val_df)):
        raw_name = names[row_i]
        gold_vid = golds[row_i]
        s_id = sample_ids[row_i]
        is_kr = is_korean_name(raw_name)

        cs = str(callsigns[row_i]).strip().upper() if pd.notna(callsigns[row_i]) else ""
        gt = gts[row_i]
        igt = igts[row_i]
        ton = igt if igt > 0 else (gt if gt > 0 else -1.0)
        alt_ton = gt if (gt > 0 and gt != ton) else -1.0
        kf = str(flags[row_i]).strip() if pd.notna(flags[row_i]) else "UNK"
        kt = str(vtypes[row_i]).strip() if pd.notna(vtypes[row_i]) else "UNK"

        try:
            arrival_yr = int(float(years[row_i])) if pd.notna(years[row_i]) else None
        except (ValueError, TypeError):
            arrival_yr = None

        if arrival_yr is not None and arrival_yr <= 2018:
            yb = "OLD"
        elif arrival_yr is not None and arrival_yr <= 2022:
            yb = "MID"
        else:
            yb = "NEW"

        # Check callsign direct lookup (if enabled)
        if False and cs and cs in callsign_lookup:
            pred_vid = callsign_lookup[cs]
            is_correct = (pred_vid == gold_vid)
            rows_detail.append({
                "sample_id": s_id,
                "hangul": is_kr,
                "vessel_name": raw_name,
                "gold_vessel_id": gold_vid,
                "pred_vessel_id": pred_vid,
                "is_correct": is_correct,
                "gold_in_candidates": True,
                "candidates_count": 1,
                "gold_rank": 1 if is_correct else 0,
                "error_type": "None" if is_correct else "Ranking Error",
            })
            continue

        # Non-Hangul vs Hangul candidate retrieval
        if not is_kr:
            comp, wtoks, ntoks = normalize_vessel_name(raw_name, kr_mode="EN_V3")
            nstr = "|".join(ntoks)
            nstr_ord = "|".join(sorted(ntoks))
            query_alpha1 = "".join(wtoks)
            sorted_wtoks = sorted(wtoks)
            sorted_query_alpha = "".join(sorted_wtoks)
            query_skel1 = consonant_skeleton(query_alpha1)
            query_phon1 = en_to_phonetic(query_alpha1)

            cand_set = set()
            if query_alpha1 in cat_idx.exact_alpha_idx:
                cand_set.update(cat_idx.exact_alpha_idx[query_alpha1])
            if sorted_query_alpha and sorted_query_alpha in cat_idx.exact_sorted_alpha_idx:
                cand_set.update(cat_idx.exact_sorted_alpha_idx[sorted_query_alpha])
            for w in wtoks:
                if len(w) >= 4:
                    cand_set.update(cat_idx.word_idx.get(w[:4], []))
            if query_alpha1:
                cand_set.update(cat_idx.prefix3_idx.get(query_alpha1[:3], []))
                if len(query_alpha1) >= 4:
                    cand_set.update(cat_idx.suffix3_idx.get(query_alpha1[-3:], []))
                if len(cand_set) < 200:
                    cand_set.update(cat_idx.prefix2_idx.get(query_alpha1[:2], []))
            if query_skel1:
                cand_set.update(cat_idx.skel3_idx.get(query_skel1[:3], []))
            if query_phon1:
                cand_set.update(cat_idx.phon3_idx.get(query_phon1[:3], []))
            if ton > 0:
                cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.10, max_cands=500))
                cand_set.update(cat_idx.query_l_win_top_osa(ton, query_alpha1, sorted_query_alpha, rel_tol=0.10, top_k=40))
                if alt_ton > 0:
                    cand_set.update(cat_idx.query_tonnage_window(alt_ton, rel_tol=0.10, max_cands=400))
                if len(cand_set) < 200:
                    cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.20, max_cands=500))
            if nstr and nstr in cat_idx.num_idx:
                num_cands = cat_idx.num_idx[nstr]
                if len(num_cands) <= 500:
                    cand_set.update(num_cands)
            if not cand_set:
                if ton > 0:
                    cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.35, max_cands=300))
                if not cand_set:
                    cand_set.add(0)

            # Score candidates
            t_lifts = type_lift.get(kt, {})
            f_lifts = flag_lift.get(f"{kf}|{yb}", flag_lift.get(f"{kf}|ALL", {}))
            n_q_words = len(wtoks)

            scored_cands = []
            for c_i in cand_set:
                c_alpha = cat_idx.word_Alpha[c_i]
                c_s_alpha = cat_idx.sorted_word_Alpha[c_i]
                c_skel = cat_idx.skeletons[c_i]
                c_nstr = cat_idx.num_strs[c_i]
                c_nstr_ord = cat_idx.num_strs_ord[c_i]
                c_gt = cat_idx.gt[c_i]
                c_wtoks = cat_idx.word_tokens[c_i]
                c_s_wtoks = cat_idx.sorted_word_tokens[c_i]

                if query_alpha1 and c_alpha:
                    sim_alpha = 1.0 if query_alpha1 == c_alpha else OSA.normalized_similarity(query_alpha1, c_alpha)
                else:
                    sim_alpha = 0.0

                if n_q_words >= 2 and len(c_wtoks) >= 2 and sorted_query_alpha and c_s_alpha:
                    sim_s_alpha = 1.0 if sorted_query_alpha == c_s_alpha else OSA.normalized_similarity(sorted_query_alpha, c_s_alpha)
                    if 0.96 * sim_s_alpha > sim_alpha:
                        sim_alpha = 0.96 * sim_s_alpha

                sim_skel = OSA.normalized_similarity(query_skel1, c_skel) if (query_skel1 and c_skel) else 0.0

                typo_bonus = 0.0
                if n_q_words > 0 and n_q_words == len(c_wtoks):
                    w_sims = []
                    for a, b in zip(wtoks, c_wtoks):
                        if a == b:
                            w_sims.append(1.0)
                        else:
                            w_sims.append(OSA.normalized_similarity(a, b))
                    sim_tok = sum(w_sims) / n_q_words
                else:
                    sim_tok = sim_alpha

                name_score = 8.0 * sim_alpha + 1.0 * sim_skel + 2.0 * sim_tok + min(0.75, typo_bonus)
                if comp and comp == cat_idx.compact_names[c_i]:
                    name_score += 2.5

                num_score = 0.0
                if (nstr and c_nstr and nstr == c_nstr) or (nstr_ord and c_nstr_ord and nstr_ord == c_nstr_ord):
                    num_score = 3.2
                elif nstr and c_nstr and nstr != c_nstr:
                    num_score = -4.0
                elif nstr and not c_nstr_ord:
                    num_score = -2.2
                elif not nstr_ord and c_nstr:
                    num_score = -1.2

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
                    else:
                        ton_score = -3.5 * min(2.5, rel * 6.0)

                ct = cat_idx.types[c_i]
                cf = cat_idx.flags[c_i]
                t_val = t_lifts.get(ct, -0.15)
                f_val = f_lifts.get(cf, 0.0) if kf != "UNK" else 0.0
                if yb == "OLD" and f_val < 0:
                    f_val *= 0.25
                meta_score = 0.40 * t_val + 0.32 * f_val

                total_score = name_score + num_score + ton_score + meta_score
                scored_cands.append((total_score, c_i))

            scored_cands.sort(key=lambda x: (-x[0], cat_idx.vessel_ids[x[1]]))
            best_vid = cat_idx.vessel_ids[scored_cands[0][1]]

            cand_vids = {cat_idx.vessel_ids[c_i] for _, c_i in scored_cands}
            gold_in_cands = (gold_vid in cand_vids)
            is_correct = (best_vid == gold_vid)

            grank = 0
            for r_idx, (_, c_i) in enumerate(scored_cands, start=1):
                if cat_idx.vessel_ids[c_i] == gold_vid:
                    grank = r_idx
                    break

            err_type = "None" if is_correct else ("Ranking Error" if gold_in_cands else "Retrieval Miss")
            rows_detail.append({
                "sample_id": s_id,
                "hangul": False,
                "vessel_name": raw_name,
                "gold_vessel_id": gold_vid,
                "pred_vessel_id": best_vid,
                "is_correct": is_correct,
                "gold_in_candidates": gold_in_cands,
                "candidates_count": len(cand_set),
                "gold_rank": grank,
                "error_type": err_type,
            })

        else:
            # Hangul row
            comp, wtoks, ntoks, kr_ord_nstr, cond_a_2char_ho = parse_kr_row_features(raw_name, kr_mode="AR06")
            nstr = "|".join(ntoks)
            cands_pairs = [kr_word_to_en_candidates(w, syl_top2, word_map) for w in wtoks]
            en_wtoks1 = [p[0] for p in cands_pairs]
            en_wtoks2 = [p[1] for p in cands_pairs]
            query_alpha1 = "".join(en_wtoks1)
            query_alpha2 = "".join(en_wtoks2)
            query_phon_words = [
                en_to_phonetic(w_en) if (w in word_map or not re_hangul.search(w_en)) else kr_to_phonetic(w)
                for w, w_en in zip(wtoks, en_wtoks1)
            ]
            query_phon_full = "".join(query_phon_words)
            query_skel1 = consonant_skeleton(query_alpha1)
            query_skel2 = consonant_skeleton(query_alpha2)

            expanded_wtoks = []
            for w in wtoks:
                acr = extract_kr_acronym_prefix(w)
                if acr is not None and acr[1]:
                    expanded_wtoks.extend([acr[0], acr[1]])
                elif acr is not None and not acr[1]:
                    expanded_wtoks.append(w)
                else:
                    expanded_wtoks.append(w)

            idea_b_word_cands = [generate_idea_b_word_variants(w, syl_top3, word_top2) for w in expanded_wtoks]
            idea_b_full_variants = []
            for v_idx in range(8):
                full_v = "".join(wc[min(v_idx, len(wc) - 1)] for wc in idea_b_word_cands if wc)
                if full_v and full_v not in idea_b_full_variants:
                    idea_b_full_variants.append(full_v)

            full_kr_concat = "".join(wtoks)
            if full_kr_concat in IDEA_B_WORD_VARS:
                for fv_direct in IDEA_B_WORD_VARS[full_kr_concat]:
                    if fv_direct not in idea_b_full_variants:
                        idea_b_full_variants.insert(0, fv_direct)

            wtoks_no_ord_cands = None
            if kr_ord_nstr and len(expanded_wtoks) >= 2:
                wtoks_no_ord_cands = idea_b_word_cands[:-1]
                for v_idx in range(3):
                    full_v_no_ord = "".join(wc[min(v_idx, len(wc) - 1)] for wc in wtoks_no_ord_cands if wc)
                    if full_v_no_ord and full_v_no_ord not in idea_b_full_variants:
                        idea_b_full_variants.append(full_v_no_ord)

            raw_clean = str(raw_name).strip()
            ends_with_ho_raw = raw_clean.endswith("호") and bool(wtoks) and not ntoks
            ho_preserved_alpha = (query_alpha1 + "HO") if ends_with_ho_raw else None
            ho_preserved_phon = (query_phon_full + en_to_phonetic("HO")) if ends_with_ho_raw else None
            if cond_a_2char_ho and ho_preserved_alpha and ho_preserved_alpha not in idea_b_full_variants:
                idea_b_full_variants.append(ho_preserved_alpha)

            cand_set = set()
            if query_alpha1 in cat_idx.exact_alpha_idx:
                cand_set.update(cat_idx.exact_alpha_idx[query_alpha1])
            if ton > 0:
                cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.098, max_cands=None))
            if alt_ton > 0:
                cand_set.update(cat_idx.query_tonnage_window(alt_ton, rel_tol=0.098, max_cands=None))
            if query_alpha1:
                cand_set.update(cat_idx.prefix3_idx.get(query_alpha1[:3], []))
            if query_skel1:
                cand_set.update(cat_idx.skel3_idx.get(query_skel1[:3], []))
            if query_phon_full:
                cand_set.update(cat_idx.phon3_idx.get(query_phon_full[:3], []))
            for w in en_wtoks1:
                if len(w) >= 3:
                    cand_set.update(cat_idx.word_idx.get(w[:4], []))
            for fv in idea_b_full_variants[:7]:
                if fv in cat_idx.exact_alpha_idx:
                    cand_set.update(cat_idx.exact_alpha_idx[fv])
                if len(fv) >= 3:
                    cand_set.update(cat_idx.prefix3_idx.get(fv[:3], []))
            for wc in idea_b_word_cands:
                for w in wc[:3]:
                    if len(w) >= 3:
                        cand_set.update(cat_idx.word_idx.get(w[:4], []))
            if not cand_set:
                if ton > 0:
                    cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.25, max_cands=300))
                if not cand_set:
                    cand_set.add(0)

            t_lifts = type_lift.get(kt, {})
            f_lifts = flag_lift.get(f"{kf}|{yb}", flag_lift.get(f"{kf}|ALL", {}))
            n_q_words = len(wtoks)

            scored_cands = []
            for c_i in cand_set:
                c_alpha = cat_idx.word_Alpha[c_i]
                c_skel = cat_idx.skeletons[c_i]
                c_pfull = cat_idx.phon_full[c_i]
                c_wtoks = cat_idx.word_tokens[c_i]
                c_pwords = cat_idx.phon_words[c_i]
                c_nstr = cat_idx.num_strs[c_i]
                c_nstr_ord = cat_idx.num_strs_ord[c_i]
                c_gt = cat_idx.gt[c_i]

                sim_a1 = OSA.normalized_similarity(query_alpha1, c_alpha) if (query_alpha1 and c_alpha) else 0.0
                sim_a2 = OSA.normalized_similarity(query_alpha2, c_alpha) if (query_alpha2 and c_alpha) else 0.0
                sim_alpha = max(sim_a1, sim_a2)

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
                        OSA.normalized_similarity(consonant_skeleton(fv), c_skel)
                        for fv in idea_b_full_variants[:5]
                    ) if c_skel else 0.0
                    sim_skel = max(sim_skel, best_b_skel)
                    best_b_phon = max(
                        OSA.normalized_similarity(en_to_phonetic(fv), c_pfull)
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

                if n_b_words := len(idea_b_word_cands):
                    if n_b_words == len(c_wtoks):
                        bw_sims = []
                        for wc_list, cw, cpw in zip(idea_b_word_cands, c_wtoks, c_pwords):
                            best_w = max(OSA.normalized_similarity(cand_w, cw) for cand_w in wc_list)
                            best_pw = max(
                                OSA.normalized_similarity(en_to_phonetic(cand_w), cpw)
                                for cand_w in wc_list[:4]
                            ) if cpw else 0.0
                            bw_sims.append(max(best_w, 0.95 * best_pw))
                        sim_tok = max(sim_tok, sum(bw_sims) / n_b_words) if matched_word_count else (sum(bw_sims) / n_b_words)
                        matched_word_count = True

                name_score = 7.0 * sim_alpha + 1.2 * sim_skel + 2.0 * sim_phon + 2.0 * sim_tok

                num_score = 0.0
                if (nstr and c_nstr and nstr == c_nstr) or (kr_ord_nstr and c_nstr_ord and kr_ord_nstr == c_nstr_ord):
                    num_score = 3.2
                elif nstr and c_nstr and nstr != c_nstr:
                    num_score = -4.0
                elif nstr and not c_nstr:
                    num_score = -2.2
                elif not nstr and c_nstr:
                    num_score = -1.2

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

                ct = cat_idx.types[c_i]
                cf = cat_idx.flags[c_i]

                # Soft meta
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
                    meta_score = 0.0

                total_score = name_score + num_score + ton_score + meta_score
                scored_cands.append((total_score, c_i))

            scored_cands.sort(key=lambda x: (-x[0], cat_idx.vessel_ids[x[1]]))

            # Lattice refinement
            if len(scored_cands) > 1 and re_hangul.search(full_kr_concat):
                top_k = min(15, len(scored_cands))
                refined = []
                for base_sc, c_i in scored_cands[:top_k]:
                    c_alpha = cat_idx.word_Alpha[c_i]
                    c_wtoks = cat_idx.word_tokens[c_i]
                    lat_sim = lattice_word_similarity(full_kr_concat, c_alpha, syl_top3)
                    if kr_ord_nstr and len(wtoks) >= 2 and c_idx_has_num(cat_idx, c_i):
                        lat_sim = max(lat_sim, lattice_word_similarity("".join(wtoks[:-1]), c_alpha, syl_top3))
                    allow_ho_lat = ends_with_ho_raw and (cond_a_2char_ho or (len(c_wtoks) > 0 and c_wtoks[-1] == "HO"))
                    if allow_ho_lat and c_alpha.endswith("HO"):
                        lat_sim = max(lat_sim, lattice_word_similarity(full_kr_concat + "호", c_alpha, syl_top3))
                    refined.append((base_sc + 1.8 * lat_sim, c_i))
                refined.sort(key=lambda x: (-x[0], cat_idx.vessel_ids[x[1]]))
                scored_cands = refined + scored_cands[top_k:]

            best_vid = cat_idx.vessel_ids[scored_cands[0][1]]
            cand_vids = {cat_idx.vessel_ids[c_i] for _, c_i in scored_cands}
            gold_in_cands = (gold_vid in cand_vids)
            is_correct = (best_vid == gold_vid)

            grank = 0
            for r_idx, (_, c_i) in enumerate(scored_cands, start=1):
                if cat_idx.vessel_ids[c_i] == gold_vid:
                    grank = r_idx
                    break

            err_type = "None" if is_correct else ("Ranking Error" if gold_in_cands else "Retrieval Miss")
            rows_detail.append({
                "sample_id": s_id,
                "hangul": True,
                "vessel_name": raw_name,
                "gold_vessel_id": gold_vid,
                "pred_vessel_id": best_vid,
                "is_correct": is_correct,
                "gold_in_candidates": gold_in_cands,
                "candidates_count": len(cand_set),
                "gold_rank": grank,
                "error_type": err_type,
            })

    eval_sec = time.perf_counter() - t0_eval
    df_diag = pd.DataFrame(rows_detail)

    # Summary metrics
    n_total = len(df_diag)
    df_kr = df_diag[df_diag["hangul"]]
    df_en = df_diag[~df_diag["hangul"]]

    acc_total = df_diag["is_correct"].mean()
    acc_kr = df_kr["is_correct"].mean()
    acc_en = df_en["is_correct"].mean()
    weighted_acc = 0.7 * acc_kr + 0.3 * acc_en

    recall_total = df_diag["gold_in_candidates"].mean()
    recall_kr = df_kr["gold_in_candidates"].mean()
    recall_en = df_en["gold_in_candidates"].mean()

    miss_kr = int((~df_kr["gold_in_candidates"]).sum())
    rank_err_kr = int((df_kr["gold_in_candidates"] & ~df_kr["is_correct"]).sum())
    correct_kr = int(df_kr["is_correct"].sum())

    miss_en = int((~df_en["gold_in_candidates"]).sum())
    rank_err_en = int((df_en["gold_in_candidates"] & ~df_en["is_correct"]).sum())
    correct_en = int(df_en["is_correct"].sum())

    print("\n" + "=" * 70)
    print("=== [3] Fold 0 Baseline Diagnostic Results (AR-06 on Fold 0) ===")
    print("=" * 70)
    print(f"Total Rows: {n_total:,} | Hangul: {len(df_kr):,} | Non-Hangul: {len(df_en):,}")
    print(f"Evaluation Time: {eval_sec:.2f}s ({eval_sec/n_total*1000:.2f}ms/row)")
    print(f"\n[Accuracy & Official Weighted Score]")
    print(f"  * Hangul Accuracy:     {acc_kr*100:.3f}% ({correct_kr}/{len(df_kr)})")
    print(f"  * Non-Hangul Accuracy: {acc_en*100:.3f}% ({correct_en}/{len(df_en)})")
    print(f"  * Official Weighted:   {weighted_acc*100:.3f}%")
    print(f"\n[Candidate Inclusion Rate (Retrieval Coverage)]")
    print(f"  * Hangul Coverage:     {recall_kr*100:.3f}% ({int(df_kr['gold_in_candidates'].sum())}/{len(df_kr)})")
    print(f"  * Non-Hangul Coverage: {recall_en*100:.3f}% ({int(df_en['gold_in_candidates'].sum())}/{len(df_en)})")
    print(f"  * Total Coverage:      {recall_total*100:.3f}% ({int(df_diag['gold_in_candidates'].sum())}/{n_total})")
    print(f"\n[Error Dissection: Retrieval Miss vs Ranking Error]")
    print(f"  * Hangul Errors:     Total {len(df_kr)-correct_kr} | Retrieval Miss (누락): {miss_kr}행 | Ranking Error (순위 밀림): {rank_err_kr}행")
    print(f"  * Non-Hangul Errors: Total {len(df_en)-correct_en} | Retrieval Miss (누락): {miss_en}행 | Ranking Error (순위 밀림): {rank_err_en}행")
    print("=" * 70)

    # Save output artifacts
    out_dir = ROOT / "reports/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fold0_baseline_diagnostics.csv"
    df_diag.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = out_dir / "fold0_baseline_diagnostics.json"
    summary = {
        "benchmark_id": "DIAG-FOLD0-BASELINE-01",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_rows": n_total,
        "hangul_rows": len(df_kr),
        "non_hangul_rows": len(df_en),
        "eval_seconds": round(eval_sec, 2),
        "metrics": {
            "hangul_accuracy": round(acc_kr, 5),
            "hangul_correct": correct_kr,
            "non_hangul_accuracy": round(acc_en, 5),
            "non_hangul_correct": correct_en,
            "weighted_accuracy": round(weighted_acc, 5),
            "hangul_candidate_coverage": round(recall_kr, 5),
            "non_hangul_candidate_coverage": round(recall_en, 5),
            "total_candidate_coverage": round(recall_total, 5),
        },
        "errors": {
            "hangul": {
                "total_errors": len(df_kr) - correct_kr,
                "retrieval_miss": miss_kr,
                "ranking_error": rank_err_kr,
            },
            "non_hangul": {
                "total_errors": len(df_en) - correct_en,
                "retrieval_miss": miss_en,
                "ranking_error": rank_err_en,
            },
        },
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed diagnostics to:\n - {csv_path}\n - {json_path}")


if __name__ == "__main__":
    run_diagnostics()
