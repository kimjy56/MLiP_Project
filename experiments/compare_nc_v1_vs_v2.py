"""Compare reverse channel V1 (AS-HCX-NC-01) vs V2 (AS-HCX-NC-02).

Evaluates:
1. Target likelihood and separation margin on known noise/transliteration test cases.
2. Direct ranking on the 12 baseline error target cases (including renamed/sister ships).
3. Standalone accuracy on 32-row pilot (16 Hangul, 16 Non-Hangul) without baseline heuristic rule score.
Runs strictly under 2 threads.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "AR-06") not in sys.path:
    sys.path.insert(0, str(ROOT / "AR-06"))

import numpy as np
import pandas as pd
import torch

from script import CatalogIndex, fit_matching_model, predict_rows
from name_decoder.noisy_channel import NoisyChannelScorer
from name_decoder.factored_noisy_channel import FactoredNoisyChannelMatcher

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

V1_MODEL_PATH = ROOT / "artifacts/AS-HCX-NC-01/model"
V2_MODEL_PATH = ROOT / "artifacts/AS-HCX-NC-02/model"
REPORTS_DIR = ROOT / "reports/noisy_channel"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORTS_DIR / "comparison_v1_vs_v2.json"

KEY_CASES = [
    {
        "case_name": "HEMISPHERE (transposition typo)",
        "observed_name": "M/V HEMISPEHRE",
        "correct_catalog": "HEMISPHERE",
        "wrong_catalog": "Zhe Hai You 5",
    },
    {
        "case_name": "SEASTAR VULCAN (letter deletion)",
        "observed_name": "SEASTAR VULAN",
        "correct_catalog": "SEASTAR VULCAN",
        "wrong_catalog": "Seastar Valiant",
    },
    {
        "case_name": "ALWINE (letter deletion)",
        "observed_name": "ALWNE",
        "correct_catalog": "ALWINE",
        "wrong_catalog": "Alani",
    },
    {
        "case_name": "MARSGRACHT (Hangul phonetics)",
        "observed_name": "마스그라치",
        "correct_catalog": "MARSGRACHT",
        "wrong_catalog": "MAASGRACHT",
    },
    {
        "case_name": "SEA WOLF (Hangul phonetics)",
        "observed_name": "씨 울프",
        "correct_catalog": "SEA WOLF",
        "wrong_catalog": "Gulf No. 16",
    },
    {
        "case_name": "MORNING LENA (sister ship tie)",
        "observed_name": "모닝 레나",
        "correct_catalog": "MORNING LENA",
        "wrong_catalog": "Morning Cara",
    },
]


def evaluate_key_cases(scorer_v1, scorer_v2):
    results = []
    for case in KEY_CASES:
        obs = case["observed_name"]
        corr = case["correct_catalog"]
        wrong = case["wrong_catalog"]

        s1_corr = scorer_v1.score_pair(corr, obs)
        s1_wrong = scorer_v1.score_pair(wrong, obs)
        m1 = s1_corr.mean_log_probability - s1_wrong.mean_log_probability

        s2_corr = scorer_v2.score_pair(corr, obs)
        s2_wrong = scorer_v2.score_pair(wrong, obs)
        m2 = s2_corr.mean_log_probability - s2_wrong.mean_log_probability

        results.append({
            "case": case["case_name"],
            "observed": obs,
            "correct_catalog": corr,
            "wrong_catalog": wrong,
            "v1": {
                "correct_mean_logp": s1_corr.mean_log_probability,
                "wrong_mean_logp": s1_wrong.mean_log_probability,
                "margin": m1,
            },
            "v2": {
                "correct_mean_logp": s2_corr.mean_log_probability,
                "wrong_mean_logp": s2_wrong.mean_log_probability,
                "margin": m2,
            },
            "margin_diff": m2 - m1,
            "v2_sharper": m2 > m1,
        })
        time.sleep(0.01)
    return results


def evaluate_target_12_cases(matcher, target_df, target_cands, name_by_id):
    results = []
    hits = 0
    for idx, (_, row) in enumerate(target_df.iterrows()):
        sid = row["sample_id"]
        obs_name = row["vessel_name"]
        gold_vid = row["target_vessel_id"]
        gold_name = name_by_id.get(gold_vid, "")
        cands = target_cands[idx]
        cand_vids = [c[1] for c in cands]

        ranked = matcher.rank_candidates(row.to_dict(), cand_vids)
        top1_vid = ranked[0][1]
        top1_name = name_by_id.get(top1_vid, "")
        is_hit = (top1_vid == gold_vid)
        if is_hit:
            hits += 1

        gold_rank = -1
        for r_idx, (_, c_vid, _) in enumerate(ranked):
            if c_vid == gold_vid:
                gold_rank = r_idx + 1
                break

        margin = 0.0
        if len(ranked) > 1:
            margin = ranked[0][0] - ranked[1][0]

        results.append({
            "sample_id": sid,
            "obs_name": str(obs_name),
            "gold_name": gold_name,
            "top1_name": top1_name,
            "gold_rank": gold_rank,
            "is_hit": is_hit,
            "top1_margin": margin,
            "components": ranked[0][2],
        })
        time.sleep(0.01)
    return hits, results


def evaluate_pilot_factored(matcher, pilot_df, pilot_cands):
    correct_ko, total_ko = 0, 0
    correct_en, total_en = 0, 0

    for idx, (_, row) in enumerate(pilot_df.iterrows()):
        gold_id = row["target_vessel_id"]
        vname = str(row["vessel_name"]) if pd.notna(row["vessel_name"]) else ""
        is_ko = bool(re.search("[가-힣]", vname))
        cands = pilot_cands[idx]
        cand_vids = [c[1] for c in cands]

        ranked = matcher.rank_candidates(row.to_dict(), cand_vids)
        top1_vid = ranked[0][1]
        is_hit = (top1_vid == gold_id)

        if is_ko:
            total_ko += 1
            if is_hit:
                correct_ko += 1
        else:
            total_en += 1
            if is_hit:
                correct_en += 1

        time.sleep(0.01)

    ko_acc = correct_ko / total_ko if total_ko else 0.0
    en_acc = correct_en / total_en if total_en else 0.0
    weighted_acc = 0.7 * ko_acc + 0.3 * en_acc

    return {
        "hangul_accuracy": ko_acc,
        "hangul_correct": correct_ko,
        "hangul_total": total_ko,
        "non_hangul_accuracy": en_acc,
        "non_hangul_correct": correct_en,
        "non_hangul_total": total_en,
        "weighted_accuracy": weighted_acc,
    }


def main():
    print("Loading datasets...")
    data_dir = ROOT / "data"
    train_df = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
    catalog_df = pd.read_csv(data_dir / "vessel_catalog.csv", encoding="utf-8-sig")
    folds_df = pd.read_csv(ROOT / "eval_kit/folds_group5_seed42.csv", dtype={"sample_id": str, "fold": int}, encoding="utf-8-sig")
    name_by_id = dict(zip(catalog_df["vessel_id"], catalog_df["vessel_name"]))

    merged = train_df.merge(folds_df[["sample_id", "fold"]], on="sample_id", how="inner")
    fit_df = merged[merged["fold"] != 0].reset_index(drop=True)
    val_df = merged[merged["fold"] == 0].reset_index(drop=True)

    print("Fitting baseline spec transition model and index...")
    cat_idx = CatalogIndex(catalog_df)
    base_model = fit_matching_model(fit_df, catalog_df, kr_mode="AR06", en_mode="EN_V3", use_train_lookup=False)

    # 1. Target 12 error cases setup
    target_sids = [
        "TRAIN_07693", "TRAIN_11797",  # 마스그라치 (한글 오답 2건)
        "TRAIN_03728", "TRAIN_03799", "TRAIN_05971", "TRAIN_06864", "TRAIN_07851",
        "TRAIN_08761", "TRAIN_09393", "TRAIN_10022", "TRAIN_10986", "TRAIN_11270"
    ]
    target_df = merged[merged["sample_id"].isin(target_sids)].copy().reset_index(drop=True)
    _, _, target_cands = predict_rows(
        target_df, base_model, cat_idx, use_train_lookup=False,
        gold_ids=target_df["target_vessel_id"].tolist(), top_candidates=35
    )

    # 2. Pilot 32 rows setup
    val_ko = val_df[val_df.vessel_name.str.contains(r"[\uac00-\ud7a3]")].head(16)
    val_en = val_df[~val_df.vessel_name.str.contains(r"[\uac00-\ud7a3]")].head(16)
    pilot_df = pd.concat([val_ko, val_en], ignore_index=True)
    _, _, pilot_cands = predict_rows(
        pilot_df, base_model, cat_idx, use_train_lookup=False,
        gold_ids=pilot_df["target_vessel_id"].tolist(), top_candidates=35
    )

    print(f"Loading V1 Scorer: {V1_MODEL_PATH}")
    s1 = NoisyChannelScorer.load(V1_MODEL_PATH)
    matcher_v1 = FactoredNoisyChannelMatcher(
        scorer=s1, catalog_df=catalog_df, model_dict=base_model,
        ton_sigma=0.035, w_name=1.0, w_ton=1.5, w_flag=0.8, w_type=0.4
    )

    print(f"Loading V2 Scorer: {V2_MODEL_PATH}")
    s2 = NoisyChannelScorer.load(V2_MODEL_PATH)
    matcher_v2 = FactoredNoisyChannelMatcher(
        scorer=s2, catalog_df=catalog_df, model_dict=base_model,
        ton_sigma=0.035, w_name=1.0, w_ton=1.5, w_flag=0.8, w_type=0.4
    )

    print("\n" + "=" * 60)
    print("   [1. Key Noise Cases Separation Margin]")
    print("=" * 60)
    case_res = evaluate_key_cases(s1, s2)
    for c in case_res:
        print(f"[{c['case']}]")
        print(f"  Observed: {c['observed']}")
        print(f"  V1 Margin: {c['v1']['margin']:+.3f} (corr: {c['v1']['correct_mean_logp']:.2f}, wrong: {c['v1']['wrong_mean_logp']:.2f})")
        print(f"  V2 Margin: {c['v2']['margin']:+.3f} (corr: {c['v2']['correct_mean_logp']:.2f}, wrong: {c['v2']['wrong_mean_logp']:.2f})")
        print(f"  Delta: {c['margin_diff']:+.3f} ({'Sharper' if c['v2_sharper'] else 'Worse'})")

    print("\n" + "=" * 60)
    print("   [2. Target 12 Error Cases (Direct Ranking)]")
    print("=" * 60)
    v1_hits, t1_details = evaluate_target_12_cases(matcher_v1, target_df, target_cands, name_by_id)
    v2_hits, t2_details = evaluate_target_12_cases(matcher_v2, target_df, target_cands, name_by_id)
    print(f"V1 Rank 1 Hits: {v1_hits} / {len(target_df)} ({v1_hits/len(target_df)*100:.1f}%)")
    print(f"V2 Rank 1 Hits: {v2_hits} / {len(target_df)} ({v2_hits/len(target_df)*100:.1f}%)")
    for r1, r2 in zip(t1_details, t2_details):
        print(f"  {r1['sample_id']} ('{r1['obs_name']}'): V1 Rank={r1['gold_rank']}(m={r1['top1_margin']:.2f}) | V2 Rank={r2['gold_rank']}(m={r2['top1_margin']:.2f})")

    print("\n" + "=" * 60)
    print("   [3. 32-Row Independent Pilot Standalone Accuracy (Zero Heuristic Rule)]")
    print("=" * 60)
    pilot_v1 = evaluate_pilot_factored(matcher_v1, pilot_df, pilot_cands)
    print(f"V1: KO={pilot_v1['hangul_accuracy']*100:.2f}%, EN={pilot_v1['non_hangul_accuracy']*100:.2f}%, Weighted={pilot_v1['weighted_accuracy']*100:.2f}%")

    pilot_v2 = evaluate_pilot_factored(matcher_v2, pilot_df, pilot_cands)
    print(f"V2: KO={pilot_v2['hangul_accuracy']*100:.2f}%, EN={pilot_v2['non_hangul_accuracy']*100:.2f}%, Weighted={pilot_v2['weighted_accuracy']*100:.2f}%")

    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "key_noise_cases": case_res,
        "target_12_cases": {
            "v1_hits": v1_hits,
            "v2_hits": v2_hits,
            "total": len(target_df),
            "details_v1": t1_details,
            "details_v2": t2_details,
        },
        "pilot_factored_32": {
            "v1": pilot_v1,
            "v2": pilot_v2,
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
