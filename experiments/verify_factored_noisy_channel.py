"""Verification of Factored Noisy Channel (LLM Name Likelihood + Physical Spec Likelihoods).

Evaluates:
1. Direct ranking on the 10 famous baseline errors (including renamed ships & sister ships).
2. Standalone accuracy on 128-row pilot (64 Hangul, 64 Non-Hangul) without baseline heuristic name score.
3. Verification that physical specifications (tonnage, flag, type) allow the reverse model
   to directly pick the gold ship even when names differ (renamed vessels).
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import json
from pathlib import Path
import re
import sys
import time

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "AR-06") not in sys.path:
    sys.path.insert(0, str(ROOT / "AR-06"))

import numpy as np
import pandas as pd

from script import CatalogIndex, fit_matching_model, predict_rows
from name_decoder.noisy_channel import NoisyChannelScorer
from name_decoder.factored_noisy_channel import FactoredNoisyChannelMatcher


def is_kr(val: str) -> bool:
    return bool(re.search("[가-힣]", str(val)))


def run_factored_nc_verification():
    print("=" * 80)
    print("   [Factored Noisy Channel: Spec-Enhanced Reverse Model Direct Matching]")
    print("=" * 80)

    data_dir = ROOT / "data"
    train_path = data_dir / "train.csv"
    catalog_path = data_dir / "vessel_catalog.csv"
    folds_path = ROOT / "eval_kit/folds_group5_seed42.csv"

    # 1. Load data
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog_df = pd.read_csv(catalog_path, encoding="utf-8-sig")
    folds_df = pd.read_csv(folds_path, dtype={"sample_id": str, "fold": int}, encoding="utf-8-sig")
    name_by_id = dict(zip(catalog_df["vessel_id"], catalog_df["vessel_name"]))

    merged = train_df.merge(folds_df[["sample_id", "fold"]], on="sample_id", how="inner")
    fit_df = merged[merged["fold"] != 0].reset_index(drop=True)
    val_df = merged[merged["fold"] == 0].reset_index(drop=True)

    # 2. Build baseline catalog and fit Fold 0 specs
    cat_idx = CatalogIndex(catalog_df)
    model = fit_matching_model(fit_df, catalog_df, kr_mode="AR06", en_mode="EN_V3", use_train_lookup=False)

    # 3. Load Noisy Channel LLM scorer
    scorer = NoisyChannelScorer.load(ROOT / "artifacts/AS-HCX-NC-01/model")

    # 4. Initialize FactoredNoisyChannelMatcher
    matcher = FactoredNoisyChannelMatcher(
        scorer=scorer,
        catalog_df=catalog_df,
        model_dict=model,
        ton_sigma=0.035,
        w_name=1.0,
        w_ton=1.5,
        w_flag=0.8,
        w_type=0.4,
    )
    print("FactoredNoisyChannelMatcher initialized successfully.")

    # ---------------------------------------------------------
    # Test 1: Evaluation on 10 Baseline Error Cases + Marsgracht
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("   [Test 1] Direct Evaluation on 12 Target Error Cases (Sister ships & Renamed)")
    print("=" * 80)

    target_sids = [
        "TRAIN_07693", "TRAIN_11797",  # 마스그라치 (한글 오답 2건)
        "TRAIN_03728", "TRAIN_03799", "TRAIN_05971", "TRAIN_06864", "TRAIN_07851",
        "TRAIN_08761", "TRAIN_09393", "TRAIN_10022", "TRAIN_10986", "TRAIN_11270"
    ]
    target_df = merged[merged["sample_id"].isin(target_sids)].copy()

    # Get baseline candidates for these rows
    _, _, target_cands = predict_rows(
        target_df, model, cat_idx, use_train_lookup=False,
        gold_ids=target_df["target_vessel_id"].tolist(), top_candidates=35
    )

    factored_direct_correct = 0
    factored_results = []

    for idx, (_, row) in enumerate(target_df.iterrows()):
        sid = row["sample_id"]
        obs_name = row["vessel_name"]
        gold_vid = row["target_vessel_id"]
        gold_name = name_by_id[gold_vid]
        cands = target_cands[idx]
        cand_vids = [c[1] for c in cands]

        # Pure factored ranking
        ranked = matcher.rank_candidates(row.to_dict(), cand_vids)
        top1_vid = ranked[0][1]
        top1_name = name_by_id[top1_vid]
        is_hit = (top1_vid == gold_vid)
        if is_hit:
            factored_direct_correct += 1

        # Check gold rank
        gold_rank = -1
        for r_idx, (_, c_vid, _) in enumerate(ranked):
            if c_vid == gold_vid:
                gold_rank = r_idx + 1
                break

        res_item = {
            "sample_id": sid,
            "obs_name": obs_name,
            "gold_name": gold_name,
            "gold_vid": gold_vid,
            "top1_name": top1_name,
            "top1_vid": top1_vid,
            "is_correct": is_hit,
            "gold_rank": gold_rank,
            "top1_components": ranked[0][2],
        }
        factored_results.append(res_item)

        status = "[HIT +RESCUE]" if is_hit else "[MISS]"
        print(f"  {status} {sid}: '{obs_name}' -> Top1: '{top1_name}' (Rank {gold_rank}) | Gold: '{gold_name}'", flush=True)
        print(f"      Components: name={ranked[0][2]['llm_name_logp']:.2f}, ton={ranked[0][2]['ton_logp']:.2f}, flag={ranked[0][2]['flag_logp']:.2f}, type={ranked[0][2]['type_logp']:.2f} => Total={ranked[0][2]['total_logp']:.2f}", flush=True)
        time.sleep(0.02)  # Cooldown to protect CPU from thermal overload

    print(f"\nTarget Errors Direct Pure-Factored NC Correct: {factored_direct_correct}/{len(target_sids)} ({factored_direct_correct/len(target_sids)*100:.1f}%)", flush=True)

    # ---------------------------------------------------------
    # Test 2: Evaluation on 64-row pilot (32 Hangul, 32 Non-Hangul) - Thermal Safe
    # ---------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("   [Test 2] Standalone Pure-Factored NC Accuracy on Pilot (Thermal Safe)", flush=True)
    print("=" * 80, flush=True)

    val_kr = val_df[val_df["vessel_name"].apply(is_kr)].reset_index(drop=True)
    val_en = val_df[~val_df["vessel_name"].apply(is_kr)].reset_index(drop=True)
    n_sample = 32  # 32 KR, 32 EN = 64 total rows for thermal safety
    pilot_df = pd.concat([val_kr.iloc[:n_sample], val_en.iloc[:n_sample]], ignore_index=True)
    pilot_gold = pilot_df["target_vessel_id"].tolist()
    pilot_is_kr = pilot_df["vessel_name"].apply(is_kr).to_numpy()

    # Get Top-8 candidates from screening
    _, _, pilot_cands = predict_rows(
        pilot_df, model, cat_idx, use_train_lookup=False,
        gold_ids=pilot_gold, top_candidates=8
    )

    pure_preds = []
    t_start = time.perf_counter()
    for idx, (_, row) in enumerate(pilot_df.iterrows()):
        cand_vids = [c[1] for c in pilot_cands[idx]]
        ranked = matcher.rank_candidates(row.to_dict(), cand_vids)
        pure_preds.append(ranked[0][1])
        time.sleep(0.015)  # Cooldown between rows

    t_eval = time.perf_counter() - t_start
    corr = (np.array(pure_preds) == np.array(pilot_gold))
    kr_acc = corr[pilot_is_kr].mean()
    en_acc = corr[~pilot_is_kr].mean()
    weighted_acc = 0.7 * kr_acc + 0.3 * en_acc

    print(f"\nPure Factored Noisy Channel (No Baseline Heuristic Score at all):")
    print(f"  - Hangul Accuracy     : {kr_acc*100:.2f}% ({corr[pilot_is_kr].sum()}/{n_sample})")
    print(f"  - Non-Hangul Accuracy : {en_acc*100:.2f}% ({corr[~pilot_is_kr].sum()}/{n_sample})")
    print(f"  - Weighted Accuracy   : {weighted_acc*100:.3f}% ({corr.sum()}/{len(pilot_df)})")
    print(f"  - Inference Time      : {t_eval:.2f}s ({t_eval/len(pilot_df):.3f}s/row)")

    # Save verification report
    out_dir = ROOT / "reports/factored_noisy_channel"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "factored_nc_verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_id": "FACTORED-NC-VERIFY-01",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "weights": {
                "w_name": matcher.w_name,
                "w_ton": matcher.w_ton,
                "w_flag": matcher.w_flag,
                "w_type": matcher.w_type,
                "ton_sigma": matcher.ton_sigma,
            },
            "pilot_results": {
                "weighted_accuracy": round(weighted_acc, 5),
                "hangul_accuracy": round(kr_acc, 5),
                "non_hangul_accuracy": round(en_acc, 5),
                "correct_count": int(corr.sum()),
                "total_count": len(pilot_df),
                "eval_seconds": round(t_eval, 2),
            },
            "target_errors_results": factored_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    run_factored_nc_verification()
