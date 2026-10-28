# -*- coding: utf-8 -*-
"""experiments/evaluate_lexicon_ablation.py — Ablation Study: Manual Lexicon (IDEA_B_WORD_VARS) Removal vs Retention.

Compares:
  Condition 1 (Retained): Current DPR-MULTIVIEW-02 with manual dictionary (IDEA_B_WORD_VARS active)
  Condition 2 (Ablated): Pure Dense + Learned Model without manual dictionary (IDEA_B_WORD_VARS = {})

Evaluates on Heldout Fold 0 (2,372 rows) across:
  - Clean Condition (ko_map OFF)
  - Severe Noise Condition (ko_map OFF, seed 20261022)
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_kit.robustness import PRESETS, RobustnessPerturber
import experimental_matchers.dpr_hybrid.script as dpr_script
from experimental_matchers.dpr_hybrid.script import (
    CatalogIndex,
    fit_matching_model,
    is_korean_name,
    predict_rows,
)
from phonetic_encoder.hybrid_matcher import HybridDenseLexicalMatcher


def calc_metrics(preds: List[str], gold: List[str], is_ko: np.ndarray) -> Dict[str, Any]:
    correct = np.asarray(preds, dtype=object) == np.asarray(gold, dtype=object)
    h_acc = float(correct[is_ko].mean()) if is_ko.any() else None
    n_acc = float(correct[~is_ko].mean()) if (~is_ko).any() else None
    w_acc = 0.7 * h_acc + 0.3 * n_acc if (h_acc is not None and n_acc is not None) else None
    return {
        "hangul_accuracy": h_acc,
        "hangul_correct": int(correct[is_ko].sum()),
        "hangul_rows": int(is_ko.sum()),
        "non_hangul_accuracy": n_acc,
        "non_hangul_correct": int(correct[~is_ko].sum()),
        "non_hangul_rows": int((~is_ko).sum()),
        "weighted_accuracy": w_acc,
    }


def find_diffs(
    df: pd.DataFrame,
    preds_a: List[str],
    preds_b: List[str],
    gold: List[str],
    is_ko: np.ndarray,
) -> Dict[str, Any]:
    """Find differences between Mode A (Retained) and Mode B (Ablated)."""
    fixes = []   # A was wrong, B is right
    breaks = []  # A was right, B is wrong
    for i in range(len(df)):
        g = gold[i]
        pa = preds_a[i]
        pb = preds_b[i]
        sid = str(df.iloc[i].get("sample_id", f"row_{i}"))
        vname = str(df.iloc[i].get("vessel_name", ""))
        ko_flag = bool(is_ko[i])

        if pa != g and pb == g:
            fixes.append({
                "sample_id": sid,
                "vessel_name": vname,
                "is_korean": ko_flag,
                "gold_id": g,
                "retained_pred": pa,
                "ablated_pred": pb,
            })
        elif pa == g and pb != g:
            breaks.append({
                "sample_id": sid,
                "vessel_name": vname,
                "is_korean": ko_flag,
                "gold_id": g,
                "retained_pred": pa,
                "ablated_pred": pb,
            })

    return {
        "num_fixes": len(fixes),
        "num_breaks": len(breaks),
        "net_gain": len(fixes) - len(breaks),
        "fixes": fixes,
        "breaks": breaks,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Manual Lexicon Ablation.")
    parser.add_argument("--catalog-csv", default=str(ROOT / "data/vessel_catalog.csv"))
    parser.add_argument("--train-csv", default=str(ROOT / "data/train.csv"))
    parser.add_argument("--folds-csv", default=str(ROOT / "eval_kit/folds_group5_seed42.csv"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(ROOT / "artifacts/DPR-SYMPHON-01/fold_0_encoder.pt"))
    parser.add_argument("--catalog-embeds", default=str(ROOT / "artifacts/DPR-SYMPHON-01/catalog_embeds_multiview_fp16.npy"))
    parser.add_argument("--output-json", default=str(ROOT / "reports/matching_improvements/DPR-MULTIVIEW-02/lexicon_ablation_study.json"))
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--dense-weight", type=float, default=2.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading data and indices (device: {device})...")

    catalog_df = pd.read_csv(args.catalog_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    train_df = pd.read_csv(args.train_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    folds_df = pd.read_csv(args.folds_csv, dtype={"sample_id": str, "fold": int}, encoding="utf-8-sig")

    merged = train_df.merge(folds_df[["sample_id", "fold"]], on="sample_id")
    fit_df = merged[merged.fold != args.fold].reset_index(drop=True)
    val_df = merged[merged.fold == args.fold].reset_index(drop=True)

    print(f"Fit data: {len(fit_df):,} rows (folds != {args.fold}). Val data: {len(val_df):,} rows (fold {args.fold}).")

    print("Fitting lexical matcher model...")
    model = fit_matching_model(fit_df, catalog_df, kr_mode="AR06", en_mode="EN_V3", use_train_lookup=False)
    cat_idx = CatalogIndex(catalog_df, full_ko_map={})

    print("Initializing HybridDenseLexicalMatcher...")
    dense_matcher = HybridDenseLexicalMatcher(
        model_checkpoint=args.checkpoint,
        catalog_embeds_npy=args.catalog_embeds,
        catalog_csv=args.catalog_csv,
        device=device,
    )

    gold = val_df["target_vessel_id"].tolist()
    is_ko = val_df["vessel_name"].apply(is_korean_name).to_numpy()

    # Original Manual Lexicon
    ORIG_IDEA_B = dict(dpr_script.IDEA_B_WORD_VARS)
    print(f"Original IDEA_B_WORD_VARS contains {len(ORIG_IDEA_B):,} manual word mappings.")

    results: Dict[str, Any] = {
        "experiment_id": "DPR-ABLATE-LEXICON-01",
        "heldout_fold": args.fold,
        "rows": len(val_df),
        "seed": args.seed,
        "dense_weight": args.dense_weight,
        "manual_dict_size": len(ORIG_IDEA_B),
        "evaluations": {},
    }

    # -------------------------------------------------------------
    # 1. CLEAN HELDOUT EVALUATION
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(">>> 1. Clean Heldout Fold 0 Evaluation (ko_map OFF)")
    print("=" * 65)

    # A. Retained (IDEA_B_WORD_VARS ON)
    dpr_script.IDEA_B_WORD_VARS = dict(ORIG_IDEA_B)
    t0 = time.perf_counter()
    preds_clean_retained, _ = predict_rows(
        val_df, model, cat_idx,
        use_train_lookup=False, gold_ids=gold,
        dense_matcher=dense_matcher, dense_mode="hybrid", dense_weight=args.dense_weight,
    )
    time_clean_retained = time.perf_counter() - t0
    m_clean_retained = calc_metrics(preds_clean_retained, gold, is_ko)
    m_clean_retained["inference_time_sec"] = time_clean_retained

    # B. Ablated (IDEA_B_WORD_VARS OFF = {})
    dpr_script.IDEA_B_WORD_VARS = {}
    t1 = time.perf_counter()
    preds_clean_ablated, _ = predict_rows(
        val_df, model, cat_idx,
        use_train_lookup=False, gold_ids=gold,
        dense_matcher=dense_matcher, dense_mode="hybrid", dense_weight=args.dense_weight,
    )
    time_clean_ablated = time.perf_counter() - t1
    m_clean_ablated = calc_metrics(preds_clean_ablated, gold, is_ko)
    m_clean_ablated["inference_time_sec"] = time_clean_ablated

    diff_clean = find_diffs(val_df, preds_clean_retained, preds_clean_ablated, gold, is_ko)
    print(f"  [Retained (Manual Dict ON )] Weighted: {m_clean_retained['weighted_accuracy']*100:.4f}% | Hangul: {m_clean_retained['hangul_accuracy']*100:.4f}% | Time: {time_clean_retained:.2f}s")
    print(f"  [Ablated  (Manual Dict OFF)] Weighted: {m_clean_ablated['weighted_accuracy']*100:.4f}% | Hangul: {m_clean_ablated['hangul_accuracy']*100:.4f}% | Time: {time_clean_ablated:.2f}s")
    print(f"  --> Fixes (Ablated right): {diff_clean['num_fixes']}, Breaks (Ablated wrong): {diff_clean['num_breaks']}, Net Gain: {diff_clean['net_gain']}")
    if diff_clean["breaks"]:
        for b in diff_clean["breaks"][:5]:
            print(f"      * Break: {b['sample_id']} '{b['vessel_name']}' -> Retained: {b['retained_pred']}, Ablated: {b['ablated_pred']} (Gold: {b['gold_id']})")

    results["evaluations"]["clean"] = {
        "retained": m_clean_retained,
        "ablated": m_clean_ablated,
        "diff": diff_clean,
    }

    # -------------------------------------------------------------
    # 2. SEVERE NOISE EVALUATION
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f">>> 2. Severe Noise Evaluation (ko_map OFF, seed {args.seed})")
    print("=" * 65)

    perturber = RobustnessPerturber(deepcopy(PRESETS["severe"]), seed=args.seed)
    noisy_df, summ, _ = perturber.perturb_dataset(val_df, catalog_df, return_row_audits=True)
    noise_rate = summ.get("any_noise_applied_rate", 0.0)
    print(f"Noise applied rate: {noise_rate * 100:.2f}%")
    test_is_ko = noisy_df["vessel_name"].apply(is_korean_name).to_numpy()

    # A. Retained (IDEA_B_WORD_VARS ON)
    dpr_script.IDEA_B_WORD_VARS = dict(ORIG_IDEA_B)
    t0 = time.perf_counter()
    preds_severe_retained, _ = predict_rows(
        noisy_df, model, cat_idx,
        use_train_lookup=False, gold_ids=gold,
        dense_matcher=dense_matcher, dense_mode="hybrid", dense_weight=args.dense_weight,
    )
    time_severe_retained = time.perf_counter() - t0
    m_severe_retained = calc_metrics(preds_severe_retained, gold, test_is_ko)
    m_severe_retained["inference_time_sec"] = time_severe_retained

    # B. Ablated (IDEA_B_WORD_VARS OFF = {})
    dpr_script.IDEA_B_WORD_VARS = {}
    t1 = time.perf_counter()
    preds_severe_ablated, _ = predict_rows(
        noisy_df, model, cat_idx,
        use_train_lookup=False, gold_ids=gold,
        dense_matcher=dense_matcher, dense_mode="hybrid", dense_weight=args.dense_weight,
    )
    time_severe_ablated = time.perf_counter() - t1
    m_severe_ablated = calc_metrics(preds_severe_ablated, gold, test_is_ko)
    m_severe_ablated["inference_time_sec"] = time_severe_ablated

    diff_severe = find_diffs(noisy_df, preds_severe_retained, preds_severe_ablated, gold, test_is_ko)
    print(f"  [Retained (Manual Dict ON )] Weighted: {m_severe_retained['weighted_accuracy']*100:.4f}% | Hangul: {m_severe_retained['hangul_accuracy']*100:.4f}% | Time: {time_severe_retained:.2f}s")
    print(f"  [Ablated  (Manual Dict OFF)] Weighted: {m_severe_ablated['weighted_accuracy']*100:.4f}% | Hangul: {m_severe_ablated['hangul_accuracy']*100:.4f}% | Time: {time_severe_ablated:.2f}s")
    print(f"  --> Fixes (Ablated right): {diff_severe['num_fixes']}, Breaks (Ablated wrong): {diff_severe['num_breaks']}, Net Gain: {diff_severe['net_gain']}")
    if diff_severe["breaks"]:
        for b in diff_severe["breaks"][:5]:
            print(f"      * Break: {b['sample_id']} '{b['vessel_name']}' -> Retained: {b['retained_pred']}, Ablated: {b['ablated_pred']} (Gold: {b['gold_id']})")
    if diff_severe["fixes"]:
        for f in diff_severe["fixes"][:5]:
            print(f"      * Fix: {f['sample_id']} '{f['vessel_name']}' -> Retained: {f['retained_pred']}, Ablated: {f['ablated_pred']} (Gold: {f['gold_id']})")

    results["evaluations"]["severe"] = {
        "noise_applied_rate": noise_rate,
        "retained": m_severe_retained,
        "ablated": m_severe_ablated,
        "diff": diff_severe,
    }

    # Restore original state
    dpr_script.IDEA_B_WORD_VARS = dict(ORIG_IDEA_B)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] Lexicon Ablation Study Report saved to: {out_path}")


if __name__ == "__main__":
    main()
