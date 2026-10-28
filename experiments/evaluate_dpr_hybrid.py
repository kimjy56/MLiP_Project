"""
Head-to-head evaluation: Legacy AS-06 vs DPR-Hybrid on Heldout Fold 0 (SPEC-DPR-01).

Experiment ID: DPR-HYBRID-01

Evaluates:
  1. Legacy Baseline (dense_mode='none')
  2. DPR-Hybrid (dense_mode='hybrid', dense_weight=2.5)
  Under controlled zero-lookup conditions (use_train_lookup=False) on all 2,372 heldout Fold 0 rows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experimental_matchers.dpr_hybrid.script import (
    CatalogIndex,
    fit_matching_model,
    predict_rows,
)
from phonetic_encoder.hybrid_matcher import HybridDenseLexicalMatcher


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPR-Hybrid on Heldout Fold 0.")
    parser.add_argument("--catalog-csv", default="data/vessel_catalog.csv", help="Path to catalog CSV")
    parser.add_argument("--train-csv", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--folds-csv", default="eval_kit/folds_group5_seed42.csv", help="Path to folds CSV")
    parser.add_argument("--fold", type=int, default=0, help="Fold to hold out")
    parser.add_argument("--checkpoint", default="artifacts/DPR-SYMPHON-01/fold_0_encoder.pt", help="Bi-Encoder model checkpoint")
    parser.add_argument("--catalog-embeds", default="artifacts/DPR-SYMPHON-01/catalog_embeds_fp16.npy", help="Catalog embeds NPY")
    parser.add_argument("--output-json", default="reports/matching_improvements/DPR-HYBRID-01/dpr_hybrid_fold0_comparison.json", help="Output report JSON")
    parser.add_argument("--dense-weight", type=float, default=2.5, help="Weight for dense similarity in Stage-2 scoring")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/5] Loading catalog and initializing index on {device}...")
    df_cat = pd.read_csv(args.catalog_csv)
    cat_idx = CatalogIndex(df_cat)

    print(f"[2/5] Loading train data and splitting Fold {args.fold}...")
    df_train = pd.read_csv(args.train_csv)
    df_folds = pd.read_csv(args.folds_csv)
    df_merged = df_train.merge(df_folds, on=["sample_id", "target_vessel_id"])

    df_trn = df_merged[df_merged["fold"] != args.fold].reset_index(drop=True)
    df_val = df_merged[df_merged["fold"] == args.fold].reset_index(drop=True)
    print(f"Training on Folds != {args.fold} ({len(df_trn):,} rows), evaluating on Fold {args.fold} ({len(df_val):,} rows).")

    # Fit lexical matching model on training folds only (Zero Leakage)
    print("[3/5] Fitting lexical matching model on training folds...")
    model = fit_matching_model(df_trn, df_cat)

    # Initialize Dense Matcher
    print("[4/5] Initializing Hybrid Dense Matcher...")
    dense_matcher = HybridDenseLexicalMatcher(
        model_checkpoint=args.checkpoint,
        catalog_embeds_npy=args.catalog_embeds,
        catalog_csv=args.catalog_csv,
        device=device,
    )

    gold_ids = df_val["target_vessel_id"].tolist()
    re_hangul = __import__("re").compile(r"[가-힣]")
    is_kr_mask = df_val["vessel_name"].astype(str).apply(lambda x: bool(re_hangul.search(x))).to_numpy()

    print(f"\n[5/5] Running Head-to-Head Comparison on {len(df_val):,} Heldout Rows (Zero-Lookup)...")

    # A. Baseline AS-06 (dense_mode="none")
    print("  Running Baseline AS-06 (dense_mode='none')...")
    t0 = time.time()
    preds_base, _ = predict_rows(df_val, model, cat_idx, use_train_lookup=False, dense_mode="none")
    time_base = time.time() - t0

    # B. DPR-Hybrid (dense_mode="hybrid")
    print(f"  Running DPR-Hybrid (dense_mode='hybrid', weight={args.dense_weight})...")
    t1 = time.time()
    preds_hybrid, _ = predict_rows(
        df_val,
        model,
        cat_idx,
        use_train_lookup=False,
        dense_matcher=dense_matcher,
        dense_mode="hybrid",
        dense_weight=args.dense_weight,
    )
    time_hybrid = time.time() - t1

    # Metrics calculation
    correct_base = np.array([p == g for p, g in zip(preds_base, gold_ids)])
    correct_hybrid = np.array([p == g for p, g in zip(preds_hybrid, gold_ids)])

    acc_base_kr = float(correct_base[is_kr_mask].mean())
    acc_hybrid_kr = float(correct_hybrid[is_kr_mask].mean())

    acc_base_en = float(correct_base[~is_kr_mask].mean())
    acc_hybrid_en = float(correct_hybrid[~is_kr_mask].mean())

    weighted_base = 0.7 * acc_base_kr + 0.3 * acc_base_en
    weighted_hybrid = 0.7 * acc_hybrid_kr + 0.3 * acc_hybrid_en

    # Fix / Break analysis
    fixed_indices = np.where(~correct_base & correct_hybrid)[0]
    broken_indices = np.where(correct_base & ~correct_hybrid)[0]

    fixes: List[Dict[str, Any]] = []
    for idx in fixed_indices:
        fixes.append({
            "sample_id": df_val.iloc[idx]["sample_id"],
            "vessel_name": df_val.iloc[idx]["vessel_name"],
            "gold_id": gold_ids[idx],
            "pred_base": preds_base[idx],
            "pred_hybrid": preds_hybrid[idx],
            "is_korean": bool(is_kr_mask[idx]),
        })

    breaks: List[Dict[str, Any]] = []
    for idx in broken_indices:
        breaks.append({
            "sample_id": df_val.iloc[idx]["sample_id"],
            "vessel_name": df_val.iloc[idx]["vessel_name"],
            "gold_id": gold_ids[idx],
            "pred_base": preds_base[idx],
            "pred_hybrid": preds_hybrid[idx],
            "is_korean": bool(is_kr_mask[idx]),
        })

    print("\n========================= HELD-OUT FOLD 0 COMPARISON =========================")
    print(f"Condition: Zero-Lookup (use_train_lookup=False), Full 2,372 heldout rows")
    print(f"Korean Subset (239 rows, 70% weight):")
    print(f"  Baseline AS-06:  {acc_base_kr*100:.4f}% ({correct_base[is_kr_mask].sum()} / {is_kr_mask.sum()})")
    print(f"  DPR-Hybrid:     {acc_hybrid_kr*100:.4f}% ({correct_hybrid[is_kr_mask].sum()} / {is_kr_mask.sum()})")
    print(f"  Delta:          {acc_hybrid_kr*100 - acc_base_kr*100:+.4f}%p")
    print(f"Non-Korean Subset (2,133 rows, 30% weight):")
    print(f"  Baseline AS-06:  {acc_base_en*100:.4f}% ({correct_base[~is_kr_mask].sum()} / {(~is_kr_mask).sum()})")
    print(f"  DPR-Hybrid:     {acc_hybrid_en*100:.4f}% ({correct_hybrid[~is_kr_mask].sum()} / {(~is_kr_mask).sum()})")
    print(f"  Delta:          {acc_hybrid_en*100 - acc_base_en*100:+.4f}%p")
    print(f"Official Weighted Accuracy (0.7*KO + 0.3*EN):")
    print(f"  Baseline AS-06:  {weighted_base*100:.4f}%")
    print(f"  DPR-Hybrid:     {weighted_hybrid*100:.4f}%")
    print(f"  Delta:          {weighted_hybrid*100 - weighted_base*100:+.4f}%p")
    print(f"\nFixes: {len(fixes)} rows, Breaks: {len(breaks)} rows (Net: {len(fixes) - len(breaks):+d} rows)")
    print(f"Inference Time: Baseline {time_base:.2f}s, DPR-Hybrid {time_hybrid:.2f}s (+{time_hybrid - time_base:.2f}s)")
    print("==============================================================================")

    # Save detailed JSON report
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_id": "DPR-HYBRID-01",
        "heldout_fold": args.fold,
        "dense_weight": args.dense_weight,
        "korean": {
            "num_rows": int(is_kr_mask.sum()),
            "baseline_accuracy": acc_base_kr,
            "hybrid_accuracy": acc_hybrid_kr,
            "delta_pct_points": (acc_hybrid_kr - acc_base_kr) * 100,
        },
        "non_korean": {
            "num_rows": int((~is_kr_mask).sum()),
            "baseline_accuracy": acc_base_en,
            "hybrid_accuracy": acc_hybrid_en,
            "delta_pct_points": (acc_hybrid_en - acc_base_en) * 100,
        },
        "official_weighted": {
            "baseline_accuracy": weighted_base,
            "hybrid_accuracy": weighted_hybrid,
            "delta_pct_points": (weighted_hybrid - weighted_base) * 100,
        },
        "net_gain_rows": len(fixes) - len(breaks),
        "num_fixes": len(fixes),
        "num_breaks": len(breaks),
        "fixes": fixes,
        "breaks": breaks,
        "inference_time": {
            "baseline_sec": time_base,
            "hybrid_sec": time_hybrid,
            "delta_sec": time_hybrid - time_base,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved comparison report to {out_path}")


if __name__ == "__main__":
    main()
