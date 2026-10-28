# -*- coding: utf-8 -*-
"""Evaluation script for KO-FUSION-01 on heldout Fold 0.

Implements SPEC-KO-FUSION-01 Section 9.2.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from experimental_matchers.ko_fusion.ranker import LGBMReranker
from experimental_matchers.ko_fusion.cache import CandidateCacheReader


def now_kst() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def main():
    parser = argparse.ArgumentParser(description="Evaluate KO-FUSION-01 on heldout fold")
    parser.add_argument("--experiment-id", type=str, default="KO-FUSION-01")
    parser.add_argument("--cache-root", type=str, default="artifacts/KO-FUSION-01")
    parser.add_argument("--outer-fold", type=int, default=0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    started_at = now_kst()

    fold_dir = Path(args.cache_root).resolve() / f"outer_{args.outer_fold}"
    val_cache_dir = fold_dir / "cache" / "validation"
    ranker_file = fold_dir / "ranker.txt"
    ranker_manifest_file = fold_dir / "ranker_manifest.json"

    print(f"=== [1] Loading validation cache for outer fold {args.outer_fold} ===")
    X_val, lsc_val, labels_val, groups_val, queries_val, val_manifest = CandidateCacheReader.load_cache(val_cache_dir)
    with open(val_cache_dir / "candidate_ids.jsonl", "r", encoding="utf-8") as f:
        cands_val = [json.loads(line) for line in f if line.strip()]

    print(f"  -> Validation cache: {len(queries_val)} queries, {len(X_val)} candidate pairs")

    print(f"=== [2] Loading trained LGBMReranker from {ranker_file.name} ===")
    ranker = LGBMReranker.load_model(ranker_file)
    with open(ranker_manifest_file, "r", encoding="utf-8") as f:
        ranker_meta = json.load(f)

    print("=== [3] Evaluating baseline vs ranker predictions ===")
    cand_offset = 0
    records = []

    for q_idx, q in enumerate(queries_val):
        sid = q["sample_id"]
        gold = q["gold_vessel_id"]
        base_pred = q["baseline_pred_vessel_id"]
        route = q["route"]
        n_cands = q["candidate_count"]

        if n_cands > 0:
            cands = cands_val[q_idx]
            q_X = X_val[cand_offset : cand_offset + n_cands]
            q_l_sc = lsc_val[cand_offset : cand_offset + n_cands]
            cand_offset += n_cands

            if route == "hangul_candidates":
                q_r_sc = ranker.predict(q_X)
                items = [(-float(r), -float(l), vid) for vid, l, r in zip(cands, q_l_sc, q_r_sc)]
                items.sort()
                pred = items[0][2]
                r_score_top1 = -items[0][0]
                l_score_top1 = -items[0][1]
            else:
                pred = base_pred
                r_score_top1 = None
                l_score_top1 = None
        else:
            pred = base_pred
            r_score_top1 = None
            l_score_top1 = None

        is_kr = (route == "hangul_candidates")
        base_ok = (base_pred == gold)
        ranker_ok = (pred == gold)

        records.append({
            "sample_id": sid,
            "route": route,
            "is_hangul": is_kr,
            "gold_vessel_id": gold,
            "baseline_pred": base_pred,
            "ranker_pred": pred,
            "baseline_correct": base_ok,
            "ranker_correct": ranker_ok,
            "candidate_count": n_cands,
            "top1_ranker_score": r_score_top1,
            "top1_legacy_score": l_score_top1,
        })

    eval_df = pd.DataFrame(records)

    # Detailed metrics
    kr_mask = eval_df["is_hangul"]
    non_mask = ~kr_mask

    kr_total = int(kr_mask.sum())
    non_total = int(non_mask.sum())

    kr_base_correct = int(eval_df.loc[kr_mask, "baseline_correct"].sum())
    kr_ranker_correct = int(eval_df.loc[kr_mask, "ranker_correct"].sum())

    non_base_correct = int(eval_df.loc[non_mask, "baseline_correct"].sum())
    non_ranker_correct = int(eval_df.loc[non_mask, "ranker_correct"].sum())

    kr_fixes = int((~eval_df.loc[kr_mask, "baseline_correct"] & eval_df.loc[kr_mask, "ranker_correct"]).sum())
    kr_breaks = int((eval_df.loc[kr_mask, "baseline_correct"] & ~eval_df.loc[kr_mask, "ranker_correct"]).sum())

    non_fixes = int((~eval_df.loc[non_mask, "baseline_correct"] & eval_df.loc[non_mask, "ranker_correct"]).sum())
    non_breaks = int((eval_df.loc[non_mask, "baseline_correct"] & ~eval_df.loc[non_mask, "ranker_correct"]).sum())

    kr_base_acc = kr_base_correct / kr_total
    kr_ranker_acc = kr_ranker_correct / kr_total
    non_base_acc = non_base_correct / non_total
    non_ranker_acc = non_ranker_correct / non_total

    base_weighted = 0.7 * kr_base_acc + 0.3 * non_base_acc
    ranker_weighted = 0.7 * kr_ranker_acc + 0.3 * non_ranker_acc

    finished_at = now_kst()
    total_sec = time.perf_counter() - t_start

    print("\n" + "=" * 76)
    print(f"=== Heldout Fold {args.outer_fold} Evaluation Results ({args.experiment_id}) ===")
    print("=" * 76)
    print(f"  * Total Rows: {len(eval_df):,} (Korean: {kr_total}, Non-Korean: {non_total})")
    print(f"  * Baseline Weighted Acc: {base_weighted:.5f} (KO: {kr_base_acc*100:.3f}% [{kr_base_correct}/{kr_total}], Non-KO: {non_base_acc*100:.3f}% [{non_base_correct}/{non_total}])")
    print(f"  * Ranker   Weighted Acc: {ranker_weighted:.5f} (KO: {kr_ranker_acc*100:.3f}% [{kr_ranker_correct}/{kr_total}], Non-KO: {non_ranker_acc*100:.3f}% [{non_ranker_correct}/{non_total}])")
    print(f"  * Net Change: {ranker_weighted - base_weighted:+.5f}")
    print(f"  * Korean: {kr_fixes} fixes, {kr_breaks} breaks (Net: {kr_fixes - kr_breaks:+d})")
    print(f"  * Non-Korean: {non_fixes} fixes, {non_breaks} breaks (Net: {non_fixes - non_breaks:+d}) [Non-Korean bypass preserved: {non_breaks == 0 and non_fixes == 0}]")
    print("=" * 76)

    # Save CSV artifacts
    pred_dir = fold_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(pred_dir / "paired_predictions.csv", index=False, encoding="utf-8")
    eval_df[["sample_id", "baseline_pred"]].rename(columns={"baseline_pred": "pred_vessel_id"}).to_csv(
        pred_dir / "predictions_baseline.csv", index=False, encoding="utf-8"
    )
    eval_df[["sample_id", "ranker_pred"]].rename(columns={"ranker_pred": "pred_vessel_id"}).to_csv(
        pred_dir / "predictions_ranker.csv", index=False, encoding="utf-8"
    )

    # Prepare experiment log JSON
    exp_log = {
        "experiment_id": args.experiment_id,
        "plan_id": "SPEC-KO-FUSION-01",
        "status": "evaluated_single_fold",
        "recorded_at_kst": finished_at,
        "started_at_kst": started_at,
        "finished_at_kst": finished_at,
        "purpose": "Korean candidate selection score learning with LightGBM LambdaRank",
        "hypothesis": "Learning combination of 28 candidate features improves Korean top-1 selection over fixed rule weights.",
        "baseline_experiment_id": "AS-09",
        "changed_factors": [
            "Learned LightGBM 4.6.0 LambdaRank on Korean candidate features",
            "Fixed AS-09 candidate set and unrounded float64 legacy scores",
            "Nested cross-fit model selection on Inner B",
        ],
        "fixed_conditions": [
            f"clean heldout fold {args.outer_fold}",
            "AS-09 bi-encoder dense top-35",
            "AS-09 lexical retrieval rules",
            "AS-09 lookup early return bypass",
            "non-hangul predictions strictly unchanged",
        ],
        "code": {
            "git_revision_at_run": "5bd552f",
            "file_sha256": {
                "as09_adapter.py": hashlib.sha256(Path("experimental_matchers/ko_fusion/as09_adapter.py").read_bytes()).hexdigest(),
                "features.py": hashlib.sha256(Path("experimental_matchers/ko_fusion/features.py").read_bytes()).hexdigest(),
                "cache.py": hashlib.sha256(Path("experimental_matchers/ko_fusion/cache.py").read_bytes()).hexdigest(),
                "ranker.py": hashlib.sha256(Path("experimental_matchers/ko_fusion/ranker.py").read_bytes()).hexdigest(),
            },
        },
        "data": {
            "split_id": f"folds_group5_seed42 fold{args.outer_fold}",
            "seed": 42,
            "validation_rows": len(eval_df),
            "hangul_rows": kr_total,
            "non_hangul_rows": non_total,
        },
        "model": {
            "type": "LightGBM LambdaRank",
            "selected_config": ranker_meta.get("selected_config"),
            "feature_count": 28,
            "ranker_manifest": str(ranker_manifest_file),
        },
        "metrics": {
            "baseline": {
                "hangul_correct": kr_base_correct,
                "hangul_rows": kr_total,
                "hangul_accuracy": kr_base_acc,
                "non_hangul_correct": non_base_correct,
                "non_hangul_rows": non_total,
                "non_hangul_accuracy": non_base_acc,
                "weighted_accuracy": base_weighted,
            },
            "ranker": {
                "hangul_correct": kr_ranker_correct,
                "hangul_rows": kr_total,
                "hangul_accuracy": kr_ranker_acc,
                "non_hangul_correct": non_ranker_correct,
                "non_hangul_rows": non_total,
                "non_hangul_accuracy": non_ranker_acc,
                "weighted_accuracy": ranker_weighted,
                "hangul_fixes": kr_fixes,
                "hangul_breaks": kr_breaks,
                "hangul_net": kr_fixes - kr_breaks,
            },
            "net_difference": {
                "weighted_accuracy_delta": ranker_weighted - base_weighted,
                "hangul_accuracy_delta": kr_ranker_acc - kr_base_acc,
            },
        },
        "cost": {
            "total_seconds": total_sec,
        },
        "artifacts": [
            str(pred_dir),
            str(ranker_file),
        ],
    }

    log_path = Path("project_log/experiments") / f"{args.experiment_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(exp_log, f, indent=2, ensure_ascii=False)
    print(f"Formal experiment log saved to {log_path}")


if __name__ == "__main__":
    main()
