# -*- coding: utf-8 -*-
"""Model selection and final fitting for KO-FUSION-01.

Implements SPEC-KO-FUSION-01 Section 7.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from experimental_matchers.ko_fusion.ranker import LGBMReranker, SEARCH_GRID
from experimental_matchers.ko_fusion.cache import CandidateCacheReader


def evaluate_on_selection_set(
    ranker: LGBMReranker,
    queries_meta: List[Dict[str, Any]],
    cands_records: List[List[str]],
    X_all: np.ndarray,
    legacy_scores_all: np.ndarray,
    group_sizes: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate accuracy, fixes, breaks on Inner B selection set."""
    kr_total = 0
    kr_correct_base = 0
    kr_correct_ranker = 0
    kr_fixes = 0
    kr_breaks = 0

    non_total = 0
    non_correct_base = 0
    non_correct_ranker = 0

    cand_offset = 0
    group_idx = 0

    for q_idx, q in enumerate(queries_meta):
        gold = q["gold_vessel_id"]
        base_pred = q["baseline_pred_vessel_id"]
        route = q["route"]
        n_cands = q["candidate_count"]

        if n_cands > 0:
            cands = cands_records[q_idx]
            q_X = X_all[cand_offset : cand_offset + n_cands]
            q_l_sc = legacy_scores_all[cand_offset : cand_offset + n_cands]
            cand_offset += n_cands
            group_idx += 1

            if route == "hangul_candidates":
                q_r_sc = ranker.predict(q_X)
                items = [(-float(r), -float(l), vid) for vid, l, r in zip(cands, q_l_sc, q_r_sc)]
                items.sort()
                pred = items[0][2]
            else:
                pred = base_pred
        else:
            pred = base_pred

        is_kr = (route == "hangul_candidates")
        base_ok = (base_pred == gold)
        ranker_ok = (pred == gold)

        if is_kr:
            kr_total += 1
            if base_ok:
                kr_correct_base += 1
            if ranker_ok:
                kr_correct_ranker += 1
            if not base_ok and ranker_ok:
                kr_fixes += 1
            elif base_ok and not ranker_ok:
                kr_breaks += 1
        else:
            non_total += 1
            if base_ok:
                non_correct_base += 1
            if ranker_ok:
                non_correct_ranker += 1

    kr_acc_base = (kr_correct_base / kr_total) if kr_total > 0 else 1.0
    kr_acc_ranker = (kr_correct_ranker / kr_total) if kr_total > 0 else 1.0
    non_acc_base = (non_correct_base / non_total) if non_total > 0 else 1.0
    non_acc_ranker = (non_correct_ranker / non_total) if non_total > 0 else 1.0

    w_acc_base = 0.7 * kr_acc_base + 0.3 * non_acc_base
    w_acc_ranker = 0.7 * kr_acc_ranker + 0.3 * non_acc_ranker

    return {
        "weighted_accuracy": w_acc_ranker,
        "baseline_weighted_accuracy": w_acc_base,
        "hangul_accuracy": kr_acc_ranker,
        "baseline_hangul_accuracy": kr_acc_base,
        "non_hangul_accuracy": non_acc_ranker,
        "baseline_non_hangul_accuracy": non_acc_base,
        "hangul_rows": kr_total,
        "non_hangul_rows": non_total,
        "hangul_fixes": kr_fixes,
        "hangul_breaks": kr_breaks,
        "hangul_net": kr_fixes - kr_breaks,
    }


def clip_large_groups(X: np.ndarray, y: np.ndarray, scores: np.ndarray, groups: np.ndarray, max_size: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LightGBM requires query group size <= 10000. Truncate large groups by legacy score while preserving positive label."""
    if not (groups > max_size).any():
        return X, y, groups

    new_X, new_y, new_groups = [], [], []
    offset = 0
    clipped = 0
    for g in groups:
        q_X = X[offset : offset + g]
        q_y = y[offset : offset + g]
        q_s = scores[offset : offset + g]
        offset += g
        if g <= max_size:
            new_X.append(q_X)
            new_y.append(q_y)
            new_groups.append(g)
        else:
            clipped += 1
            order = np.argsort(-q_s)
            pos_indices = np.where(q_y == 1)[0]
            selected = list(order[:max_size])
            for p in pos_indices:
                if p not in selected:
                    selected[-1] = p
            selected = sorted(selected)
            new_X.append(q_X[selected])
            new_y.append(q_y[selected])
            new_groups.append(len(selected))

    print(f"  -> LightGBM max_position constraint: clipped {clipped} queries to <= {max_size} candidates (preserved positives)")
    return np.vstack(new_X), np.concatenate(new_y), np.array(new_groups, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Train KO-FUSION-01 LightGBM ranker")
    parser.add_argument("--experiment-id", type=str, default="KO-FUSION-01")
    parser.add_argument("--cache-root", type=str, default="artifacts/KO-FUSION-01")
    parser.add_argument("--outer-fold", type=int, default=0)
    args = parser.parse_args()

    cache_dir = Path(args.cache_root).resolve() / f"outer_{args.outer_fold}" / "cache"
    train_cache_dir = cache_dir / "train"
    train_full_cache_dir = cache_dir / "train_full"
    sel_cache_dir = cache_dir / "selection"
    out_dir = Path(args.cache_root).resolve() / f"outer_{args.outer_fold}"

    print(f"=== [1] Loading Inner A training cache ===")
    X_train, lsc_train, y_train, groups_train, _, train_manifest = CandidateCacheReader.load_cache(train_cache_dir)
    X_train, y_train, groups_train = clip_large_groups(X_train, y_train, lsc_train, groups_train, max_size=10000)
    print(f"  -> A cache ready for fit: {len(X_train)} candidate pairs, {len(groups_train)} queries")

    print(f"=== [2] Loading Inner B selection cache ===")
    X_sel, lsc_sel, _, groups_sel, queries_sel, sel_manifest = CandidateCacheReader.load_cache(sel_cache_dir)
    with open(sel_cache_dir / "candidate_ids.jsonl", "r", encoding="utf-8") as f:
        cands_sel = [json.loads(line) for line in f if line.strip()]
    print(f"  -> B cache loaded: {len(queries_sel)} queries ({sel_manifest['total_candidates']} candidate pairs)")

    print(f"\n=== [3] Evaluating {len(SEARCH_GRID)} configurations on Inner B ===")
    results: List[Dict[str, Any]] = []

    for cfg_idx, cfg in enumerate(SEARCH_GRID):
        reg_lambda = cfg["reg_lambda"]
        n_est = cfg["n_estimators"]

        t0 = time.perf_counter()
        ranker = LGBMReranker(params={"reg_lambda": reg_lambda}, n_estimators=n_est)
        ranker.fit(X_train, y_train, groups_train)
        fit_sec = time.perf_counter() - t0

        eval_res = evaluate_on_selection_set(
            ranker,
            queries_sel,
            cands_sel,
            X_sel,
            lsc_sel,
            groups_sel,
        )
        eval_res["reg_lambda"] = reg_lambda
        eval_res["n_estimators"] = n_est
        eval_res["fit_seconds"] = fit_sec
        results.append(eval_res)

        print(
            f"  [{cfg_idx+1}/6] reg_lambda={reg_lambda:4.1f}, n_est={n_est:3d} | "
            f"W_Acc={eval_res['weighted_accuracy']:.5f} (Base={eval_res['baseline_weighted_accuracy']:.5f}) | "
            f"KO_Acc={eval_res['hangul_accuracy']*100:.2f}% | "
            f"Fixes={eval_res['hangul_fixes']}, Breaks={eval_res['hangul_breaks']} (Net={eval_res['hangul_net']:+d}) | "
            f"Time={fit_sec:.2f}s"
        )

    # Sort candidates by:
    # 1. Highest weighted_accuracy (-w_acc)
    # 2. Minimum breaks (+breaks)
    # 3. Smaller n_estimators (+n_est)
    # 4. Larger reg_lambda (-reg_lambda)
    def _rank_key(r):
        return (-r["weighted_accuracy"], r["hangul_breaks"], r["n_estimators"], -r["reg_lambda"])

    results.sort(key=_rank_key)
    best_cfg = results[0]

    print("\n" + "=" * 76)
    print(
        f"Selected Best Configuration on Inner B:\n"
        f"  * reg_lambda = {best_cfg['reg_lambda']}\n"
        f"  * n_estimators = {best_cfg['n_estimators']}\n"
        f"  * Weighted Accuracy on B = {best_cfg['weighted_accuracy']:.5f} (Base = {best_cfg['baseline_weighted_accuracy']:.5f})\n"
        f"  * Korean Net = {best_cfg['hangul_net']:+d} ({best_cfg['hangul_fixes']} fixes / {best_cfg['hangul_breaks']} breaks)"
    )
    print("=" * 76)

    # Train final model on Full T
    X_full, lsc_full, y_full, groups_full, _, _ = CandidateCacheReader.load_cache(train_full_cache_dir)
    X_full, y_full, groups_full = clip_large_groups(X_full, y_full, lsc_full, groups_full, max_size=10000)
    print(f"  -> Full T ready for fit: {len(X_full)} candidate pairs, {len(groups_full)} queries")

    t0_final = time.perf_counter()
    final_ranker = LGBMReranker(
        params={"reg_lambda": best_cfg["reg_lambda"]},
        n_estimators=best_cfg["n_estimators"],
    )
    final_ranker.fit(X_full, y_full, groups_full)
    final_fit_sec = time.perf_counter() - t0_final
    print(f"  -> Final model fit in {final_fit_sec:.2f}s")

    # Save model and manifest
    ranker_file = out_dir / "ranker.txt"
    final_ranker.save_model(ranker_file)

    ranker_manifest = {
        "experiment_id": args.experiment_id,
        "outer_fold": args.outer_fold,
        "selected_config": {
            "reg_lambda": best_cfg["reg_lambda"],
            "n_estimators": best_cfg["n_estimators"],
        },
        "selection_metrics_on_b": best_cfg,
        "all_configurations_tested": results,
        "final_fit_seconds": final_fit_sec,
        "ranker_file": str(ranker_file),
    }
    with open(out_dir / "ranker_manifest.json", "w", encoding="utf-8") as f:
        json.dump(ranker_manifest, f, indent=2, ensure_ascii=False)

    print(f"Model saved to {ranker_file} and manifest to {out_dir / 'ranker_manifest.json'}")


if __name__ == "__main__":
    main()
