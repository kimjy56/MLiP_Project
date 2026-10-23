"""Run full 5-fold OOF evaluation using CatBoost on the official G5-s42 split."""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

from train import read_data, evaluate, QUERY_COLUMNS, training_pair_for_row
from model.matcher import CatalogIndex, make_assets, prepare_row, hangul, worker_count, runtime_int
from train_oof import validate_folds, sha256, now_kst
from decision_models.catboost_model import CatBoostDecisionModel
from decision_models.matcher_adapter import DecisionMatcher


def main():
    exp_id = "CB-01"
    output_dir = ROOT / f"artifacts/oof/{exp_id}"
    record_path = ROOT / f"project_log/experiments/{exp_id}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting 5-Fold OOF Evaluation with CatBoost (CPU exact mode) ===")
    start_total = time.monotonic()

    # Load data
    train_df, catalog = read_data(ROOT / "data")
    folds_df = pd.read_csv(ROOT / "eval_kit/folds_group5_seed42.csv", dtype={"sample_id": str, "target_vessel_id": str})
    joined = validate_folds(train_df, catalog, folds_df)

    oof_parts = []
    fold_summaries = []

    for fold_k in range(5):
        t0_fold = time.monotonic()
        print(f"\n--- [Fold {fold_k}/5] ---")
        val_df = joined[joined.fold == fold_k].drop(columns="fold").copy()
        real_df = joined[joined.fold != fold_k].drop(columns="fold").copy()
        fold_out = output_dir / f"fold_{fold_k}"
        fold_out.mkdir(parents=True, exist_ok=True)

        # 1. Assets & Index
        assets = make_assets(real_df, catalog, name_mode="all", phonetic_search=24, retrieval_channels=["S", "P", "T"], use_train_lookup=True)
        assets_path = fold_out / "assets.json"
        assets_path.write_text(json.dumps(assets, ensure_ascii=False), encoding="utf-8")
        index = CatalogIndex(catalog, assets)

        # 2. Build training pairs
        cols = ['vessel_name', 'gross_tonnage', 'international_gross_tonnage', 'flag_name', 'vessel_type', 'target_vessel_id']
        unique_train = real_df.drop_duplicates(cols)
        records = unique_train.to_dict('records')
        kh = unique_train.vessel_name.map(hangul).mean()

        def _process_train_row(row):
            return training_pair_for_row(row, index, kh)

        matrices, labels, weights = [], [], []
        workers = worker_count('VESSEL_FEATURE_WORKERS')
        batch_size = runtime_int('VESSEL_TRAIN_BATCH', 64)
        print(f"Generating training pairs ({len(records)} rows) using {workers} workers...")
        t_feat = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for start_r in range(0, len(records), batch_size):
                batch = [prepare_row(r) for r in records[start_r:start_r + batch_size]]
                index.prepare_retrieval(batch)
                for f, l, w, was_ret, s_id, c_ids in executor.map(_process_train_row, batch):
                    matrices.append(f)
                    labels.extend(l)
                    weights.extend(w)
        feat_sec = time.monotonic() - t_feat
        x_train = np.concatenate(matrices)
        y_train = np.asarray(labels, dtype=np.float32)
        w_train = np.asarray(weights, dtype=np.float32)
        matrices.clear()
        print(f"Pairs ready: {len(y_train)} pairs in {feat_sec:.2f}s")

        # 3. Fit CatBoost
        print("Fitting CatBoost (CPU exact mode, 350 trees, depth=6)...")
        t_fit = time.monotonic()
        cb = CatBoostDecisionModel(task_type="CPU", thread_count=6)
        cb.fit(x_train, y_train, w_train, index.feature_names)
        cb.save(fold_out)
        fit_sec = time.monotonic() - t_fit
        print(f"CatBoost fit finished in {fit_sec:.2f}s")

        # 4. Predict & Evaluate
        print("Evaluating validation fold...")
        t_pred = time.monotonic()
        matcher = DecisionMatcher(catalog, assets_path, cb)
        metrics, predictions = evaluate(matcher, val_df, set(real_df.target_vessel_id))
        pred_sec = time.monotonic() - t_pred

        oof_parts.append(pd.DataFrame({
            "sample_id": val_df.sample_id.to_numpy(),
            "pred_vessel_id": predictions,
            "fold": fold_k
        }))

        fold_time = time.monotonic() - t0_fold
        print(f"Fold {fold_k} Result: Weighted={metrics['weighted_accuracy']:.5f} (Hangul={metrics['hangul_accuracy']:.4f}, NonHangul={metrics['non_hangul_accuracy']:.4f}) in {fold_time:.1f}s (predict: {pred_sec:.1f}s)")
        fold_summaries.append({
            "fold": fold_k,
            "metrics": metrics,
            "fit_seconds": fit_sec,
            "pred_seconds": pred_sec,
            "total_seconds": fold_time
        })

    # Combine OOF
    oof = pd.concat(oof_parts, ignore_index=True)
    oof = oof.set_index("sample_id").loc[train_df.sample_id].reset_index()
    oof_path = output_dir / f"oof_{exp_id}.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8")
    oof_sha = sha256(oof_path)
    total_sec = time.monotonic() - start_total
    print(f"\nAll 5 folds complete in {total_sec:.1f}s ({total_sec/60:.2f} min)!")
    print(f"Saved OOF to {oof_path} (SHA-256: {oof_sha})")

    # Run official evaluator
    print("\nRunning official evaluate.py vs AUG-04 (LightGBM baseline)...")
    eval_proc = subprocess.run([
        sys.executable, str(ROOT / "eval_kit/evaluate.py"),
        "--train", str(ROOT / "data/train.csv"),
        "--folds", str(ROOT / "eval_kit/folds_group5_seed42.csv"),
        "--oof", str(oof_path),
        "--compare", str(ROOT / "artifacts/oof/AUG-04/oof_AUG-04.csv")
    ], capture_output=True, text=True, encoding="utf-8")
    print(eval_proc.stdout)

    # Save experiment record
    record = {
        "experiment_id": exp_id,
        "status": "completed",
        "recorded_at_kst": now_kst(),
        "purpose": "5-Fold OOF evaluation of CatBoost classifier with optimized candidate retrieval",
        "hypothesis": "CatBoost with exact split binning and robust symmetric trees outperforms LightGBM without candidate recall penalty",
        "model": {
            "type": "CatBoostDecisionModel",
            "task_type": "CPU",
            "thread_count": 6,
            "iterations": 350,
            "depth": 6,
            "learning_rate": 0.05
        },
        "retrieval": {
            "channels": ["S", "P", "T"],
            "phonetic_search": 24,
            "score_cutoff": 35,
            "skip_non_hangul_phonetic": True
        },
        "oof": {
            "path": str(oof_path),
            "rows": len(oof),
            "sha256": oof_sha
        },
        "fold_results": fold_summaries,
        "elapsed_seconds": total_sec,
        "official_evaluation_stdout": eval_proc.stdout
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Experiment saved to {record_path}")


if __name__ == "__main__":
    main()
