"""CatBoost hyperparameter tuning script (Ultra-fast: Features precomputed once).
Explores depth, l2_leaf_reg, learning_rate/iterations, and reports local metric + predicted LB score.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from train import read_data, training_pair_for_row, QUERY_COLUMNS
from model.matcher import CatalogIndex, make_assets, prepare_row, hangul, worker_count, runtime_int
from train_oof import validate_folds
from decision_models.catboost_model import CatBoostDecisionModel

LATIN_LB_BASELINE = 3314


def main():
    print("=== CatBoost Parameter Tuning Experiment on Fold 0 ===")
    train_df, catalog = read_data(ROOT / "data")
    folds_df = pd.read_csv(ROOT / "eval_kit/folds_group5_seed42.csv", dtype={"sample_id": str, "target_vessel_id": str})
    joined = validate_folds(train_df, catalog, folds_df)

    val_df = joined[joined.fold == 0].drop(columns="fold").copy()
    real_df = joined[joined.fold != 0].drop(columns="fold").copy()

    # 1. Assets & Index (Prepared once)
    print("1. Building Assets & Index for Fold 0...")
    assets = make_assets(
        real_df, catalog,
        name_mode="all", phonetic_search=24,
        retrieval_channels=["S", "P", "T"],
        use_train_lookup=False,
        korean_expansion=True
    )
    index = CatalogIndex(catalog, assets)

    # 2. Build training pairs once
    cols = ['vessel_name', 'gross_tonnage', 'international_gross_tonnage', 'flag_name', 'vessel_type', 'target_vessel_id']
    unique_train = real_df.drop_duplicates(cols)
    records = unique_train.to_dict('records')
    kh = unique_train.vessel_name.map(hangul).mean()

    def _process_train_row(row):
        return training_pair_for_row(row, index, kh)

    matrices, labels, weights = [], [], []
    workers = worker_count('VESSEL_FEATURE_WORKERS')
    batch_size = runtime_int('VESSEL_TRAIN_BATCH', 64)
    print(f"2. Generating training pairs ({len(records)} rows) using {workers} workers...")
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
    print(f"   Training pairs ready: {len(y_train)} pairs in {feat_sec:.2f}s\n")

    # 3. Precompute validation candidates & features once
    print(f"3. Precomputing validation candidates & features ({len(val_df)} rows)...")
    t_val_start = time.monotonic()
    val_records = val_df.loc[:, QUERY_COLUMNS].to_dict('records')
    val_rows = [prepare_row(r, index.korean_expansion) for r in val_records]
    index.prepare_retrieval(val_rows, use_direct=False)

    val_built = []
    for r in val_rows:
        cands = index.candidates(r, max_candidates=30)
        feats = index.features(r, cands) if len(cands) else None
        val_built.append((cands, feats))

    val_matrices = [f for _, f in val_built if f is not None and len(f)]
    val_x = np.concatenate(val_matrices) if val_matrices else np.empty((0, len(index.feature_names)), dtype=np.float32)
    cand_counts = [len(c) for c, _ in val_built]
    t_val_sec = time.monotonic() - t_val_start
    print(f"   Validation features ready: {len(val_x)} candidate pairs in {t_val_sec:.2f}s\n")

    y_true = np.asarray(val_df.target_vessel_id.tolist())
    ko_mask = val_df.vessel_name.map(hangul).values

    # Parameter grid to test
    param_grid = [
        {"desc": "기본값 (Baseline: Depth=6, L2=3)", "params": {"depth": 6, "iterations": 350, "learning_rate": 0.05, "l2_leaf_reg": 3}},
        {"desc": "깊이 축소 (Depth=4, L2=3)", "params": {"depth": 4, "iterations": 350, "learning_rate": 0.05, "l2_leaf_reg": 3}},
        {"desc": "깊이 확대 (Depth=8, L2=3)", "params": {"depth": 8, "iterations": 350, "learning_rate": 0.05, "l2_leaf_reg": 3}},
        {"desc": "규제 강화 (Depth=6, L2=7)", "params": {"depth": 6, "iterations": 350, "learning_rate": 0.05, "l2_leaf_reg": 7}},
        {"desc": "규제 대폭 강화 (Depth=6, L2=15)", "params": {"depth": 6, "iterations": 350, "learning_rate": 0.05, "l2_leaf_reg": 15}},
        {"desc": "규제 완화 (Depth=6, L2=1)", "params": {"depth": 6, "iterations": 350, "learning_rate": 0.05, "l2_leaf_reg": 1}},
        {"desc": "정밀 학습 (Depth=6, LR=0.03, Iter=500)", "params": {"depth": 6, "iterations": 500, "learning_rate": 0.03, "l2_leaf_reg": 3}},
        {"desc": "경량 정밀 (Depth=4, L2=7)", "params": {"depth": 4, "iterations": 350, "learning_rate": 0.05, "l2_leaf_reg": 7}},
        {"desc": "고규제 정밀 (Depth=6, LR=0.03, L2=7)", "params": {"depth": 6, "iterations": 500, "learning_rate": 0.03, "l2_leaf_reg": 7}},
    ]

    results = []
    print(f"{'설정 설명':<34} | {'가중점수':<8} | {'한글 정확도':<14} | {'비한글 정확도':<14} | {'오답':<4} | {'예측 LB 점수':<14} | {'학습/평가'}")
    print("-" * 118)

    base_ko_correct = None

    for item in param_grid:
        desc = item["desc"]
        p = item["params"]

        # Fit CatBoost
        t0 = time.monotonic()
        cb = CatBoostDecisionModel(task_type="CPU", thread_count=6, **p)
        cb.fit(x_train, y_train, w_train, index.feature_names)
        fit_sec = time.monotonic() - t0

        # Predict using precomputed features
        t_pred = time.monotonic()
        scores = cb.predict(val_x)
        preds = []
        offset = 0
        for cands, cnt in zip([c for c, _ in val_built], cand_counts):
            if cnt > 0:
                best_idx = cands[int(np.argmax(scores[offset:offset + cnt]))]
                preds.append(index.ids[best_idx])
                offset += cnt
            else:
                preds.append(index.ids[0])
        preds = np.asarray(preds)
        pred_sec = time.monotonic() - t_pred

        correct = (y_true == preds)
        ko_acc = correct[ko_mask].mean()
        latin_acc = correct[~ko_mask].mean()
        weighted_score = 0.7 * ko_acc + 0.3 * latin_acc
        total_errors = int((~correct).sum())
        ko_corr_cnt = int(correct[ko_mask].sum())
        ko_tot_cnt = int(ko_mask.sum())
        latin_corr_cnt = int(correct[~ko_mask].sum())
        latin_tot_cnt = int((~ko_mask).sum())

        if base_ko_correct is None:
            base_ko_correct = ko_corr_cnt

        delta_ko = ko_corr_cnt - base_ko_correct
        estimated_public_korean = 434 + (delta_ko * 3)
        estimated_public_korean = max(404, min(500, estimated_public_korean))
        predicted_lb = (estimated_public_korean / 760.0) + (3.0 * LATIN_LB_BASELINE / 35680.0)

        res = {
            "desc": desc,
            "params": p,
            "weighted_score": round(weighted_score, 5),
            "ko_acc": round(ko_acc, 5),
            "ko_correct": f"{ko_corr_cnt}/{ko_tot_cnt}",
            "latin_acc": round(latin_acc, 5),
            "latin_correct": f"{latin_corr_cnt}/{latin_tot_cnt}",
            "total_errors": total_errors,
            "predicted_lb_score": round(predicted_lb, 10),
            "estimated_public_korean": estimated_public_korean,
            "fit_seconds": round(fit_sec, 2),
            "pred_seconds": round(pred_sec, 2),
        }
        results.append(res)
        print(f"{desc:<34} | {weighted_score:.5f} | {ko_acc*100:6.2f}% ({ko_corr_cnt:>3}/{ko_tot_cnt}) | {latin_acc*100:6.2f}% ({latin_corr_cnt:>4}/{latin_tot_cnt}) | {total_errors:>3}건 | {predicted_lb:.10f} | {fit_sec:.1f}s / {pred_sec:.2f}s")

    report_file = ROOT / "reports/catboost_param_tuning.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTuning results saved to: {report_file}")


if __name__ == "__main__":
    main()
