"""Feature Combination and Ablation Experiment.
Tests 8 feature subsets using the optimal CatBoost model (depth=6, l2=7).
Precomputes Fold 0 features once and slices feature columns for ultra-fast, robust evaluation.
Reports local metrics and predicted Public LB scores.
"""
from __future__ import annotations

import concurrent.futures
import json
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
    print("=== Feature Combination Ablation Study on Fold 0 ===")
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
    all_feature_names = index.feature_names
    print(f"   Full features ({len(all_feature_names)}): {all_feature_names}\n")

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
    x_train_full = np.concatenate(matrices)
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
    val_x_full = np.concatenate(val_matrices) if val_matrices else np.empty((0, len(all_feature_names)), dtype=np.float32)
    cand_counts = [len(c) for c, _ in val_built]
    t_val_sec = time.monotonic() - t_val_start
    print(f"   Validation features ready: {len(val_x_full)} candidate pairs in {t_val_sec:.2f}s\n")

    y_true = np.asarray(val_df.target_vessel_id.tolist())
    ko_mask = val_df.vessel_name.map(hangul).values

    # Feature groups
    name_feats = ["ratio", "roman_ratio", "length_ratio", "name_length", "catalog_name_length",
                  "token_count_gap", "token_roman_idf", "token_roman_min", "token_roman_last"]
    digit_feats = ["digits_equal", "digits_missing", "number_sequence_equal", "number_conflict"]
    phone_feats = ["phonetic_ratio", "phonetic_partial", "weighted_phone_full", "weighted_phone_token_idf",
                   "weighted_phone_token_min", "weighted_phone_last", "weighted_phone_residual", "phone_shared_edge"]
    tonnage_feats = ["gross_gap", "international_gap", "best_gap", "gross_exact", "international_exact",
                     "gross_missing", "international_missing"]
    meta_feats = ["type_probability", "flag_probability", "flag_missing", "is_hangul"]

    # Combinations to test
    combinations = [
        {
            "desc": "전체 특징 (Full 32개, Baseline)",
            "selected": all_feature_names
        },
        {
            "desc": "선종/국적 메타 제외 (No Metadata, 28개)",
            "selected": [f for f in all_feature_names if f not in meta_feats]
        },
        {
            "desc": "톤수 특징 제외 (No Tonnage, 25개)",
            "selected": [f for f in all_feature_names if f not in tonnage_feats]
        },
        {
            "desc": "발음 특징 제외 (No Phonetic, 25개)",
            "selected": [f for f in all_feature_names if f not in phone_feats]
        },
        {
            "desc": "이름 + 숫자만 사용 (Name + Digit Only, 13개)",
            "selected": [f for f in all_feature_names if f in (name_feats + digit_feats)]
        },
        {
            "desc": "이름 + 톤수만 사용 (Name + Tonnage, 16개)",
            "selected": [f for f in all_feature_names if f in (name_feats + tonnage_feats)]
        },
        {
            "desc": "메타 제외 + 핵심 톤수만 (Name + Phone + Core Tonnage, 22개)",
            "selected": [f for f in all_feature_names if f in (name_feats + digit_feats + phone_feats + ["gross_gap", "best_gap", "gross_exact"])]
        },
        {
            "desc": "발음 + 메타 제외 (Name + Digit + Tonnage, 20개)",
            "selected": [f for f in all_feature_names if f in (name_feats + digit_feats + tonnage_feats)]
        },
    ]

    results = []
    print(f"{'특징 조합 설명':<44} | {'특징수':<5} | {'가중점수':<8} | {'한글 정확도':<14} | {'비한글 정확도':<14} | {'오답':<4} | {'예측 LB 점수':<14}")
    print("-" * 128)

    base_ko_correct = 232  # standard baseline

    for combo in combinations:
        desc = combo["desc"]
        selected_cols = combo["selected"]
        col_indices = [all_feature_names.index(c) for c in selected_cols]

        # Slice features
        x_tr = x_train_full[:, col_indices]
        v_x = val_x_full[:, col_indices]

        # Train CatBoost optimal model
        t0 = time.monotonic()
        cb = CatBoostDecisionModel(task_type="CPU", thread_count=6, depth=6, l2_leaf_reg=7, learning_rate=0.05, iterations=350)
        cb.fit(x_tr, y_train, w_train, selected_cols)
        fit_sec = time.monotonic() - t0

        # Predict
        scores = cb.predict(v_x)
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

        correct = (y_true == preds)
        ko_acc = correct[ko_mask].mean()
        latin_acc = correct[~ko_mask].mean()
        weighted_score = 0.7 * ko_acc + 0.3 * latin_acc
        total_errors = int((~correct).sum())
        ko_corr_cnt = int(correct[ko_mask].sum())
        ko_tot_cnt = int(ko_mask.sum())
        latin_corr_cnt = int(correct[~ko_mask].sum())
        latin_tot_cnt = int((~ko_mask).sum())

        delta_ko = ko_corr_cnt - base_ko_correct
        estimated_public_korean = 434 + (delta_ko * 3)
        estimated_public_korean = max(390, min(500, estimated_public_korean))
        predicted_lb = (estimated_public_korean / 760.0) + (3.0 * LATIN_LB_BASELINE / 35680.0)

        res = {
            "desc": desc,
            "feature_count": len(selected_cols),
            "weighted_score": round(weighted_score, 5),
            "ko_acc": round(ko_acc, 5),
            "ko_correct": f"{ko_corr_cnt}/{ko_tot_cnt}",
            "latin_acc": round(latin_acc, 5),
            "latin_correct": f"{latin_corr_cnt}/{latin_tot_cnt}",
            "total_errors": total_errors,
            "predicted_lb_score": round(predicted_lb, 10),
            "estimated_public_korean": estimated_public_korean,
            "fit_seconds": round(fit_sec, 2),
            "selected_features": selected_cols
        }
        results.append(res)
        print(f"{desc:<44} | {len(selected_cols):>4}개 | {weighted_score:.5f} | {ko_acc*100:6.2f}% ({ko_corr_cnt:>3}/{ko_tot_cnt}) | {latin_acc*100:6.2f}% ({latin_corr_cnt:>4}/{latin_tot_cnt}) | {total_errors:>3}건 | {predicted_lb:.10f}")

    report_file = ROOT / "reports/feature_combination_tuning.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFeature ablation results saved to: {report_file}")


if __name__ == "__main__":
    main()
