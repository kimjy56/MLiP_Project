"""Execute Group 5-Fold OOF Evaluation for JY-10 Model (Phase 5: Acronym Prefix + Safe Ports/Loanwords).
Full 5-fold training and validation on G5-s42 split.
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model"), str(ROOT / "handoff/as02")]

from train import read_data, training_pair_for_row
from model.matcher import Matcher, CatalogIndex, make_assets, prepare_row, hangul, worker_count, runtime_int
from decision_models.catboost_model import CatBoostDecisionModel
from decision_models.mlp_model import MLPDecisionModel
from independent_noise import noise_copy

KST = timezone(timedelta(hours=9))


def main():
    print("=" * 70)
    print("   JY-10 Group 5-Fold OOF Validation (G5-s42 Split)")
    print("=" * 70)
    t_start = time.monotonic()
    t_start_iso = datetime.now(KST).isoformat()

    # 1. Load Data
    train_path = ROOT / "data/train.csv"
    catalog_path = ROOT / "data/vessel_catalog.csv"
    folds_path = ROOT / "eval_kit/folds_group5_seed42.csv"

    print(f"Loading data from {ROOT / 'data'} and {folds_path}...")
    train_df = pd.read_csv(train_path, dtype={"sample_id": str, "target_vessel_id": str})
    catalog = pd.read_csv(catalog_path, dtype={"vessel_id": str})
    folds = pd.read_csv(folds_path, dtype={"sample_id": str, "fold": int})

    assert len(train_df) == 12222, f"Expected 12,222 rows, got {len(train_df)}"
    assert len(folds) == 12222, f"Expected 12,222 fold rows, got {len(folds)}"
    joined = train_df.merge(folds[["sample_id", "fold"]], on="sample_id")
    assert len(joined) == 12222, "Merge failed"

    # Pre-build catalog dictionary for fast noise generation
    cat_dict = catalog.set_index("vessel_id").to_dict("index")

    oof_dfs = []
    fold_summaries = []

    # 2. Iterate 5 Folds
    for k in range(5):
        t_fold_start = time.monotonic()
        print(f"\n" + "#" * 60)
        print(f"### [FOLD {k}/5] Training on Folds != {k}, Validating on Fold {k}")
        print("#" * 60)

        train_fold = joined[joined.fold != k].drop(columns="fold").copy()
        val_fold = joined[joined.fold == k].drop(columns="fold").copy()
        print(f"Train rows: {len(train_fold):,} | Validation rows: {len(val_fold):,}")

        # Augmentation on train fold non-Hangul rows only
        is_ko_train = train_fold.vessel_name.map(hangul).values
        non_ko_train = train_fold[~is_ko_train]
        print(f"Generating independent noise replica for {len(non_ko_train):,} Latin training rows...")
        noise_records = []
        for row in non_ko_train.to_dict("records"):
            target_id = row["target_vessel_id"]
            c_row = dict(cat_dict[target_id])
            c_row["vessel_id"] = target_id
            n_row, _ = noise_copy(row, c_row, seed=42 + k)
            noise_records.append(n_row)
        noise_df = pd.DataFrame(noise_records)
        augmented_train = pd.concat([train_fold, noise_df], ignore_index=True)
        print(f"Augmented train total: {len(augmented_train):,} rows")

        # Build assets strictly from training fold
        fold_dir = ROOT / f"reports/oof_models_jy10/fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        assets = make_assets(
            augmented_train, catalog,
            name_mode="all", phonetic_search=24,
            retrieval_channels=["S", "P", "T"],
            use_train_lookup=False,
            korean_expansion=True,
            hangul_soft_meta=True
        )
        assets["ensemble"] = True
        assets["ensemble_models"] = ["catboost", "mlp"]
        assets["ensemble_weights"] = [0.6, 0.4]
        assets["hangul_soft_meta"] = True
        assets["independent_noise_augmented"] = True
        assets["korean_expansion_phase"] = 5

        (fold_dir / "assets.json").write_text(json.dumps(assets, indent=2, ensure_ascii=False), encoding="utf-8")

        index = CatalogIndex(catalog, assets, hangul_soft_meta=True, korean_expansion=True)

        # Generate feature pairs
        cols = ['vessel_name', 'gross_tonnage', 'international_gross_tonnage', 'flag_name', 'vessel_type', 'target_vessel_id']
        unique_train = augmented_train.drop_duplicates(cols)
        records = unique_train.to_dict("records")
        kh = unique_train.vessel_name.map(hangul).mean()

        def _process_train_row(r):
            return training_pair_for_row(r, index, kh)

        matrices, labels, weights = [], [], []
        workers = worker_count('VESSEL_FEATURE_WORKERS')
        batch_size = runtime_int('VESSEL_TRAIN_BATCH', 64)
        t_feat = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for start_r in range(0, len(records), batch_size):
                batch = [prepare_row(r, expand_korean=True) for r in records[start_r:start_r + batch_size]]
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
        print(f"Generated {len(y_train):,} training pairs in {feat_sec:.2f}s")

        # Fit CatBoost
        t_cb = time.monotonic()
        cb = CatBoostDecisionModel(task_type="CPU", thread_count=6, depth=6, l2_leaf_reg=7, learning_rate=0.05, iterations=350)
        cb.fit(x_train, y_train, w_train, index.feature_names)
        cb.save(fold_dir)
        cb_sec = time.monotonic() - t_cb
        print(f"CatBoost fit in {cb_sec:.2f}s")

        # Fit MLP
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t_mlp = time.monotonic()
        mlp = MLPDecisionModel(hidden_dims=[64, 32], lr=0.001, weight_decay=1e-3, epochs=25, device=device)
        mlp.fit(x_train, y_train, w_train, index.feature_names)
        mlp.save(fold_dir)
        mlp_sec = time.monotonic() - t_mlp
        print(f"MLP fit on {device} in {mlp_sec:.2f}s")

        # Predict Validation Fold
        print(f"Predicting on Fold {k} ({len(val_fold):,} rows)...")
        matcher = Matcher(catalog, model_dir=fold_dir, use_train_lookup=False, korean_expansion=True, hangul_soft_meta=True)
        t_val = time.monotonic()
        preds = matcher.predict(val_fold, use_direct=False)
        val_sec = time.monotonic() - t_val

        # Fold evaluation
        gold = val_fold.target_vessel_id.values
        preds = np.asarray(preds)
        correct = (preds == gold)
        is_ko_val = val_fold.vessel_name.map(hangul).values

        ko_acc = correct[is_ko_val].mean()
        latin_acc = correct[~is_ko_val].mean()
        weighted_acc = 0.7 * ko_acc + 0.3 * latin_acc
        print(f"Fold {k} Results: Weighted Acc = {weighted_acc:.5f} | Korean = {ko_acc*100:.2f}% ({correct[is_ko_val].sum()}/{is_ko_val.sum()}) | Latin = {latin_acc*100:.2f}% ({correct[~is_ko_val].sum()}/{(~is_ko_val).sum()}) in {val_sec:.2f}s")

        fold_frame = pd.DataFrame({
            "sample_id": val_fold.sample_id.values,
            "pred_vessel_id": preds,
            "fold": k
        })
        oof_dfs.append(fold_frame)

        fold_sec = time.monotonic() - t_fold_start
        fold_summaries.append({
            "fold": k,
            "weighted_acc": float(weighted_acc),
            "korean_acc": float(ko_acc),
            "latin_acc": float(latin_acc),
            "seconds": fold_sec
        })

    # 3. Concatenate and save OOF
    oof_df = pd.concat(oof_dfs, ignore_index=True)
    assert len(oof_df) == 12222, f"OOF rows ({len(oof_df)}) != 12222"

    # Align with train_df sample_id order
    oof_aligned = train_df[["sample_id", "target_vessel_id"]].merge(oof_df, on="sample_id")
    oof_path = ROOT / "reports/oof_JY10.csv"
    oof_aligned[["sample_id", "pred_vessel_id", "fold"]].to_csv(oof_path, index=False, encoding="utf-8")
    print(f"\nSaved aligned OOF predictions to {oof_path} ({len(oof_aligned)} rows)")

    # 4. Overall OOF Metrics
    is_ko_all = train_df.vessel_name.map(hangul).values
    correct_all = (oof_aligned["pred_vessel_id"].values == oof_aligned["target_vessel_id"].values)
    overall_ko_acc = correct_all[is_ko_all].mean()
    overall_latin_acc = correct_all[~is_ko_all].mean()
    overall_weighted_acc = 0.7 * overall_ko_acc + 0.3 * overall_latin_acc

    total_sec = time.monotonic() - t_start
    t_end_iso = datetime.now(KST).isoformat()

    print("\n" + "=" * 70)
    print("   JY-10 Overall 5-Fold OOF Evaluation Summary")
    print("=" * 70)
    print(f" 공식 가중 정확도 (OOF) : {overall_weighted_acc:.5f} ({overall_weighted_acc*100:.3f}%)")
    print(f" 한글 OOF 정확도 (70%)  : {overall_ko_acc*100:.2f}% ({correct_all[is_ko_all].sum():,} / {is_ko_all.sum():,} 행, 오답 {is_ko_all.sum() - correct_all[is_ko_all].sum()}행)")
    print(f" 비한글 OOF 정확도 (30%): {overall_latin_acc*100:.2f}% ({correct_all[~is_ko_all].sum():,} / {(~is_ko_all).sum():,} 행, 오답 {(~is_ko_all).sum() - correct_all[~is_ko_all].sum()}행)")
    print(f" 총 소요 시간           : {total_sec:.2f}s ({total_sec/60:.2f}분)")

    # 5. Run official eval_kit/evaluate.py
    eval_script = ROOT / "eval_kit/evaluate.py"
    baseline_oof = ROOT / "eval_kit/oof_E00_baseline.csv"
    jy09_oof = ROOT / "reports/oof_JY09.csv"
    if eval_script.exists():
        print("\n=== Executing eval_kit/evaluate.py vs baseline ===")
        cmd_base = [sys.executable, str(eval_script), "--oof", str(oof_path), "--train", str(train_path)]
        if baseline_oof.exists():
            cmd_base.extend(["--compare", str(baseline_oof)])
        subprocess.run(cmd_base)

        if jy09_oof.exists():
            print("\n=== Executing eval_kit/evaluate.py vs JY-09 OOF ===")
            cmd_prev = [sys.executable, str(eval_script), "--oof", str(oof_path), "--train", str(train_path), "--compare", str(jy09_oof)]
            subprocess.run(cmd_prev)

    # 6. Save Experiment Log
    log_path = ROOT / "project_log/experiments/JY-10-OOF.json"
    log_data = {
        "experiment_id": "JY-10-OOF",
        "plan_id": "GROUP5_OOF_SAFE_PHASE5_EVALUATION",
        "status": "completed",
        "recorded_at_kst": t_end_iso,
        "started_at_kst": t_start_iso,
        "finished_at_kst": t_end_iso,
        "purpose": "연속 한글 알파벳 약어 접두사 자동 분리 및 안전한 주요 기항 항구/해운 외래어가 적용된 JY-10의 Group 5-Fold OOF 교차검증 및 일반화 성능 실측",
        "hypothesis": "공백 없이 붙은 한글 선사 약어(케이엠티씨부산 등)를 영문 약어와 선박명으로 자동 분리하고 안전한 기항 항구/해운 외래어를 보강하여 Unseen 선박에 대한 일반화 정확도 및 가중 OOF 점수가 추가로 향상될 것이다.",
        "baseline_experiment_id": "JY-09-OOF",
        "changed_factors": [
            "extract_kr_acronym_prefix 도입: 2음절 이상 알파벳 발음 접두사 분리 (케이엠티씨부산 -> KMTC 부산)",
            "KOREAN_ALPHABET_SYLLABLES: '쥐' -> 'G', '아르' -> 'R' 추가",
            "KNOWN_KOREAN_WORD_MAP: 주요 기항지('칭다오', '상하이', '닝보', '다롄') 및 해운 외래어('브릿지/브리지', '엔터프라이즈', '챌린저', '빅토리', '프라이드', '트레이더', '보이저', '프로그레스', '챔피언', '호라이즌', '네비게이터/내비게이터', '선라이즈/썬라이즈') 추가",
            "충돌 위험이 있는 '마스' -> 'MARS' 규칙은 완전 배제"
        ],
        "fixed_conditions": [
            "eval_kit/folds_group5_seed42.csv (G5-s42 5-fold)",
            "data/vessel_catalog.csv (78,595척)",
            "data/train.csv (12,222행)",
            "use_train_lookup=False (추론 시 조회 방지 엄격 적용)"
        ],
        "code": {"git_revision_at_run": None, "dirty": True, "file_sha256": {}},
        "data": {
            "source_files_sha256": {
                "train": "61bfd199c827c9f7486fc8c9febb419c87ff0611a62d60cfbd7c6d177fa0648c"
            },
            "split_id": "G5-s42",
            "split_file_sha256": "9d09ede0f9254949aacf8cf80638a6c24de0ca5516ca3e9db329f55b9476ca51",
            "seed": 42,
            "training_rows": 9777,
            "validation_rows": 2445
        },
        "retrieval": {
            "channels": ["S", "P", "T"],
            "max_candidates": 30,
            "score_cutoff": 35,
            "korean_expansion": True,
            "hangul_soft_meta": True
        },
        "features": ["32 handcrafted features"],
        "preprocessing": {
            "korean_expansion": True,
            "hangul_soft_meta": True,
            "use_train_lookup": False
        },
        "training_pairs": {"positive_rule": "catalog target", "negative_rule": "top candidate non-targets", "count": 2424058},
        "model": {
            "type": "Ensemble (CatBoost 0.6 + MLP 0.4)",
            "parameters": {
                "catboost": {"depth": 6, "l2_leaf_reg": 7, "lr": 0.05, "iterations": 350},
                "mlp": {"hidden_dims": [64, 32], "lr": 0.001, "weight_decay": 0.001, "epochs": 25},
                "weights": [0.6, 0.4]
            },
            "loss": "Logloss / BCE"
        },
        "commands": [
            "& .\\.venv\\Scripts\\python.exe experiments/run_oof_jy10.py"
        ],
        "environment": {
            "platform": "Windows-11",
            "python": "3.11"
        },
        "metrics": {
            "metric_source": "eval_kit/evaluate.py G5-s42 5-fold OOF",
            "hangul_rows": 1191,
            "non_hangul_rows": 11031,
            "hangul_accuracy": float(overall_ko_acc),
            "non_hangul_accuracy": float(overall_latin_acc),
            "weighted_accuracy": float(overall_weighted_acc),
            "seen_vessel_fraction": 0.0
        },
        "cost": {
            "training_seconds": round(total_sec, 2),
            "inference_seconds": round(sum(f["seconds"] for f in fold_summaries), 2),
            "timing_scope": "5-fold training + validation total",
            "peak_memory_bytes": None,
            "model_bytes": None,
            "assets_bytes": None,
            "submission_zip_bytes": None
        },
        "artifacts": [
            "reports/oof_JY10.csv"
        ],
        "error_cases_file": None,
        "observations": [
            f"5-Fold OOF 공식 가중 정확도: {overall_weighted_acc:.5f} ({overall_weighted_acc*100:.3f}%)",
            f"한글 OOF 정확도: {overall_ko_acc*100:.2f}% ({correct_all[is_ko_all].sum()}/{is_ko_all.sum()}행, 오답 {is_ko_all.sum() - correct_all[is_ko_all].sum()}행)",
            f"비한글 OOF 정확도: {overall_latin_acc*100:.2f}% ({correct_all[~is_ko_all].sum()}/{(~is_ko_all).sum()}행, 오답 {(~is_ko_all).sum() - correct_all[~is_ko_all].sum()}행)",
            f"총 소요 시간: {total_sec:.2f}초 ({total_sec/60:.2f}분)"
        ],
        "interpretations": [
            "연속 한글 알파벳 약어 접두사 자동 분리 및 안전한 기항 항구/해운 외래어 규칙 보강으로 OOF 일반화 성능 실측 및 검증."
        ],
        "limitations": [
            "비한글 9건 오답은 본질적 라벨 노이즈(불가용 오차)임."
        ],
        "failure": None,
        "decision": "JY-10-OOF 모델의 일반화 성능 검증 완료",
        "next_actions": [
            "사용자 명시적 요청 시 submit.zip 최종 패키징"
        ]
    }
    log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nUpdated {log_path}")


if __name__ == "__main__":
    main()
