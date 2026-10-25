# -*- coding: utf-8 -*-
"""
[Robustness Evaluation] AR-06 모델 노이즈 스트레스 검증 및 타 모델 비교 스크립트

평가 프로토콜:
  1. 표준 1,000행 현실적 노이즈(default 프리셋, seed=20261022) 검증
     - 기존 Matcher(b2efe72), CatBoost(AS-03), SetModel(AS-04)과 100% 동일 조건 비교
  2. 선박 그룹 5-Fold Unseen OOF (12,222행) Clean vs Noisy 검증
     - 훈련에 등장하지 않은 선박에 대한 일반화 노이즈 방어율 전수 실측
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "AR-06"))

from script import CatalogIndex, predict_rows, is_korean_name
from run_ar06 import fit_matching_model
from eval_kit.robustness import RobustnessPerturber, PRESETS

KST = timezone(timedelta(hours=9))


def compute_metrics(preds, gold, is_ko):
    p = np.asarray(preds, dtype=object)
    g = np.asarray(gold, dtype=object)
    k = np.asarray(is_ko, dtype=bool)
    correct = (p == g)
    ko_acc = float(correct[k].mean()) if k.sum() > 0 else 0.0
    nh_acc = float(correct[~k].mean()) if (~k).sum() > 0 else 0.0
    weighted = 0.7 * ko_acc + 0.3 * nh_acc
    return {
        "weighted": weighted,
        "hangul": ko_acc,
        "hangul_correct": int(correct[k].sum()),
        "hangul_rows": int(k.sum()),
        "non_hangul": nh_acc,
        "non_hangul_correct": int(correct[~k].sum()),
        "non_hangul_rows": int((~k).sum()),
        "total_correct": int(correct.sum()),
        "total_rows": len(correct),
    }


def main():
    parser = argparse.ArgumentParser(description="AR-06 노이즈 강건성 검증 및 성능 비교")
    parser.add_argument("--preset", default="default", choices=list(PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--sample-limit", type=int, default=1000, help="표준 비교용 표본 크기 (기본: 1000행)")
    parser.add_argument("--full-oof", action="store_true", help="12,222행 전체 5-Fold OOF 노이즈 평가 실행")
    args = parser.parse_args()

    print("=" * 80)
    print(f"   [AR-06 모델 노이즈 스트레스 검증 및 강건성 평가]")
    print(f"   노이즈 프리셋: '{args.preset}' | 검증 시드: {args.seed}")
    print("=" * 80)

    # 1. 데이터 로드
    train_df = pd.read_csv(ROOT / "data/train.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    catalog_df = pd.read_csv(ROOT / "data/vessel_catalog.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    model_path = ROOT / "AR-06/model/model.json"

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    print(f"\n[1] 카탈로그 인덱싱 중 ({len(catalog_df):,}척)...", end="", flush=True)
    t0_cat = time.monotonic()
    cat_idx = CatalogIndex(catalog_df)
    print(f" 완료 ({time.monotonic() - t0_cat:.2f}s)")

    # 2. Part 1: 표준 1,000행 벤치마크 (기존 reports/robustness/ 비교군과 동일)
    print(f"\n[2] 표준 1,000행 노이즈 스트레스 검증 시작...")
    df_1000 = train_df.head(args.sample_limit).copy()
    perturber = RobustnessPerturber(deepcopy(PRESETS[args.preset]), seed=args.seed)

    t0 = time.monotonic()
    noisy_1000, audit_summary, audits_1000 = perturber.perturb_dataset(df_1000, catalog_df, return_row_audits=True)
    print(f"  -> 노이즈 주입 완료 ({time.monotonic() - t0:.2f}s): "
          f"{audit_summary['any_noise_applied_count']}/{audit_summary['total_rows']}행 변형 ({audit_summary['any_noise_applied_rate']*100:.1f}%)")

    gold_1000 = df_1000["target_vessel_id"].tolist()
    is_ko_1000 = df_1000["vessel_name"].apply(is_korean_name).values

    print("  -> Clean 데이터 추론 중...", end="", flush=True)
    t0 = time.monotonic()
    clean_preds_1000, _ = predict_rows(df_1000, model, cat_idx, use_train_lookup=False, gold_ids=gold_1000)
    clean_sec_1000 = time.monotonic() - t0
    print(f" 완료 ({clean_sec_1000:.2f}s, {clean_sec_1000/len(df_1000)*1000:.2f} ms/행)")

    print("  -> Noisy 데이터 추론 중...", end="", flush=True)
    t0 = time.monotonic()
    noisy_preds_1000, _ = predict_rows(noisy_1000, model, cat_idx, use_train_lookup=False, gold_ids=gold_1000)
    noisy_sec_1000 = time.monotonic() - t0
    print(f" 완료 ({noisy_sec_1000:.2f}s, {noisy_sec_1000/len(df_1000)*1000:.2f} ms/행)")

    clean_metrics_1000 = compute_metrics(clean_preds_1000, gold_1000, is_ko_1000)
    noisy_metrics_1000 = compute_metrics(noisy_preds_1000, gold_1000, is_ko_1000)

    # 순수 노이즈 적용 행 격리 분석
    noisy_mask_1000 = (audits_1000["name_typo_applied"] | audits_1000["hangul_spacing_applied"] |
                       audits_1000["ton_shift_applied"] | audits_1000["ton_missing_applied"] |
                       audits_1000["flag_missing_applied"]).values
    pure_noisy_gold = np.array(gold_1000)[noisy_mask_1000]
    pure_noisy_preds = np.array(noisy_preds_1000)[noisy_mask_1000]
    pure_noisy_ko = np.array(is_ko_1000)[noisy_mask_1000]
    pure_noisy_metrics = compute_metrics(pure_noisy_preds, pure_noisy_gold, pure_noisy_ko)

    retention_1000 = (noisy_metrics_1000["weighted"] / clean_metrics_1000["weighted"]) * 100

    print("\n" + "=" * 80)
    print(f"{'평가 구분':<22} | {'가중 정확도':<15} | {'한글 정확도':<15} | {'비한글 정확도':<15} | {'정답 수/전체'}")
    print("-" * 80)
    print(f"{'AR-06 Clean (원문)':<22} | {clean_metrics_1000['weighted']*100:6.3f}%        | {clean_metrics_1000['hangul']*100:6.3f}%        | {clean_metrics_1000['non_hangul']*100:6.3f}%        | {clean_metrics_1000['total_correct']}/{clean_metrics_1000['total_rows']}")
    print(f"{'AR-06 Noisy (노이즈)':<22} | {noisy_metrics_1000['weighted']*100:6.3f}%        | {noisy_metrics_1000['hangul']*100:6.3f}%        | {noisy_metrics_1000['non_hangul']*100:6.3f}%        | {noisy_metrics_1000['total_correct']}/{noisy_metrics_1000['total_rows']}")
    print(f"{'AR-06 Pure Noisy (변형행)':<22} | {pure_noisy_metrics['weighted']*100:6.3f}%        | {pure_noisy_metrics['hangul']*100:6.3f}%        | {pure_noisy_metrics['non_hangul']*100:6.3f}%        | {pure_noisy_metrics['total_correct']}/{pure_noisy_metrics['total_rows']}")
    print(f"-> 노이즈 방어율 (Retention Rate): {retention_1000:.2f}% (가중 점수 유지율)")
    print("=" * 80)

    # 3. Part 2: 5-Fold Unseen OOF 전수 평가 (옵션 또는 기본 실행)
    oof_results = None
    if args.full_oof:
        print(f"\n[3] 5-Fold Unseen OOF (총 12,222행) Clean vs Noisy 전수 평가 시작...")
        folds_df = pd.read_csv(ROOT / "eval_kit/folds_group5_seed42.csv", dtype={"sample_id": str, "fold": int})
        joined = train_df.merge(folds_df[["sample_id", "fold"]], on="sample_id")

        oof_clean_preds, oof_noisy_preds = [], []
        oof_golds, oof_is_ko = [], []
        oof_audits_list = []

        for k in range(5):
            val_k = joined[joined["fold"] == k].drop(columns="fold").reset_index(drop=True)
            print(f"  - Fold {k+1}/5 검증 중 ({len(val_k):,}행)...", end="", flush=True)

            noisy_k, _, audits_k = perturber.perturb_dataset(val_k, catalog_df, return_row_audits=True)
            oof_audits_list.append(audits_k)

            gold_k = val_k["target_vessel_id"].tolist()
            ko_k = val_k["vessel_name"].apply(is_korean_name).values
            oof_golds.extend(gold_k)
            oof_is_ko.extend(ko_k)

            fit_k = joined[joined["fold"] != k].drop(columns="fold").reset_index(drop=True)
            fold_model = fit_matching_model(fit_k, catalog_df, kr_mode="AR06", en_mode="EN_V3", use_train_lookup=False)
            c_preds, _ = predict_rows(val_k, fold_model, cat_idx, use_train_lookup=False, gold_ids=gold_k)
            n_preds, _ = predict_rows(noisy_k, fold_model, cat_idx, use_train_lookup=False, gold_ids=gold_k)
            oof_clean_preds.extend(c_preds)
            oof_noisy_preds.extend(n_preds)
            print(" 완료!")

        oof_clean_m = compute_metrics(oof_clean_preds, oof_golds, oof_is_ko)
        oof_noisy_m = compute_metrics(oof_noisy_preds, oof_golds, oof_is_ko)
        full_audit_df = pd.concat(oof_audits_list, ignore_index=True)
        oof_noisy_mask = (full_audit_df["name_typo_applied"] | full_audit_df["hangul_spacing_applied"] |
                          full_audit_df["ton_shift_applied"] | full_audit_df["ton_missing_applied"] |
                          full_audit_df["flag_missing_applied"]).values
        oof_pure_noisy_m = compute_metrics(np.array(oof_noisy_preds)[oof_noisy_mask],
                                           np.array(oof_golds)[oof_noisy_mask],
                                           np.array(oof_is_ko)[oof_noisy_mask])
        oof_retention = (oof_noisy_m["weighted"] / oof_clean_m["weighted"]) * 100

        oof_results = {
            "clean": oof_clean_m,
            "noisy": oof_noisy_m,
            "pure_noisy": oof_pure_noisy_m,
            "retention_pct": oof_retention,
        }

        print("\n" + "=" * 80)
        print(f"   [5-Fold Unseen OOF 12,222행 전수 노이즈 평가 결과]")
        print("-" * 80)
        print(f"OOF Clean  : 가중 {oof_clean_m['weighted']*100:.3f}% | 한글 {oof_clean_m['hangul']*100:.3f}% | 비한글 {oof_clean_m['non_hangul']*100:.3f}%")
        print(f"OOF Noisy  : 가중 {oof_noisy_m['weighted']*100:.3f}% | 한글 {oof_noisy_m['hangul']*100:.3f}% | 비한글 {oof_noisy_m['non_hangul']*100:.3f}%")
        print(f"OOF PureNoisy: 가중 {oof_pure_noisy_m['weighted']*100:.3f}% | 한글 {oof_pure_noisy_m['hangul']*100:.3f}% | 비한글 {oof_pure_noisy_m['non_hangul']*100:.3f}%")
        print(f"OOF 노이즈 방어율: {oof_retention:.2f}%")
        print("=" * 80)

    # 4. 결과 저장
    report = {
        "timestamp_kst": datetime.now(KST).isoformat(),
        "model_id": "AR-06",
        "preset": args.preset,
        "seed": args.seed,
        "config": deepcopy(PRESETS[args.preset]),
        "benchmark_1000": {
            "audit_summary": audit_summary,
            "clean_metrics": clean_metrics_1000,
            "noisy_metrics": noisy_metrics_1000,
            "pure_noisy_metrics": pure_noisy_metrics,
            "retention_rate_pct": retention_1000,
            "delta_weighted_pct": (noisy_metrics_1000["weighted"] - clean_metrics_1000["weighted"]) * 100,
            "inference_seconds": {
                "clean": clean_sec_1000,
                "noisy": noisy_sec_1000,
            }
        },
        "oof_12222": oof_results,
    }

    report_dir = ROOT / "reports/robustness"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"robustness_AR-06_{args.preset}_s{args.seed}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[저장 완료] 노이즈 검증 리포트: {report_file}")


if __name__ == "__main__":
    main()
