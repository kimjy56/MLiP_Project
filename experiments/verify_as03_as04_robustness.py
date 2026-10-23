"""Fair 5-Fold Unseen OOF Robustness Stress Test: AS-03 (JY-10 CatBoost) vs AS-04 (AS-SET-09 Set Model).

Evaluates both models on completely Unseen fold validation rows (12,222 rows total)
under both Clean and Noisy (seed=20261022) conditions, and analyzes pure noise survival rates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))

from model.matcher import Matcher, CatalogIndex, hangul
from decision_models.matcher_adapter import DecisionMatcher
from decision_models.set_model import SetDecisionModel
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
        'weighted': weighted,
        'hangul': ko_acc,
        'hangul_correct': int(correct[k].sum()),
        'hangul_total': int(k.sum()),
        'non_hangul': nh_acc,
        'non_hangul_correct': int(correct[~k].sum()),
        'non_hangul_total': int((~k).sum()),
        'total_correct': int(correct.sum()),
        'total_rows': len(correct)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default='default', choices=list(PRESETS.keys()))
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--max-rows-per-fold', type=int, default=None, help='빠른 진단용 fold당 표본 제한')
    args = parser.parse_args()

    print("=" * 80)
    print(f"   [AS-03 vs AS-04 Unseen 5-Fold OOF Robustness Stress Test]")
    print(f"   노이즈 프리셋: '{args.preset}' | 검증 시드: {args.seed} (학습 시드와 완전 분리)")
    print("=" * 80)

    # 1. Load Data
    train_df = pd.read_csv(ROOT / 'data/train.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    catalog = pd.read_csv(ROOT / 'data/vessel_catalog.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    folds_df = pd.read_csv(ROOT / 'eval_kit/folds_group5_seed42.csv', dtype={'sample_id': str, 'fold': int})
    joined = train_df.merge(folds_df[['sample_id', 'fold']], on='sample_id')

    perturber = RobustnessPerturber(PRESETS[args.preset], seed=args.seed)

    all_audits = []
    as03_clean_preds, as03_noisy_preds = [], []
    as04_clean_preds, as04_noisy_preds = [], []
    all_golds, all_is_ko = [], []

    assets_file = ROOT / 'model/assets.json'

    # 2. Iterate 5 Folds
    t_start = time.monotonic()
    for k in range(5):
        val = joined[joined.fold == k].drop(columns='fold').reset_index(drop=True)
        if args.max_rows_per_fold and args.max_rows_per_fold < len(val):
            val = val.head(args.max_rows_per_fold).copy()

        print(f"\n--- Fold {k+1}/5 검증 ({len(val):,}개 Unseen 보류 행) ---", flush=True)

        # Perturb Unseen Validation Fold
        noisy_val, summary, audits = perturber.perturb_dataset(val, catalog, return_row_audits=True)
        all_audits.append(audits)

        gold = val.target_vessel_id.tolist()
        is_ko = val.vessel_name.map(hangul).values
        all_golds.extend(gold)
        all_is_ko.extend(is_ko)

        # AS-03 (JY-10 CatBoost fold k model)
        as03_model_dir = ROOT / f'reports/oof_models_jy10/fold_{k}'
        matcher_as03 = Matcher(catalog, as03_model_dir, use_train_lookup=False)
        print("  [AS-03 CatBoost] Clean 추론...", end='', flush=True)
        t0 = time.monotonic()
        p_as03_c = matcher_as03.predict(val, use_direct=False)
        print(f" 완료 ({time.monotonic()-t0:.1f}s) | Noisy 추론...", end='', flush=True)
        t0 = time.monotonic()
        p_as03_n = matcher_as03.predict(noisy_val, use_direct=False)
        print(f" 완료 ({time.monotonic()-t0:.1f}s)")
        as03_clean_preds.extend(p_as03_c)
        as03_noisy_preds.extend(p_as03_n)

        # AS-04 (AS-SET-09 Set Model fold k model)
        as04_set_dir = ROOT / f'artifacts/AS-SET-09/fold_{k}/set'
        set_scorer = SetDecisionModel.load(as04_set_dir, device='cpu')
        matcher_as04 = DecisionMatcher(catalog, assets_file, set_scorer)
        print("  [AS-04 SetModel] Clean 추론...", end='', flush=True)
        t0 = time.monotonic()
        p_as04_c = matcher_as04.predict(val, use_direct=False)
        print(f" 완료 ({time.monotonic()-t0:.1f}s) | Noisy 추론...", end='', flush=True)
        t0 = time.monotonic()
        p_as04_n = matcher_as04.predict(noisy_val, use_direct=False)
        print(f" 완료 ({time.monotonic()-t0:.1f}s)")
        as04_clean_preds.extend(p_as04_c)
        as04_noisy_preds.extend(p_as04_n)

    total_sec = time.monotonic() - t_start
    print(f"\n전체 5-Fold 추론 완료 (총 소요 시간: {total_sec:.1f}초)")

    # 3. Overall Metrics Evaluation
    all_is_ko = np.asarray(all_is_ko, dtype=bool)
    as03_c_met = compute_metrics(as03_clean_preds, all_golds, all_is_ko)
    as03_n_met = compute_metrics(as03_noisy_preds, all_golds, all_is_ko)
    as04_c_met = compute_metrics(as04_clean_preds, all_golds, all_is_ko)
    as04_n_met = compute_metrics(as04_noisy_preds, all_golds, all_is_ko)

    audit_df = pd.concat(all_audits, ignore_index=True)
    noisy_mask = (
        audit_df.name_typo_applied |
        audit_df.hangul_spacing_applied |
        audit_df.ton_shift_applied |
        audit_df.ton_missing_applied |
        audit_df.flag_missing_applied
    ).values

    n_noisy = int(noisy_mask.sum())
    n_clean_subset = len(noisy_mask) - n_noisy

    as03_pure_noisy_met = compute_metrics(
        np.array(as03_noisy_preds)[noisy_mask],
        np.array(all_golds)[noisy_mask],
        all_is_ko[noisy_mask]
    )
    as04_pure_noisy_met = compute_metrics(
        np.array(as04_noisy_preds)[noisy_mask],
        np.array(all_golds)[noisy_mask],
        all_is_ko[noisy_mask]
    )

    # 5. Output Comparison Table
    print("\n" + "=" * 82)
    print("                 [1. 전체 5-Fold OOF 성능 비교 (Clean vs Noisy)]")
    print("=" * 82)
    print(f"{'모델':<12} | {'조건':<8} | {'가중 점수':<14} | {'한글 정확도':<16} | {'비한글 정확도':<16} | {'총 오답'}")
    print("-" * 82)
    print(f"{'AS-03 (Cat)':<10} | {'Clean':<8} | {as03_c_met['weighted']*100:>12.3f}% | {as03_c_met['hangul']*100:>14.3f}% | {as03_c_met['non_hangul']*100:>14.3f}% | {as03_c_met['total_rows']-as03_c_met['total_correct']}행")
    print(f"{'AS-03 (Cat)':<10} | {'Noisy':<8} | {as03_n_met['weighted']*100:>12.3f}% | {as03_n_met['hangul']*100:>14.3f}% | {as03_n_met['non_hangul']*100:>14.3f}% | {as03_n_met['total_rows']-as03_n_met['total_correct']}행")
    d_as03 = as03_n_met['weighted'] - as03_c_met['weighted']
    print(f" {'-> AS-03 변화량:':<18} | {d_as03*100:>+11.3f}%p | {(as03_n_met['hangul']-as03_c_met['hangul'])*100:>+13.3f}%p | {(as03_n_met['non_hangul']-as03_c_met['non_hangul'])*100:>+13.3f}%p | 방어율: {(as03_n_met['weighted']/as03_c_met['weighted'])*100:.2f}%")
    print("-" * 82)
    print(f"{'AS-04 (Set)':<10} | {'Clean':<8} | {as04_c_met['weighted']*100:>12.3f}% | {as04_c_met['hangul']*100:>14.3f}% | {as04_c_met['non_hangul']*100:>14.3f}% | {as04_c_met['total_rows']-as04_c_met['total_correct']}행")
    print(f"{'AS-04 (Set)':<10} | {'Noisy':<8} | {as04_n_met['weighted']*100:>12.3f}% | {as04_n_met['hangul']*100:>14.3f}% | {as04_n_met['non_hangul']*100:>14.3f}% | {as04_n_met['total_rows']-as04_n_met['total_correct']}행")
    d_as04 = as04_n_met['weighted'] - as04_c_met['weighted']
    print(f" {'-> AS-04 변화량:':<18} | {d_as04*100:>+11.3f}%p | {(as04_n_met['hangul']-as04_c_met['hangul'])*100:>+13.3f}%p | {(as04_n_met['non_hangul']-as04_c_met['non_hangul'])*100:>+13.3f}%p | 방어율: {(as04_n_met['weighted']/as04_c_met['weighted'])*100:.2f}%")
    print("=" * 82)

    print("\n" + "=" * 82)
    print(f"          [2. 순수 노이즈 적용 행 격리 심층 분석 (총 {n_noisy:,}행 / {len(noisy_mask):,}행 중)]")
    print("=" * 82)
    print(f"{'모델':<12} | {'노이즈 집단 가중점수':<18} | {'한글 정확도':<16} | {'비한글 정확도':<16} | {'노이즈 오답수'}")
    print("-" * 82)
    print(f"{'AS-03 (Cat)':<10} | {as03_pure_noisy_met['weighted']*100:>16.3f}% | {as03_pure_noisy_met['hangul']*100:>14.3f}% | {as03_pure_noisy_met['non_hangul']*100:>14.3f}% | {as03_pure_noisy_met['total_rows']-as03_pure_noisy_met['total_correct']}행")
    print(f"{'AS-04 (Set)':<10} | {as04_pure_noisy_met['weighted']*100:>16.3f}% | {as04_pure_noisy_met['hangul']*100:>14.3f}% | {as04_pure_noisy_met['non_hangul']*100:>14.3f}% | {as04_pure_noisy_met['total_rows']-as04_pure_noisy_met['total_correct']}행")
    diff_noisy = as04_pure_noisy_met['weighted'] - as03_pure_noisy_met['weighted']
    print("-" * 82)
    print(f"순수 노이즈 환경 격차: AS-04가 AS-03 대비 {diff_noisy*100:+.3f}%p 더 우수한 방어력 기록")
    print("=" * 82)

    # 6. Save JSON Report
    report = {
        'timestamp_kst': datetime.now(KST).isoformat(),
        'preset': args.preset,
        'seed': args.seed,
        'total_evaluated_rows': len(all_golds),
        'pure_noisy_rows': n_noisy,
        'pure_clean_rows': n_clean_subset,
        'as03': {
            'clean': as03_c_met,
            'noisy': as03_n_met,
            'pure_noisy_slice': as03_pure_noisy_met,
            'retention_pct': (as03_n_met['weighted'] / as03_c_met['weighted']) * 100.0,
            'delta_pct': d_as03 * 100.0
        },
        'as04': {
            'clean': as04_c_met,
            'noisy': as04_n_met,
            'pure_noisy_slice': as04_pure_noisy_met,
            'retention_pct': (as04_n_met['weighted'] / as04_c_met['weighted']) * 100.0,
            'delta_pct': d_as04 * 100.0
        }
    }
    rep_dir = ROOT / 'reports/robustness'
    rep_dir.mkdir(parents=True, exist_ok=True)
    out_file = rep_dir / f"compare_as03_as04_{args.preset}_s{args.seed}.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n[리포트 저장 완료] {out_file}")


if __name__ == '__main__':
    main()
