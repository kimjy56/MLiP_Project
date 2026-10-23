#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""robustness.py — 재사용 가능한 모델 강건성 및 스트레스 테스트 도구 (DACON 236771)

노이즈(철자 오타, 톤수 변동/결측, 국적 결측, 한글 띄어쓰기 변형 등) 주입을 통해
모델이 실전 환경의 데이터 결함에 얼마나 견고한지(일반화 및 방어율)를 정량 평가합니다.

사용법:
  # 1. JY-10 (현재 루트 model) 기본 노이즈 스트레스 테스트
  python eval_kit/robustness.py --model model --preset default --seed 20261022

  # 2. SetDecisionModel (AS-SET-09) 톤수 왜곡 스트레스 테스트
  python eval_kit/robustness.py --set-model artifacts/AS-SET-09 --preset tonnage_only --seed 20261022

  # 3. 노이즈 주입된 데이터셋만 CSV로 생성 (다른 검증에 재사용)
  python eval_kit/robustness.py --generate-only --input data/train.csv --output data/train_noisy.csv --seed 20261022
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))

from model.matcher import Matcher, CatalogIndex, hangul, feature_names
from decision_models.matcher_adapter import DecisionMatcher

KST = timezone(timedelta(hours=9))


# ==============================================================================
# 1. 재사용 가능한 노이즈 섭동 엔진 (RobustnessPerturber)
# ==============================================================================

PRESETS = {
    'default': {
        'description': '실제 현장/대회 평가서버 수준의 종합 현실적 노이즈',
        'typo_prob': 0.365,
        'ton_shift_prob': 0.370,
        'ton_shift_range': (0.03, 0.05),
        'ton_missing_prob': 0.070,
        'flag_missing_prob': 0.170,
        'hangul_meta_noise': True,
        'hangul_spacing_prob': 0.150,
    },
    'mild': {
        'description': '경미한 측정 오차 수준의 가벼운 노이즈',
        'typo_prob': 0.150,
        'ton_shift_prob': 0.200,
        'ton_shift_range': (0.01, 0.03),
        'ton_missing_prob': 0.030,
        'flag_missing_prob': 0.080,
        'hangul_meta_noise': True,
        'hangul_spacing_prob': 0.080,
    },
    'severe': {
        'description': '심각한 데이터 결측 및 왜곡 스트레스 테스트',
        'typo_prob': 0.500,
        'ton_shift_prob': 0.500,
        'ton_shift_range': (0.05, 0.12),
        'ton_missing_prob': 0.200,
        'flag_missing_prob': 0.300,
        'hangul_meta_noise': True,
        'hangul_spacing_prob': 0.300,
    },
    'tonnage_only': {
        'description': '오직 톤수(계측 오차 및 결측) 섭동 집중 스트레스 테스트',
        'typo_prob': 0.0,
        'ton_shift_prob': 0.500,
        'ton_shift_range': (0.03, 0.08),
        'ton_missing_prob': 0.200,
        'flag_missing_prob': 0.0,
        'hangul_meta_noise': True,
        'hangul_spacing_prob': 0.0,
    },
    'name_only': {
        'description': '오직 선박명 오타 및 띄어쓰기 변형 집중 스트레스 테스트',
        'typo_prob': 0.500,
        'ton_shift_prob': 0.0,
        'ton_shift_range': (0.0, 0.0),
        'ton_missing_prob': 0.0,
        'flag_missing_prob': 0.0,
        'hangul_meta_noise': False,
        'hangul_spacing_prob': 0.350,
    },
    'missing_only': {
        'description': '오직 톤수 및 국적 정보 결측(NaN) 집중 스트레스 테스트',
        'typo_prob': 0.0,
        'ton_shift_prob': 0.0,
        'ton_shift_range': (0.0, 0.0),
        'ton_missing_prob': 0.350,
        'flag_missing_prob': 0.350,
        'hangul_meta_noise': True,
        'hangul_spacing_prob': 0.0,
    }
}


class RobustnessPerturber:
    """결정론적 해시 기반의 난수를 사용하여 데이터셋에 현실적 노이즈를 주입하는 클래스."""

    def __init__(self, config: Dict[str, Any], seed: int = 20261022):
        self.cfg = config
        self.seed = seed

    def _rng(self, sample_id: str, kind: str) -> random.Random:
        raw = f'{self.seed}|{sample_id}|{kind}'.encode('utf-8')
        return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:16], 'big'))

    def perturb_row(self, row: Dict[str, Any], catalog_gt: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """단일 행에 대해 설정된 확률에 따라 독립적으로 노이즈를 적용합니다."""
        out = dict(row)
        sid = str(row.get('sample_id', 'UNKNOWN'))
        vessel_name = str(out.get('vessel_name', ''))
        is_ko = bool(re.search(r'[가-힣]', vessel_name))

        audit = {
            'sample_id': sid,
            'is_hangul': is_ko,
            'name_typo_applied': False,
            'hangul_spacing_applied': False,
            'ton_shift_applied': False,
            'ton_missing_applied': False,
            'flag_missing_applied': False,
        }

        # 1. 영문 선박명 오타 (Delete, Duplicate, Transpose)
        if not is_ko and self.cfg.get('typo_prob', 0) > 0:
            if self._rng(sid, 'typo').random() < self.cfg['typo_prob']:
                rng_ops = self._rng(sid, 'typo_ops')
                name = vessel_name
                matches = list(re.finditer(r'[A-Za-z]{6,}', name))
                if matches:
                    match = rng_ops.choice(matches)
                    word = match.group()
                    op = rng_ops.choice(['delete', 'duplicate', 'transpose'])
                    if op == 'transpose':
                        positions = [i for i in range(2, len(word) - 3) if word[i] != word[i + 1]]
                        if positions:
                            pos = rng_ops.choice(positions)
                            changed = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
                            name = name[:match.start()] + changed + name[match.end():]
                    else:
                        pos = rng_ops.randrange(2, len(word) - 2)
                        changed = word[:pos] + (word[pos] * 2 if op == 'duplicate' else '') + word[pos + 1:]
                        name = name[:match.start()] + changed + name[match.end():]
                    if name != vessel_name:
                        out['vessel_name'] = name
                        audit['name_typo_applied'] = True

        # 2. 한글 선박명 띄어쓰기 변형 (공백 제거 또는 임의 분리)
        if is_ko and self.cfg.get('hangul_spacing_prob', 0) > 0:
            if self._rng(sid, 'hangul_space').random() < self.cfg['hangul_spacing_prob']:
                rng_space = self._rng(sid, 'hangul_space_ops')
                name = vessel_name
                if ' ' in name:
                    # 공백 제거
                    name = re.sub(r'\s+', '', name)
                elif len(name) >= 4:
                    # 중간 공백 임의 삽입 (예: 동진보이저 -> 동진 보이저)
                    split_idx = rng_space.randint(2, len(name) - 2)
                    name = name[:split_idx] + ' ' + name[split_idx:]
                if name != vessel_name:
                    out['vessel_name'] = name
                    audit['hangul_spacing_applied'] = True

        # 3. 톤수 섭동 (한글 선박도 hangul_meta_noise 설정 시 적용)
        allow_meta = (not is_ko) or self.cfg.get('hangul_meta_noise', True)

        if allow_meta:
            # 톤수 결측 검사
            if self.cfg.get('ton_missing_prob', 0) > 0:
                if self._rng(sid, 'ton_missing').random() < self.cfg['ton_missing_prob']:
                    out['gross_tonnage'] = ''
                    out['international_gross_tonnage'] = ''
                    audit['ton_missing_applied'] = True

            # 톤수 결측이 아닌 경우 톤수 오차(shift) 주입
            if not audit['ton_missing_applied'] and self.cfg.get('ton_shift_prob', 0) > 0:
                if self._rng(sid, 'ton_shift').random() < self.cfg['ton_shift_prob']:
                    rng_shift = self._rng(sid, 'ton_shift_val')
                    min_s, max_s = self.cfg.get('ton_shift_range', (0.03, 0.05))
                    factor = 1.0 + rng_shift.choice([-1.0, 1.0]) * rng_shift.uniform(min_s, max_s)

                    base_ton = catalog_gt
                    if base_ton is None or base_ton <= 0:
                        try:
                            val = float(str(row.get('gross_tonnage', '')).strip() or 0)
                            if val > 0:
                                base_ton = val
                        except (ValueError, TypeError):
                            base_ton = None

                    if base_ton is not None and base_ton > 0:
                        shifted_val = round(base_ton * factor, 2)
                        out['gross_tonnage'] = str(shifted_val)
                        out['international_gross_tonnage'] = str(shifted_val)
                        audit['ton_shift_applied'] = True

            # 국적 결측 검사
            if self.cfg.get('flag_missing_prob', 0) > 0:
                if self._rng(sid, 'flag_missing').random() < self.cfg['flag_missing_prob']:
                    out['flag_name'] = ''
                    audit['flag_missing_applied'] = True

        return out, audit

    def perturb_dataset(self, df: pd.DataFrame, catalog: Optional[pd.DataFrame] = None, return_row_audits: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """전체 DataFrame에 노이즈를 일괄 적용하고 통계 감사 리포트를 반환합니다."""
        cat_dict = {}
        if catalog is not None and 'vessel_id' in catalog.columns and 'gross_tonnage' in catalog.columns:
            for _, r in catalog.iterrows():
                try:
                    gt_val = float(r['gross_tonnage']) if r['gross_tonnage'] != '' else -1.0
                except (ValueError, TypeError):
                    gt_val = -1.0
                cat_dict[str(r['vessel_id'])] = gt_val

        records = df.to_dict('records')
        noisy_records = []
        audits = []

        for r in records:
            tgt = str(r.get('target_vessel_id', ''))
            c_gt = cat_dict.get(tgt)
            n_row, audit = self.perturb_row(r, catalog_gt=c_gt)
            noisy_records.append(n_row)
            audits.append(audit)

        noisy_df = pd.DataFrame(noisy_records)
        audit_df = pd.DataFrame(audits)

        n_total = len(df)
        summary = {
            'total_rows': n_total,
            'seed': self.seed,
            'typo_applied_count': int(audit_df['name_typo_applied'].sum()),
            'typo_applied_rate': float(audit_df['name_typo_applied'].mean()),
            'hangul_spacing_applied_count': int(audit_df['hangul_spacing_applied'].sum()),
            'hangul_spacing_applied_rate': float(audit_df['hangul_spacing_applied'].mean()),
            'ton_shift_applied_count': int(audit_df['ton_shift_applied'].sum()),
            'ton_shift_applied_rate': float(audit_df['ton_shift_applied'].mean()),
            'ton_missing_applied_count': int(audit_df['ton_missing_applied'].sum()),
            'ton_missing_applied_rate': float(audit_df['ton_missing_applied'].mean()),
            'flag_missing_applied_count': int(audit_df['flag_missing_applied'].sum()),
            'flag_missing_applied_rate': float(audit_df['flag_missing_applied'].mean()),
            'any_noise_applied_count': int((
                audit_df['name_typo_applied'] |
                audit_df['hangul_spacing_applied'] |
                audit_df['ton_shift_applied'] |
                audit_df['ton_missing_applied'] |
                audit_df['flag_missing_applied']
            ).sum()),
        }
        summary['any_noise_applied_rate'] = summary['any_noise_applied_count'] / max(1, n_total)

        if return_row_audits:
            return noisy_df, summary, audit_df
        return noisy_df, summary


# ==============================================================================
# 2. 공식 산식 평가 및 강건성 비교 함수
# ==============================================================================

def compute_metrics(preds: List[str], gold: List[str], is_ko: np.ndarray) -> Dict[str, Any]:
    preds_arr = np.asarray(preds, dtype=object)
    gold_arr = np.asarray(gold, dtype=object)
    correct = (preds_arr == gold_arr)

    ko_acc = float(correct[is_ko].mean()) if is_ko.sum() > 0 else 0.0
    latin_acc = float(correct[~is_ko].mean()) if (~is_ko).sum() > 0 else 0.0
    weighted = 0.7 * ko_acc + 0.3 * latin_acc

    return {
        'weighted': weighted,
        'hangul': ko_acc,
        'hangul_correct': int(correct[is_ko].sum()),
        'hangul_rows': int(is_ko.sum()),
        'non_hangul': latin_acc,
        'non_hangul_correct': int(correct[~is_ko].sum()),
        'non_hangul_rows': int((~is_ko).sum()),
        'total_correct': int(correct.sum()),
        'total_rows': len(correct)
    }


def evaluate_matcher_predictions(matcher: Matcher, df: pd.DataFrame) -> List[str]:
    return matcher.predict(df, use_direct=False)


# ==============================================================================
# 3. CLI 엔트리포인트 및 보고서 생성기
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="재사용 가능한 모델 강건성 및 노이즈 스트레스 테스트 도구")
    parser.add_argument('--input', type=Path, default=ROOT / 'data/train.csv', help='입력 데이터셋 (train.csv)')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'data/vessel_catalog.csv', help='카탈로그 데이터셋')
    parser.add_argument('--model', type=Path, default=None, help='테스트할 표준 Matcher 모델 디렉토리 (예: model)')
    parser.add_argument('--set-model', type=Path, default=None, help='테스트할 SetDecisionModel 디렉토리 (예: artifacts/AS-SET-09/fold_0/set)')
    parser.add_argument('--sample-limit', type=int, default=None, help='빠른 진단용 표본 수 제한 (예: 500)')
    parser.add_argument('--preset', type=str, default='default', choices=list(PRESETS.keys()), help='노이즈 프리셋')
    parser.add_argument('--seed', type=int, default=20261022, help='노이즈 생성 시드 (학습 시드와 분리 권장)')
    parser.add_argument('--generate-only', action='store_true', help='추론 없이 노이즈 CSV만 생성')
    parser.add_argument('--output', type=Path, default=None, help='생성할 노이즈 CSV 경로')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'reports/robustness', help='평가 리포트 저장 폴더')

    args = parser.parse_args()

    # 1. 프리셋 및 설정 로드
    cfg = deepcopy(PRESETS[args.preset])
    perturber = RobustnessPerturber(cfg, seed=args.seed)

    print("=" * 75)
    print(f"   [Robustness Stress Test] 프리셋: '{args.preset}' (시드: {args.seed})")
    print(f"   설명: {cfg['description']}")
    print("=" * 75)

    # 2. 데이터 로드
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    cat = pd.read_csv(args.catalog, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if args.sample_limit and args.sample_limit < len(df):
        df = df.head(args.sample_limit).copy()
        print(f"[Sample Limit] 상위 {args.sample_limit:,}행으로 평가 표본 제한 적용.")
    print(f"데이터 로드 완료: {len(df):,}개 쿼리 행, {len(cat):,}개 카탈로그 선박")

    # 3. 노이즈 데이터 생성
    t0 = time.monotonic()
    noisy_df, audit_summary = perturber.perturb_dataset(df, cat)
    gen_time = time.monotonic() - t0

    print(f"\n노이즈 주입 완료 ({gen_time:.2f}초):")
    print(f" - 전체 변형 행수: {audit_summary['any_noise_applied_count']:,} / {audit_summary['total_rows']:,} ({audit_summary['any_noise_applied_rate']*100:.1f}%)")
    print(f" - 영문명 오타 적용: {audit_summary['typo_applied_count']:,}건 ({audit_summary['typo_applied_rate']*100:.1f}%)")
    print(f" - 한글 띄어쓰기 변형: {audit_summary['hangul_spacing_applied_count']:,}건 ({audit_summary['hangul_spacing_applied_rate']*100:.1f}%)")
    print(f" - 톤수 오차 주입: {audit_summary['ton_shift_applied_count']:,}건 ({audit_summary['ton_shift_applied_rate']*100:.1f}%)")
    print(f" - 톤수 결측 주입: {audit_summary['ton_missing_applied_count']:,}건 ({audit_summary['ton_missing_applied_rate']*100:.1f}%)")
    print(f" - 국적 결측 주입: {audit_summary['flag_missing_applied_count']:,}건 ({audit_summary['flag_missing_applied_rate']*100:.1f}%)")

    # 4. generate-only 옵션인 경우 저장 후 종료
    if args.generate_only:
        out_path = args.output or (args.report_dir / f"train_noisy_{args.preset}_s{args.seed}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        noisy_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n[저장 완료] 노이즈 주입 CSV: {out_path}")
        return

    # 5. 모델 평가 (Clean vs Noisy)
    if args.set_model:
        from decision_models.set_model import SetDecisionModel
        set_dir = args.set_model
        if (set_dir / 'fold_0/set').exists():
            set_dir = set_dir / 'fold_0/set'
        print(f"\n--- SetDecisionModel 로드 및 추론 평가 (대상: {set_dir}) ---")
        set_scorer = SetDecisionModel.load(set_dir, device='cpu')
        assets_file = ROOT / 'model/assets.json'
        matcher = DecisionMatcher(cat, assets_file, set_scorer)
        target_name = f"SetModel_{set_dir.parent.name}"
    else:
        model_dir = args.model or (ROOT / 'model')
        if not (model_dir / 'assets.json').exists():
            print(f"\n[안내] 유효한 model 디렉토리가 지정되지 않았습니다: {model_dir}")
            print("  `--generate-only` 로 노이즈 데이터만 생성하거나, `--model <경로>`를 지정하십시오.")
            return
        print(f"\n--- Matcher 로드 및 추론 평가 (대상: {model_dir}) ---")
        matcher = Matcher(cat, model_dir, use_train_lookup=False)
        target_name = f"Matcher_{model_dir.name}"

    gold = df['target_vessel_id'].tolist()
    is_ko = df['vessel_name'].map(hangul).values

    print("1) Clean 원본 데이터 추론 중...")
    t_clean = time.monotonic()
    clean_preds = evaluate_matcher_predictions(matcher, df)
    clean_metrics = compute_metrics(clean_preds, gold, is_ko)
    clean_dur = time.monotonic() - t_clean

    print("2) Noisy 스트레스 데이터 추론 중...")
    t_noisy = time.monotonic()
    noisy_preds = evaluate_matcher_predictions(matcher, noisy_df)
    noisy_metrics = compute_metrics(noisy_preds, gold, is_ko)
    noisy_dur = time.monotonic() - t_noisy

    # 6. 강건성 결과 집계
    w_clean = clean_metrics['weighted']
    w_noisy = noisy_metrics['weighted']
    delta_w = w_noisy - w_clean
    retention_w = (w_noisy / max(1e-6, w_clean)) * 100.0

    ko_clean = clean_metrics['hangul']
    ko_noisy = noisy_metrics['hangul']
    delta_ko = (ko_noisy - ko_clean) * 100.0

    latin_clean = clean_metrics['non_hangul']
    latin_noisy = noisy_metrics['non_hangul']
    delta_latin = (latin_noisy - latin_clean) * 100.0

    print("\n" + "=" * 75)
    print("                     [강건성 스트레스 테스트 결과]")
    print("=" * 75)
    print(f"{'평가 지표':<18} | {'Clean (원본)':<14} | {'Noisy (스트레스)':<16} | {'변화량 (Δ)':<12} | {'방어율'}")
    print("-" * 75)
    print(f"{'공식 가중 정확도':<16} | {w_clean*100:>12.3f}% | {w_noisy*100:>14.3f}% | {delta_w*100:>+10.3f}%p | {retention_w:>6.2f}%")
    print(f"{'한글 정확도 (70%)':<16} | {ko_clean*100:>12.3f}% | {ko_noisy*100:>14.3f}% | {delta_ko:>+10.3f}%p | {(ko_noisy/max(1e-6, ko_clean))*100:>6.2f}%")
    print(f"{'비한글 정확도 (30%)':<14} | {latin_clean*100:>12.3f}% | {latin_noisy*100:>14.3f}% | {delta_latin:>+10.3f}%p | {(latin_noisy/max(1e-6, latin_clean))*100:>6.2f}%")
    print("-" * 75)
    print(f"오답 수 비교: Clean {clean_metrics['total_rows'] - clean_metrics['total_correct']}행 -> Noisy {noisy_metrics['total_rows'] - noisy_metrics['total_correct']}행 (+{noisy_metrics['total_rows'] - noisy_metrics['total_correct'] - (clean_metrics['total_rows'] - clean_metrics['total_correct'])}행 증가)")
    print("=" * 75)

    # 7. 보고서 저장
    report_data = {
        'timestamp_kst': datetime.now(KST).isoformat(),
        'model_dir': str(model_dir),
        'preset': args.preset,
        'seed': args.seed,
        'config': cfg,
        'audit_summary': audit_summary,
        'clean_metrics': clean_metrics,
        'noisy_metrics': noisy_metrics,
        'retention_rate_pct': retention_w,
        'delta_weighted_pct': delta_w * 100.0,
        'delta_hangul_pct': delta_ko,
        'delta_non_hangul_pct': delta_latin
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_file = args.report_dir / f"robustness_{model_dir.name}_{args.preset}_s{args.seed}.json"
    report_file.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n[보고서 저장 완료] {report_file}")


if __name__ == '__main__':
    main()
