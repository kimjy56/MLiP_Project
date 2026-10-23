"""Run comprehensive comparison test for Korean expansion & toggles on official data."""
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

from matcher import Matcher, hangul

def main():
    print("=" * 70)
    print(" [비교 테스트] 한글 표기 확장(korean_expansion) 효과 정밀 검증")
    print("=" * 70)

    catalog = pd.read_csv(ROOT / 'data/vessel_catalog.csv')
    train = pd.read_csv(ROOT / 'data/train.csv')
    folds = pd.read_csv(ROOT / 'eval_kit/folds_group5_seed42.csv')

    ko_mask = train.vessel_name.fillna('').apply(hangul)
    ko_df = train[ko_mask].copy().reset_index(drop=True)
    latin_sample = train[~ko_mask].sample(n=1000, random_state=42).copy().reset_index(drop=True)

    test_df = pd.concat([ko_df, latin_sample], ignore_index=True)
    print(f"평가 대상: 총 {len(test_df)}행 (한글 전체 {len(ko_df)}행 + 비한글 표본 {len(latin_sample)}행)")

    configs = [
        {
            'name': '기준 판 (Baseline)',
            'korean_expansion': False,
            'hangul_ignore_type_flag': False,
            'relax_tonnage_penalty': False,
        },
        {
            'name': '후보 판 (korean_expansion ON)',
            'korean_expansion': True,
            'hangul_ignore_type_flag': False,
            'relax_tonnage_penalty': False,
        },
        {
            'name': '후보 판 2 (korean_expansion + ignore_type_flag)',
            'korean_expansion': True,
            'hangul_ignore_type_flag': True,
            'relax_tonnage_penalty': False,
        },
    ]

    results = []
    golds = test_df.target_vessel_id.tolist()
    ko_indices = list(range(len(ko_df)))
    latin_indices = list(range(len(ko_df), len(test_df)))

    for cfg in configs:
        print(f"\n>> 실행 중: {cfg['name']} ...", flush=True)
        t0 = time.monotonic()
        matcher = Matcher(
            catalog, ROOT / 'model',
            use_train_lookup=False,
            korean_expansion=cfg['korean_expansion'],
            hangul_ignore_type_flag=cfg['hangul_ignore_type_flag'],
            relax_tonnage_penalty=cfg['relax_tonnage_penalty']
        )
        preds, candidates, direct = matcher.predict_details(test_df)
        elapsed = time.monotonic() - t0

        # Metrics
        ko_correct = sum(preds[i] == golds[i] for i in ko_indices)
        latin_correct = sum(preds[i] == golds[i] for i in latin_indices)
        
        # Candidate recall in Top-30 candidates
        ko_recall = sum(golds[i] in candidates[i] for i in ko_indices)
        latin_recall = sum(golds[i] in candidates[i] for i in latin_indices)

        results.append({
            'name': cfg['name'],
            'ko_acc': ko_correct / len(ko_indices),
            'ko_correct': ko_correct,
            'ko_total': len(ko_indices),
            'ko_recall': ko_recall / len(ko_indices),
            'latin_acc': latin_correct / len(latin_indices),
            'latin_correct': latin_correct,
            'latin_total': len(latin_indices),
            'latin_recall': latin_recall / len(latin_indices),
            'elapsed_sec': elapsed,
            'preds': preds
        })
        print(f"   소요시간: {elapsed:.2f}초 | 한글 정확도: {ko_correct}/{len(ko_indices)} ({ko_correct/len(ko_indices)*100:.2f}%)")

    # Detailed comparison
    base = results[0]
    cand = results[1]

    ko_changed = sum(base['preds'][i] != cand['preds'][i] for i in ko_indices)
    ko_c2w = sum(base['preds'][i] == golds[i] and cand['preds'][i] != golds[i] for i in ko_indices)
    ko_w2c = sum(base['preds'][i] != golds[i] and cand['preds'][i] == golds[i] for i in ko_indices)

    latin_changed = sum(base['preds'][i] != cand['preds'][i] for i in latin_indices)

    print("\n" + "=" * 70)
    print(" [비교 결과 요약: Baseline vs korean_expansion]")
    print("=" * 70)
    print(f"- 한글 전체(1,191건) 정확도: {base['ko_acc']*100:.2f}% -> {cand['ko_acc']*100:.2f}% ({cand['ko_acc'] - base['ko_acc']:+.2f}%p)")
    print(f"- 한글 후보 포함률(Top-30 Recall): {base['ko_recall']*100:.2f}% -> {cand['ko_recall']*100:.2f}% ({cand['ko_recall'] - base['ko_recall']:+.2f}%p)")
    print(f"- 한글 변경 행 수: {ko_changed}건")
    print(f"  * 맞음 -> 틀림 (퇴보): {ko_c2w}건")
    print(f"  * 틀림 -> 맞음 (개선): {ko_w2c}건")
    print(f"- 비한글 변경 행 수: {latin_changed}건 (0건이어야 정상)")
    print(f"- 1,191건 추론 속도: {cand['elapsed_sec']:.2f}초 ({cand['elapsed_sec']/len(test_df)*1000:.2f} ms/행)")
    print("=" * 70)

if __name__ == '__main__':
    main()
