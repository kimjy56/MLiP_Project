"""Evaluate the final JY-04 model on all 12,222 rows and compare with JY-03."""
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
    print(" [JY-04 최종 모델 전체 데이터 성능 평가]")
    print("=" * 70)

    train = pd.read_csv(ROOT / 'data/train.csv')
    catalog = pd.read_csv(ROOT / 'data/vessel_catalog.csv')

    ko_mask = train.vessel_name.fillna('').apply(hangul).to_numpy()
    golds = train.target_vessel_id.to_numpy()

    print(f"평가 데이터: 총 {len(train)}행 (한글 {ko_mask.sum()}행, 비한글 {(~ko_mask).sum()}행)")
    
    t0 = time.monotonic()
    matcher = Matcher(catalog, ROOT / 'model')
    preds, candidate_ids, direct_flags = matcher.predict_details(train)
    elapsed = time.monotonic() - t0

    preds = np.asarray(preds)
    correct = (preds == golds)

    ko_acc = float(correct[ko_mask].mean())
    latin_acc = float(correct[~ko_mask].mean())
    weighted_acc = 0.7 * ko_acc + 0.3 * latin_acc

    ko_correct_cnt = int(correct[ko_mask].sum())
    ko_total_cnt = int(ko_mask.sum())
    latin_correct_cnt = int(correct[~ko_mask].sum())
    latin_total_cnt = int((~ko_mask).sum())
    total_correct = int(correct.sum())

    # Candidate recall
    recall = np.array([g in cands for g, cands in zip(golds, candidate_ids)])
    ko_recall = float(recall[ko_mask].mean())
    latin_recall = float(recall[~ko_mask].mean())

    print("\n--- [JY-04 종합 성능 지표] ---")
    print(f"공식 산식 가중 점수: {weighted_acc:.5f} ({weighted_acc*100:.3f}%)")
    print(f"한글 정확도 (가중치 0.7): {ko_acc*100:.2f}% ({ko_correct_cnt} / {ko_total_cnt}행, 오답 {ko_total_cnt - ko_correct_cnt}행)")
    print(f"비한글 정확도 (가중치 0.3): {latin_acc*100:.2f}% ({latin_correct_cnt} / {latin_total_cnt}행, 오답 {latin_total_cnt - latin_correct_cnt}행)")
    print(f"전체 단순 정확도: {total_correct/len(train)*100:.2f}% ({total_correct} / {len(train)}행, 총 오답 {len(train) - total_correct}행)")
    print(f"Top-30 후보 포함률: 한글 {ko_recall*100:.2f}%, 비한글 {latin_recall*100:.2f}%")
    print(f"추론 총 소요시간: {elapsed:.2f}초 (행당 {elapsed/len(train)*1000:.2f} ms)")
    print("=" * 70)

    # Save summary
    res = {
        'model_name': 'JY-04',
        'weighted_score': weighted_acc,
        'korean_accuracy': ko_acc,
        'korean_correct': ko_correct_cnt,
        'korean_total': ko_total_cnt,
        'korean_recall': ko_recall,
        'latin_accuracy': latin_acc,
        'latin_correct': latin_correct_cnt,
        'latin_total': latin_total_cnt,
        'latin_recall': latin_recall,
        'total_rows': len(train),
        'elapsed_seconds': elapsed,
        'ms_per_row': elapsed / len(train) * 1000,
    }
    import json
    (ROOT / 'reports/jy04_evaluation_summary.json').write_text(json.dumps(res, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
