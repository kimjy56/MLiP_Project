"""DACON 236771 Leaderboard Score Decoder.

Metric: 0.7 * (Korean / 532) + 0.3 * (Latin / 3568)
       = Korean / 760 + 3 * Latin / 35680
"""
import sys
import argparse

LATIN_DEFAULT = 3314  # Baseline Latin correct count (team baseline)

def decode(score: float, latin_fixed: int = LATIN_DEFAULT):
    latin_contribution = 3.0 * latin_fixed / 35680.0
    korean_raw = (score - latin_contribution) * 760.0
    korean_round = round(korean_raw)
    korean_acc = korean_round / 532.0 * 100.0
    latin_acc = latin_fixed / 3568.0 * 100.0
    reconstructed_score = (korean_round / 760.0) + (3.0 * latin_fixed / 35680.0)
    return {
        "score": score,
        "latin_fixed": latin_fixed,
        "latin_acc": latin_acc,
        "korean_raw": korean_raw,
        "korean_correct": korean_round,
        "korean_acc": korean_acc,
        "reconstructed_score": reconstructed_score,
        "residual": abs(score - reconstructed_score)
    }

def main():
    parser = argparse.ArgumentParser(description="Decode DACON LB Score into Korean/Latin counts")
    parser.add_argument("score", type=float, help="Official DACON Score (e.g. 0.9010119188)")
    parser.add_argument("--latin", type=int, default=LATIN_DEFAULT, help="Fixed Latin correct count (default: 3314)")
    args = parser.parse_args()

    res = decode(args.score, args.latin)
    print("=" * 55)
    print(f" 입력 점수(Score) : {res['score']:.10f}")
    print(f" 비한글 정답 수  : {res['latin_fixed']} / 3,568 ({res['latin_acc']:.2f}%)")
    print(f" 한글 정답 수    : {res['korean_correct']} / 532 ({res['korean_acc']:.2f}%) [raw: {res['korean_raw']:.4f}]")
    print(f" 복원된 정확 점수: {res['reconstructed_score']:.10f} (오차: {res['residual']:.2e})")
    print("=" * 55)

if __name__ == "__main__":
    main()
