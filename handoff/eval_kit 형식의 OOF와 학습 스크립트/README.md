# eval_kit 형식의 OOF 및 학습 스크립트 (10/21 인계 6절 AS-3)

이 디렉터리는 인계 문서(10-21_인계.md) 6절 AS의 **3번 작업("eval_kit 형식의 OOF와 학습 스크립트 정리")**에 관련된 필수 파일만을 선별하여 구성한 전용 공유 패키지입니다.

---

## 1. 포함된 파일 목록 (관련 파일 5개)

| 파일명 | 파일 크기 | 설명 |
| :--- | :---: | :--- |
| **`oof_AS-02.csv`** | 335 KB | `eval_kit` 공식 규격 12,222행 OOF 예측 파일 (`sample_id,pred_vessel_id,fold`, SHA256: `4834ffa7...`) |
| **`oof_AR02-reproduced.csv`** | 335 KB | 1:1 비교 기준인 AR-02 재현 12,222행 OOF 예측 파일 (SHA256: `69e5c9bd...`) |
| **`leaderboard_AS.csv`** | 954 B | `eval_kit` 공식 리더보드 규격 CSV 요약 한 줄 기록 |
| **`train_oof.py`** | 3.8 KB | G5-s42 5-Fold 학습 및 OOF 생성을 단 한 번에 재현하는 독립 스크립트 |
| **`script.py`** | 34 KB | AS-02 모델 및 매칭 엔진 (한글 행 국적·선종 점수 OFF / 비한글 불변) |

---

## 2. eval_kit 공식 검증 결과 요약

```text
evaluate.py 1.0 — OOF oof_AS-02.csv (12,222행, sha 4834ffa7)
------------------------------------------------------------------------
한글      97.481%  [95.62, 98.96]       1,191행    382척
비한글    99.927%  [99.87, 99.97]      11,031행  5,471척
가중      0.98215  [0.96907, 0.99252]   (0.7한글 + 0.3비한글)
오답    한글 30행 · 비한글 8행
------------------------------------------------------------------------
짝지은 비교: Δ = oof_AS-02.csv - oof_AR02-reproduced.csv (sha 69e5c9bd)
  한글          -0.08%p  [-1.10, 0.92]           갈린 행 13 (고침 6 / 망침 7)
  비한글         +0.00%p  [0.00, 0.00]            갈린 행 0 (고침 0 / 망침 0)
  가중         -0.00059  [-0.00770, 0.00644]     판정: 동등
```

- **한글 퇴보율**: **-0.08%p** (인계서의 1%p 퇴보 제한 기준 안전하게 통과)
- **비한글 예측**: AR-02 대비 **100% 동일 (0행 변경)** 보존
- **판정**: 95% 신뢰구간이 0을 포함하므로 통계적 **동등**

---

## 3. 재현 및 검증 명령어

### 1) eval_kit 공식 평가 실행
```bash
# AS-02 OOF 단독 평가
python eval_kit/evaluate.py --oof handoff/eval_kit_oof_and_train_scripts/oof_AS-02.csv --train data/train.csv

# AR-02 기준 대비 짝지은 비교 평가
python eval_kit/evaluate.py --oof handoff/eval_kit_oof_and_train_scripts/oof_AS-02.csv --train data/train.csv --compare handoff/eval_kit_oof_and_train_scripts/oof_AR02-reproduced.csv
```

### 2) 5-Fold 학습 및 OOF 재생성
```bash
# 5개 Fold 순차 학습, OOF 생성 및 자동 평가 수행
python handoff/eval_kit_oof_and_train_scripts/train_oof.py
```

---

## 4. 단톡 보고 문구

```text
AS-02 | 미제출(OOF 검증완료) | AR-02 | 한글 국적·선종 점수 OFF (아이디어 D) | 비한글 변경 없음 | G5 한글 22행 변경(개선 6 / 악화 7, 판정: 동등)
```
