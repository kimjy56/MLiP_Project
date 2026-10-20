# 팀 공유 패키지: 실험 AS-01

- **실험 ID**: `AS-01`
- **기준 fold**: `eval_kit/folds_group5_seed42.csv` (G5-s42, SHA-256: `9d09ede0...`)
- **설정**:
  - `name_mode`: `all` (한글 음차 및 구조어 정규화 적용)
  - `phonetic_search`: `24`
  - `use_train_lookup`: `True`
  - `retrieval_channels`: `('S', 'P', 'T')`
- **검증 환경**: Python 3.11 / LightGBM 4.6.0

---

## 1. 평가 성적 (eval_kit/evaluate.py 1.0 기준)

| 지표 | 수치 | 95% 신뢰구간 (선박 부트스트랩 B=2000) | 오답 수 |
|---|---|---|---|
| **가중 정확도** | **0.97918** | [0.96547, 0.99069] | - |
| **한글 정확도 (0.7)** | **97.061%** | [95.097, 98.706] | 35 / 1,191행 |
| **비한글 정확도 (0.3)** | **99.918%** | [99.864, 99.964] | 9 / 11,031행 |

### Fold별 가중 점수
- Fold 0: 한글 96.7% · 비한글 99.91% · 가중 0.9763
- Fold 1: 한글 98.8% · 비한글 99.87% · 가중 0.9909
- Fold 2: 한글 93.2% · 비한글 100.0% · 가중 0.9521
- Fold 3: 한글 100.0% · 비한글 99.91% · 가중 0.9997
- Fold 4: 한글 96.6% · 비한글 99.91% · 가중 0.9757

---

## 2. 파일 목록 및 SHA-256 해시

- `oof_AS-01.csv` (12,222행)
  - SHA-256: `1eb6beb0eff062bd28eb2836d257f1768c6a2a0291bb3bcf814e60a5ab5ab3eb`
- `leaderboard_AS.csv`
  - 팀 리더가 공통 `leaderboard.csv`에 병합할 수 있는 표준 요약 줄
- `AS-01.json`
  - 코드 커밋 해시, 환경 정보, fold별 진단 메트릭 상세 기록

---

## 3. 재현 및 검증 명령어

### 평가 점수 검증
```bash
python eval_kit/evaluate.py --oof share/AS-01/oof_AS-01.csv --train data/train.csv
```

### 전체 5-fold OOF 재현 실행
```bash
python train_oof.py --experiment-id AS-01 --folds-sha256 9d09ede0f9254949aacf8cf80638a6c24de0ca5516ca3e9db329f55b9476ca51
```
