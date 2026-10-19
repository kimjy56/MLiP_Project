# eval_kit — 팀 공통 평가 묶음 (DACON 236771)

버전 1.0 · 2026-10-19 배포 · 팀 병합 후 팀 안에서만 공유 · 규칙 전체는 `TEAM_RULES.md`

이 묶음 하나로 세 사람이 **같은 fold, 같은 산식**으로 점수를 잰다. 점수 비교는 이 묶음으로 잰 것만 인정한다.

## 1. 들어 있는 것

| 파일 | 내용 |
|---|---|
| `evaluate.py` | 공통 평가 함수. 고치지 않는다(버그가 의심되면 리더에게) |
| `selftest_evaluate.py` | evaluate.py 자체 시험 33개. 처음 한 번 돌려 환경을 확인한다 |
| `check_submission.py` | 제출 zip 점검(구조·패키지·실행·행 독립·시간). 올리기 전에 각자, 제출 전에 리더가 돌린다(§5) |
| `folds_group5_seed42.csv` | **기준 fold.** 선박 그룹 5-fold(처음 보는 선박 시험). 모든 실험은 이것으로 |
| `oof_E00_baseline.csv` | 기준점 OOF(주최 baseline을 위 fold로 다시 학습). evaluate.py가 0.47750을 내는지 확인용 |
| `aux/folds_group5_seed{0,1,2}.csv` | 보조 시드 fold(같은 생성 방식). 최종 후보 재확인 때만 |
| `aux/folds_row5_seed42.csv` | 행 분할 5-fold(아는 선박이 섞인 시험, 검증 행의 71.2%가 아는 선박). 사전 조회 효과를 볼 때만. **기준 점수와 합치거나 맞비교하지 않는다** |

fold 파일 형식: `sample_id, target_vessel_id, fold`(0~4). 기준 fold는 선박 단위로 무작위 배정했고 한글 선박(382척)을 다섯 묶음에 고르게 나눴다(묶음당 약 2,444행, 한글 233~244행).

## 2. 처음 한 번 (5분)

```bash
# 1) 자체 시험: 마지막 줄이 "전부 통과"여야 한다
python selftest_evaluate.py --train <train.csv 경로>

# 2) 기준점 재현: 가중 0.47750, 한글 26.196%, 비한글 98.042%가 나와야 한다
python evaluate.py --oof oof_E00_baseline.csv --train <train.csv 경로>
```

- 필요한 패키지는 numpy와 pandas뿐이다(pandas 2.0.3 ~ 3.x, Python 3.9 이상). 제출 코드는 서버와 같은 Python 3.11에서 확인해야 하므로 3.11 환경을 권장한다(§7).
- 출력 맨 위에 `원본 해시 일치`가 떠야 한다. `원본과 다름(!)`이면 train.csv를 DACON에서 다시 받는다. **CSV를 엑셀로 열고 저장하면 바뀐다**(인코딩·숫자 표기).
- Windows 터미널에서 한글이 깨지면 `python -X utf8 evaluate.py ...`로 실행한다.

## 3. OOF 만들기

OOF = 각 행을 "그 행의 선박을 학습에 쓰지 않은 모델"로 예측한 결과 12,222행이다. 기준 fold 파일을 **읽어서** 쓴다.

```python
import pandas as pd

folds = pd.read_csv("folds_group5_seed42.csv", dtype={"sample_id": str, "fold": int})
train = pd.read_csv("train.csv", encoding="utf-8-sig").merge(folds[["sample_id", "fold"]], on="sample_id")

parts = []
for k in range(5):
    fit, val = train[train["fold"] != k], train[train["fold"] == k]
    model = my_fit(fit, catalog)                         # 사전·대응표·확률·튜닝은 fit에서만 만든다
    pred, gold_rank = my_predict(model, val, catalog)    # 예측에는 val의 정답을 쓰지 않는다
    parts.append(pd.DataFrame({"sample_id": val["sample_id"], "fold": k,
                               "pred_vessel_id": pred, "gold_rank": gold_rank}))
pd.concat(parts).to_csv("oof_O-01.csv", index=False)
```

| 컬럼 | 필수 | 뜻 |
|---|---|---|
| `sample_id` | 필수 | train 12,222행 전부, 중복 없이 (행 순서는 상관없다) |
| `pred_vessel_id` | 필수 | 예측한 `vessel_id`. 글자 그대로 비교한다(앞뒤 공백도 오답) |
| `fold` | 권장 | 그 행을 예측한 모델의 검증 fold 번호. evaluate.py가 기준 fold와 대조해 다르면 오류를 낸다 |
| `gold_rank` | 선택 | 정답 선박이 후보 안에서 몇 위였는지(1 = 1위, 0 = 후보에 없음). 사전 조회로 맞힌 행은 1. 넣으면 검색 recall과 오답 분해(검색 누락 / 순위 오류)가 나온다 |

주의: **fold를 직접 다시 나누지 않는다.** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`처럼 "seed 42"를 맞춰도 다른 코드로 나누면 다른 fold다(재확인: 기준 파일과 같은 묶음에 든 선박 20.7%, 무관한 배정 수준). fold가 다르면 점수를 비교할 수 없다.

## 4. 평가하기

```bash
# 기본: 한글·비한글·가중 + 선박 단위 95% 신뢰구간 + fold별 + (gold_rank 있으면) recall
python evaluate.py --oof oof_O-02.csv --train <train.csv>

# 짝지은 비교: Δ = oof_O-02 - oof_O-01, 판정(개선/동등/악화)
python evaluate.py --oof oof_O-02.csv --train <train.csv> --compare oof_O-01.csv

# 기록: 요약 줄을 leaderboard.csv에 덧붙인다(없으면 머리줄과 함께 만든다)
python evaluate.py --oof oof_O-02.csv --train <train.csv> --compare oof_O-01.csv \
    --id O-02 --note "숫자 표기 정규화" --train-sec 95 --infer-sec 41 --zip-mb 3.2 --leaderboard leaderboard_O.csv

# 구간별 정확도(톤수·국적 결측, 선박당 행 수, 도착연도)
python evaluate.py --oof oof_O-02.csv --train <train.csv> --slices

# 보조 시험: 행 분할(아는 선박) — 기준 점수와 섞지 않는다
python evaluate.py --oof oof_O-02_row.csv --train <train.csv> --folds aux/folds_row5_seed42.csv

# 변형 시험처럼 일부 행만 예측한 OOF
python evaluate.py --oof oof_O-02_numvar.csv --train <train.csv> --subset --compare oof_O-02.csv
```

출력 읽는 법
- **가중** = 0.7 × 한글 + 0.3 × 비한글(대회 산식). 대괄호는 선박 단위 부트스트랩 95% 신뢰구간이다(한글 행·비한글 행을 따로 선박 단위로 다시 뽑음, B=2000, seed 0).
- **판정**(TEAM_RULES §4): Δ가중의 95% CI 하한 > 0이면 **개선**, 상한 < 0이면 **악화**, 0을 포함하면 **동등**(그때는 더 빠르고 작은 쪽). 한글은 382척뿐이라 한글 1~2%p 차이는 개선이 아니다.
- **갈린 행**: 두 OOF에서 정오가 다른 행 수(고침/망침). 상위권에서는 이 숫자가 몇 행이다.
- 맨 아래 **요약 줄**이 leaderboard.csv 한 줄이다. 공유 폴더에서 동기화 충돌을 피하려면 각자 `leaderboard_<접두>.csv`에 쌓고 리더가 합친다.
- 종료 코드 2 = 입력 오류(중복·누락 행, 없는 sample_id, 컬럼 없음, fold 불일치 등). 이때는 점수를 내지 않는다.

## 5. 제출 zip 점검

```bash
python check_submission.py --zip submit_AS-01.zip --train <train.csv 경로>
```

- 서버와 같은 Python 3.11 환경(§7)에서 돌린다. 다른 버전이면 맨 위에 주의가 뜬다.
- 하는 일: ① zip 맨 위에 `script.py`·`requirements.txt`·`model/`만 있는지(폴더째 압축, 역슬래시 경로, 불필요한 파일) ② requirements.txt에 서버에 이미 있는 패키지를 적었는지 ③ 코드에 네트워크·외부 AI·train.csv 읽기·개인 경로·pickle 흔적이 있는지 ④ train에서 뽑은 300행(정답 열 제거)으로 실제 실행해 `output/submission.csv` 확인, data/에 파일을 쓰는지 ⑤ 행 독립: 전체 / 순서 섞기(다른 작업 폴더) / 절반 / 한 행씩 실행의 예측이 같은지(난수 때문에 실행마다 바뀌는 코드도 여기서 잡힌다) ⑥ 3,000행·30,000행 실행 시간 어림.
- 마지막 줄이 `결과: 실패 0 … 올려도 된다`여야 올린다. 점수는 참고용이다(뽑은 행은 학습에 쓴 행이라 높게 나온다).
- zip 안의 코드를 실제로 실행하는 도구다. 남이 준 zip은 script.py를 먼저 훑어보고 돌린다.
- zip은 Python으로 만든다(Windows PowerShell의 `Compress-Archive`는 경로에 역슬래시를 넣어 서버에서 깨질 수 있다).

```python
import zipfile
from pathlib import Path
src = Path("my_submission")            # 안에 script.py, requirements.txt, model/
with zipfile.ZipFile("submit_AS-01.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, p.relative_to(src).as_posix())
```

## 6. 지킬 것

- OOF·leaderboard는 행 단위 결과라 팀 안에서만 주고받는다. 공개 저장소·AI 채팅에 올리지 않는다.
- 사전·대응표·확률·튜닝은 학습 fold에서만 만든다. 검증 fold의 정답으로 만든 값이 하나라도 들어가면 그 OOF는 무효다.
- 점수는 기준 fold(`G5-s42`) OOF로만 순위를 매긴다. 보조 fold·행 분할·변형 시험은 따로 적는다.
- 한 fold나 단일 분할 숫자로 방법을 고르지 않는다(baseline도 fold별 한글이 19.2~34.8%로 흔들린다).

## 7. 설치 안내 (서버와 같은 개발 환경, 권장)

평가 서버: Ubuntu 22.04, Python 3.11.17, pandas 2.0.3, numpy 1.26.4, scipy 1.15.3, scikit-learn 1.5.2, lightgbm 4.6.0, catboost 1.2.8, rapidfuzz 3.14.3, Unidecode 1.4.0, joblib 1.5.2.

```bash
# Windows (Python 3.11이 없으면 python.org에서 3.11.x 설치 후)
py -3.11 -m venv .venv311
.venv311\Scripts\activate
# macOS / Linux
python3.11 -m venv .venv311 && source .venv311/bin/activate

pip install numpy==1.26.4 pandas==2.0.3 scipy==1.15.3 scikit-learn==1.5.2 lightgbm==4.6.0 \
    catboost==1.2.8 rapidfuzz==3.14.3 Unidecode==1.4.0 joblib==1.5.2
python -c "import sys, pandas, numpy, sklearn; print(sys.version, pandas.__version__, numpy.__version__, sklearn.__version__)"
```

torch·transformers는 임베딩·언어모델을 쓰기로 할 때만 설치한다(서버에는 torch 2.7.1+cu128, transformers 4.57.6이 있다). 서버에 있는 패키지는 제출 zip의 `requirements.txt`에 넣지 않는다.

## 8. 변경 이력
- 1.0 (2026-10-19): 최초 배포. 자체 시험 33개 통과(서버와 같은 Python 3.11.9 · pandas 2.0.3 · numpy 1.26.4에서, 그리고 Python 3.14 · pandas 3.0.3 · numpy 2.3.5에서). E00 재현 0.47750 [0.44598, 0.51008]. check_submission.py 1.0 추가(정상 zip 통과, 폴더째 압축·data/ 쓰기·실행마다 다른 예측·작업 폴더 의존을 잡는 것 확인).
