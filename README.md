# 선박 정보 매칭 베이스라인

공식 `train.csv`와 `vessel_catalog.csv`만 사용한 Python 제출 프로젝트입니다.
실제 `test.csv`는 제공되지 않았으며, 테스트 예측 파일은 평가 서버에서 생성합니다.
`sample_submission.csv`의 3,000행을 테스트 전체 크기로 가정하지 않습니다.

이 저장소에는 코드, 분석 리포트, 집계 통계와 그래프를 포함합니다. 원본 데이터,
학습 가중치·매핑 사전, 제출 ZIP, 행 단위 예측 파일은 Git에서 제외합니다.
재현하려면 공식 `open.zip`의 CSV를 `data/`에 배치한 뒤 아래 학습 명령을 실행합니다.
제출 ZIP은 학습 후 생성되는 산출물입니다.

최신 종합 분석은 `analysis/REPORT_8_SECTIONS.md`, 재현 코드는
`analysis/eda.py`, `analysis/comprehensive_analysis.py`,
`analysis/pair_pipeline.py`에 있습니다. 과거 실험 보고서의 수치는 저장 당시의
결과이며, 실행 시간·환경 경로는 현재 체크아웃의 환경과 다를 수 있습니다.

## 구현

1. 공식 학습 데이터에서 모호하지 않은 `call_sign → target_vessel_id` 매핑을 만듭니다.
   입력 행의 호출부호가 매핑에 있으면 해당 ID를 선택합니다. 익명화 식별정보를
   복원하거나 외부 선박 DB와 대조하지 않습니다.
2. 그 외에는 정규화 이름·로마자 표기·간단한 발음 유사 키·톤수로 후보를 검색합니다.
   한글 로마자 변환은 서버에 제공된 범용 Unidecode 라이브러리를 사용합니다.
   학습 데이터의 별칭도 후보에 추가합니다.
3. 이름 유사도, 숫자 일치, 톤수 상대 오차와 결측, 학습 데이터에서 추정한
   선종·국적 대응 확률 등 23개 특징으로 LightGBM 이진 분류 모델이 후보를 평가합니다.
   점수가 가장 높은 후보를 선택하며, 동점은 고정된 카탈로그 ID 정렬 순서로 처리합니다.

호출부호·별칭·선종·국적 매핑은 학습 시에만 생성합니다. 추론에서는 테스트 행의
집계·빈도·순위·다수결·누적 별칭·분포 기반 보정을 사용하지 않습니다.
`sample_id`와 `arrival_year`는 예측 특징으로 사용하지 않습니다.

## 검증 설계

- 시드: `20261016`.
- 선박 ID의 20%를 그룹 분할로 분리하여 처음 보는 선박용 검증 세트로 둡니다.
- 나머지 기록의 20%를 한글 여부로 층화 추출하여 행 단위 검증 세트로 둡니다.
- 남은 기록만으로 검증 모델과 모든 학습 사전을 만듭니다.
- 두 검증 세트 모두 학습에 사용하지 않습니다. 행 단위 검증에는 기존 선박과
  새 선박이 섞이므로 결과에 학습 선박 비율을 함께 기록합니다.
- 한글 여부는 제공된 공식 베이스라인과 동일하게 `[가-힣]` 포함 여부로 판정합니다.
- 지표는 `0.7 × 한글 정확도 + 0.3 × 비한글 정확도`입니다.
- 검증 후 제출 모델은 공개 학습 데이터 전체로 다시 학습합니다.

검증 수치는 `reports/validation.json`, 각 기록의 분할은 `reports/split.csv`에 있습니다.
오류 분석용 예측 CSV는 공개 train의 검증 기록만 포함합니다.
이 수치는 로컬 검증 결과이며 공식 Public/Private 점수가 아닙니다.

## 파일

- `submit.zip`: 평가 서버에 제출할 파일. 최상위는 `script.py`, `requirements.txt`, `model/`입니다.
- `script.py`: `data/test.csv`, `data/vessel_catalog.csv`를 읽고 `output/submission.csv`를 생성합니다.
- `model/matcher.py`, `model/ranker.txt`, `model/assets.json`: 추론 구현·모델·고정 학습 자산.
- `train.py`: 검증 및 전체 데이터 재학습.
- `verify_submission.py`: 제출 ZIP 생성과 독립 실행 검사.
- `reports/`: 데이터 분석, 로컬 검증, 제출 검사 결과.
- `official_baseline/`: 필요 시 원래 업로드 ZIP에서 별도로 추출하는 공식 베이스라인 원본. Git에 포함하지 않으며 공식 pickle 모델은 실행하지 않았습니다.

## 재현

Python 3.11의 별도 가상환경에서 다음 명령을 실행합니다.

```bash
python -m pip install -r requirements-local.txt
python train.py
python verify_submission.py
```

이미 생성된 제출 모델을 실행할 때는 다음과 같습니다.

```bash
python script.py
```

학습 없이 ZIP 검사만 하려면 `python verify_submission.py`를 실행합니다.
검증만 하려면 `python train.py --validate-only`, 전체 재학습만 하려면
`python train.py --final-only`를 실행합니다.

평가 서버용 `requirements.txt`는 기본 설치 패키지를 재설치하지 않도록 주석만 포함합니다.
실제 로컬 검증은 Python 3.12와 일부 다른 numpy/pandas/scikit-learn 버전에서 수행했으므로,
평가 서버와 완전히 동일한 환경에서 실행했다고 주장하지 않습니다.
LightGBM 가중치는 네이티브 텍스트, 사전은 JSON으로 저장하여 Python pickle 버전 의존을 피했습니다.

## 제출 검사와 한계

깨끗한 임시 디렉터리에 ZIP을 풀어 네트워크 연결을 차단한 상태에서 진입점을 실행합니다.
공개 학습 기록 일부의 호출부호를 미등록 값으로 대체하여 후보 검색 경로도 검사합니다.
입력 파일 보존, 출력 컬럼·순서·카탈로그 ID, 전체·부분·단독·순서 변경 입력의
예측 일치 및 빈 입력 처리를 확인합니다. 결과는 `reports/submission_checks.json`에 있습니다.

실제 비공개 테스트, 평가 서버 실행 시간, 공식 점수는 아직 확인하지 않았습니다.
발음 키는 단순 규칙이므로 한글 음역의 예외와 약칭에 한계가 있으며,
호출부호가 학습에 없는 선박의 성능은 별도 그룹 검증 결과로 판단해야 합니다.
새 모델의 검증 결과를 기존 결과와 비교할 때 동일한 분할을 유지하십시오.
