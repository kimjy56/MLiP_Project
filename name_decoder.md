# 선박 정보 매칭 프로젝트

팀 규칙 v4에 맞춘 조회 설정·고정 5-fold OOF 실행 방법은
[모델 코드 적용 안내](docs/TEAM_RULES_IMPLEMENTATION.md)를 참고하세요.
팀 공통 OOF 생성은 `train_oof.py`, 기존 개발 holdout·최종 학습은 `train.py`입니다.
실제 CSV·배포 fold·평가기를 준비해야 팀 검증을 실행할 수 있습니다.

ByT5로 **관측 선박명 → 카탈로그 선박명**을 학습하는 별도 진입점은
`train_name_decoder.py`입니다. `name_decoder/`는 예측 모듈이며,
실제 학습 루프는 `experiments/train_name_decoder.py`에 있습니다.
학습·단일 이름 예측 명령과 현재 실행 제한은 [선명 해독기 안내](docs/NAME_DECODER.md)를 참고하세요.
## 로컬 폴더 구성

```text
실전기계학습/
├─ train.py, script.py, docs/, ...  # 코드·문서: Git 동기화
├─ data/                          # 각 PC에서 직접 받은 CSV, Git 제외
│  ├─ train.csv
│  └─ vessel_catalog.csv
├─ .venv-server/                  # 각 PC에서 별도 설치, Git 제외
└─ model/
   ├─ *.py                       # 모델 코드: Git 동기화
   ├─ assets.json                # 학습 산출물: Git 제외
   └─ ranker.txt                 # 학습 산출물: Git 제외
```

2026-10-23 확인: 현재 PC의 data/에는 공식 train.csv와 vessel_catalog.csv가 있다.
ByT5 사전학습·학습 완료 가중치는 별도로 준비해야 한다.
.venv-server/는 environment/setup_windows.ps1을 실행하면 실제 환경과 함께 생성된다.
모델 파일 구분은 [model/README.md](model/README.md)를 참고한다.

여러 컴퓨터에서의 코드·데이터 분리와 동기화 절차는
[다중 PC 작업 안내](docs/MULTI_PC_SETUP.md)를 참고하세요.
Windows 설치는 `environment/setup_windows.ps1`을 사용합니다.

팀 운영·공통 검증 기준은 [TEAM_RULES.md](TEAM_RULES.md), 이번 정리 내역은
[팀 규칙 변경 기록](docs/TEAM_RULES_CHANGES.md)에서 확인합니다.
공통 평가 묶음 안내는 [eval_kit/README.md](eval_kit/README.md)에 정리했습니다.
현재는 안내 문서만 있으며 평가기·고정 fold 등은 별도로 받아야 합니다.

로컬 Python·패키지는 평가 서버 버전에 맞춘 `.venv-server`를 사용합니다.
활성화·설치·호환성 검사는 [환경 안내](environment/README.md)를 참고하세요.
제출 ZIP은 사용자가 요청할 때만 생성·갱신합니다.

프로젝트 진행 기록은 [project_log/README.md](project_log/README.md)에서 확인합니다.
수업 평가 기준, 실험대장, 선택 이유, 대표 오답, 보고서·발표 구성과 기록 양식을 관리합니다.

## 최신 선박명 개선 결과

문자 n-gram TF-IDF 검색을 구현하고, 문자열(S)·n-gram(N)·발음(P)·제원(T)의
모든 16개 조합을 기존 두 개발 분할에서 비교했습니다(E007/E008).
30개 조합별 모델을 학습했고, 모두 끈 NONE은 두 분할에서 고정 기준선으로 평가했습니다.
37개 비교 특징과 LightGBM 설정은 고정했습니다.

네 경로 모두 사용한 **SNPT**가 두 분할 평균으로 가장 높았습니다.
기존 SPT→SNPT의 새 선박 가중 정확도는 분할1 97.596%→98.620%,
분할2 97.590%→97.604%, 단순 평균 97.593%→98.112%입니다.
혼합 검증 분할1은 99.611%→99.255%로 하락했습니다. 두 분할은 일부 관측이
겹치는 개발 검증이며 공식 점수나 독립 최종 테스트 결과가 아닙니다.

전체 16조합·n-gram 추가 8쌍·오류 변화·한계는
[네 검색 경로 모든 조합 결과](docs/RETRIEVAL_COMBINATIONS.md)에 기록했습니다.
[이전 선박명 개선 결과](docs/NAME_IMPROVEMENT_RESULTS.md)는 E004~E006의 역사적 기록입니다.
이전 `reports/validation.json`과 `docs/MODEL_DETAILS.md`는 23개 특징 베이스라인의
역사적 기록입니다. 최신 전체 학습·ZIP 검사는 `project_log/experiments/E009.json`을 확인합니다.

현재 작업공간의 개선 제출 파일 경로는
`/workspace/vessel_matching/ngram_submission/submit.zip`입니다.
E009 전체 학습·오프라인·행 독립성·공개 12,222행 실행 비용 검사를 완료했습니다. 기존
`/workspace/vessel_matching/submit.zip`과 `name_improved_submission/submit.zip`은 보존합니다.

최신 구성을 재현하려면 공식 CSV를 `data/`에 놓고 다음을 실행합니다.

```bash
python train.py --name-mode all --phonetic-search 24 --retrieval-channels S N P T
python verify_submission.py
```

기본 `python train.py`는 비교를 위한 기존 baseline 구성을 유지합니다.

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
3. 기존 baseline은 이름 유사도, 숫자 일치, 톤수 상대 오차와 결측, 학습 데이터에서 추정한
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
- `model/name_processing.py`: 원래 표기 보존을 전제로 한 단어·숫자·가중 발음 비교와 보조 검색 키.
- `model/ngram_retrieval.py`: 공식 카탈로그에만 fit하는 문자 2~4gram TF-IDF 검색.
- `experiments/name_ablation.py`: 고정된 조건의 비교와 두 번째 그룹 분할 재검증.
- `experiments/retrieval_factorial.py`: 네 검색 경로의 모든 조합별 학습·평가.
- `reports/retrieval_combinations.csv`: 조합·분할·검증 집단·조회 조건별 전체 128행 집계.
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
