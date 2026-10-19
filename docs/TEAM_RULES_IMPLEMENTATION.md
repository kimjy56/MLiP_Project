# 팀 규칙 v4 코드 반영

2026-10-19 KST. 이 문서는 구현과 실행 범위를 설명한다. 공식 규칙 확인이나 팀 공통 검증 완료를 의미하지 않는다.

| 규칙 | 구현 |
|---|---|
| §3 고정 그룹 5-fold | `train_oof.py`: 배포된 fold를 읽고 5회 학습. 분할을 생성하지 않음 |
| §3 ID·그룹·해시 확인 | 12,222행, 중복·누락·정답 일치, 카탈로그 포함, fold 0~4, 선박 그룹 분리 및 전달받은 SHA-256 검사 |
| §3 학습 fold 내 사전 생성 | 각 fold의 나머지 4개 fold로만 `fit`·사전 생성. 카탈로그는 전체 사용 |
| §3 정답과 예측 분리 | 검증 호출은 `QUERY_COLUMNS`만 전달. 예측 확정 후 정답으로 진단 |
| §4 OOF 형식 | `sample_id,pred_vessel_id,fold`, 원래 train 순서 복원·전체 행 대조 |
| §5 입력 경로 | 파일별 명시 경로 → `data/` → `open/` 순서. 지정 경로가 잘못됐으면 임의 대체하지 않음 |
| §5 CPU 예산 | 순차 fold 학습, 학습 수치 연산 4스레드, 모델 초기화·추론 1스레드 |
| Windows 한글 경로 | LightGBM 모델 텍스트를 Python의 UTF-8 파일 입출력으로 저장·읽기 |
| §6 조회 통합 설정 | `use_train_lookup`이 호출부호 직접 조회와 학습 이름 별칭 후보 추가를 함께 제어 |
| §6 행 독립성 | 고정 카탈로그·학습 자산으로 행별 예측. 반복·역순·부분·단독·빈 입력 검사 |
| §8 기록 | 학습 전 계획, 실행 중 fold별 결과, 코드·데이터·모델 해시 및 실패/중단 기록 |
| §11 ZIP | OOF 실행은 ZIP이나 최종 전체 학습 모델을 생성하지 않음 |

## 조회 설정

`train.py`와 `train_oof.py`의 `--no-use-train-lookup`은 사전 조회를 끈 설정을 `assets.json`에 저장한다. 기본값은 기존 구성과 동일하게 켜짐이다. 이 선택은 사전 조회에 관한 운영진 승인 확인을 뜻하지 않는다.

추론 시 설정을 지정하지 않으면 저장한 값을 사용한다. 이전 버전의 자산에 설정이 없으면 기존 동작인 켜짐을 유지한다. 일시적으로 덮어쓸 때는 PowerShell에서 다음처럼 실행한다.

```powershell
$env:USE_TRAIN_LOOKUP = '0'
.\.venv-server\Scripts\python.exe script.py
Remove-Item Env:USE_TRAIN_LOOKUP
```

`0`/`false`는 끄기, `1`/`true`는 켜기다. 잘못된 값은 오류로 중단한다. 선종·국적 대응 확률은 학습 fold로 만든 특징이므로 이 조회 스위치의 대상이 아니다. 과거 실험용 `predict(use_direct=False)`는 호출부호만 끄므로 통합 설정으로 대신 사용하지 않는다.

## 팀 공통 OOF 실행

먼저 공식 CSV 두 개를 `data/`에, 팀이 배포한 평가 묶음을 `data/eval_kit/`에 둔다. 기본 fold 경로는 `--data-dir` 아래의 `eval_kit/folds_group5_seed42.csv`다. 배포자가 제공한 SHA-256 및 배정받은 팀 실험 ID를 사용한다. 아래 `O-01`은 형식 예시로, 팀 ID를 새로 배정한 것이 아니다.

```powershell
.\.venv-server\Scripts\python.exe train_oof.py --experiment-id O-01 --folds-sha256 <배포본_SHA256> --name-mode all --phonetic-search 24 --retrieval-channels S N P T --no-use-train-lookup
```

`<배포본_SHA256>`은 실제 64자리 해시로 바꾼다. 위 예시는 조회를 끈 SNPT 구성을 평가한다. 조회를 켜려면 `--use-train-lookup`을 사용하며 별도 실험 ID로 비교한다.

- OOF: `artifacts/oof/O-01/oof_O-01.csv`
- fold별 모델: `artifacts/oof/O-01/fold_0/model/`부터 `fold_4/model/`까지
- 설정·집계·해시·오류 기록: `project_log/experiments/O-01.json`

모델·OOF는 Git 제외 경로에 저장된다. 동일 ID나 기존 출력 폴더가 있으면 덮어쓰지 않고 중단한다. 중단·실패한 실험도 기록을 보존하고 새 ID로 재시도한다. 로그의 로컬 진단 점수는 팀 공통 평가 결과를 대체하지 않는다. `gold_rank`는 배포 평가기의 정확한 정의를 확인하기 전에는 출력하지 않는다.

팀 평가 파일을 수령하면 먼저 `selftest_evaluate.py`와 E00 재현을 확인한다. 그 뒤 배포본의 실제 CLI를 확인하고 다음 형식으로 평가한다.

```powershell
.\.venv-server\Scripts\python.exe -X utf8 data/eval_kit/evaluate.py --oof artifacts/oof/O-01/oof_O-01.csv --train data/train.csv
```

전체 OOF·fold별 변동·선박 단위 부트스트랩 CI·보조 분할·비용을 함께 검토한다. 한 fold나 최고 개발 점수를 독립 최종 성능으로 보고하지 않는다. 평가기·fold는 이 작업에서 새로 만들지 않았다.

## 기존 학습과 호환성

`train.py`는 기존 두 개발 holdout 및 최종 전체 학습 용도로 유지한다. 팀 공통 5-fold 검증은 `train_oof.py`를 사용한다. 기존 보고서를 보존하려면 실행마다 새로운 `--output-root`를 사용한다. 예를 들어:

```powershell
.\.venv-server\Scripts\python.exe train.py --final-only --output-root artifacts/final/run_01 --name-mode all --phonetic-search 24 --retrieval-channels S N P T --no-use-train-lookup
```

생성된 `model/assets.json`·`model/ranker.txt`를 한 쌍으로 관리하고 해시·실험 ID를 확인한 뒤 추론용 `model/`에 배치한다. 제출 ZIP은 별도 요청 때만 생성한다.

## 확인 범위

지정 Windows 가상환경의 합성 테스트로 저장/로딩, fold 격리, 잘못된 분할 거부, 조회 통합 설정, 정답과 예측 분리, 실제 진입점의 순서·부분·단독·빈 입력 및 파일 보존을 확인한다. 합성 데이터의 소형 학습은 실행 검사이며 성능 비교가 아니다.

코드 구현 당시에는 실제 공식 CSV·고정 fold·평가기가 없었다. 이후 사용자가 `data/eval_kit/`에 평가 묶음을 배치했다. 공식 CSV 두 개는 아직 없어 팀 OOF, 평가기 자체 시험/E00 재현, 공식 점수 및 Ubuntu/L4 실행은 미검증이다. 구현 검사 기록은 `project_log/team_rules_implementation.json`, 묶음 수령 검사는 `project_log/eval_kit_received.json`에 기록한다.

원본 `data/train.csv`를 준비한 뒤 최초 검사는 다음 순서로 실행한다.

```powershell
.\.venv-server\Scripts\python.exe -X utf8 data/eval_kit/selftest_evaluate.py --train data/train.csv
.\.venv-server\Scripts\python.exe -X utf8 data/eval_kit/evaluate.py --oof data/eval_kit/oof_E00_baseline.csv --train data/train.csv
```
