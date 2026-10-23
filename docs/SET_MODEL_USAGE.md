# 후보 집합 모델 사용법

2026-10-22 KST. N2 구현. 기존 제출 모델과 별도인 실험용 모델이며 성능 개선이 확인된 판은 아니다.

## 구현

- `decision_models/set_model.py`: 학습·저장·로드·후보 그룹별 점수 계산. 학습한 후보 표현 32차원의 평균/최댓값을 같은 입력 행 안에서만 계산하고 각 후보에 결합한다. 이후 32차원 은닉층을 거쳐 후보별 점수를 만든다.
- 현재 32개 입력 특징에서 파라미터 4,193개. 기존 64→32 MLP는 4,225개다. 손실은 후보 쌍 가중 BCE를 유지하며, 행 단위 softmax 변경은 포함하지 않았다.
- `model/matcher.py`: `assets.json`의 `decision_model: set_mlp`일 때 새 모델을 로드한다. 점수 계산에 각 입력 행의 후보 수를 함께 전달한다. 기존 LightGBM/CatBoost/MLP 자산은 기존 경로를 사용한다.
- `experiments/train_set_model.py`: 기존 후보·특징 생성기를 재사용하는 학습 및 CSV 추론 진입점. 기존 한글 확장/soft-meta와 비한글 독립 잡음을 사용하며 조회는 끈다. 잡음을 끄려면 `--no-independent-noise`를 지정한다. 학습 결과에는 설정을 기록한다.

한글과 비한글 모두를 처리할 수 있는 독립 모델이다. 기존 앙상블에 자동으로 섞거나 기존 비한글 경로를 교체하지 않았다. 한글만 교체하는 실험은 이후 기준판을 고정해 별도로 구성한다.

## 환경

프로젝트 기준 환경은 `.venv-server/Scripts/python.exe`다. 이번 PC에서는 지정 Python 3.11.17 Windows 배포본을 설치 도구가 찾지 못하여, 사용자의 명시적 허용에 따라 기존 `.venv/Scripts/python.exe`(3.11.15, torch 2.7.1+cu128)로 검증했다. 아래 예시는 이번 PC의 PowerShell 실행 방식이며 서버 버전 일치 확인을 대신하지 않는다.

## 공식 데이터로 작은 실행 확인

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/train_set_model.py train --data-dir data --output artifacts/set_smoke_new --holdout-fold 0 --max-train-rows 128 --max-valid-rows 32 --epochs 3 --device cpu --save-pairs
```

기존 팀 fold 파일과 SHA-256을 검사하고 fold 0 밖에서 학습 행을, fold 0 안에서 검증 행을 고른다. 크기 제한은 한글/비한글을 포함시키기 위한 실행 표본이며 평가 분포를 재현하지 않는다. 전체 공식 카탈로그를 사용하고 사전·스케일러는 선택된 학습 자료로만 만든다.

기존 출력 폴더에는 덮어쓰지 않으므로 실행마다 새 경로를 지정한다. 작은 학습 결과는 배포용 모델이나 성능 비교 결과로 쓰지 않는다. `--save-pairs`는 행 경계를 포함한 학습 특징 캐시를 로컬 artifacts에 보관한다.

## 전체 학습

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/train_set_model.py train --data-dir data --output artifacts/set_full_new --epochs 25 --device cpu --save-pairs
```

`--device cuda`로 GPU 학습도 가능하다. 전체 학습은 이 구현 작업에서 실행하지 않았다. `--holdout-fold 0`을 추가하고 크기 제한을 생략하면 해당 fold의 개발 검증이 되며, 이것도 전체 G5 OOF를 대신하지 않는다. 후보 검색 실패 시 정답 보충은 학습에서만 기존 방식을 따른다. 검증에는 정답 후보를 삽입하지 않는다. 학습/추론 후보 수 차이는 현재 파이프라인에서 계승하므로 이후 구조 비교에서 동일 조건을 유지하고 별도 민감도 검증이 필요하다.

출력:

- `model/assets.json`: 해당 학습 자료에서 만든 검색/전처리 자산과 모델 종류.
- `model/set_model.pt`, `set_scaler.npz`, `set_config.json`: 가중치, 학습 스케일러, 구조·학습 설정.
- `run.json`: 실행 범위, 데이터/분할 해시, 단계별 시간, 비용/검증 집계, 자산 해시.
- 선택적 `pairs.npz`: `x, y, w, feature_names, group_sizes`. 같은 입력의 후보는 연속해 저장한다. 증강 복사본은 별도 그룹이다.

## CSV 추론

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/train_set_model.py predict --model artifacts/set_full_new/model --test data/test.csv --catalog data/vessel_catalog.csv --output output/set_submission.csv
```

추론은 CPU로 모델을 로드한다. train 또는 검증 정답을 읽지 않고 `sample_id,target_vessel_id`만 출력한다. 기존 출력 파일에는 덮어쓰지 않는다. ZIP을 생성하지 않는다.

## 같은 특징 캐시로 단독 실행

```powershell
.\.venv\Scripts\python.exe -m decision_models.set_model train --pairs artifacts/set_smoke_new/pairs.npz --model artifacts/set_cached_new --epochs 25 --device cpu
.\.venv\Scripts\python.exe -m decision_models.set_model predict --pairs artifacts/set_smoke_new/pairs.npz --model artifacts/set_cached_new --output artifacts/set_cached_scores.npy
```

이 명령은 점수 모델만 저장한다. 전체 선박 매칭에는 같은 특징을 만든 `assets.json`과 공식 카탈로그가 별도로 필요하다. 일반 MLP와 비교할 때는 같은 캐시 및 epochs/가중치/스케일링/학습 후보를 맞춘다. 기본 CLI 파라미터가 서로 같다고 가정하지 않는다.

Python API는 `fit(..., group_sizes=...)`, `predict_groups(x, group_sizes)` 또는 한 행만 명시하는 `predict_group(x)`다. 경계 없이 `predict(x)`를 호출하면 오류를 내므로 여러 테스트 행을 하나의 집합으로 잘못 섞는 일을 방지한다.

## 검증 계약과 한계

- 패딩은 평균/최댓값에서 제외한다. 학습 스케일러는 추론에서 갱신하지 않는다.
- 추론은 입력 행마다 독립 실행한다. 후보 특징을 정렬해 합산 순서도 고정한 후 원래 후보 순서로 점수를 돌려준다. 후보 중복 특징은 같은 점수를 받는다. 최종 동점 처리는 기존 카탈로그 ID 정렬을 따른다.
- 단독/전체/부분/순서 변경·다른 행 내용 변경, 후보 순서 변경, 저장/복원, 직접 조회와 모델 예측 혼합, 기존 모델 경로를 검사한다.
- per-query 추론을 선택해 다른 행의 배치 크기에 따른 수치 차이를 피한다. 대규모 실행 속도는 별도로 측정해야 한다.
- 실측 기록은 `project_log/experiments/AS-SET-01.json` 참조. 소규모 학습 완료와 전체 OOF 성능 향상은 구분한다. 기존 모델 파일과 ZIP은 교체하지 않았다.
