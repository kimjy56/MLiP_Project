# 합성 발음 실험 실행 안내

현재 결과는 Pronunciation_PRONUNCIATION_RESULTS.md에 있다. 발음 확대와 역검색 필터 모두 기본 적용하지 않고 AS-06을 유지한다. 아래는 새 실험 ID로 재현하기 위한 개발 명령이다. 제출 ZIP 생성·업로드 기능은 없다.

## 현재 작업공간 조건

- PowerShell, 프로젝트 루트 C:\Projects\ML_midterm에서 실행한다.
- `.venv-server\Scripts\python.exe`를 사용한다. 현재 Python3.11.15와 서버 목표3.11.17의 차이를 실험 기록에 남겼다.
- CUDA, 기존 `artifacts/LM-LFM25-FULL-R24-01/model`, `AR-06/catalog_full_ko_map.json`, 공식 data/train.csv·vessel_catalog.csv·eval_kit/folds_group5_seed42.csv가 필요하다.
- 매칭 비교기는 현재 experimental_matchers/jamo_scoring 기준 실험 JSON·캐시·fit 모델을 검증해 활용한다. 다른 PC에는 Git에서 제외된 이 자산이 없을 수 있다. 기존 자산은 docs/MULTI_PC_SETUP.md 절차에 따라 별도로 준비한다.
- 기존 완료·실패 실험 ID를 재사용하지 않는다. 아래 NEW01도 이미 사용했다면 다른 ID로 바꾼다.

## 발음 보강

```powershell
.\.venv-server\Scripts\python.exe -X utf8 experiments/build_catalog_hangul_diverse.py --experiment-id KO-DIVERSE-NEW01 --scope deficient
```

`deficient`는 기존 발음이 하나 이하인 공식 카탈로그 이름만 처리한다. 검증 정답은 선택에 사용하지 않는다. `all`은 전체 정규화 이름을 새로 생성하는 별도 범위이며, 이번 성능 비교에서는 실행하지 않았다.

기존 발음은 보존하고, sampling·출력 검사·원자모 중복 제거·숫자 signature 일치·고정 품질 규칙으로 최대4개까지 저장한다. 후보 품질 규칙의 세부값은 새 실험 JSON의 policy와 generation_config.json에 기록한다. 모델은 eval/inference 모드이며 가중치를 학습하지 않는다.

중단 후에는 새 실험 ID와 `--resume-from artifacts/이전ID`를 사용한다. 모델·입력·prompt·정책·소스 해시가 같아야 하며 배치 이름과 seed도 대조한다. 코드나 설정이 바뀌었다면 기존 배치를 섞지 않고 새로 생성한다. GPU가 없으면 명시적으로 실패하며 긴 CPU 생성으로 자동 전환하지 않는다.

## 추가 발음의 역검색 검사

```powershell
.\.venv-server\Scripts\python.exe -X utf8 experiments/filter_catalog_hangul_ownership.py --experiment-id KO-OWNERSHIP-NEW01 --source-experiment-id KO-DIVERSE-NEW01
```

원래 AS-06 발음은 보존한다. 추가 발음만 정규화 자모 OSA로 기존 사전에 역검색해 원래 이름이 유일한 최근접일 때 허용한다. 동률·다른 이름 우세·원래 anchor 부재는 제외한다. 숫자 signature가 같은 그룹에서만 비교한다. 모델 예측 확률을 검증하는 절차는 아니며, 원래 사전이 잘못된 경우 유효한 새 발음도 제외할 수 있다.

## 동일 엔진 비교

```powershell
.\.venv-server\Scripts\python.exe -X utf8 experiments/evaluate_pronunciation_maps.py --experiment-id EVAL-KO-DIVERSE-NEW01 --map-experiment-id KO-DIVERSE-NEW01
.\.venv-server\Scripts\python.exe -X utf8 experiments/evaluate_pronunciation_maps.py --experiment-id EVAL-KO-OWNERSHIP-NEW01 --map-experiment-id KO-OWNERSHIP-NEW01
```

AS-06 legacy 점수, lookup OFF, 외부 fold0 2,372행을 사용한다. 원본/default20261022/severe20261022·20261026·20261027의 다섯 조건을 평가한다. 공식 평가기의 subset 짝지은 비교와 선박 단위 bootstrap2,000회 seed0을 사용한다. 사전 변경 때문에 후보 집합은 바뀔 수 있다. 기존 캐시가 검증되지 않으면 원본 사전 예측을 다시 계산한다.

코드 테스트:

```powershell
.\.venv-server\Scripts\python.exe -X utf8 -m unittest tests.test_pronunciation_quality tests.test_pronunciation_ownership -v
```

## 결과 위치와 제한

- `project_log/experiments/<ID>.json`: 실행 설정·환경·결과·해석·한계·파일 해시.
- `artifacts/<생성ID>/catalog_full_ko_map.json`: 고정 검색 자산. model 가중치나 ZIP과 구분한다.
- `artifacts/<생성ID>/generation_config.json`, `batches/`: 생성 설정·배치 seed·필터 결과. 마지막 사전의 해시는 실험 JSON에 기록한다.
- `artifacts/<평가ID>/`: 공개 보류의 행별 예측·오답 사례. Git 제외 경로다.
- `reports/matching_improvements/<평가ID>/`: 집계 JSON·공식 평가 출력.

기존 생성기는 fold0을 제외한 학습 기록을 갖지만, 수동 사전과 과거 baseline map의 생성 이력 한계가 남아 있다. 같은 사전을 모든 fold에 적용해 누수 없는 전체5-fold OOF로 제시하지 않는다. 새로운 개발 후보의 local 결과를 AS-06 서버0.9122153056과 같은 범위의 수치로 비교하지 않는다.
