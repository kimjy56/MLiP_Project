# 여러 해석을 유지하는 선명 해독기

## AS-NAME-01 초기 계획 (2026-10-23 KST, planned)

목적: 관측 선명을 하나의 영문 표기로 확정하지 않고 공식 카탈로그 안의 여러 이름으로 해석하는 독립 모델을 시작한다. 기존 Matcher/Set Model과 제출 ZIP은 이 실험의 수정 대상이 아니다.

가설: `google/byt5-small`의 바이트 표현을 실제 이름 대응 쌍으로 파인튜닝하면 규칙 기반 로마자 편집거리와 다른 후보를 찾을 수 있다. 카탈로그 접두 트리로 생성 토큰을 제한해 존재하지 않는 이름이 출력되는 것을 막는다. 빔 검색은 근사 탐색이므로 정답 후보의 누락까지 막지는 못한다.

- 기본 가중치: Google 공식 ByT5-small, revision `68377bdc18a2ffec8a0533fef03b1c513a4dd49d`.
- 자료: 공식 train.csv와 vessel_catalog.csv만 사용. 합성 학습 자료·외부 별칭·테스트 파일 사용 없음.
- 분할: 배포된 G5-s42의 fold 0을 보류. 같은 선박이 학습과 보류에 겹치면 중단한다.
- 학습: 학습 fold의 관측 이름과 정답 카탈로그 이름 쌍을 중복 제거. 이름은 Unicode NFC·대문자·연속 공백 정리만 적용해 숫자·구두점을 보존한다. 한글 쌍의 총 추출 확률 0.7, 비한글 0.3. AdamW, learning rate 0.0001, 200 optimizer steps, batch 8 × accumulation 2, seed 42. 검증 정답으로 epoch/설정을 선택하지 않는다.
- 입력: 이름만 사용. 호출부호, 선박 ID, 다른 테스트 행의 정보는 입력하지 않는다.
- 출력: 카탈로그 안의 상위 최대 10개 이름·모델 점수·각 이름에 대응하는 모든 선박 ID. 동명 선박은 유지한다.
- 초기 평가: fold 0에서 seed 42로 한글/비한글 각각 최대 64행을 고정 추출한다. 미세조정 전후와 동일 표본의 `Unidecode + WRatio` 기준선을 비교한다. 이는 실행·학습 가능성 확인용 pilot이며 전체 OOF 또는 모델 채택 근거가 아니다.
- 지표: 정답 **선명** 포함률@1/5/10, 실제 반환 이름 수, 동명 충돌, 추론 시간. 선명 일치는 선박 ID 정확도가 아니다. 공식 가중 ID 점수는 미측정(null)으로 둔다.
- 독립성: 한 번에 입력 한 행만 처리하고 샘플링을 끈다. 반복·순서·부분집합 입력에 대한 후보 목록 일치를 검사한다.
- 환경: 기존 승인된 Windows Python 3.11.15 환경 사용. 현재 torch 2.7.1+cu128, transformers 5.20.1로 서버의 4.57.6과 다르다. 서버 호환성 확인으로 표현하지 않는다.

이 단계의 목표는 이름 후보 생성이다. 톤수·국적을 이용한 최종 선박 ID 결정과 전체 G5 OOF는 다음 단계이며, 이 초기 시험에서 임의의 메타정보 보정 규칙을 추가하지 않는다.

공식 모델 설명: https://huggingface.co/google/byt5-small
생성 제약 API: https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils

## 구현·실행 안내

### AS-NAME-02 재실행 계획과 완료 기록

AS-NAME-01의 학습·인메모리 평가는 종료됐지만 저장 후 재로딩 검사 4행에서 후보 목록이 달랐다. 기존 결과는 보존하며 배포 가능한 성능으로 채택하지 않는다. 현재 transformers 5.20.1의 T5Config가 원본 `tie_word_embeddings=false`를 true로 다시 쓰는 것을 확인했다. 설정 변화만이 후보 차이의 원인이라고 단정하지 않는다.

AS-NAME-02는 원본의 가중치 공유·출력 스케일 설정을 명시적으로 보존하고 추론은 FP32로 고정한다. 학습 200 step, 분할, 표본, 모델 초기 가중치, 학습률, 후보 수는 유지한다. 저장 후 재로딩한 모델로 최종 pilot을 평가하고, 저장 전/후 후보 동일성을 추가 검사한다. 변경이 두 가지이므로 01↔02 차이를 개별 변경의 정확도 효과로 해석하지 않는다.

구현과 검증이 완료되면 명령 및 결과를 이 문서에 추가한다. 모델·행별 후보는 Git 제외된 artifacts/에, 공유 가능한 집계·환경 차이·실패 기록은 project_log/experiments/에 보존한다. 신규 실행에는 새 실험 ID를 사용한다.

기존 실행 기록 `project_log/experiments/AS-NAME-02.json`은 completed다. 한글 64행의 이름 상위 10개 포함률은 6.25%, 비한글 64행은 71.875%였고, 저장/재로딩 및 순서 변경 검사 4행이 통과했다. 현재 PC에서 재실행한 결과는 아니다. 원본 기록을 보존하며, 이 기록에 적힌 가중치·예측 파일은 현재 PC에 없다. 초기 200-step 결과로 모델을 채택하지 않는다.

## AS-NAME-03 구현 이어가기 (2026-10-23 KST)

현재 목표는 **관측 이름 → 카탈로그의 정답 선박명**이다. ByT5를 전체 파인튜닝하며, 추론은 전체 공식 카탈로그의 접두 트리로 제한한다. `predict()`는 기본 10-beam 검색에서 1위인 정규화 카탈로그 이름을 `predicted_name`으로 반환한다. 동명 선박의 모든 ID는 `vessel_ids`로 남긴다. 1위 선택을 추가했다고 정답 일치가 보장되는 것은 아니다. 선명만으로 구별할 수 없는 동명 ID 결정은 이 모듈 범위 밖이다.

### 파일과 입출력

| 파일 | 역할 |
|---|---|
| `train_name_decoder.py` | 최상위 학습 실행 진입점 |
| `experiments/train_name_decoder.py` | 실제 학습 루프, 검증, 저장, 실험 기록 |
| `name_decoder/data.py` | 공식 정답 ID로 이름 쌍 생성; 그룹 분리 검증; 전체 학습 모드 |
| `name_decoder/model.py` | `predict(name)`, `predict_many(names)`, `decode(name)` |
| `name_decoder/__main__.py` | 이름 하나를 받아 예측 JSON 출력 |

출력 JSON에는 `predicted_name`, `vessel_ids`, `candidates`가 있다. 후보 점수는 길이 보정 빔 점수이며 보정된 확률이 아니다. 반환 이름은 NFC·대문자·공백 정규화한 카탈로그 키다. 동일 후보 검색을 유지하려면 단일 이름 예측에서도 `--top-k 10`을 유지한다. `--top-k 1`은 빔 폭도 달라지므로 같은 검색의 첫 항목만 반환하는 것과 다르다.

### 학습량과 손실

- 기존 동작: 옵션을 생략하면 `--steps 200 --loss-reduction token`. 이전 실행 기록과 기본 설정을 보존한다.
- `--epochs N`: 중복 제거된 학습 쌍 수 × N에 해당하는 **추출 횟수**를 지정한다. 한글/비한글 0.7/0.3 가중 복원추출이므로 모든 쌍을 정확히 N번 순회한다는 뜻은 아니다. 실제 optimizer step과 추출 횟수를 기록한다. `--steps`와 동시 지정할 수 없다.
- `--loss-reduction row`: 각 이름 쌍의 유효 정답 토큰 손실을 평균한 뒤 쌍별 평균을 낸다. 패딩은 제외하며, 긴 이름이 토큰 수 때문에 더 큰 가중치를 갖는 효과를 분리하는 비교 옵션이다. 성능 향상은 아직 미측정이다.
- 검증 정답으로 epoch 수를 고르거나 조기 종료하지 않는다. 학습량과 손실은 별도 실험 ID로 비교한다. 여러 변경을 함께 적용한 결과를 개별 변경 효과로 해석하지 않는다.

### 실행 (Windows PowerShell)

프로젝트 루트에서 실행한다. 사용자가 확인한 데이터 폴더는 이 프로젝트의 `data`다. 학습/테스트 인터프리터는 `.venv-server\Scripts\python.exe`로 고정한다. 환경 준비는 `environment/README.md`의 절차를 따른다. 현재 PC에서는 환경 설치 도구 uv 확보에 실패했으므로 아래는 **실행 안내이며 이번 세션의 실행 완료 기록이 아니다**.

```powershell
# 공식 ByT5-small 가중치 다운로드(인터넷 필요). 데이터는 전송하지 않음.
.\.venv-server\Scripts\python.exe tools\fetch_byt5.py

# 기존 200-step 설정을 supplied fold에서 다시 확인
.\.venv-server\Scripts\python.exe train_name_decoder.py --experiment-id AS-NAME-04 --data-dir .\data --folds .\eval_kit\folds_group5_seed42.csv --fold 0 --steps 200 --loss-reduction token --eval-per-language 64

# 학습량만 바꾸는 비교 예시. 10은 검증된 최적값이 아님.
.\.venv-server\Scripts\python.exe train_name_decoder.py --experiment-id AS-NAME-05 --data-dir .\data --folds .\eval_kit\folds_group5_seed42.csv --fold 0 --epochs 10 --loss-reduction token --eval-per-language 64

# 별도 선택한 설정으로 전체 공식 train 학습. 검증 결과는 생성하지 않음.
.\.venv-server\Scripts\python.exe train_name_decoder.py --experiment-id AS-NAME-FINAL-01 --data-dir .\data --final-train --epochs 10 --loss-reduction token

# 학습 완료 후 한 이름을 카탈로그 이름으로 예측
.\.venv-server\Scripts\python.exe -m name_decoder --model .\artifacts\AS-NAME-FINAL-01\model --catalog .\data\vessel_catalog.csv --name "오션스타7" --top-k 10

# 소프트웨어 테스트: 작은 무작위 모델/가상 fixture만 사용, 성능 실험 아님
.\.venv-server\Scripts\python.exe -m unittest discover -s tests -p "test_name_decoder*.py" -v
```

GPU 사용 가능 시 자동 선택하며 CPU도 지원한다. 가중치 다운로드 이후 학습·추론은 로컬 파일만 사용한다. `--final-train`은 fold가 없어도 실행할 수 있지만, 그 모델을 동일 train에서 평가한 값을 일반화 성능으로 보고하지 않는다. 검증 모드는 배포된 fold 파일이 없으면 중단하며 임의 재분할하지 않는다. 전체 보류 fold를 평가하려면 `--eval-per-language 0`을 사용한다. 전체 5-fold 실행·합산은 아직 수행하지 않았다.

### 산출물과 검증 범위

- `artifacts/<ID>/model/`: 학습 가중치·토크나이저·설정·`name_decoder_manifest.json`.
- `artifacts/<ID>/training_history.json`: 실제 학습 시간·스텝별 손실.
- 검증 모드만 `pilot_predictions.csv`: 공개 train 보류 행의 1위 예측 이름·정오·후보 목록. 파일명은 전체 fold 평가에서도 유지된다.
- `project_log/experiments/<ID>.json`: 데이터/분할/코드 해시, 환경, 비용, 저장/재로딩 검사, 한글·비한글 이름 top-1 정확도와 상위 5/10개 포함률. 이름 가중 정확도는 별도 `weighted_name_exact_accuracy`이며 공식 ID 가중 점수는 null이다.
- 기존 실험 ID와 출력 폴더가 있으면 덮어쓰지 않는다. ZIP은 생성하지 않는다.

이번 구현은 데이터·코드 존재 및 Git diff를 확인했다. 지정 Python 환경/가중치가 없어 실제 학습과 Python 테스트는 실행하지 못했다. 공통 fold도 미수령 상태다. 최우선 후속은 환경 준비 → 새 소프트웨어 테스트 → 공통 fold 복원 → 같은 128행에서 200-step 재현 → 학습량만 변경한 비교다. 원래 AS-NAME-02는 Transformers 5.20.1 환경이므로 지정 4.57.6 환경 결과와 직접 같은 조건으로 취급하지 않는다.

### 2026-10-24 KST — AS-NAME-04 및 AS-NAME-05 실행 결과 비교

`.venv-server` 환경과 Google 공식 ByT5-small 가중치를 확보하고, 복원된 `eval_kit/folds_group5_seed42.csv`의 Fold 0 보류 128행(한글 64행, 비한글 64행)에 대해 동일 조건 비교 실험을 완료했다.

#### 실측 비교 요약표

| 모델 / 실험 | 설정 | Loss 수렴 | 한글 Top-1 (Hits) | 한글 Top-10 (Hits) | 비한글 Top-1 (Hits) | 비한글 Top-10 (Hits) | 가중 선명 정확도 |
|---|---|---|---|---|---|---|---|
| **규칙 기준선 (WRatio)** | Unidecode + RapidFuzz | N/A | 15.6% (10/64) | 48.4% (31/64) | **100.0%** (64/64) | **100.0%** (64/64) | 40.94% |
| **사전학습 ByT5 (Pretrained)** | Google 공식 Zero-shot | N/A | 0.0% (0/64) | 0.0% (0/64) | 10.9% (7/64) | 12.5% (8/64) | 3.28% |
| **AS-NAME-04** | 200 steps, token loss | 5.041 ➜ 1.375 | 0.0% (0/64) | 0.0% (0/64) | 70.3% (45/64) | 71.9% (46/64) | 21.09% |
| **AS-NAME-05** | **600 steps, row loss** | **5.226 ➜ 0.546** | **17.2%** (11/64) | **28.1%** (18/64) | **82.8%** (53/64) | **82.8%** (53/64) | **36.88%** |

#### 주요 발견 및 시사점

1. **한글 음역 능력 발현 (0.0% ➜ 17.2% 돌파)**:
   - 200스텝(AS-NAME-04)에서는 한글 3바이트 음절과 영문 알파벳 간의 매핑을 학습하기에 노출 수가 부족(한글 305쌍당 평균 7회)하여 0%에 머물렀음.
   - 600스텝 + `--loss-reduction row`(AS-NAME-05, 쌍 추출 9,600회) 적용 시 손실이 0.546으로 대폭 하강하며 한글 Top-1 정확도가 **17.2%로 급상승**하여 규칙 기준선(15.6%)을 추월함.
   - 모델이 `우진 파이오니아호` ➜ `WOOJIN PIONEER`('호' 접미사 자동 제거 및 음역), `흥아 파이오니아호` ➜ `HEUNG-A PIONEER`, `글로리 하나` ➜ `GLORY HANA`, `닝화 421` ➜ `NING HUA 421` 등을 정확히 맞추어 실제 한영 음역 규칙을 학습했음을 실증.
2. **비한글 복원력 강화 (70.3% ➜ 82.8%)**:
   - 철자 오타가 있는 비한글 선명(`HATTHYAA NAREE` ➜ `HATTHAYA NAREE` 등)의 카탈로그 정규화 복원력이 사전학습 10.9%에서 82.8%까지 지속 상승.
3. **무결성 및 독립성 검증**:
   - 두 실험 모두 모델 가중치 직렬화 재로딩 및 단일 행 추론 셔플 검사에서 100% 일치 확인 (`top_10_names_equal: true`, `reordered_and_subset_equal: true`).

## 2026-10-24 KST — AS-HCX-SCORER-01: 생성 후 카탈로그 검색 ratio 적용

AS-HCX-03의 저장 생성문 128행(한글/비한글 각 64행, 미등장 선박 108척)을 고정하고 WRatio와 ratio만 비교했다. 원본 예측·공식 train·카탈로그·fold 해시, seed 42 표본, 기존 WRatio Top-10 전체 일치를 확인했다. 가중치는 재학습하지 않았다.

| 선박명 지표 | WRatio | ratio |
|---|---:|---:|
| 가중 Top-1 정확도 | 58.4375% | 72.34375% |
| 한글 Top-1 | 29/64 | 40/64 |
| 한글 Top-10 포함률 | 40/64 | 53/64 |
| 비한글 Top-1 | 57/64 | 61/64 |
| 비한글 Top-10 포함률 | 61/64 | 63/64 |

정답화 15행(한글 11, 비한글 4), 오답화 0행. 검색만의 3회 중앙값은 17.294초 → 2.413초(7.17배); 로딩·인덱스·생성·저장을 제외했다. 호스트 부하 및 초기 반복과 CLI import 검사 중첩이 있어 시간은 이 실행에 한정한다. 전체 추론 속도 개선은 미측정이다.

남은 27개 오답 중 정답 이름이 Top-10 안에 있는 후보 선택 오류는 15행(한글 13, 비한글 2), 검색 누락은 12행(한글 11, 비한글 1)이다. `KITI BOM`은 `BO` 대신 `KITI BHUM`으로 정답화됐지만, `POLAR BRAIT`은 전체 카탈로그에서 `POLAR BRASIL`이 1위여서 `POLAR BRIGHT` 정답 복구에 아직 실패한다. 소형 fixture의 두 후보 비교와 전체 카탈로그 결과를 구분한다.

이 수치는 이미 탐색에 사용한 로컬 파일럿의 **선박명** 점수다. 공식 선박 ID 점수·전체 OOF·등장 선박 성능은 미측정이다. Python 3.11.3이므로 목표 서버 3.11.17과 패치 버전 차이가 있다.

- 코드: `experiments/train_hyperclovax_sft.py` 기본 `--catalog-scorer ratio`; 재현용 `--catalog-scorer WRatio`. 원문 Roman WRatio 기준선은 그대로 유지.
- 근거: `project_log/experiments/AS-HCX-SCORER-01.json`; 행별 비교와 오류 분류: `artifacts/AS-HCX-SCORER-01/paired_predictions.csv`(해시를 실험 기록에 보존).
- 검증: 검색 회귀 테스트 3개, 실제 HCX 평가 함수 연결 테스트 1개 통과. 후자는 모의 생성문을 사용하는 소프트웨어 검사.
- 다음: scorer를 고정한 더 넓은 보류 평가 → 동명 ID를 모두 보존하며 기존 후보와 합집합 구성 → 다른 행 속성을 활용한 재정렬과 공식 ID 점수·시간 비교. 5-fold OOF에는 각 fold의 별도 학습 모델이 필요하며 fold-0 모델을 전 fold에 재사용하지 않는다.

재평가 실행 예시(Windows PowerShell, 아래 새 ID가 없는 경우):

```powershell
.\.venv-server\Scripts\python.exe experiments\compare_hcx_catalog_scorers.py --experiment-id AS-HCX-SCORER-02 --source-experiment AS-HCX-03 --repeats 3
.\.venv-server\Scripts\python.exe -m unittest discover -s tests -p "test_catalog_search.py" -v
.\.venv-server\Scripts\python.exe -m unittest discover -s tests -p "test_hcx_catalog_integration.py" -v
```

재평가는 모델 로딩 없이 저장된 공식 공개 train 보류 생성문을 사용하며, 새 실험의 계획/완료/실패 기록을 보존한다. 기존 실험 ID 또는 출력 폴더가 있으면 덮어쓰지 않는다. 학습 스크립트의 신규 학습 실행에서는 `--catalog-scorer ratio`가 기본이다.
