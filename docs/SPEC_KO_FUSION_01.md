# SPEC-KO-FUSION-01 — AS-09 한글 후보 재정렬 학습

버전: **1.0** · 작성일: **2026-10-29 KST**  
문서 상태: **명세 작성 완료 / 구현·학습·성능 검증은 planned**  
상위 계획: [AS09_IMPROVEMENT_PLAN.md](AS09_IMPROVEMENT_PLAN.md) 1번  
명세 기록: `SPEC-KO-FUSION-01` · 향후 실행 실험: `KO-FUSION-01`

## 1. 목표와 변경 범위

AS-09 구조로 생성한 **동일한 한글 후보 집합**에서 기존 고정 점수 대신 학습된 후보 점수로 최종 `vessel_id`를 선택한다. 이름·숫자·톤수·메타정보·Dense 정보가 충돌할 때의 선택을 개선하는 것이 가설이다. 개선 폭과 서버 목표 점수는 미측정이다.

이번 사용자의 요청은 명세 작성이다. 이 문서의 CLI·모듈·설정은 앞으로 구현할 계약이며 현재 구현됐다는 뜻이 아니다.

| 항목 | KO-FUSION-01 계약 |
|---|---|
| 변경 | 한글 일반 후보 비교 경로의 최종 선택 점수 |
| 고정 | AS-09 검색 규칙, Dense Top-35, 한글 map 사용 조건, lookup 조건, 전처리, 기존 tie-break |
| 인코더 | AS-09/DPR-SYMPHON-02 구조·학습 절차 기준. 비교하는 두 방식은 같은 분할의 동일 인코더·사전 자산을 공유 |
| 비한글 | 기존 AS-09 예측을 그대로 사용. 랭커 학습 대상에서 제외 |
| 호출부호·이름 lookup | 기존 조기 반환 유지. 이 경로의 한글 행도 최종 평가에는 포함 |
| 후보 확장·Dense 범위 확대 | 이번 실험에 포함하지 않음. 후속 `DENSE-COVERAGE-01`과 구분 |
| 신규 생성 모델·tail 가산·예외 사전 | 이번 실험에 포함하지 않음 |
| 데이터 | 공식 공개 train·catalog만 사용. 검증 정답은 사후 라벨 결합·평가에만 사용 |
| 제출물 | 명세·구현·검증 요청을 ZIP 생성 권한으로 확대하지 않음 |

**후보 고정은 분할·조건별 두 예측 방식 사이의 고정**이다. 서로 다른 fold의 학습 자산과 후보가 같아야 한다는 뜻은 아니다. clean과 노이즈 입력도 각각 자기 기준선과 비교한다.

## 2. 기준선·출처 확인 선행 조건

### 2.1 제출 코드 기준점

- 파일: 저장소 루트 `submit_AS-09.zip`
- SHA256: `3c89f0d60d8c14b369b7f4846951cbce53acc39652f2359f72b29466000b0325`
- 사용자 제공 서버 결과: 제출 `117799`, 가중 점수 `0.9201100425`, 표시 시간 `497초`
- ZIP 내부 `script.py`, `model/phonetic_encoder/`, `model/model.json`을 기준 코드로 동결한다. 현재 작업 트리에는 DPR03 변경이 섞여 있으므로 현재 파일을 AS-09 원본으로 간주하지 않는다.
- 실제 서버 업로드 바이트와의 동일성, 언어별 서버 정확도·오답은 미확인이다. 자세한 범위: [AS09_SERVER_RESULT.md](AS09_SERVER_RESULT.md).

### 2.2 전수 자산과 검증 자산 구분

AS-09 ZIP의 `full_encoder.pt`, 전수 조회표·통계·map을 공개 보류에 그대로 적용해 OOF라고 보고하지 않는다. 검증용 `fit_assets(S)`는 공식 학습 기록 중 **선박 집합 S에 속한 행만** 사용해 다음을 구성해야 한다.

1. 조회표·단어/음절 학습표·국적/선종 통계.
2. 지도 학습된 인코더와 mining/checkpoint 선택 자료.
3. 학습된 발음 생성기와 그 출력 map. 전체 공식 카탈로그를 읽는 것은 허용하되 S 밖의 관측 이름·정답 대응은 쓰지 않는다.

범용 사전학습 원본과 라벨을 쓰지 않는 공식 카탈로그 인덱스는 공유할 수 있다. 모든 자산은 입력 집합 해시·코드·설정·가중치·생성 map 해시로 연결한다. 내부 cross-fit에도 같은 원칙을 적용한다.

과거 LFM map은 생성 시점 모델 바인딩에 이력 한계가 있고, 과거 학습 소스와 현재 코드의 차이도 기록돼 있다. 구현 첫 단계에서 다음 상태를 명시한다.

| 상태 | 허용되는 진행·표현 |
|---|---|
| `verified` | 검증용 fit 절차와 split 격리가 확인됨. 같은 자산으로 기준선/랭커 통제 비교 가능 |
| `source_only_verified` | 코드 추출·기능 parity 작업 가능. 누수 없는 성능 비교 완료로 표시 불가 |
| `blocked_lineage` | 누락 자산·설정·이력을 기록하고 성능 단계 중단. DPR03이나 전수 map으로 조용히 대체하지 않음 |

재구성이 원래 AS-09 학습과 달라지면 새 기준선 ID와 차이를 기록한다. 그때 확인되는 것은 **재구성 기준선에 대한 랭커 효과**이며 원래 서버 AS-09의 개선량이 아니다.

## 3. 후보 추출과 예측 계약

### 3.1 추출 위치

ZIP 내부 `script.py`에서 한글 후보의 `total_score`를 계산하고 `scored_cands`에 추가하는 지점에 추적 출력을 연결한다. 후보 생성·점수 계산식을 고치지 않고 값만 수집한다.

기존 `top_candidates` 반환 API는 후보를 잘라내고 점수를 소수점 6자리로 반올림한다. **랭커 학습 캐시로 그대로 사용하면 안 된다.** 최종 후보 전체와 반올림 전 float64 값을 별도 추적 경로로 제공한다. 기존 API는 호환성을 유지한다.

### 3.2 세 가지 행 경로

| `route` | 동작 |
|---|---|
| `non_hangul` | 기존 예측 반환. ranker 호출 0회 |
| `lookup` | 기존 호출부호/이름 조회의 예측 반환. ranker 호출 0회 |
| `hangul_candidates` | 기존 후보 전체에 특징을 붙여 기준선/랭커 선택을 각각 수행 |

후보가 1개면 같은 ID를 반환한다. 원래의 빈 후보 fallback이 만든 후보도 그대로 유지한다. 계측 코드가 새 fallback 후보를 만들면 안 된다.

`non_hangul/lookup`에서는 예측·route metadata만 저장하고 ranker용 후보·점수 배열은 길이 0, 특징 배열은 `(0, 28)`로 둔다. 그 행의 검색 후보가 없었다는 뜻으로 해석하지 않는다. 후보 해시의 비교 대상은 `hangul_candidates` 경로이며 bypass 예측은 별도의 정확 일치 검사 대상이다.

### 3.3 API 초안

아래는 구현할 인터페이스이며 완성된 코드가 아니다.

```python
@dataclass(frozen=True)
class CandidateTrace:
    sample_id: str
    route: str
    baseline_pred_vessel_id: str
    candidate_ids: tuple[str, ...]       # ID 문자열 오름차순, 중복 없음
    legacy_scores: np.ndarray           # float64, 반올림 전
    features: np.ndarray                # float64, shape=(N, 28)
    candidate_set_sha256: str
    feature_schema_sha256: str
    asset_manifest_sha256: str

def trace_row(row, assets) -> CandidateTrace: ...
def label_trace(trace, gold_vessel_id) -> LabeledQuery: ...
def select_with_ranker(trace, ranker) -> str: ...
```

`trace_row`에 gold ID를 인자로 주지 않는다. 특징 캐시 생성이 끝난 뒤 별도 라벨 결합 단계에서 정답을 연결한다. 원본 코드의 `gold_ids` 진단 인자는 후보 생성에 영향을 주지 않아야 하며 추적 경로에서는 사용하지 않는다.

랭커 출력은 확률로 해석하지 않는다. 최종 tie-break는 `(-ranker_score, -legacy_score, vessel_id)`로 고정한다. 비교 기준선은 원래의 `(-legacy_score, vessel_id)`를 사용한다. 모든 ranker 점수가 같으면 원래 기준선 선택과 같아진다.

## 4. 특징 스키마 v1 — 28개

순서는 아래 표 순서로 고정한다. 버전·dtype·결측 규칙과 함께 JSON으로 저장한다. 표의 하위 항목은 적힌 순서대로 펼친다.

| 그룹 | 순서대로 저장할 특징 | 개수·정의 |
|---|---|---|
| 기존 점수 | `legacy_total_score`, `legacy_name_score`, `legacy_num_score`, `legacy_ton_score`, `legacy_meta_score`, `legacy_dense_score` | 6. 원래 계산 결과를 직접 캡처 |
| 기존 이름 정보 | `sim_alpha`, `sim_phon`, `sim_skel`, `sim_tok`, `word_cnt_bonus`, `first_match` | 6. 자모/map 등의 max 반영을 마친 최종 값 |
| 숫자 상태 | `query_num_present`, `candidate_num_present`, `num_compatible` | 3. 기존 분기의 `nstr/kr_ord_nstr`, `c_nstr/c_nstr_ord` 기준 존재 여부. compatible은 `legacy_num_score > 0` |
| 톤수 | `gt_missing`, `igt_missing`, `catalog_gt_missing`, `rel_gt`, `rel_igt` | 5. 유효값은 finite 양수. rel은 `abs(input-catalog_gt)/max(1,catalog_gt)`. 비교 불가하면 NaN |
| 메타 결측 | `flag_missing`, `type_missing` | 2. 기존 정규화가 UNK로 처리하는지 여부 |
| 기존 Dense | `in_dense_top35`, `dense_sim`, `dense_rank_recip`, `dense_zscore`, `dense_margin_top1` | 5. 원래 Dense 결과만 사용. 밖의 후보는 in=0, 나머지 4개 NaN |
| 기존 순위 | `legacy_rank_recip` | 1. 원래 점수 및 ID tie-break 순위의 역수 |

추가 규칙:

- Dense Top-35 밖의 cosine을 새로 계산하지 않는다. NaN은 누락 상태이며 실제 유사도 0과 구분한다. 기존 `legacy_dense_score=0`은 그대로 보존한다.
- `dense_zscore`의 평균·표준편차와 margin은 원래 Top-35 정의 그대로다. 최종 후보 합집합으로 재정규화하지 않는다.
- GT와 IGT는 따로 특징화하되 **기존 톤수 점수 계산에는 손대지 않는다.** 이 특징들은 기존 입력 값으로부터 계산하는 명시된 파생 특징이다.
- raw 이름, sample/vessel/callsign ID, 정답 ID, fold 번호, 데이터 순서, 특정 선박명 매핑을 모델 입력으로 넣지 않는다. ID와 fold는 결합·추적용 metadata에만 둔다.
- 새 tail 유사도, 선명 빈도, 후보 출처별 새 검색 점수, 별도 자모 채널은 v1에 넣지 않는다. 후속 스키마/실험으로 분리한다.
- NaN 이외의 ±Inf, 알 수 없는 feature 순서, shape 불일치는 오류로 중단한다. 자동 0 대체나 열 재정렬을 하지 않는다.

## 5. 캐시와 라벨 형식

권장 저장 형식은 JSON metadata + NumPy 배열이다. 새 Parquet 의존성은 필수로 추가하지 않는다. 큰 캐시는 chunk/memmap으로 처리하고 후보를 잘라 메모리를 줄이지 않는다.

```text
artifacts/KO-FUSION-01/
  reference/manifest.json
  splits/outer_manifest.json
  splits/outer_0_inner_selection.csv
  splits/outer_0_crossfit.csv
  outer_0/
    cache/<role>/<partition>/
      queries.jsonl          # sample_id, route, baseline_pred, query_offsets 등
      candidate_ids.jsonl    # query별 ID 배열, 같은 이름의 서로 다른 ID 보존
      X.npy                  # (전체 후보 수, 28), float64
      legacy_scores.npy
      labels.npy             # 생성 후 라벨 결합 단계에서만 작성
      group_sizes.npy
      manifest.json
    ranker.txt
    ranker_manifest.json
    feature_schema.json
  oof_KO-FUSION-01.csv
  oof_KO-FUSION-01_baseline.csv
  diagnostics/
```

- query = 공개 train의 `sample_id` 한 개다. 한 선박의 여러 관측은 서로 다른 query이지만 같은 분할 그룹에 속한다.
- 후보는 query 안에서 연속 배치한다. `sum(group_sizes) == len(X)`를 검증한다. 학습 시 query 단위로 섞어도 내부 후보를 다른 query와 섞지 않는다.
- 정답 후보가 있으면 정확히 1개를 1, 나머지를 0으로 라벨링한다. 후보 ID 중복이면 실패한다.
- **정답이 후보 밖인 query에는 gold를 추가하지 않는다.** 양성이 없는 query와 단일 후보 query는 랭커 fit에서 제외하되 평가·검색 누락 집계·분모에는 전부 남긴다. 제외 사유별 수를 기록한다.
- `route=lookup/non_hangul`도 query metadata·예측·평가 분모에 남기되 랭커 학습용 후보 배열에서는 제외한다.
- `candidate_set_sha256`: ID 문자열 정렬 목록을 UTF-8 JSON(`ensure_ascii=False`, `separators=(',', ':')`)으로 직렬화한 뒤 SHA256. 점수는 별도의 manifest/배열 해시로 묶는다.
- 캐시는 train/catalog/fold 파일, split별 fit 선박 목록, 코드, upstream 모델·map·설정, feature schema, 노이즈 조건/seed 해시가 모두 맞을 때만 재사용한다.
- 원본·후보·OOF·공개 보류 사례는 Git 제외 내부 자산으로 저장하고 경로·해시만 기록한다. 비공개 평가 행·개별 예측은 디버그 캐시에 저장하지 않는다.
- 전체 OOF의 필수 열은 `sample_id, fold, pred_vessel_id`다. 비한글 검색 전체를 다시 계측하지 않는 v1에서는 전체 OOF에 `gold_rank` 열을 억지로 채우지 않는다. 한글 일반 후보의 최종 정답 순위는 별도 진단 파일에 저장하고 필요하면 해당 subset을 공식 평가기의 `--subset`으로 검사한다. 조회 오류와 비한글 오답을 임의의 검색 누락으로 집계하지 않는다.

## 6. 분할과 학습: upstream까지 제외하는 nested cross-fit

외부 fold는 배포된 `eval_kit/folds_group5_seed42.csv`를 **그대로** 읽는다. 같은 seed로 새로 만들지 않는다. sample_id 결합 누락·중복, ID 정합, 선박 그룹 분리와 파일 해시를 먼저 확인한다.

### 6.1 외부 fold k의 내부 모델 선택

각 k=0..4에서 `T = fold != k`, `V = fold == k`로 둔다.

1. **T를 A/B로 분리**: T 안에서 `StratifiedGroupKFold(5, shuffle=True, random_state=42)`를 적용한다. strata는 한글 여부, groups는 정답 선박 ID다. 내부 fold 0을 B(랭커 선택용), 나머지를 A로 둔다. 분할 파일·선박 목록·해시를 저장한다.
2. **A의 랭커 학습 특징 생성**: A를 같은 strata/groups 원칙의 3-fold(`random_state=20261029`)로 나눈다. 각 `I_j`에 대해 `fit_assets(A \ I_j)`를 만들어 I_j를 추적한다. 합친 결과가 A의 upstream OOF 특징이다.
3. **B의 선택 특징 생성**: `fit_assets(A)`를 만들어 B를 추적한다. B의 관측·정답은 A의 자산과 A의 랭커 학습 특징 생성에 접근하지 않는다.
4. §7의 제한된 후보 설정을 A 특징으로 학습하고 B의 **행별 최종 정확도**로 선택한다. 후보 누락·lookup·비한글을 포함한 B 전체를 평가한다. V는 이 과정에서 열지 않는다.

**금지되는 지름길**: T 전체의 cross-fit 특징을 먼저 만든 뒤 그 특징만 A/B로 나누어 모델 선택하지 않는다. A용 upstream 모델이 B의 정답을 학습하는 간접 경로가 생길 수 있다. 랭커 선택용 B는 upstream fit 단계에서도 제외해야 한다.

### 6.2 선택 후 외부 fold V 예측

1. 선택된 feature schema·파라미터·트리 수를 고정한다.
2. T를 3-fold cross-fit하여 T 전체의 랭커 학습 특징을 다시 만든다. 이 단계의 fit은 T 내부만 사용한다.
3. 고정 설정으로 T의 cross-fit 특징에 최종 랭커를 fit한다.
4. `fit_assets(T)`를 만들어 V의 trace를 **한 번 생성**한다. 기준선과 랭커가 이 trace를 공유한다.
5. V의 두 예측을 확정한 뒤 라벨을 결합해 평가한다. V 결과에 따라 해당 fold의 가중치·특징·checkpoint를 다시 선택하지 않는다.

외부 fold별 내부 선택 결과가 달라도 같은 사전 지정 선택 절차를 적용한 nested 평가로 기록한다. 모든 fold에 가장 잘 맞는 설정을 외부 결과로 고른 뒤 같은 OOF를 독립 검증이라고 부르지 않는다. 이미 공개 train/fold0을 개발에 사용한 이력도 유지한다.

### 6.3 학습 분포와 증강

v1 랭커 학습은 **clean 공개 학습 관측만** 사용한다. class/언어 오버샘플링, 정답 기반 어려운 행 선별, 임의 hard-negative 축소를 하지 않는다. 각 적격 query의 원래 후보 전체를 사용한다.

노이즈 증강을 학습에 추가하려면 후속 새 실험 ID로 분리한다. 이번 clean 학습의 성능과 합산하거나 증강 효과로 재표기하지 않는다. upstream 인코더의 원래 학습 증강은 동결한 AS-09 절차에 따라 처리하며, 랭커 학습 증강과 구분한다.

## 7. LightGBM 설정과 내부 선택

프로젝트 지정 **LightGBM 4.6.0**의 ranking API를 사용한다. query group 크기 배열을 전달하고 relevance는 0/1로 둔다. 모든 후보 sample_weight는 1이다. 한글만 학습하므로 0.7을 일괄 곱해 언어별 가중 학습이라고 부르지 않는다.

```json
{
  "objective": "lambdarank",
  "metric": "ndcg",
  "eval_at": [1],
  "label_gain": [0, 1],
  "lambdarank_truncation_level": 5,
  "lambdarank_norm": true,
  "learning_rate": 0.03,
  "num_leaves": 7,
  "max_depth": 3,
  "min_child_samples": 20,
  "max_bin": 63,
  "reg_alpha": 0.0,
  "reg_lambda": 10.0,
  "subsample": 1.0,
  "subsample_freq": 0,
  "colsample_bytree": 1.0,
  "random_state": 42,
  "deterministic": true,
  "force_col_wise": true,
  "n_jobs": 6,
  "verbosity": -1
}
```

최초 검색 공간은 **reg_lambda ∈ {10, 30}**, **트리 수 ∈ {32, 64, 128}**, 총 6개다. 나머지는 고정한다. 128개까지 학습한 두 모델의 prefix를 평가할 수 있지만 저장된 모델의 실제 트리 수와 사용한 `num_iteration`을 일치시킨다. 조기 종료로 실제 트리 수가 작아지면 실제 값을 기록한다.

모델 선택 점수는 다음 순서다.

1. B 전체에서 `0.7*한글 Top-1 정확도 + 0.3*비한글 정확도` 최대.
2. 같은 점수면 기준선 정답을 오답으로 바꾼 한글 Break 수 최소.
3. 그다음 트리 수가 작은 설정, 그다음 reg_lambda가 큰 설정.

NDCG@1은 보조 지표다. 양성 없는 query·lookup 행을 뺀 NDCG를 최종 가중 정확도로 대체하지 않는다. 새 confidence gate나 `baseline + lambda*ranker` 혼합 계수는 v1에서 추가하지 않는다.

랭커가 기준선보다 나쁜 경우도 외부 검증을 완료해 결과를 보존한다. 외부 fold마다 결과를 보고 유리한 쪽으로 바꾸는 oracle 선택을 하지 않는다. 최종 성능 채택은 §10에서 별도로 판단한다.

## 8. 실행 모듈과 산출 계약

아래 경로는 **신규 구현 대상 제안**이다. 기존 `model/matcher.py`, DPR03 실행 경로, AS-09 ZIP 자체를 수정하는 방식으로 통합하지 않는다.

| 모듈 | 책임 |
|---|---|
| `experimental_matchers/ko_fusion/as09_adapter.py` | 동결한 AS-09 소스 로딩, trace 계측, 기존 동작 parity |
| `experimental_matchers/ko_fusion/features.py` | 28개 특징과 schema/hash 계약 |
| `experimental_matchers/ko_fusion/cache.py` | query 연속 저장, manifest 검증, 라벨 결합 |
| `experimental_matchers/ko_fusion/ranker.py` | ranker fit/load/select와 결정적 tie-break |
| `experiments/prepare_ko_fusion.py` | 출처 감사·nested split·분할별 upstream 자산/특징 준비 |
| `experiments/train_ko_fusion.py` | 내부 6설정 비교와 고정된 외부 fold용 랭커 fit |
| `experiments/evaluate_ko_fusion.py` | baseline/learned paired 예측·공식 평가·보고 |
| `tests/test_ko_fusion.py` | §9의 동작·누수·후보 보존 검사 |

upstream 학습 코드는 감사된 별도 `fit_assets` 어댑터에서 호출한다. 이 명세 때문에 기존 학습 코드를 전면 리팩터링하지 않는다. 환경 검사만을 위해 ZIP 생성 기능이 있는 `verify_submission.py`를 실행하지 않는다.

### 예정 CLI — 현재 실행 명령이 아님

Windows PowerShell 예시다. `ACTUAL_DATA_DIR`는 실행 시 확인한 실제 로컬 데이터 경로로 지정한다. 데이터 경로를 과거 기록에서 추정하지 않는다.

```powershell
$fusionDataDir = 'ACTUAL_DATA_DIR'
.\.venv-server\Scripts\python.exe -B -X utf8 experiments/prepare_ko_fusion.py `
  --experiment-id KO-FUSION-01 --reference-zip submit_AS-09.zip `
  --data-dir $fusionDataDir --folds eval_kit/folds_group5_seed42.csv `
  --outer-folds 0 1 2 3 4 --schema ko_fusion_v1
.\.venv-server\Scripts\python.exe -B -X utf8 experiments/train_ko_fusion.py `
  --experiment-id KO-FUSION-01 --cache-root artifacts/KO-FUSION-01
.\.venv-server\Scripts\python.exe -B -X utf8 experiments/evaluate_ko_fusion.py `
  --experiment-id KO-FUSION-01 --cache-root artifacts/KO-FUSION-01 `
  --data-dir $fusionDataDir --folds eval_kit/folds_group5_seed42.csv
```

Linux는 동일 CLI에 `.venv-server/bin/python`을 사용한다. Windows 현재 Python 3.11.15와 목표 3.11.17의 차이는 실제 환경 보고서로 남긴다. eval_kit 실제 README 1.0과 사용자 기준 1.1의 차이도 적용 완료로 덮지 않는다.

## 9. 구현·검증 완료 기준

### 9.1 성능 실험에 들어가기 전

- [ ] 원본 ZIP·동결 소스·분할별 upstream 자산 provenance 확인.
- [ ] ranker 비활성 adapter와 동결한 원본의 후보 집합·반올림 전 기존 점수·Top-1 일치. 점수 재구성 오차는 float64 `atol=1e-10, rtol=0`으로 확인하고 ID는 정확 일치.
- [ ] public fit/보류의 한글·비한글·lookup·숫자·톤수 결측/충돌·동명 ID·빈 이름을 포함한 사전 고정 parity 표본 저장. 서로 다른 동명 ID가 합쳐지면 실패.
- [ ] 6자리 반올림 때문에 근접 후보 순위가 바뀌는 합성 경계 사례로 full-precision 캐시를 확인.
- [ ] gold ID를 바꾸거나 제거해도 trace와 후보 집합/특징 해시가 같음.
- [ ] split별 upstream 학습 선박, ranker 학습 선박, 평가 선박의 허용 관계를 검사. B/V 접근이 생기면 중단.
- [ ] cache provenance·schema·후보 해시 불일치가 재사용 거부로 이어짐. 잘못된 모델 파일/특징 열은 명시적 오류.
- [ ] ranker 비활성·활성 모두 비한글 및 lookup 행의 ID 차이 0. 단일 후보와 모든 ranker 점수가 같은 tie 사례에서 기존 선택 보존.

### 9.2 전체 외부 평가

- [ ] clean 기준선/랭커 각각 공개 train 전체 12,222행 OOF를 구성. sample_id·fold·ID 범위·중복·누락 검증.
- [ ] 모든 한글 일반 후보 query의 candidate_set_sha256 일치. baseline과 ranker는 같은 trace를 사용. bypass query는 기존 예측 정확 일치.
- [ ] 한글/비한글/가중 정확도, 표본 수, fold별 결과, Fix/Break/유지, lookup/일반 후보/누락 사유 집계.
- [ ] 한글 일반 후보 subset에서 baseline/learned 각각의 최종 정답 순위를 계산. 같은 후보 집합이므로 이 subset의 후보 포함률은 같고 최종 선택 정확도만 달라질 수 있음. 나머지 경로의 미계측 검색 지표는 null/미측정으로 기록.
- [ ] 공식 `eval_kit/evaluate.py`를 수정하지 않고 paired 비교. 예측 차이의 선박 단위 bootstrap 95% CI와 scope 기록.
- [ ] `RobustnessPerturber`의 기존 default/severe 설정을 hash로 동결. seed 20261022, 20261026, 20261027로 조건별 같은 입력을 두 방식에 제공. 노이즈 결과를 모델 선택에 쓰지 않고 clean과 분리 보고.
- [ ] 톤수 결측, 숫자 불일치 등 관측 구간별 결과와 선박 수를 보고. 같은 선박의 반복 변형을 독립 표본으로 합산하지 않음.
- [ ] 서로 다른 공개 query가 충분히 들어간 고정 표본에서 전체/역순/부분집합/단독/반복 예측의 ID 불일치 0. 한글 일반 후보와 lookup을 모두 포함.
- [ ] 특징·ranker 점수는 해당 행의 후보에만 의존. 테스트 전체 빈도·시간·다른 행의 누적 상태를 입력하지 않음.
- [ ] 로딩·인덱스·특징 생성·ranker 계산·저장 시간, 전체 nested upstream 학습 비용, ranker 학습 비용, 모델·검색 자산 크기를 각각 측정. 6 vCPU 이내로 제한.

기존 캐시·모델을 쓸 수 없다면 그 상태를 기록한다. 소규모 실행·기능 검사 통과를 전체 OOF 또는 성능 목표 달성으로 표시하지 않는다.

## 10. 실행 완료와 채택 판단

**실행 완료**는 §9의 산출물·검사가 끝나고 성공/실패·한계가 기록된 상태다. 성능이 하락해도 실행 결과 자체는 완료로 남긴다.

**정확도 개선 채택 근거**는 전체 clean OOF의 paired 가중 차이 95% CI 하한 > 0, 비한글 ID 차이 0, 후보 보존·행 독립성·출처 검사 통과가 모두 필요하다. 스트레스 조건의 유의한 악화가 있으면 기본 채택을 보류하고 trade-off를 보고한다. CI가 0을 포함하면 차이가 불명확한 것이며 통계적 동등성 입증으로 쓰지 않는다.

외부 OOF를 여러 번 보며 설계를 바꾸면 개발 선택 편향이 생긴다. 이 이력과 공개 train 전체 EDA 이력을 명시한다. 로컬 채택 기준 통과가 서버 상승을 보장하지 않는다. 한글 단일 fold만으로 채택하지 않는다.

최종 전수 학습은 구현·검증 결과를 보고 요청받은 후 별도 기록으로 수행한다. 그때도 cross-fit 특징으로 전수 랭커를 학습하고, 전수 upstream 모델은 실제 추론용으로 fit한다. 전수 학습 특징을 같은 전수 upstream 모델로 생성하는 지름길을 쓰지 않는다. 새 ZIP은 별도 명시 요청 시에만 생성한다.

## 11. 기록·인수인계

실제 실행은 `project_log/templates/experiment.json`으로 `KO-FUSION-01` 새 기록을 실행 전에 만든다. 준비·상위 자산 재학습·실패 재시도는 별도 식별자로 보존한다. 과거 AS-09 제출이나 계획 문서를 새 실행 결과로 덮어쓰지 않는다.

필수 기록: 코드/ZIP/데이터/분할/feature schema/가중치 해시, upstream 학습 집합, seed, 원본 명령, 실제 환경, 후보 설정, 제외 query 수, 선택한 6설정 중 결과, 외부 OOF·조건별 성적·CI·Fix/Break, 시간 포함 범위, 자산 위치·크기, 오류·재시도·한계. 미측정 값은 null로 둔다.

공개 보류의 대표 오답만 내부 진단에 저장한다. 확인된 검색 누락/후보 선택/조회 오류와 원인 가설을 구분한다. 문서·현재 상태·의사결정을 같은 작업에서 갱신한다. AS 접두사는 실제 제출 파일을 가리키며 이번 실험에 다음 AS 번호를 미리 붙이지 않는다.

구현 순서: **출처 확인 → adapter parity → trace/cache → nested 학습 자료 → ranker 내부 선택 → 외부 5-fold/강건성/독립성 평가 → 채택 판단**.

## 12. 참고

- [상위 개선 계획](AS09_IMPROVEMENT_PLAN.md), [AS-09 확인 범위](AS09_SERVER_RESULT.md), [DPR03 실험 한계](SPEC_DPR_03_IMPLEMENTATION.md)
- [공통 규칙](TEAM_RULES.md), [평가 도구](../eval_kit/README.md), [기록 템플릿](../project_log/templates/experiment.json)
- [LightGBM 4.6.0 LGBMRanker](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRanker.html): query group·라벨·평가 API
- [LightGBM 4.6.0 파라미터](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html): LambdaRank·결정성 설정

이 문서의 수치 설정은 사전 지정한 **실험 설정**이며 최적값이나 측정된 성능이 아니다.
