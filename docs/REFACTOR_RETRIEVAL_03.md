# REFACTOR-RETRIEVAL-03 — 동작 보존 리팩토링

기록: 2026-10-27 KST. 구조 개선과 호환성 검증이며 모델 정확도 개선 실험이 아니다. 새 학습·튜닝·OOF 생성·제출 ZIP 생성/갱신·커밋·push는 수행하지 않았다.

## 적용 문서와 초기 상태

- 실제 규칙 파일 docs/TEAM_RULES.md의 **v4**와 10/22 적용 보충 문서를 확인했다. 실험 ID의 AS 예시는 이후 사용자 명명 지시보다 우선하지 않는다.
- eval_kit/README.md의 실제 표기는 **1.0**이다. 요청한 1.1과 다르고 DECISIONS.md D14의 과거 “1.1 반영” 기록과도 다르다. 현재 파일을 1.1로 재표기하거나 원문 복구를 완료했다고 주장하지 않는다. 정답을 추론에서 제외하고 ZIP은 요청 시에만 생성하는 현재 사용자 지시를 적용했다. 평가기와 fold는 수정하지 않았다.
- 시작 HEAD는 5bd552f. model/matcher.py, LM 코드, 상태 문서 등에 기존 미커밋 변경이 있었으며 실험/인코더 코드 다수가 미추적 상태였다. 비교 기준은 HEAD가 아닌 **작업 시작 당시 실제 파일**이다.
- 초기 Git 상태·추적 파일 해시는 reports/refactoring/REFACTOR-RETRIEVAL-03/에, 전체 초기 diff와 수정 대상 3개 소스의 바이트 동일 사본은 artifacts/REFACTOR-RETRIEVAL-03/에 보존했다. refactor_only.patch는 사용자 변경을 제외한 이번 소스 차이다.

## 호출 관계에 따른 분류

123개 Python 소스의 import/함수/클래스 목록과 해시를 저장했다. 정적 목록은 실행 관측을 대신하지 않는다. 아래는 실제 import, 동적 파일 로딩, CLI 및 기존 실행 기록을 함께 근거로 한 구분이다.

| 역할 | 진입점·연결 | 구분 근거 |
|---|---|---|
| 기존 ML 학습·OOF·추론 | train.py → model.matcher; train_oof.py → train; 루트 script.py → top-level matcher.Matcher | 루트 추론기는 model/을 import 경로에 추가하며 저장 assets의 scorer에 따라 분류기를 로드한다. 최신 제출 ZIP 엔진이라고 간주하지 않는다. |
| 공통 ML 표현·검색 | model/matcher.py → name_processing, ngram_retrieval, torch_indexer, decision_models | 여러 학습·비교 스크립트가 사용한다. 기존 미커밋 dense 기능도 작업 기준에 포함했다. |
| 이름 생성·역방향 점수 | train_name_decoder.py → experiments/train_name_decoder.py; train_noisy_channel.py → name_decoder의 data/channel_training/internal_dev/noisy_channel/lora | LM-LFM25-FULL-R24-01 기록의 실제 명령과 import를 확인했다. 이름 정규화는 ML 표현과 의미가 달라 합치지 않는다. |
| 신경 발음 인코더 | experiments/train_bi_encoder.py, evaluate_bi_encoder_oof.py, export_catalog_embeddings.py → phonetic_encoder | 저장 checkpoint/NPY를 HybridDenseLexicalMatcher도 로드한다. 학습 중 이름 기반 평가와 별도 ID 기반 평가는 분모/결과 키가 다르다. |
| 독립 DPR 엔진 | evaluate_dpr_hybrid.py → experimental_matchers/dpr_hybrid/script.py + phonetic_encoder.hybrid_matcher | AR 계열과 비슷하지만 수동 사전, 한글 지도 스위치, dense 합집합·점수 의미가 다르다. |
| 독립 Jamo/발음 실험 | evaluate_jamo_scoring.py → 보존 AS-06 source + experimental_matchers/jamo_scoring/script.py; evaluate_pronunciation_maps.py → Jamo legacy mode | source/map 해시와 별도 실험 ID에 연결된다. DPR 기본 엔진으로 대체하지 않는다. |
| AR-06 현재 참조 | evaluate_lm_rank_gpu.py, name_decoder/internal_dev.py → AR-06/script.py 동적 로딩 | 현재 호출처가 있다. AR-06/run_ar06.py는 ZIP 작성이 포함된 실행기로 이번에는 실행하지 않았다. |
| 과거 재현/제출 스냅샷 | train_AS-06/07/08, AR-06/train_AR-06, artifacts/AS-06, Jamo reference, 기존 ZIP | 제출/실험 기록과 해시 연결. 현재 코드와 통합하거나 수정하지 않았다. |
| 생성물 | data/, artifacts/, model/의 JSON·가중치, NPY, cache, OOF, 보고서 | 폴더 전체를 생성물로 취급하지 않는다. model/*.py는 실행 소스이며 reports/에도 검증 근거가 있다. |

기계 판독 근거: [소스 목록](../reports/refactoring/REFACTOR-RETRIEVAL-03/source_inventory.json), [동일 함수 42개 그룹](../reports/refactoring/REFACTOR-RETRIEVAL-03/identical_functions.json). AST가 동일해도 전역 상수와 설정이 다를 수 있어 자동 통합 근거로 사용하지 않았다.

## 문제·수정·검증

| 문제와 근거 | 영향 | 이번 변경 | 검증 |
|---|---|---|---|
| train_bi_encoder.evaluate_retrieval와 evaluate_bi_encoder_oof.evaluate_dense_retrieval가 같은 argpartition → argsort 복제 | 정렬 수정 시 후보 순서가 따로 바뀔 수 있음 | phonetic_encoder/retrieval.py의 top_score_indices로 두 연산만 추출 | ties/NaN/Inf/dtype/범위 오류와 기존 평가 결과 필드 정확 일치 |
| CatalogIndex.candidates와 features가 dense 조회·ID 매핑·캐시 각각 구현 | unknown ID·빈 캐시 처리 차이 위험 | 내부 _retrieve_dense_map 추출 | 실제 checkpoint/전체 catalog에서 64행 후보·특징 일치; 중복 ID/없는 ID/빈 이름/None/빈 캐시 재조회 |
| 작업 트리와 HEAD가 다름 | 사용자 변경이 이번 성과로 섞임 | 실제 파일 snapshot과 별도 차이 파일 | 최초 해시 대조, 수정 파일 허용 목록, 보호 파일 검사 |
| 오래된 문서 경로·실행 상태 | 진입점과 ZIP 부작용 오인 | 루트 README 연결 및 이 문서/상태/결정/실험 기록 | 링크·JSON/INDEX·현재 상태 확인 |

phonetic_encoder/hybrid_matcher.py에도 같은 partial sort가 있다. 기존 ZIP 빌더는 모듈명을 명시적으로 복사하므로 새 helper 의존성을 넣으면 패키징 경로도 수정해야 한다. 이 경로와 제출 스냅샷은 보존했다. 새 helper는 학습 중 평가와 별도 평가 두 곳만 사용한다.

## 유지한 계약

- 정규화·자료형·결측, 검색 조건·후보 집합·순서, 특징 이름·순서·차원·float32 값, 점수·가중치·임계값·최고점 선택 유지.
- NumPy 기본 정렬과 partial-sort 동점 유지. 안정 정렬/ID 동점 규칙으로 변경하지 않았다. count=0·음수·범위 초과도 기존 NumPy 동작 유지.
- None 이름의 dense 조회 전 문자열 "None" 변환, 빈 dense 결과의 재조회, 공백 이름에서 캐시 키를 만들지 않는 기존 동작 유지.
- CLI 인자·기본값·help, 평가 함수별 다른 분모/빈 결과 키, 난수·fold·fit 범위·학습/체크포인트 선택, 모델 직렬화·로드 경로 유지.
- 새 외부 의존성이나 진입점 이동 없음. 공식 평가 산식과 평가 코드 변경 없음.

## 실행 결과

지정 .venv-server/Scripts/python.exe를 사용했다. 실제 **Python 3.11.15**, 목표 **3.11.17**이다. reports/server_environment.json은 이전 Linux 3.11.17 기록이며 이번 Windows 실행 증거로 쓰지 않았다.

16개 지정 패키지 중 11개가 버전/import 조건을 통과했다. 차이는 joblib 1.6.0(목표 1.5.2), accelerate 1.15.0(1.9.0), safetensors 0.8.0(0.6.2), sentencepiece 누락(0.2.0), tqdm 4.70.1(4.66.5)이다. 환경 검사 종료 코드는 1이다. CPU/GPU 단순 텐서 검사는 통과했지만 전후 예측 비교는 CPU에서 수행했다. 패키지와 기존 환경은 변경하지 않았다.

| 검증 | 실제 결과 |
|---|---|
| 수정 전 관련 테스트 | 16개 통과 |
| 수정 후 기존+새 경계 테스트 | 23개 통과(기존 16 + 새 7) |
| 기존 dense 통합 테스트 | 2개 통과; 실제 저장 checkpoint 로드 |
| 공개 train 고정 표본 | 배포 G5 fold 0에서 sample_id 정렬 후 한글 32/비한글 32행. 정답/fold는 예측 전에 제거 |
| 저장 분류기, 조회 off/on | 후보·특징·점수·순위·최종 ID 전후 정확 일치; 직접 조회 성공 각각 0/64, 64/64 |
| dense 경로 | 실제 encoder + 기존 catalog embedding + 전체 catalog, 64행(결측/혼합 이름 포함) 후보·특징 정확 일치 |
| dense 최종 선택 | 동점 합성 점수기로 첫 후보 선택 확인. 실제 dense 결합 모델 정확도 평가 아님 |
| 행 독립/순서 복원 | 전후 각 엔진의 전체·17/47행 분할·역순·64개 단건·빈 입력 일치. saved classifier 조회 off/on 및 dense 경로 |
| CLI·시그니처 | 두 수정 CLI의 전후 help 동일, direct/-m 실행 동일, public signatures 동일 |
| 부동소수점 | rtol=atol=0. 점수 차이 0, 최종 ID 차이 0 |
| 보호 파일 | 데이터/fold/평가기/가중치/스냅샷/ZIP/기존 실험 기록 **229개 해시 불변** |
| 구문 점검 | 기존 123개 소스 compile 통과. 엔트리포인트/ZIP 빌더 실행 없음 |

검증 도구 전체 시간은 수정 전 약 64.984초, 수정 후 약 64.141초다. 로딩·인덱스·특징·점수·행 독립 검사·CLI를 포함하며 속도 개선 근거가 아니다. 정확도·후보 포함률·seen/unseen 성능·학습 시간·메모리·서버 성능은 미측정이다.

최초 baseline 도구가 Set scorer에 predict를 호출해 중단됐다. 기존 predict_groups 인터페이스로 도구를 고쳐 재실행했고 기대값/허용오차는 바꾸지 않았다. 모델 소스 수정 전 도구 오류이며 시도 내역을 실험 JSON에 보존했다.

## 재현 명령

PowerShell, 프로젝트 루트 기준:

    .\.venv-server\Scripts\python.exe -B -X utf8 -m unittest tests.test_retrieval_refactor tests.test_matcher_refactor tests.test_name_processing tests.test_retrieval_channels tests.test_switches -v
    .\.venv-server\Scripts\python.exe -B -X utf8 -m unittest tests.test_dense_features -v
    .\.venv-server\Scripts\python.exe -B -X utf8 tools/verify_retrieval_refactor.py

검증 도구는 --before, --data-dir, --folds, --protected, --output을 받는다. 기본 입력은 현재 저장소의 data/와 기존 fold다. 다른 PC에는 실제 입력·snapshot·모델 자산을 준비해 경로를 지정한다. 집계·해시만 저장하며 개별 입력/예측은 메모리에서만 비교한다. ZIP 빌더를 호출하지 않는다.

## 남은 문제·다음 작업

1. **기존 패키지 import 실패**: 새 프로세스의 import model.matcher는 No module named 'name_processing'로 실패하며 변경 전 snapshot도 같다. model/ngram_retrieval.py의 top-level import와 matcher fallback이 관련된다. 기존 진입점처럼 model/을 경로에 추가하면 검증은 통과한다. fallback/import 예외 처리는 별도 호환성 수정으로 분리한다.
2. **목표 환경·문서 버전**: 의도된 eval_kit 1.1 원문을 확인/복구하고, 기존 환경을 보존하는 별도 환경에서 Python 3.11.17/16버전을 검사해야 한다. 이번 환경을 서버 동등으로 부르지 않는다.
3. **미검증 범위**: 전체 5-fold/비공개 test/서버 GPU 비교 없음. 공개 train은 과거 전체 EDA·최종 학습에 사용됐으므로 이번 표본은 독립 최종 테스트가 아니다. 기존 지도/생성기 fold 계보 경고는 기존 감사 기록을 유지하며 이번에 재학습·재검증하지 않았다.
4. **큰 엔진 통합 보류**: AR/Jamo/DPR의 전역 설정·정규화·후보/점수 차이와 제출 구조 때문에 통합하지 않았다. 동일 함수 AST만으로 전체 의미 보존을 증명할 수 없다.

근거: [실험 기록](../project_log/experiments/REFACTOR-RETRIEVAL-03.json), [전후 비교](../reports/refactoring/REFACTOR-RETRIEVAL-03/verification.json), [기존 문제](../reports/refactoring/REFACTOR-RETRIEVAL-03/preexisting_issues.json).

권장 논리적 커밋:

- refactor(retrieval): share exact Top-K ordering and dense candidate mapping
- test(retrieval): verify snapshot parity and row-independent predictions
- docs(refactor): record retrieval audit, environment gaps and handoff

model/matcher.py와 기존 상태 문서에는 이전 변경도 있으므로 파일 전체를 무조건 stage하지 말고 refactor_only.patch와 이번 문서 추가 부분을 기준으로 검토한다. 자동 stage/commit/push는 하지 않았다.
