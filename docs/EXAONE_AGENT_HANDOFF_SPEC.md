# EXAONE 음역 사전 학습·생성·평가 인수인계 명세서

- 문서 번호: `SPEC-LEX-EXAONE-HANDOFF-01`
- 기준 명세: `SPEC-LEX-01 Rev.2` 및 현재 구현·실행 기록
- 작성일: 2026-10-28 KST
- 대상: DACON 236771 행별 선박 ID 매칭 / `MLiP_Project`
- 담당 범위: EXAONE 작업을 이어받는 에이전트
- 이번 인수인계의 완료 지점: **요청한 실제 EXAONE 모델 확보 → fold 분리 LoRA 학습 → 신규 정적 사전 생성 → EXAONE 단독 평가 → DPR 담당자에게 자산 전달**

이 문서는 EXAONE 후속 작업의 실행 지시서다. 문서 작성 단계에서는 새 모델 다운로드·학습·생성·평가를 실행하지 않았다. 아래에서 `planned`로 표시한 작업은 수신 에이전트가 실제 실행 후 상태를 갱신한다. DPR-SYMPHON-03 학습·점수 함수 변경과 제출 ZIP 생성은 담당 범위에 포함하지 않는다.

## 1. 시작 상태와 먼저 읽을 자료

현재 확인한 작업공간은 Windows의 `C:\Projects\ML_midterm`이며 Git 원격은 `kimjy56/MLiP_Project`다. 확인 시 HEAD는 `5bd552f`이고 미커밋·미추적 변경이 많다. HEAD만으로 EXAONE 구현을 복원할 수 있다고 가정하지 않는다. 다른 PC·Linux 작업공간에서는 프로젝트와 데이터의 실제 경로를 다시 확인한다. 원본 데이터를 채팅에 업로드하도록 요청하지 않는다.

먼저 다음을 읽고, 현재 상태와 작업 범위를 한 줄로 알린 뒤 시작한다.

1. 루트 `AGENTS.md`
2. `docs/PROJECT_STATUS.md` — 루트에는 현재 `PROJECT_STATUS.md`가 없다.
3. `docs/TEAM_RULES.md` — 실제 위치와 버전 v4 확인
4. `eval_kit/README.md` — 사용자 적용 기준 1.1과 현재 파일 표기 1.0의 차이 유지
5. `project_log/README.md`, `project_log/EVALUATION_CRITERIA.md`, `project_log/DECISIONS.md`
6. `docs/SPEC_LEX_01_IMPLEMENTATION.md`, `docs/EXAONE_NETWORK_DIAGNOSIS.md`
7. `project_log/experiments/IMPL-EXAONE-LEX-01.json`, `LEX-EXAONE-PURE-01.json`, `NET-EXAONE-01.json`

규칙 문서에 남은 과거 AS 번호 관행보다 사용자 최신 지시를 우선한다. `AS`는 실제 제출 ZIP의 접두사이며, 이번 모델·학습·생성 실험에 다음 AS 번호를 부여하지 않는다.

### 현재 확인된 것

| 항목 | 상태·근거 |
|---|---|
| 요청 모델 | `LGAI-EXAONE/EXAONE-4.0-1.2B` |
| EXAONE 전용 코드 | 구현되어 있음. 아래 파일을 수정·보완하며 이어간다. |
| 과거 기능 검사 | 32개 통과, 실제 생성 사전이 필요한 지정 사례 3개 skip. 이번 문서 작업에서 재실행한 결과는 아니다. |
| 실제 EXAONE 학습 | `LEX-EXAONE-PURE-01`: `blocked_missing_base`, optimizer 0단계 |
| 다운로드 | `EXAONE-BASE-FETCH-01` 실패. 이후 `NET-EXAONE-01`에서 네트워크 접근 거부 진단 |
| 기본 모델 디렉터리 | 이번 확인 시 `artifacts/pretrained/exaone-4.0-1.2b` 없음. 다른 캐시 존재 여부는 재개 시 별도 확인 |
| 새 학습 자산 | 이번 확인 시 `artifacts/LEX-EXAONE-PURE-02` 없음 |
| 생성 사전·정확도 | 실제 전수 사전과 EXAONE 매칭 점수 없음. 생성·평가 기록은 `planned` |
| 제출 기준선 | 사용자 전달 AS-09: `0.9201100425`, 표시 시간 497초. 기존 사전·Dense 등을 쓴 별도 시스템 결과이며 EXAONE 성과가 아님 |

### 기존 데이터 감사 집계

아래는 기존 `EXAONE-DATA-AUDIT-02` 등의 기록이다. 재개 시 같은 파일 해시·정규화 정책에서 재현되는지 확인한다.

| 항목 | 기록된 수 |
|---|---:|
| 공식 train 전체 | 12,222행 |
| 외부 fold 0 학습 / 보류 | 9,850 / 2,372행 |
| 보류 한글 / 비한글 | 239 / 2,133행 |
| 신규 고유 학습 쌍 | 1,573개 |
| 전체 이름 / 동일 어절 수 정렬 / 일반 규칙 쌍 | 269 / 328 / 976개 |
| 카탈로그 행 / 정규화 고유 이름 | 78,595 / 78,592개 |
| 카탈로그 고유 영문 토큰 | 35,418개 |

명세 초안의 약 28,000단어·약 5MB는 측정값이 아니다. 카탈로그의 같은 이름을 가진 여러 `vessel_id`는 별도 후보로 보존한다.

## 2. 에이전트 간 담당 범위

| EXAONE 에이전트가 담당할 파일·자산 | 작업 |
|---|---|
| `name_decoder/deterministic_ko_rules.py` | 일반 철자 독음·숫자·로마자 규칙 점검 |
| `name_decoder/exaone_data.py` | 공식 fold 검사·fit 전용 학습 쌍 |
| `name_decoder/exaone_translit.py` | 실제 base 로딩, LoRA, 제약 디코딩, adapter 복원 |
| `name_decoder/exaone_lexicon.py` | 단어 후보·이름 후보의 두 Track 조합 |
| `experiments/exaone_common.py` | 실행 ID·원자 저장·해시·실패 기록 |
| `experiments/train_exaone_translit.py` | 실제 학습 |
| `experiments/build_exaone_ko_lexicon.py` | pilot·전수 생성·resume |
| `experiments/evaluate_pure_exaone_oof.py` | EXAONE 단독 평가·계보 검사 |
| `experimental_matchers/exaone_lexicon.py` | 독립 정적 JSON matcher |
| `tools/fetch_exaone.py`, `tests/test_exaone*.py` | 모델 확보·관련 검증 |
| 별도 EXAONE 실험 ID의 `artifacts/`, `reports/`, 기록 JSON | 모델·사전·비용·검증 근거 |

DPR 담당 파일인 `phonetic_encoder/` 전체, `experiments/train_bi_encoder.py`, `experiments/export_catalog_embeddings.py`, `experiments/evaluate_bi_encoder_oof.py`, `experiments/evaluate_dpr_robustness.py`, `experimental_matchers/dpr_hybrid/`는 읽기·연동 계약 확인에만 사용한다. EXAONE 구현을 이유로 수정하거나 학습을 실행하지 않는다.

기존 `AR-06/`, `artifacts/AS-*`, `train_AS-*`, 기존 ZIP, 기존 LFM/HyperCLOVA 자산과 완료 실험 기록을 덮어쓰지 않는다. `name_decoder/__init__.py`는 공유 파일이므로 공개 API·지연 import 회귀를 확인하고 필요한 경우에만 최소 수정한다. 공용 상태 문서는 자기 작업 항목만 추가한다. 다른 에이전트의 변경을 되돌리거나 `git add .`, 전체 checkout/reset을 실행하지 않는다.

두 에이전트가 같은 GPU를 쓰는 경우 실제 학습 프로세스·VRAM을 먼저 확인한다. DPR 학습을 종료하지 말고 GPU 사용 순서를 조정한다. EXAONE 오프라인 생성도 GPU를 사용하므로 별도 작업으로 알린다.

## 3. 절대 지켜야 할 데이터·평가 조건

1. 기준 분할은 `eval_kit/folds_group5_seed42.csv`를 그대로 읽는다. 재생성하지 않는다. `sample_id`, 정답 ID, 중복·누락, 선박 ID의 fold 간 분리를 검사하고 해시를 기록한다.
2. fold 0 실험에서는 **fold 0의 모든 선박 ID를 학습 행·학습 쌍·학습에서 fit하는 규칙·선택 과정에서 제외**한다. 카탈로그의 무라벨 영문 이름은 전체를 생성 입력으로 쓸 수 있다.
3. 검증 신고명·정답으로 학습 쌍을 만들거나 생성 후보를 보정하지 않는다. 예측·후보·순위를 확정한 뒤 평가와 사후 진단에만 정답을 읽는다.
4. `IDEA_B_WORD_VARS`, 기존 `catalog_full_ko_map.json`, LFM/HyperCLOVA 생성 사전, 기존 학습 음역표·이름 별칭·호출부호 정답 조회를 EXAONE 단독 경로에서 읽지 않는다. 기존 엔진을 import해 전역 변수만 비우는 방식보다 현재 독립 경로를 유지한다. 읽기 감시·금지 자산 접근 실패 검사 등으로 실제 미접근을 확인한다.
5. 일반 알파벳·숫자 독음은 허용하지만 `HD→현대` 같은 회사명 의미 대응과 특정 보류 사례의 정답은 수동 규칙에 넣지 않는다. `KEO→거` 등 기존 보수적 로마자 대안도 일반 규칙·근거·적용 범위를 기록하고 지정 사례를 맞추기 위해 바꾸지 않는다.
6. 외부 선박 DB·정답 매핑·온라인 번역 API를 사용하지 않는다. 원본 train·카탈로그·비공개 평가 입력을 모델 다운로드 요청이나 외부 API에 보내지 않는다.
7. 기본 checkpoint 선택은 **사전 고정한 마지막 4 epoch**다. DPR 명세의 외부 fold 점수 기반 best 선택을 EXAONE에 가져오지 않는다. best 선택이 필요하면 fit 안에서 선박 그룹별 내부 dev를 먼저 분리하고 별도 ID·설정으로 기록한다.
8. 공개 train은 이미 EDA·여러 개발에 사용되었다. fold 0 점수를 독립 최종 테스트 또는 리더보드의 정확한 예측값으로 표현하지 않는다. 사전을 끄는 것만으로 모든 누수가 제거되었다고 주장하지 않는다.

## 4. 실행 재개의 첫 단계: 환경·네트워크·실제 모델

### 4.1 환경

학습·검증·실행은 Windows `.venv-server\Scripts\python.exe`, Linux `.venv-server/bin/python`을 사용한다. 설치 기준은 각각 `environment/windows-requirements.txt`, `environment/server-requirements.txt`이며 목표 Python은 3.11.17과 지정 16개 패키지다.

`reports/server_environment.json`은 이전 Linux 환경 기록이다. 현재 Windows 환경이 동일하다는 증거로 쓰지 않는다. 기존 구현 기록의 Windows Python은 3.11.15이며 패키지 차이도 기록되어 있다. 재개 환경을 별도 보고서로 검사하고 차이를 명시한다. 임의 패키지 업그레이드나 기존 Python 3.12 환경 삭제를 하지 않는다. 환경 설정이 필요하면 `environment/README.md`를 따른다.

환경 검사에 ZIP을 생성하는 `verify_submission.py`를 쓰지 않는다. 버전·import는 `environment/check_environment.py`, 준비된 실행 bundle 호환성은 `environment/check_model_compatibility.py`의 실제 인자를 확인해 검사한다.

### 4.2 다운로드 장애 처리

과거 진단에서는 외부 DNS 오류 11001과 직접 IP 접속 오류 `WinError 10013`, AppContainer가 확인되었다. 이는 당시 세션의 실측이며 새 세션의 현재 상태는 다시 확인해야 한다.

- 현재 환경에서 `tools/check_network.py`로 연결을 확인한다.
- 네트워크가 허용된 환경에서는 `tools/fetch_exaone.py`로 요청 모델을 받는다.
- 차단된 실행 환경에서는 제한을 우회하거나 DNS·hosts·TLS·방화벽을 완화하지 않는다. 현재 정책에서 실행할 수 없는 경우 정확한 오류와 **해당 환경에서 실행 가능한 호스트 PowerShell 명령**을 남긴다.
- 모델 다운로드만 차단된 경우 공식 snapshot을 이미 보유한 실제 경로 또는 허용된 모델 캐시를 확인한다. 존재하지 않는 경로를 추측하거나 다른 모델로 대체하지 않는다.

공식 base의 model ID·revision·가중치·config·tokenizer 해시를 연결한다. 모델 ID와 비슷한 config만으로 정확한 가중치 출처를 확인했다고 표시하지 않는다. base/tokenizer가 실제로 로드되고 짧은 비지정 예시를 제약 생성하는지 확인한 뒤 학습한다. 이 검사는 기능 확인이며 성능 실험이 아니다.

### 4.3 다른 PC에서 이어갈 때 보완할 인터페이스

현재 adapter manifest는 `base_path` 절대경로를 저장한다. `ExaoneTransliterator.from_adapter()`에는 `local_model` 재지정 기능이 있지만 **생성 CLI에는 이를 전달하는 `--local-model` 인자가 없다**. 다른 PC에서 base 경로가 달라지면 이 작은 인터페이스를 먼저 보완한다. 명세서에 적힌 경로를 맞추려고 과거 manifest를 임의 수정하지 않는다.

기본 경로는 현재대로 두고, 선택 인자로 새 base 경로를 받아 **모든 base/tokenizer 파일 해시가 manifest와 일치할 때만** 복원한다. 같은 에이전트의 관련 파일·테스트·사용 안내만 함께 갱신한다. 아직 없는 CLI 인자를 구현된 명령처럼 실행하지 않는다.

## 5. 학습 명세

### 5.1 학습 쌍

영문 카탈로그명에서 한글 신고명으로 가는 순방향 음역을 학습한다. 다음을 fit 행에서만 만들고 `(source, target)` 기준 중복을 제거한다.

| 종류 | 구성·주의 |
|---|---|
| Full-name | 학습 선박의 영문 전체 이름 → 정규화한 한글 신고명. `호`, `선` 접미사 정리는 일관된 규칙으로 적용 |
| Word-aligned | 한영 어절 수가 같은 쌍의 1:1 어절 대응. 개수 일치만으로 의미 정렬이 보장되지 않으므로 제외 건수·규칙을 기록 |
| General rule | 일반 알파벳 철자 독음·숫자·서수 등 결정론적 쌍. 대량의 선박명 가짜 음역 사전과 구분 |

현재 일반 규칙 쌍은 976개다. 영문 1~3글자 모든 조합을 학습했다는 뜻은 아니므로 실제 생성 범위를 감사한다. 범위를 늘린다면 보류 정답을 쓰지 않는 별도 변경 요소로 기록한다. DPR에서 폐기하려는 Stage A 합성 10만 건을 EXAONE 학습에 가져오지 않는다.

### 5.2 고정 설정

| 항목 | 설정 |
|---|---|
| Base | `LGAI-EXAONE/EXAONE-4.0-1.2B`, 정확한 snapshot 고정 |
| dtype | 지원 시 BF16, 그 외 CUDA FP16. 실제 dtype·GPU·라이브러리 버전 기록 |
| LoRA 대상 | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LoRA | rank 24, alpha 48, dropout 0.05. 모든 layer의 대상 모듈 수 검사 |
| Optimizer | AdamW, learning rate `2e-4`, weight decay `0.01` |
| Epoch | 4, 마지막 epoch 사전 선택 |
| Batch | 유효 64, 기본 microbatch 4, gradient accumulation. 마지막 부분 배치도 올바르게 가중 |
| Scheduler | Cosine, warmup ratio `0.05` |
| Seed / max length | 42 / 256 tokens |
| Loss | 정답 한글 토큰과 실제 EOS만 causal CE. 프롬프트·padding은 `-100` 마스킹 |

프롬프트는 `[EN] {english_name} => [KO] `이다. 학습 target 뒤에는 **실제 tokenizer의 `eos_token_id`**를 붙인다. 다른 모델의 `<|endoftext|>`를 임의로 붙이지 않는다. 과장 길이 쌍은 조용히 자르지 말고 제외 수를 기록한다.

VRAM 부족이면 유효 batch를 유지하며 microbatch를 줄이고 실제 값을 기록한다. 불가피하게 학습 설정을 바꾸면 새 ID를 사용한다. loss·gradient 유한성, optimizer step 수, trainable parameter 수, 학습 전후 adapter 변화, 저장 후 재로딩을 실제 base에서 확인한다. 무작위 tiny 모델 검사로 실제 1.2B 학습 완료를 대체하지 않는다.

## 6. 제약 디코딩과 신규 정적 사전

### 6.1 HangulOnly 디코딩

허용 출력은 `[가-힣0-9 ]+`와 종료용 실제 EOS다. 영어·한자·줄바꿈·특수 문자·replacement 문자는 생성에 허용하지 않는다. 첫 출력이 EOS 또는 공백만 되지 않도록 한다.

현재 토큰별 whitelist가 실제 tokenizer의 한국어 표현을 과도하게 차단하지 않는지 대표 음절·복수 어절·숫자·약어에서 확인한다. 개별 byte 조각 토큰을 제외한 결과와 최종 문자열 검사 결과를 기록한다. 실패를 해결하려고 허용 문자를 임의로 넓히지 않는다. EOS 없이 길이 제한에서 잘린 생성과 빈 출력은 실패로 집계하고 정답으로 채우지 않는다.

### 6.2 두 Track과 후보 상한

| Track | 생성·조합 |
|---|---|
| 단어 | 카탈로그 고유 영문 토큰 전체를 EXAONE으로 beam 4 / 최대 3개 생성. 일반 규칙 대안과 합쳐 중복 제거, 단어당 최대 3개 저장 |
| 전체 이름 | 카탈로그 고유 이름 전체를 EXAONE으로 beam 4 / 최대 2개 문맥 후보 생성 |
| 어절 조합 | 단어 사전의 어절별 상위 2개를 bounded Cartesian 조합. 순위·동점·다양성 규칙을 고정 |
| 최종 map | 문맥 후보 최대 2개와 다른 어절 조합을 합쳐 이름당 최대 4개 |

전체 이름 후보 2개와 어절 조합 2개만으로 모든 발음 조합을 보장할 수 없다. 단어 후보 순위는 규칙 우선순위·beam 순위의 정책 결과이며 보정된 발음 확률이 아니다. 숫자 보존·중복 제거·빈 후보 수·후보 개수 분포를 집계한다.

생성 기본값은 batch 128, `max_new_tokens=48`, seed 42, 최대 후보 4개다. CUDA OOM 시 물리 배치를 줄이고 cache에 실제 배치를 기록한다. 작은 pilot을 별도 ID로 실행한 뒤 전체 생성한다. pilot 이름은 정규화 정렬된 앞부분 등 정답과 무관한 규칙으로 선택한다.

cache의 source 순서·모델/adapter·코드·설정·catalog 해시가 모두 같을 때만 동일 생성 run을 `--resume`한다. 완료 출력 또는 다른 설정의 cache를 덮어쓰지 않는다.

### 6.3 전달할 자산 형식

| 파일 | 형식·필수 조건 |
|---|---|
| `word_exaone_lexicon.json` | `{canonical_english_token: [korean_variant, ...]}`, 최대 3개 |
| `catalog_exaone_ko_map.json` | `{canonical_english_name: [korean_variant, ...]}`, 최대 4개 |
| `catalog_exaone_ko_map.manifest.json` | map·word lexicon·catalog·base·adapter·학습 분할·생성 코드/설정의 식별·해시·커버리지 |
| `fold_0_lora/adapter.pt`, `manifest.json` | 학습된 adapter와 재현·복원 근거 |

위 schema는 현재 구현의 키 구조를 따른다. 영문 이름 키는 78,592개, 선박 ID는 78,595개일 수 있다. 소비자는 이름을 해당 catalog의 모든 ID로 확장한다. 빈 후보를 포함한 이름 키 수와 실제 유효 발음이 있는 이름·ID 수를 각각 보고한다.

manifest의 최소 전달 항목은 `model_id`, base revision 또는 확인 가능한 출처, base/tokenizer 파일 SHA256, adapter/학습 manifest SHA256, train/catalog/fold SHA256, fit/heldout ID 집합 해시, fold 번호, 전체 학습 여부, 생성 코드 snapshot·설정·seed, 이름·단어 수, 누락 수, 파일 크기, `legacy_assets_read=[]`다. 자기 선언만으로 누수 없음이 검증된 것으로 처리하지 않는다.

## 7. 검색·스코어링 및 DPR 연동 계약

### 7.1 1차 산출물: EXAONE 단독

현재 `experimental_matchers/exaone_lexicon.py`를 사용한다. 정적 JSON·catalog로 exact/prefix/suffix/word/3-gram 자모 색인을 만들고 자모 후보 상한은 250으로 유지한다. 이름 비교는 후보 발음별 다음 값의 최댓값이다.

`sim_jamo(q, v) = max_k [0.60 × OSA_sim(jamo_raw(q), jamo_raw(k)) + 0.40 × OSA_sim(jamo_norm(q), jamo_norm(k))]`

현재 독립 scorer는 `10 × name_similarity`에 숫자·톤수·같은 문자열 메타 점수를 더한다. **기존 AS-06/AS-09 scorer와 같은 구현이 아니다.** 처음에는 이 scorer와 설정을 고정하여 측정한다. 시스템 차이를 EXAONE 사전만의 개선량으로 보고하지 않는다. LoRA 효과를 주장하려면 같은 후보·scorer·분할의 base-only 등 대응 비교를 별도 실험으로 수행해야 한다.

기본 평가에는 `--dense-mode none`을 명시한다. 온라인 경로에서 EXAONE/torch/transformers 가중치 로딩·네트워크 호출이 없어야 한다. 온라인 LLM 호출 0회와 JSON 로딩·검색·점수 계산 시간이 0이라는 주장은 구분한다.

### 7.2 2차 전달: DPR 담당자

EXAONE 에이전트는 JSON·manifest·결과·독립 로딩 함수를 전달하고, DPR 담당자가 별도 실험에서 자모 후보와 Dense 후보를 합친다. SPEC-DPR-03의 Dense top K=60이나 `sim_best` 보정 공식을 이 작업의 기본값에 섞지 않는다.

현재 EXAONE 평가 CLI의 선택적 hybrid는 Dense top K=35이며 별도 모델 계보를 쓴다. 이를 사용할 때는 별도 실험 ID와 `--allow-existing-dpr`, checkpoint·embeddings·fold·fit ID 해시가 검증된 `--dense-provenance`가 필요하다. 결과를 EXAONE-only로 표시하지 않는다. 35/60 변경은 수신 DPR 에이전트의 명시적 통합 설정으로 남긴다.

합집합은 가변 후보군이다. **자모 250 ∪ Dense 35/60의 정답 포함률을 Recall@35/@60이라고 부르지 않는다.** 고정 K 지표는 현재 RRF 등 미리 정한 검색 순위의 상위 K에서 별도로 계산하고, Stage-2 최종 순위와 구분한다.

fold 0용 map은 해당 fold 개발 평가용이다. 다른 fold 평가에 재사용하지 않는다. full-train 모델·사전이 필요하면 검증 설정 확정 뒤 별도 ID와 `--final-train`으로 만들고 fold 0 사전을 덮어쓰지 않는다. full-train 사전으로 OOF 점수를 내지 않는다.

## 8. 검증 항목과 성능 목표

다음 검사와 목표 달성 여부를 분리해 보고한다. 목표 미달도 실제 결과·누락 원인·다음 가설을 남긴 정상적인 실험 결과다.

| 검증 항목 | 요구 |
|---|---|
| 데이터·계보 | fold/ID 분리, 변경·다른 fold·full-train map 거부, 해시·생성 계보 검사 |
| Legacy 미접근 | 기존 수동/생성 사전·학습 조회를 실제로 읽지 않음 |
| 실제 모델 | 요청 base 학습·adapter 변경·저장 후 복원 확인 |
| 생성 품질 | 유효 문자·EOS·숫자·중복·빈 출력·최대 4개·전수 범위 확인 |
| 정적 추론 | LLM import/호출 없이 JSON으로 실행 |
| 행 독립성 | 동일 공개 보류 입력의 전체/부분/순서 변경/단독/반복 예측 ID가 일치. 빈 입력·안정된 ID 동점 규칙 확인 |
| 고정 K 검색 목표 | fold 0 보류 한글 Recall@35 ≥ 99.0%: 239행 기준 최소 237행 |
| 최종 매칭 목표 | 같은 한글 집단 Top-1 ≥ 97.5%: 239행 기준 최소 234행 |
| 개별 사례 진단 | BOW CECIL→바우 세실, CNC DREAM→씨엔씨 드림, KEOYOUNG STAR 3→거영 스타 3/거영스타3 |
| ZIP checker | 이번 범위에서 미실행. ZIP 생성 없이 PASS를 주장하지 않음 |

세 지정 사례는 이미 알려진 공개 개발 사례다. 생성 후 진단만 하고, 미포함 시 실패를 기록한다. 목표를 맞추기 위한 수동 문자열 주입·해당 사례 기준 빔 선택·규칙 수정은 하지 않는다.

평가는 한글/비한글 정확도·맞힌 수/표본 수, `0.7 × 한글 + 0.3 × 비한글`, 고정 K 검색 recall, 합집합 포함률, 후보 수 분포, 검색 누락/후보 선택 오류를 분리한다. fold 0은 그룹 분할상 새 선박 집단이며 기존 선박 정확도는 미측정으로 둔다. 기존 선박 보조 시험은 별도 분할·결과로 기록한다.

`eval_kit/evaluate.py`로 fold 0은 `--subset` 평가한다. 2,372행 결과를 전체 5-fold OOF로 부르지 않는다. 전체 OOF는 각 fold에서 독립 학습·생성한 5개 map으로 12,222행을 만든 경우에만 표시한다. fold 0 하나만으로 최종 모델을 선정하지 않는다.

새 개선 비교가 있으면 같은 입력·분할·후보 정책을 명시하고 선박 단위 paired bootstrap CI를 기록한다. 차이의 CI가 0을 포함하면 개선이 확인되지 않았다고 설명한다. 로컬 점수와 서버 점수가 일치하거나 0.95 이상이 된다고 보장하지 않는다.

시간은 다운로드, 데이터 준비, base 로딩, 학습, 단어/전체 이름 생성, JSON 저장, index 구축, 검색/점수, 출력 저장으로 포함 범위를 나눈다. GPU 측정은 동기화와 warmup 조건을 기록한다. 모델/adapter·정적 검색 자산·미생성 ZIP 용량을 구분하며 미측정은 null로 둔다. AS-09 서버 표시 497초와 다른 범위의 로컬 시간으로 가속 배율을 주장하지 않는다.

## 9. 표준 재개 명령

아래는 **PowerShell / 현재 확인된 프로젝트 경로** 예시다. 다른 컴퓨터에서는 먼저 실제 경로를 확인한다. 모두 ZIP을 만들지 않는 명령이다. 실행 전 아래 ID의 기록·출력 존재와 상태를 다시 확인한다. 실패·완료 ID는 재사용하지 않고, `planned`만 현재 실행기의 허용 범위에서 실행으로 전환한다.

```powershell
Set-Location C:\Projects\ML_midterm
$PythonExe = (Resolve-Path '.\.venv-server\Scripts\python.exe').Path
$DataDir = (Resolve-Path '.\data').Path
& $PythonExe -B -X utf8 environment/check_environment.py --output reports/exaone_environment_handoff.json
& $PythonExe -B -X utf8 tools/check_network.py
```

네트워크가 허용되고 base가 없는 경우 모델 다운로드:

```powershell
& $PythonExe -B -X utf8 tools/fetch_exaone.py --experiment-id EXAONE-BASE-FETCH-02
```

base 로딩·GPU 사용 가능 여부를 확인한 다음 데이터 분리 감사와 실제 학습:

```powershell
& $PythonExe -B -X utf8 experiments/train_exaone_translit.py --experiment-id EXAONE-DATA-AUDIT-03 --data-dir $DataDir --fold 0 --prepare-only
& $PythonExe -B -X utf8 -m unittest tests.test_exaone_lexicon tests.test_exaone_generated_acceptance -v
& $PythonExe -B -X utf8 experiments/train_exaone_translit.py --experiment-id LEX-EXAONE-PURE-02 --model-id LGAI-EXAONE/EXAONE-4.0-1.2B --data-dir $DataDir --fold 0 --epochs 4 --batch-size 64 --microbatch-size 4 --offline
```

데이터 감사는 자기 출력 디렉터리를 만들기 때문에 감사 ID와 학습 ID를 공유하지 않는다. 이미 보유한 다른 base 경로는 학습에 기존 `--local-model <실제경로>`로 전달할 수 있다. unittest의 생성 사례 3개 skip은 아직 생성하지 않았다는 뜻이며 성공으로 세지 않는다.

학습·복원이 성공한 뒤 작은 생성 pilot과 별도 전수 생성:

```powershell
& $PythonExe -B -X utf8 experiments/build_exaone_ko_lexicon.py --experiment-id LEX-EXAONE-PILOT-01 --checkpoint artifacts/LEX-EXAONE-PURE-02/fold_0_lora --catalog-csv "$DataDir/vessel_catalog.csv" --limit-names 64 --batch-size 16
& $PythonExe -B -X utf8 experiments/build_exaone_ko_lexicon.py --experiment-id LEX-EXAONE-GENERATE-01 --checkpoint artifacts/LEX-EXAONE-PURE-02/fold_0_lora --catalog-csv "$DataDir/vessel_catalog.csv" --output-word-lex artifacts/LEX-EXAONE-GENERATE-01/word_exaone_lexicon.json --output-catalog-map artifacts/LEX-EXAONE-GENERATE-01/catalog_exaone_ko_map.json
```

학습의 base 절대경로가 현재 PC에 없으면 §4.3의 생성 CLI 경로 재지정 기능을 먼저 구현·검증한다. 전수 생성 중단은 동일 소스·설정·입력·모델일 때만 해당 명령에 `--resume`을 추가한다. pilot 산출물을 전수 사전으로 사용하지 않는다.

생성 후 지정 사례 검사와 EXAONE 단독 fold 0 평가:

```powershell
$env:EXAONE_ACCEPTANCE_MAP = 'artifacts/LEX-EXAONE-GENERATE-01/catalog_exaone_ko_map.json'
& $PythonExe -B -X utf8 -m unittest tests.test_exaone_generated_acceptance -v
& $PythonExe -B -X utf8 experiments/evaluate_pure_exaone_oof.py --experiment-id EVAL-EXAONE-PURE-01 --ko-map artifacts/LEX-EXAONE-GENERATE-01/catalog_exaone_ko_map.json --data-dir $DataDir --fold 0 --dense-mode none
```

각 단계의 종료 코드와 산출물을 확인한 뒤 다음 단계로 진행한다. PowerShell은 외부 명령이 비정상 종료되어도 자동으로 멈추지 않을 수 있으므로, 실패 후 나머지 명령을 연속 실행하지 않는다.

Linux에서는 확인한 프로젝트 루트에서 `.venv-server/bin/python`으로 실행하고 환경 변수 설정도 Linux 셸 문법으로 바꾼다. Windows 절대경로나 PowerShell 문법을 그대로 사용하지 않는다.

## 10. 완료 조건·기록·최종 인수인계

실행 전에 목적·가설·비교 기준·변경 요소를 `planned`로 기록한다. `EXAONE-DATA-AUDIT-03`, `LEX-EXAONE-PURE-02`, pilot, 생성, 평가를 각각 독립 실행으로 보존한다. 현재 생성·평가 ID 01은 기존 `planned` 기록이며, 시작 시 실제 의존 자산을 갱신한다. 과거 실패 01을 새 학습 성공으로 바꾸지 않는다.

각 기록에는 실행 명령, KST 시작·종료, 코드 snapshot/hash, 실제 환경, 데이터·fold hash, seed, 학습·생성·검색 조건, 결과·비용·한계를 남긴다. 실패·OOM·중단도 기록하며 미측정값을 0으로 채우지 않는다. 원본·가중치·OOF를 Git에 추가하지 말고 위치와 SHA256을 기록한다. 네트워크 권한으로 막혔다면 복구했다고 표시하지 않는다.

완료 시 체크리스트:

- [ ] 실제 base/tokenizer와 학습 adapter의 출처·hash·복원을 확인
- [ ] fold 0 분리·legacy 미접근의 실행 근거를 보존
- [ ] pilot과 별도로 전체 카탈로그 map·word lexicon·manifest를 저장
- [ ] 지정 3개 사례의 PASS/FAIL과 전체 생성 커버리지를 보고
- [ ] EXAONE-only 언어별·가중 정확도, Recall@35, 후보 포함률, 비용, 행 독립성을 보고
- [ ] `docs/PROJECT_STATUS.md`, `project_log/README.md`, `project_log/DECISIONS.md`를 실제 상태로 갱신
- [ ] DPR 담당자에게 전달할 map/manifest 실제 경로·hash·대응 fold·집계 결과·제약을 정리
- [ ] ZIP 생성·업로드·공개 push 실행 여부를 기록. 필요한 경우 별도의 사용자 명시 요청에 따름

최종 보고는 완료한 단계, 실제 점수·비용, 전달할 파일, 미완료·차단 요인과 다음 절차를 짧게 정리한다. 학습 성공만으로 전체 작업이 끝났다고 하지 말고 생성·평가까지 진행한다. 실행할 수 없는 외부 의존성이 남으면 막힌 단계와 증거를 명시한다.

권장 커밋 단위는 코드·검증·결과 문서를 구분한다. 예: `fix(exaone): support verified base relocation for generation`, `experiment(exaone): train and evaluate fold-isolated transliteration lexicon`. 기존 다른 작업을 일괄 stage하지 않는다. 이번 문서 작성에서 커밋·push는 실행하지 않았다.

## 11. 다른 에이전트에게 그대로 전달할 시작 지시문

> `docs/EXAONE_AGENT_HANDOFF_SPEC.md`를 읽고 SPEC-LEX-01 Rev.2의 EXAONE 작업을 현재 구현에서 이어받으세요. 먼저 AGENTS.md와 상태·규칙·기록을 확인한 뒤, 공식 EXAONE-4.0-1.2B 실제 파일을 확보하고 fold 0을 제외한 고정 4epoch 학습, 신규 단어·전체 카탈로그 정적 사전 생성, EXAONE 단독 평가까지 진행하세요. 기존 구현에서 필요한 결함을 수정하고 실제 실험을 완료하세요. DPR-SYMPHON-03 담당 파일·GPU 작업과 충돌하지 않도록 하고, DPR 담당자에게 JSON과 해시가 포함된 manifest를 전달할 수 있게 정리하세요. 기존 수동 사전·LFM 사전·학습 정답 조회를 사용하거나 알려진 보류 정답을 주입하지 마세요. 과거 실패 기록과 AS 제출 자산을 보존하고 제출 ZIP은 만들지 마세요. 성능 목표는 미달을 포함해 실측으로 보고하고 작업 중·종료 시 기록을 갱신하세요.
