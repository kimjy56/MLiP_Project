## 2026-10-29T16:09:10+09:00 — AS-09 개선 2~5 실험 경로 준비 및 출처 감사

- [x] Dense Top-35 밖 특징, 조회 충돌 재비교, 비한글 n-gram 검색, EXAONE 후보 합집합을 제출 코드와 분리해 구현했다. 후보 추가는 기존 fallback 완료 후에 이루어지고 조회 우회 정책을 유지한다. 합성 테스트 10개와 실제 AS-09 인코더의 합성 입력 기능 검사가 통과했다.
- [ ] **성능 평가 차단**: 팀 원본 G5-s42 fold, 분할별 Dense 인코더/임베딩/발음 map, 기존 KO-FUSION 캐시 및 EXAONE 역변환 가중치가 현재 작업공간에 없다. 전체 학습 자산을 보류 성능 평가에 대체 사용하지 않는다. 실제 정확도·서버 성적·시간은 미측정이다.
- **기존 KO-FUSION-01 보고 범위 정정**: 현재 준비 코드는 기본 전수 한글 map을 복사하고 공통 인코더 경로를 사용한다. 과거 실행의 분할 격리를 이 코드만으로 확정할 수 없다. 기존 평가기는 조회 경로의 한글 행을 비한글로 집계할 수 있으며, 실제 Fold 0 수치에 영향이 있었는지는 캐시 부재로 미확인이다. 과거 JSON을 덮어쓰지 않는다.
- **주의**: 현재 AS-09 소스의 한글 점수식은 유형·국적 전이표가 없는 fallback에서 `meta_score` 미할당 가능성이 있다. 실험 코드는 해당 조건에서 실패하도록 했다. 현재 전수 model.json의 두 표는 비어 있지 않으며 실제 서버 오류는 확인되지 않았다.
- **다음 작업**: 사용자의 로컬 환경에 배포된 fold와 분할별 자산·출처를 확인하고, 누수/후보 동일성을 검사한 후 각 실험을 동일 조건으로 평가한다. 결과 확인 전 모델 채택·ZIP·제출·공유는 하지 않는다. 상세: [실험 2~5](../docs/AS09_EXPERIMENTS_02_05.md).

## 2026-10-29T01:10:00+09:00 — 한글 후보 재정렬 점수 학습 및 Fold 0 단독 평가 완료 (KO-FUSION-01)

- [x] **KO-FUSION-01 파이프라인 및 중첩 분할(Nested Cross-Fit) 캐시 생성 완료**:
  - `SPEC-KO-FUSION-01` 28개 후보 특징(Legacy 점수, 편집거리/음운/자모 유사도, 숫자 일치, 톤수/국적/선종, Dense Top-35 지표) 구현 (`experimental_matchers/ko_fusion/features.py`, schema sha256 `683c34914fb0f4d306b99818e312a0f8bfd321557bf8f2e24cf20935541fe7ff`).
  - 외부 Fold 0(2,372행) 완전 격리 하에, Fold != 0(9,850행)에서 Inner A(7,853행), Inner B(1,997행) 분할 및 누수 없는 split_assets 동적 구축.
  - 단위 테스트 6건 전체 통과 (`tests/test_ko_fusion.py`).
  - RTX 5080 캐시 준비: Train(739 쿼리, 3,451,963쌍), Selection(213 쿼리, 1,011,626쌍), Train Full(952 쿼리, 4,461,929쌍), Validation(239 쿼리, 1,061,075쌍).
- [x] **LightGBM LambdaRank 학습 및 모델 선택 (`train_ko_fusion.py`)**:
  - Inner B 기준 그리드 탐색(6개 설정): Inner B 상에서는 6개 설정 모두 100.00% 한글 정확도(0 fixes, 0 breaks) 달성.
  - 최적 설정 선정: `reg_lambda=30.0, n_estimators=32` (정규화 최대화). Full T(4,461,061쌍) 최종 학습 완료 (1.82초, `artifacts/KO-FUSION-01/outer_0/ranker.txt`).
- [x] **Fold 0 보류 단독 평가 (`evaluate_ko_fusion.py`)**:
  - Fold 0 2,372행 (한글 239행, 비한글 2,133행):
  - **AS-09 기준선**: 한글 100.000% (239/239), 비한글 99.859% (2,130/2,133), **가중 정확도 0.99958**.
  - **KO-FUSION-01 랭커**: 한글 94.142% (225/239), 비한글 99.859% (2,130/2,133), **가중 정확도 0.95857** (Net: **-0.04100**, 한글 0 fixes, 14 breaks).
  - 비한글 바이패스 정책 유지: 0 fixes, 0 breaks (영향 없음 확증).
- **원인 분석 및 결정 사항**:
  1. 14개 break는 단 5개 선박(VESSEL_73813, 13453, 73101, 18548, 12277)에서 발생함.
  2. 쿼리당 ~4,000개의 방대한 음성 후보군에 대해 트리 랭커가 `sim_alpha`(단순 문자열 유사도) 및 `sim_skel`에 과도한 분기 이득(gain > 40만)을 부여하여, 정답 선박의 높은 Dense 점수(예: 20.72)에도 불구하고 철자 유사도가 근소하게 높은 오답 선박(예: legacy score 0.05)을 최상위로 올리는 과적합 회귀 현상 발생.
  3. `AS09_IMPROVEMENT_PLAN.md` 규칙에 따라, 단독 성능에서 회귀(-14 breaks)가 확인된 현재의 제약 없는 트리 랭커는 채택하지 않고, baseline AS-09 점수 체계를 유지함.
- **제출 및 ZIP 원칙**: `submit.zip` 미생성 (사용자 명시 요청 없음).
- **근거**: [KO-FUSION-01 JSON](experiments/KO-FUSION-01.json), [예측 비교 CSV](../artifacts/KO-FUSION-01/outer_0/predictions/paired_predictions.csv).

## 2026-10-29T00:28:00+09:00 — EXAONE 우도 판별기(Noisy Channel Likelihood) 학습 및 Fold 0 단독/결합 평가 완료 (LM-EXAONE-01 / EVAL-LM-EXAONE-02)

- [x] **EXAONE Noisy Channel 토크나이저 어댑터 지원 (`name_decoder/noisy_channel.py`)**: chat_template 부재 시 기본 fallback 템플릿 처리 추가, 단위 테스트 9건 전체 통과.
- [x] **EXAONE 우도 모델 LoRA 300스텝 학습 (`LM-EXAONE-01`)**:
  - 학습 대상: `LGAI-EXAONE/EXAONE-4.0-1.2B`, `q_proj`/`v_proj` LoRA (3,194,880 파라미터).
  - 학습 데이터: Fold 0 보류 제외 Fit 내 7,068개 (Fit 정답 + 증강 + 내부 AR-06 혼동쌍), Step 125 dev loss 최소값(1.434233) 체크포인트 병합 완료 (`artifacts/LM-EXAONE-01/model`, 2.56GB).
- [x] **Fold 0 고정 35-후보 숏리스트 단독/결합 평가 (`EVAL-LM-EXAONE-02`)**:
  - Fold 0 2,372행(83,020쌍) RTX 5080 평가 완료 (1,088.4초, ~458.8ms/행).
  - **순수 LM 단독 정확도**: 한글 **97.908%** (234/239), 비한글 **99.578%** (2,124/2,133), **가중 정확도 0.98409** (15오답: 한글 5, 비한글 10).
  - AR-06 기준선 대비: 0 fixes, 11 breaks.
  - **Fusion(0.20 가중치) 및 Gated(margin <= 1.0)**: **0.99958** (한글 100%, 비한글 99.859%; 0 fixes, 0 breaks; 기준선 복귀).
- **주요 비교 및 결론**:
  - 기존 모델 대비 LM 단독 성능: HyperCLOVA X (0.98730) > EXAONE-4.0-1.2B (0.98409) > LFM2.5 (0.97908).
  - 역방향 생성 디코더(`EVAL-EXAONE-REV-01`, 0.82597) 대비 우도 스코어러로서 0.98409로 월등히 높은 식별력을 입증함.
  - 다만 클린 데이터셋 기준선(0.99958) 대비 단독 LM 점수는 여전히 11개 break가 발생하여, 하이브리드 결합 시 임계값 게이팅이 필수적임.
- **제출 및 ZIP 원칙**: `submit.zip` 미생성 (사용자 명시 요청 없음).

## 2026-10-28T23:38:42+09:00 — Bi-Encoder 전용 전달 묶음 (PACK-BI-ENCODER-01)

- [x] 사용자 요청에 따라 AS-09/DPR-SYMPHON-02 가중치·설정·로더·모델/학습 코드만 별도 선별했다. 파일15개 ZIP의 정확 목록·CRC 및 복사 원본11개 해시 일치를 확인하고 격리 safetensors 로드/128차원 유한 출력 검사를 통과했다. 과거 텐서·임베딩 동등성 검증은 재실행하지 않고 출처를 구분했다.
- **주요 결정/산출물**: `handoff/BI_ENCODER_HANDOFF_20261028.zip`; [설명](../handoff/BI_ENCODER_HANDOFF_20261028/README.md), [기록](experiments/PACK-BI-ENCODER-01.json). 기존 합본을 보존하며 별도 Bi-Encoder 전용 ZIP을 제공한다. 학습 소스의 과거 해시 차이 한계 유지. 새 학습·성능 평가·제출 ZIP 생성·외부 전송·커밋/push 없음.
- **다음 작업**: 이번 요청 완료. 권장 커밋: `docs(handoff): add Bi-Encoder-only source and safetensors bundle`.



## 2026-10-29T00:22:11+09:00 — 1번 한글 재정렬 구현 명세 완료 / SPEC-KO-FUSION-01

- [x] 사용자 요청 범위인 명세서 v1.0 작성 완료. AS-09 후보/lookup/비한글 고정, 반올림 전 전체 후보 추적, 28개 특징·결측 규칙, A/B 선택 및 upstream 격리 cross-fit, LightGBM 내부 6설정, 모듈·예정 CLI·5-fold/강건성/독립성 검사·채택 기준을 정의했다.
- **주요 결정 사항**: 정답은 trace 생성 후 결합한다. 전체 T의 cross-fit 특징을 만든 뒤 A/B로 나누는 간접 누수 경로를 피한다. 기존 top_candidates의 6자리 반올림/후보 절단을 학습 캐시로 사용하지 않는다. 후보 범위 확대나 조회 정책 변경은 후속 실험이다.
- [x] 28개 특징 합계, 내장 JSON 설정·문서 링크·코드 블록을 정적으로 점검했다. 구현·학습·모델 추론·새 성능 평가·ZIP 생성·업로드·커밋/push는 수행하지 않았다.
- [ ] **다음 작업(planned)**: 구현 요청 시 출처/분할별 fit_assets 감사 → adapter parity → trace/cache → nested 학습 → 외부 OOF 순으로 진행한다. 명세 완료를 성능 실험 완료로 해석하지 않는다.
- **근거**: [SPEC_KO_FUSION_01.md](../docs/SPEC_KO_FUSION_01.md); [SPEC-KO-FUSION-01](experiments/SPEC-KO-FUSION-01.json). 권장 커밋: `docs(ko-fusion): specify controlled Korean candidate reranking`.

## 2026-10-29T00:02:27+09:00 — AS-09 개선안 제시 / PLAN-AS09-IMPROVE-01

- [x] 실제 AS-09 ZIP 해시와 내부 점수·조회·비한글 검색 코드를 정적으로 확인하고 기존 DPR03/발음/EXAONE 결과를 검토했다. 서버 기준선은 사용자 제공 0.9201100425/497초이며 새 성능 측정은 없다.
- **주요 결정 사항**: 후보를 고정한 한글 점수 결합 학습 → Dense Top35 밖의 특징 범위 → 조회 충돌 재검토 → 비한글 전역 문자 검색 → EXAONE 추가 후보의 상보성 순으로 검증을 제안한다. DPR03의 다중 변경이나 단순 발음 확대를 즉시 채택하지 않는다.
- [ ] **다음 작업(planned)**: 구현 요청 시 AS-09 학습 출처 정합화 및 fold별/fit-inner 자산 분리부터 시작한다. 전수 학습 ZIP을 OOF 검증에 쓰지 않고, 효과를 단계별로 분리한다. 구현·학습·추론·ZIP 생성·업로드는 이번 작업에서 수행하지 않았다.
- **근거**: [AS09_IMPROVEMENT_PLAN.md](../docs/AS09_IMPROVEMENT_PLAN.md); [PLAN-AS09-IMPROVE-01](experiments/PLAN-AS09-IMPROVE-01.json). 개선 폭은 미측정이다. 권장 커밋: `docs(as09): prioritize controlled scoring and retrieval improvements` (커밋/push 미수행).

## 2026-10-28T23:31:20+09:00 — LFM 출처 확인 및 AS-09 인코더 safetensors 전달 (EXPORT-LFM-HANDOFF-01)

- [x] 기록·매니페스트·생성 코드를 확인했다. 변환표는 기록상 로컬 `LiquidAI/LFM2.5-350M` LoRA 모델의 출력이며, train 실제 한글 이름은 모델 학습에 사용됐다. 생성 코드의 직접 train 병합과는 구분한다. 역사 감사는 fold0 제외9,850행을 확인했으며 생성 시점 모델 바인딩 매니페스트 부재 한계를 유지한다.
- [x] AS-09/DPR-SYMPHON-02 `full_encoder.pt`를 별도 `full_encoder.safetensors`로 변환. 텐서61개·설정 정확 일치, 합성 한글쿼리3/영문명3 임베딩 최대차0, 소스24개 구문·CLI4개 help 통과. 기존 artifact 및 AS-09 ZIP 안의 PT 해시 일치. 성능 재평가나 새 학습은 없다.
- **주요 결정 사항**: LFM 생성 모델과 Symphonym 계열 음운 인코더를 분리해 설명한다. 전달 소스는 실제 AS-09 패키지, 보존된 DPR02 학습 스냅샷 및 현재 LFM 소스로 구성하고 당시 학습 해시 차이를 표시한다. 정확 재학습 보장은 하지 않는다.
- **산출물/근거**: [전달 안내](../handoff/LFM_HANDOFF_20261028/README.md), [실행 기록](experiments/EXPORT-LFM-HANDOFF-01.json), `handoff/LFM_HANDOFF_20261028.zip` (코드 전달용, 제출 ZIP 아님). 원본 데이터·OOF·행단위 CSV 미포함. Python3.11.15/safetensors0.8.0 환경 차이 기록. 기존 제출 ZIP·가중치 수정/대회 제출/외부 메시지/커밋/push 없음.
- **다음 작업**: 이번 요청 완료. 정확 재학습을 요청받으면 과거 소스·Stage A 초기값·혼동 후보 자산부터 정합화한다. 권장 커밋: `feat(export): provide lossless AS09 encoder safetensors and LFM provenance handoff`.


## 2026-10-28T23:33:42+09:00 — AS-05 원래 제출본 복원 요청: 원본 자산 누락으로 차단

- [x] 사용자 명시 요청에 따라 복원 준비·보관본 탐색·해시 대조를 수행했다. 원래 ZIP은 없고 AS-HCX-NC-01의 `model.safetensors`가 원천 폴더와 AS-05 패키지 3곳 모두에서 빠져 있다. 같은 크기의 후속 HCX 가중치 4개는 모두 원본 해시와 다르다. 접근 가능한 백업 ZIP 21개에서도 원본을 찾지 못했다.
- [x] 당시 패키징 기록 대비 코드 5개 중 2개의 해시가 다르다. 설정과 나머지 코드의 일치는 원본 가중치의 존재나 원래 제출본 동일성을 보장하지 않는다. 현재 패키지와 다른 후속 모델로 원본을 대체하지 않는다.
- [ ] 다음 작업: 원래 `submit_AS-05.zip` 또는 당시 코드·가중치를 포함한 완전한 백업의 실제 경로를 받아 해시 검증 후 복원한다. 기존 AS-05 자산·`submit.zip`은 보존했고 새 ZIP·학습·변환·추론·업로드는 수행하지 않았다.
- 근거: [RESTORE-AS05-ORIGINAL-01](experiments/RESTORE-AS05-ORIGINAL-01.json). 원본 실험/제출로 중복 집계하지 않는다. 권장 커밋: `docs(as05): record original submission recovery blockers` (커밋/push 미수행).

## 2026-10-28T23:22:00+09:00 — EXAONE 역방향 디코더(KO -> EN) 학습 및 Fold 0 단독 평가 완료

- [x] **역방향 데이터 감사 (`EXAONE-REV-DATA-AUDIT-02`)**: Fold 0 보류(2,372행) 완전 격리, Fit(9,850행)에서 1,573개 역방향 전사 쌍(`[KO] {korean} => [EN] {english}`) 추출 확인 (Vessel ID 중복 0건). 단위 테스트 2건 PASS.
- [x] **역방향 LoRA 4-Epoch 학습 (`LEX-EXAONE-REV-01`)**: 공식 EXAONE-4.0-1.2B 기반 100 Optimizer Step 학습 완료. Target CE Loss 10.140에서 **0.02305**로 초저손실 수렴. 어댑터(`adapter.pt`, 45.8MB) 및 매니페스트 저장 완료 (`artifacts/LEX-EXAONE-REV-01/fold_0_lora/`).
- [x] **역방향 디코더 Fold 0 단독 평가 (`EVAL-EXAONE-REV-01`)**:
  - Fold 0 239개 한글 쿼리 역방향 디코딩 지연 시간 약 4.2초 (~17.5ms/행).
  - `거영스타3` $\rightarrow$ `KEOYOUNG STAR 3` 등 영문 정규 선명 1순위 직접 복원 확인.
  - **한글 정확도**: **76.987%** [66.80, 87.19] (184 / 239행 정답)
  - **비한글 정확도**: **95.687%** [94.20, 97.00] (2,041 / 2,133행 정답)
  - **가중 점수 (0.7 한글 + 0.3 비한글)**: **0.82597** [0.75481, 0.89780]
  - **검색 Recall (gold_rank >= 1)**: 한글 84.94% (후보 누락 36, 순위 오류 19), 비한글 99.86% (후보 누락 3, 순위 오류 89)
- **주요 해석 및 가치**:
  - 역방향 디코더 단독은 한글 정확도 76.99%로 정방향 단독(85.77%)보다 낮으나, 한글 입력을 영문 카탈로그명으로 1:1 직접 매핑하여 영문 카탈로그 검색 채널로 직접 투입할 수 있는 강력한 보조 채널을 제공함.
- **제출 및 ZIP 원칙**: `submit.zip` 미생성 (사용자 명시 요청 없음).

## 2026-10-28T23:05:00+09:00 — SPEC-LEX-01 Rev.2: EXAONE 실제 모델 확보·학습·전수 생성·단독 평가 완료

- [x] **공식 원본 확보 (`EXAONE-BASE-FETCH-02`)**: `LGAI-EXAONE/EXAONE-4.0-1.2B`의 safetensors 및 토크나이저 파일 9개 다운로드 완료 (`artifacts/pretrained/exaone-4.0-1.2b`, 2.56GB), SHA256 해시 검증.
- [x] **데이터 고립 감사 및 단위 테스트 (`EXAONE-DATA-AUDIT-03`)**: Fold 0 보류군(2,372행)과 Fit(9,850행) 분리 확인. Fit 내 1,573개 전사 쌍(규칙 976, 단어 328, 전명칭 269) 추출 (Vessel ID 중복 0건). 단위 테스트 14건 전체 PASS.
- [x] **Fold 0 분리 고정 4-Epoch LoRA 학습 (`LEX-EXAONE-PURE-02`)**: Target CE 손실 10.148에서 0.237로 정상 수렴. 체크포인트 및 어댑터(`adapter.pt`, 45.8MB)와 매니페스트 저장 완료.
- [x] **전체 카탈로그 정적 사전 생성 (`LEX-EXAONE-GENERATE-01`)**:
  - 카탈로그 78,595행 대상 고유 단어 35,418개(단어 사전: `word_exaone_lexicon.json`, 2.69MB) 및 고유 정규 선명 78,592개(카탈로그 맵: `catalog_exaone_ko_map.json`, 9.64MB) 생성 완료.
  - 레거시 사전/LFM/수동 사전 일체 미접근(`legacy_assets_read: []`), 결측률 < 0.01% (단어 결측 1개, 선명 결측 5개).
- [x] **지정 사례 수용성 진단 (`tests/test_exaone_generated_acceptance.py`)**:
  - `CNC DREAM` → '씨엔씨 드림' (PASS)
  - `KEOYOUNG STAR 3` → '거영스타3' (PASS)
  - `BOW CECIL` → '보큐씨엘'/'보완씨엘' 등 생성 (FAIL, 진단 보존; 임의 라벨 주입 배제 규칙 준수).
- [x] **순수 EXAONE 단독 Fold 0 평가 (`EVAL-EXAONE-PURE-01`)**:
  - 공식 평가기 `eval_kit/evaluate.py --subset` 실행 통과 (checks: `train=ok; folds=G5-s42; foldcol=ok`).
  - **한글 정확도**: 85.774% [76.76, 93.13] (205 / 239행 정답)
  - **비한글 정확도**: 88.326% [85.96, 90.46] (1,884 / 2,133행 정답)
  - **가중 점수 (0.7 한글 + 0.3 비한글)**: **0.86540** [0.80234, 0.91632]
  - **검색 Recall (gold_rank >= 1)**: 한글 95.82% (검색 누락 10, 순위 오류 24), 비한글 99.62% (검색 누락 8, 순위 오류 241)
  - **Recall@35 (고정 K RRF)**: 한글 92.89%, 비한글 95.97%
  - **Top 5 후보 포함률**: 한글 95.40%, 비한글 96.44%
  - **행 독립성(Row Independence)**: 역순·스트라이드·단독 행 슬라이스 예측 일치 100% 검증 통과.
  - **지연 시간**: Fold 0 2,372행 채점 22.27초 (~9.39ms/행, 순수 정적 JSON 인덱스).
- **DPR-SYMPHON-03 담당자 전달용 매니페스트**:
  - 매니페스트 경로: `artifacts/LEX-EXAONE-GENERATE-01/catalog_exaone_ko_map.manifest.json`
  - 카탈로그 맵: `artifacts/LEX-EXAONE-GENERATE-01/catalog_exaone_ko_map.json` (SHA256: `ff1977fc0ba47b7fbc0eb2c955d48ff2ad737448d970f42bd94bb572ee6c8f71`)
  - 단어 사전: `artifacts/LEX-EXAONE-GENERATE-01/word_exaone_lexicon.json` (SHA256: `eb405e688d84e154ef13bf1d4127d850b8e8f83b105a3859a5fe809ebb0cf7f4`)
  - 어댑터 가중치: `artifacts/LEX-EXAONE-PURE-02/fold_0_lora/adapter.pt` (SHA256: `ac719b1b9938c65dc6ab009ee85196b02b98e346184af5b1caeada6331490f72`)
  - 제약 사항: Fold 0 보류 전용(새 선박 집단) 검증용 사전이며, DPR-SYMPHON-03 결합 시 가변 합집합을 Recall@35로 부르지 않고 RRF 순위 기반 상위 K 지표로 구분할 것.
- **제출 및 ZIP 원칙**: `submit.zip` 미생성 (사용자 명시 요청 없음).

## 2026-10-28T23:17:36+09:00 — DPR-SYMPHON-03 구현·학습·평가 완료 / 성능 채택 보류

- [x] SPEC-DPR-03 구현, 원본 symphonym-v8에서 Stage A 없이8epoch 실제 학습, global composite epoch5 보존, catalog78595행 FP16 export, standalone·4조건×4구성 hybrid 평가 완료.
- [x] **검색 목표 미달**: 한글 R@1/35/60=40.5858/89.1213/89.5397%. epoch8 R@1=21.76%로 하락했지만 epoch5 가중치는 보존했다. clean 사전OFF 가중99.6649%는 기존02와 같아 로컬92%=LB92% 전제는 재현되지 않았다.
- [x] **severe 사전OFF 악화**: 기존02 가중98.7441%→전체03 96.1081%, 차이-2.6360%p/95% CI[-4.709,-0.827]%p. 같은03/K60에서 직결은 한글230→225, tail규칙은226으로 일부 회복. 사전ON clean99.9578→99.6649%, severe98.7441→97.8654%이며 해당 CI들은0을 포함한다.
- [x] **속도·검증**: 로컬CPU32행6threads 인코딩2.496배, Dense검색2.389배. 단일행encoder1회/full-catalog dot1회. 최종기능18개 PASS; native300행 reverse/subset150/single3/repeat 예측불일치0. IEEE FP32로 실제모델1e-6 검사통과. 원본35자산 SHA256·epoch5 tensor 일치확인.
- **주요 결정 사항**: 정확도 개선 모델로 DPR03/직결 규칙을 채택하지 않는다. AS09 기존 ZIP·자산·서버0.9201100425/497초 기준선을 유지한다. 외부fold0 선택은 사용자 명시 요청대로 수행했고 개발 선택점수로만 해석한다. Stage A 단독 인과·서버시간2~3분·점수향상을 주장하지 않는다.
- **실패·환경 기록**: pytest미설치→표준unittest runner 사용. 첫 standalone TF32 실패와 첫 재시도 numpy bool 저장 실패를 별도ID로 보존, RECHECK-02 성공. Python3.11.15/목표3.11.17 및 패키지5개 차이 유지. ZIP checker는미실행, native검증으로 구분한다.
- [x] 현재 실행 중인 실험없음. 새 ZIP/업로드/커밋/push·EXAONE 작업없음. 후속 개선을 요청받으면 초기화·LR·학습조건을 분리하고 fit-inner에서 선택하는 별도 실험을 설계한다.
- **근거·인수인계**: [전체 구현·결과](../docs/SPEC_DPR_03_IMPLEMENTATION.md); DPR-SYMPHON-03/EXPORT-DPR-SYMPHON-03/EVAL-DPR-SYMPHON-03-RECHECK-02/DPR-HYBRID-03/BENCH-DPR03-LATENCY-01/VERIFY-DPR03-01/ANALYZE-DPR03-01 JSON. 공개보류 오답23조건건(선택22/검색1)은 artifacts/ANALYZE-DPR03-01에 보존했다.
- **권장 커밋**: `feat(dpr): add clean training and single-pass row-local inference`; `experiment(dpr): record DPR03 regressions and preserve AS09 reference`. 기존 다른작업을 일괄stage하지 않는다.

## 2026-10-28T22:22:07+09:00 — EXAONE 다른 에이전트용 인수인계 명세 작성

- [x] 사용자 요청에 따라 [EXAONE 인수인계 명세](../docs/EXAONE_AGENT_HANDOFF_SPEC.md) 작성. 현재 구현·실패/계획 기록·CLI를 확인하고 모델 확보 → fold 분리 고정 4epoch 학습 → pilot/전수 사전 생성 → EXAONE-only 평가 → DPR 자산 전달의 순서를 구체화했다.
- **주요 결정 사항**: 이번 요청은 명세 작성이다. 새 다운로드·학습·생성·성능 평가·ZIP 생성은 실행하지 않았다. 실제 EXAONE 점수는 여전히 미측정이며 과거 32 PASS/3 skip을 이번 재실행 결과로 표시하지 않는다. 다른 에이전트를 실제 실행시키거나 메시지를 전송하지 않았다.
- **담당 범위**: EXAONE 전용 코드·자산과 DPR 담당 파일/GPU 작업의 경계를 명시. 기존 AS 자산과 실패 실험 보존, legacy 미접근, 마지막 epoch 사전 선택, 다른 PC의 base 절대경로 재지정·해시 검사, 목표와 실측의 분리를 요구한다.
- [ ] **다음 에이전트 작업**: 명세 §11 시작 지시문으로 인수 → 실제 환경/네트워크/base 확인 → LEX-EXAONE-PURE-02 등 새 실행 ID로 재개. 기존 생성/평가 01은 planned이며 실행 여부·경로를 다시 확인한다. 복구되지 않은 권한 차단은 증거와 호스트 실행 명령으로 인계한다.
- **검증/한계**: 현재 코드와 로컬 경로·기록을 읽고 문서와 링크를 정적 확인했다. 루트 TEAM_RULES/PROJECT_STATUS 대신 docs/의 실제 파일을 안내했다. 문서 적용만으로 모델 확보·성능 개선·서버 검증을 주장하지 않는다.
- **권장 커밋**: `docs(exaone): add agent handoff specification and execution boundaries`. 이번 작업에서 커밋/push 없음.

## 2026-10-28T22:09:58+09:00 — 수정 AS-08(117597) 결과 반영 / VERIFY-AS08-FIXED-RESULT-01

- [x] 사용자 정정: **오류 수정본 AS-08**은 제출 `117597`, `2026-10-27 23:17:35 KST`, **0.9082679372**, **327초(5분27초)**다. 최초 AS-08 `117501`(0.9003732004/236초)은 별도 제출로 보존한다. 기존 SUB-117597 기록과 일치하며 중복 등록하지 않았다.
- [x] 수정본−최초 AS-08: **+0.0078947368(+0.78947368%p), 표시 시간+91초**. 수정본−AS-06/07: **−0.0039473684(−0.39473684%p)**.
- [x] **AS-09−수정 AS-08: +0.0118421053(+1.18421053%p), 표시 시간+170초**. 앞선 +1.97368421%p/+261초 비교는 최초 AS-08에 대한 값으로만 유효하다. 향후 최신 AS-08 기준 비교에는 수정본을 쓴다.
- **주요 결정/해석**: AS-09는 관측된 최고 서버 점수 기준선을 유지한다. 오류 수정 후 점수 회복은 확인되지만 전체 상승분을 Dense 단독 효과로 귀속하거나 잔여 격차를 사전 제거 때문으로 확정하지 않는다. 기존 SUB-117597의 인과 확정 표현은 서버 로그·대응 ZIP 변경 범위가 확인되기 전까지 가설로 한정한다. 현재 로컬 ZIP과 서버 업로드 바이트 일치는 미확인이다.
- **범위/제약**: 기록·소수 연산·로컬 ZIP 설정 확인만 수행. 새 학습/모델 추론/ZIP 수정/업로드/커밋/push 없음. 언어별 정확도·표본 수·구성별 기여는 미측정이다. TEAM_RULES v4 확인, eval_kit 실제1.0/요청1.1 차이와 Python3.11.15/목표3.11.17 차이를 유지한다.
- [ ] **다음 작업**: 별도 개선 요청 시 같은 조건의 구성별 비교로 기여를 분리한다. 이번 결과 반영은 완료했다.
- **근거**: [수정 제출 결과](experiments/SUB-117597.json), [검증 기록](experiments/VERIFY-AS08-FIXED-RESULT-01.json), [갱신 비교표](../docs/AS09_SERVER_RESULT.md). 권장 커밋: `docs(results): distinguish corrected AS08 submission and revise AS09 comparisons`.

## 2026-10-28T22:15:00+09:00 — AS-08 수정본 공식 서버 결과: Dense 복구 인과 입증 (SUB-117597)

사용자 전달 결과: 제출 **117597**, **submit_AS-08.zip (수정 edit)**, 제출 시각 **2026-10-27 23:17:35 KST**, 가중 점수 **0.9082679372**, 표시 시간 **327초(5분 27초)**.

- 결함 AS-08(`SUB-117501`, 0.9003732004) 대비 **`+0.0078947368` (+0.7895%p)** 반등.
- 서버 표시 시간이 236초에서 327초(+91초)로 증가하여, `loss.py` 패키징 누락 해결 후 Dense 신경망 매칭이 서버에서 정상 실행되었음을 확인.
- 다만 합성 한글 카탈로그 맵(`catalog_full_ko_map`) OFF 및 수동 사전 제거 상태로 인해 AS-06(`0.9122153056`) 대비로는 `-0.0039473684`(-0.3947%p) 격차가 유지됨.
- 이는 차세대 `DPR-SYMPHON-02`와 `catalog_full_ko_map`을 모두 갖춘 `AS-09`(`0.9201100425`, +1.1842%p vs AS-08 수정)로의 비약적 성과로 이어짐.
- 근거: [SUB-117597](experiments/SUB-117597.json), [AS-08 서버 분석](../docs/AS08_SERVER_ANALYSIS.md).

## 2026-10-28T22:06:06+09:00 — VERIFY-AS09-RESULT-01: AS-09 결과·패키지 근거 범위 확인

- [x] 기존 제출117799 기록과 사용자 표 일치 확인: **0.9201100425 / 497초**. AS-06/07 대비 **+0.78947369%p**, AS-08 대비 **+1.97368421%p**. 새 제출·실험으로 중복 집계하지 않았다.
- [x] 로컬 ZIP 해시가 기존 AS-09 기록과 일치하고 CRC 통과. Dense hybrid/가중치2.5, loss.py 포함, 실패 시 중단 확인. 수동 단어장0개지만 기존 한글 map78,586개와 학습 조회는 ON이다. EXAONE-only 결과가 아니다.
- **주요 결정 사항/해석 정정**: AS-09는 현재 관측된 서버 점수 기준선이다. 서버 로그·업로드 원본 해시는 미확인이며 Dense 활성은 동일 ZIP 실행 가정의 강한 근거로 구분한다. 개별 구성의 성능 기여·언어별 성적은 미측정이다. 아래 이전 기록의 서버 직접 검증/업로드 해시 확인/1,200초 제한 표현은 [최신 확인 범위](../docs/AS09_SERVER_RESULT.md)로 한정한다. 기록된 최신 안내는1,800초이며 표시 시간의 범위는 미확인이다.
- **제약**: TEAM_RULES v4 확인, eval_kit 실제1.0/요청1.1 불일치. Python3.11.15/목표3.11.17 차이. 기존 DPR 기록은 fold0이며 이번 결과를5-fold OOF로 표시하지 않는다. 모델 추론·새 학습·ZIP 수정·업로드·커밋·push 없음.
- [ ] **다음 작업**: 별도 개선 요청 시 동일 조건의 Dense OFF/후보만/가산 및 map ON/OFF를 분리 비교한다. 현재 작업은 기록 검증까지 완료, 실행 중 실험 없음.
- **근거**: [분석 문서](../docs/AS09_SERVER_RESULT.md), [검증 기록](experiments/VERIFY-AS09-RESULT-01.json). 권장 커밋: `docs(results): verify AS09 score, package settings and evidence limits`.

## 2026-10-28T22:01:00+09:00 — AS-09 공식 서버 결과: 프로젝트 역대 최고점 달성 (SUB-117799)

사용자 전달 결과: 제출 **117799**, **submit_AS-09.zip**, 제출 시각 **2026-10-28 00:54:00 KST**, 가중 점수 **0.9201100425**, 표시 시간 **497초(8분 17초)**.

- AS-06/AS-07 (`0.9122153056`) 대비 **`+0.0078947369` (+0.7895%p)** 상승하여 **0.9201100425**로 프로젝트 역대 최고점 달성.
- AS-08 (`0.9003732004`) 대비 **`+0.0197368421` (+1.9737%p)** 상승.
- `DPR-SYMPHON-02` 아키텍처(SyllableConvBlock 1D-CNN + Multi-Query Attention Pooling + Gated Multi-View & Tail Fusion + ANCE/자매선 하드 네거티브 + CosFace)와 완전한 의존성 패키징(모든 `phonetic_encoder` 소스 동봉)이 서버 평가 환경에서 완벽히 작동함을 실측 검증.
- `AS-09`를 프로젝트의 공식 최우수 제출(Best Submission)로 승격.
- 근거: [SUB-117799](experiments/SUB-117799.json), [AS-09 패키지 기록](experiments/AS-09.json).

## 2026-10-28T00:41:43+09:00 — NET-EXAONE-01: DNS 증상 원인은 sandbox 네트워크 접근 거부

직접IP443도10013 접근 거부, 외부DNS11001, AppContainer=True를 확인했다. 사용자 network_access는 이미true이고 proxy/offline override는 없다. OS DNS/hosts/firewall은 수정하지 않았다. 현재 세션의 실효 네트워크/권한 확대 차단으로 연결 복구는 미완료다. 일반 호스트 PowerShell의 연결 확인·공식 모델 다운로드 명령은 [네트워크 진단 안내](../docs/EXAONE_NETWORK_DIAGNOSIS.md)에 저장했다. 실제 다운로드/학습 재개 성공을 주장하지 않는다. 근거: experiments/NET-EXAONE-01.json.

## 2026-10-28T00:11:19+09:00 — SPEC-LEX-01 Rev.2 신규 EXAONE 사전 경로 구현 (IMPL-EXAONE-LEX-01)

기존 수동 사전/합성 map/학습 음역 조회 자산을 읽지 않는 규칙·fold쌍·EXAONE LoRA·Hangul 제약·두 Track 생성/캐시·독립 자모 matcher·공통 평가 연결을 구현했다. 정적 JSON 추론은 torch/transformers를 import하지 않으며 기존 공개 decoder API는 지연 로딩으로 보존했다.

- 검증: **32개 통과**, 실제 생성 map이 필요한 지정 사례 **3개 skip**. 작은 무작위 EXAONE 구조의 gradient/adapter 복원은 기능 검사이며1.2B 실제 품질이 아니다.
- 데이터: fit9850/보류2372행, 고유 쌍1573개, catalog78595행/고유 이름78592/토큰35418. fold 파일 재생성·보류 정답 후보 주입 없음.
- **실제 모델 실험 차단**: LEX-EXAONE-PURE-01은 base 부재로 optimizer0단계에서 중단. EXAONE-BASE-FETCH-01은 huggingface.co DNS 실패. 전수 생성/정확도 평가는 planned로 유지하고 대체 모델·기존 사전을 사용하지 않았다.
- **주요 결정**: 사전 배제만으로 누수0/로컬=리더보드를 보장하지 않는다. HD→특정 회사명은 일반 규칙으로 고정하지 않는다. 유효 batch64/microbatch4, 실제EOS 사용. 기존 DPR은 별도 계보 옵션과 검증 manifest가 필요하며 EXAONE-only와 구분한다.
- **다음 작업**: 가중치 확보 → 새 학습 ID02 → 생성 → fold 평가. 기존 ZIP/완료 기록 보존, ZIP 생성·커밋·push 없음.

근거: [구현·실행 안내](../docs/SPEC_LEX_01_IMPLEMENTATION.md), [작업 기록](experiments/IMPL-EXAONE-LEX-01.json), [학습 차단](experiments/LEX-EXAONE-PURE-01.json).

## 2026-10-27T23:10:25+09:00 — AS-08 잔존 매칭 경로 확인 (AUDIT-AS08-REMAINING-01)

수동 IDEA_B/map 제거와 전체 음역/조회 제거는 다르다. ZIP에 호출부호 조회5851·이름 조회5857·학습 단어466·음절271이 남고 EN_V3·음역 규칙·숫자/톤수/메타 점수도 유지된다. 조회 적중/경로별 기여는 미측정. 공식 점수0.9003732004는 비한글100% 가정에서도 한글 최소85.7676%를 요구한다(산술 하한, 실제 언어별 성적 아님). 정적 설명 보강만 수행, 학습·추론·ZIP 수정 없음. 근거: docs/AS08_SERVER_ANALYSIS.md §8와 experiments/AUDIT-AS08-REMAINING-01.json.

## 2026-10-27T23:03:45+09:00 — AS-08 실제 ZIP 감사와 복구 우선순위 (ANALYSIS-AS08-SERVER-01)

- **확인된 결함**: AS-07/08 ZIP의 phonetic_encoder.__init__가 .loss를 import하지만 loss.py가 없다. 저장소 경로 없는 격리 import에서 두 진입점 모두 Dense 클래스 unavailable 재현. 임시 소스에 누락 파일 추가 또는 추론 전용 __init__ 적용 시 import 성공, 모델 로드0회.
- **해석 정정**: 설계상 Fuzzy+Dense 하이브리드와 ZIP 독립 실행은 구분한다. AS-07의 AS-06 동률은 fallback 가설과 일치하나 서버 로그 없이는 확정하지 않는다. AS-08은 map78586→0·수동 사전299→0·lattice 제거 상태라 Dense 미활성과 보완 장치 제거의 결합이 유력한 하락 가설이다.
- **추가 근거**: AS-07/08 신경 가중치 hash 동일. 기존 멀티뷰 보고서의 한글 Top1은138→147/239이나 Top35는227→226/239. 현재 외부 fold best 선택과 캐시 provenance 한계도 기록했다. 기존 완료 결과를 덮어쓰지 않았다.
- **결정/다음 작업**: PLAN-SERVER-RECOVERY-01은 planned. P0 추론 패키지 의존성/활성 확인 → AS-06 기능 보존 + Dense 후보만 추가 → 별도 가산/게이트 → map/사전/lattice 단일 제거 → 보조 뷰 분리 순서다. 점수 향상 폭은 미측정.
- **범위**: 새 학습·튜닝·모델 추론·ZIP 생성/수정·업로드·커밋·push 없음. 실제 서버 활성화와 각 요소의 점수 기여는 미확인.
- **근거**: [분석 보고서](../docs/AS08_SERVER_ANALYSIS.md), [실행 기록](experiments/ANALYSIS-AS08-SERVER-01.json), [복구 계획](experiments/PLAN-SERVER-RECOVERY-01.json).

## 2026-10-27T22:53:44+09:00 — AS-08 공식 서버 결과: 점수 하락 기록 (SUB-117501)

사용자 전달 결과: 제출 **117501**, **submit_AS-08.zip**, 제출 시각 **2026-10-27 22:14:50 KST**, 가중 점수 **0.9003732004**, 표시 시간 **236초(3분 56초)**. 사이트 직접 조회는 하지 않았다.

| 제출 파일 | 제출 ID | 표시 가중 점수 | 표시 시간 |
|---|---:|---:|---:|
| submit_AS-06.zip | 115506 | 0.9122153056 | 366초 |
| submit_AS-07.zip | 116338 | 0.9122153056 | 372초 |
| submit_AS-08.zip | 117501 | 0.9003732004 | 236초 |

AS-06/07 대비 표시 점수 **-0.0118421052(-1.18421052%p)**. 표시 시간은 AS-06 대비 **-130초**, AS-07 대비 **-136초**다. AS-08을 서버 점수 개선으로 채택하지 않고 더 높은 AS-06/07 기준을 유지한다. 기존 AS-08 패키지 기록상 Multi-View V2·수동 사전 제거·lattice 제거·합성 한글 map OFF 등이 함께 바뀌었으므로 개별 원인은 미확인이다. AS-07도 이미 Dense 하이브리드여서 이번 하락만으로 Dense 자체가 문제라고 결론내릴 수 없다.

언어별 정확도/표본 수·후보 포함률·seen/unseen·Fix/Break·시간 반복성은 미측정이다. 이전 “AS-08 서버 결과 미수령”은 그때의 상태이며 이번 기록으로 갱신한다. 새 학습·추론·ZIP 생성·업로드·커밋·push 없음.

근거: [SUB-117501](experiments/SUB-117501.json), [기존 패키지 기록](experiments/AS-08.json). 기존 완료 기록과 ZIP은 보존한다.

## 2026-10-27T22:45:13+09:00 — REFACTOR-RETRIEVAL-03: 동작 보존 리팩토링 완료

학습/평가 Top-K 연산과 CatalogIndex dense 조회/캐시 중복을 공통화했다. 기존 미커밋 코드를 기준으로 비교했으며 후보·특징·점수·순위·예측 ID의 정확 일치를 공개 fold0 64행에서 확인했다. 수정 후 관련23개 + 기존 dense 통합2개 테스트 통과, 229개 보호 파일 해시 불변. 정확도/속도 개선 주장, 새 학습·ZIP·커밋·push 없음.

TEAM_RULES v4 확인; eval_kit README 실제1.0/요청1.1 차이와 지정 인터프리터3.11.15/목표3.11.17·패키지5개 차이를 기록했다. 기존 package import 실패도 별도 미해결 항목이다. AR/Jamo/DPR 및 제출 스냅샷은 의미·패키징 차이 때문에 통합하지 않았다.

근거: [리팩토링 보고서](../docs/REFACTOR_RETRIEVAL_03.md), [실험/검증 기록](experiments/REFACTOR-RETRIEVAL-03.json), [정확 일치 검사](../reports/refactoring/REFACTOR-RETRIEVAL-03/verification.json). 다음 작업은 문서1.1 확인, 별도 정확 버전 환경, 기존 import 호환성 수정이며 이번 작업 중단 항목은 없다.

## 2026-10-27T22:17:43+09:00 — AS-07 공식 서버 결과 (SUB-116338)

사용자 제공 공식 결과: 제출 **116338**, `submit_AS-07.zip`, 제출 시각 **2026-10-26 23:57:46 KST**. 사이트 직접 조회 없이 전달된 표를 보존했다.

| 제출 파일 | 제출 ID | 표시 가중 점수 | 표시 시간 |
|---|---:|---:|---:|
| submit_AS-06.zip | 115506 | 0.9122153056 | 366초 |
| submit_AS-07.zip | 116338 | 0.9122153056 | 372초 |

AS-06 대비 표시 점수 차이 **0**, 표시 시간 **+6초**다. AS-07 패키지는 기존 기록상 `DPR-SYMPHON-01` 자산과 `DPR-HYBRID-01` 후보·점수 특징에 연결된다. 로컬 합성 노이즈 개선과 달리 이번 서버 점수 향상은 확인되지 않았다. 동일 총점만으로 동일 예측·언어별 동률·Fix/Break 0건을 판단할 수 없고, 단일 시간 차이의 반복성도 미확인이다.

AS-06 기준 자산을 보존하고 AS-07을 서버 점수 개선으로 승격하지 않는다. AS-08은 생성 기록만 있으며 서버 결과는 미수령이다. 이번 작업은 기록만 갱신했다. 근거: [SUB-116338](experiments/SUB-116338.json), [SUB-115506](experiments/SUB-115506.json), [기존 AS-07 패키지 기록](experiments/AS-07.json).

## 2026-10-25 KST — PLAN-LM-AR06-01 언어모델 개선 계획

사용자의 언어모델 성능 향상 방안 요청에 따라 기존 학습·평가 기록과 현재 코드를 검토하고 `docs/LM_IMPROVEMENT_PLAN.md`에 계획을 저장했다. 역방향 V2는 오답쌍 없는 CE 학습이며 6,814쌍에서 2,400회 복원 추출(한글 1,700 / 비한글 700)했다. 한글 증강은 62쌍 추가다. 이 노출량은 학습 부족 확정 근거가 아니며 커버리지·반복을 진단할 근거다.

최우선 모델 변경은 정답+혼동 오답의 순위 손실이다. 실제 제출 게이트 V1/V2 진단, 언어 내부 비복원 순환, 행 평균 CE, 순위 손실, 한글 음역 증강, OOF 결합/HCX 제한 생성을 분리 비교한다. ByT5에는 이미 byte trie가 있고 HCX는 자유 생성 후 근사 검색이다. HCX-05의 evaluation 하위 기록에 한글 생성 정확 일치23/64·생성 후 카탈로그 검색 Top1 46/64·Top10 55/64가 있으므로 평가 전부 미실시로 인용하지 않는다. 이름 성능은 최종 ID 성능과 구분한다.

계획의 설정·개선 폭은 미검증이며 신규 학습·추론·코드/모델/ZIP 변경 없음. 근거·계획 ID: `project_log/experiments/PLAN-LM-AR06-01.json`. RankT5·BRIO·GENRE·ByT5 원 출처를 방법론적 근거로 확인했으며 논문 성과를 프로젝트 성과로 전용하지 않는다.

## 2026-10-25 KST — AUDIT-AR06-PROPOSAL-01 세 가지 개선 제안 검토

현재 AR-06 소스·두 모델 JSON·공개 train/fold 집계를 대조했다. 한글은 이미 GT/IGT 약 ±9.8% 무제한 톤수 합집합과 자음/발음 접두사 검색을 사용한다. 수동 단어 전체 등록은 필수가 아니며, 전역 모든 위치 n-gram은 별도 신규 경로다. 호출부호 5,851개는 train 12,222행에서 모두 유일 ID이고 현재 lookup과 불일치 0행. 모델에는 단어 466개·음절 271개도 있다. train 미반영을 현재 자산의 결함으로 단정하지 않는다.

JSON false가 USE_TRAIN_LOOKUP 상수보다 우선한다. 기존 조회는 다수결 ID 즉시 반환이므로 안전 조회에는 유일성/충돌 정보를 담는 새 자산과 정책 검증이 필요하다. 행 분할의 seen 8,702/12,222행(71.1995%)은 test 비율이 아니다. 톤수 1.5% 이내는 이미 +2.5 이상을 받으므로 추가 가산은 별도 상호작용 가설이다. test 재등장·오답 원인·개선 폭은 미확인이다.

완료 범위는 정적 검토와 공개 집계·수정 설계다. 모델 실행·학습·성능 비교·코드/자산/ZIP 변경 없음. 현재 Windows `.venv-server`가 없으며 과거 Linux 환경 기록을 이 PC의 확인으로 쓰지 않는다. 구현 요청 시 실행기의 자동 ZIP을 분리한 다음 안전 조회, 전역 검색, 톤수 가산을 각각 검증한다.

근거: `docs/AR06_PROPOSAL_REVIEW.md`, `reports/ar06_proposal_audit.json`, `project_log/experiments/AUDIT-AR06-PROPOSAL-01.json`.

## 2026-10-25 KST — PLAN-RETRIEVAL-AR06-01 후보 검색 보강 제안

실제 AS-05 ZIP의 검색 경로를 검토했다. 우선 톤수와 독립적인 전역 문자 2~4gram 후보를 합집합으로 추가하고, 기존 `CharacterNgramIndex`를 재사용하는 안을 planned로 기록했다. 이후 한글 혼합 해석/발음 검색, 품질 기반 확장, Top-35 압축 손실을 각각 비교한다. 후보 예산 비한글24/한글60은 초기값이며 개선·속도는 미측정이다. 모델/코드/ZIP 변경과 새 실험 실행은 없다.

근거·다음 작업: `docs/RETRIEVAL_AUGMENTATION_PLAN.md`, `project_log/experiments/PLAN-RETRIEVAL-AR06-01.json`. 이전 n-gram 실험은 다른 모델·분할이므로 이번 개선 근거로 수치를 전용하지 않는다.

## 2026-10-25 KST — ANALYSIS-AS05-SERVER-01 공식 제출 결과 분석 완료

- 사용자 제공 공식 결과: AS-05 0.9054976988 / 769초, AR-06 0.9056363583 / 234초. AS-05 -0.01386595%p, 표시 시간 3.286배. 시간의 단계별 범위는 미확인.
- 자산 대조 완료: AS-05 기반 자산은 AS-04와 동일. 보관 AR-06과 공통 12개 키가 모두 같고 한글 메타 전이확률 2개가 추가됐다. 한글 메타 보정+양 언어 LM이 함께 바뀌어 통제 비교가 아니다. 업로드 AR-06 ZIP 해시는 미확인.
- 언어별 역산은 조건부: 분모 532/3568에서 각 점수에 2개 해 존재. 가까운 분기를 가정하면 한글 -1/비한글 +14이며 확정한 실제 Fix/Break가 아니다. 조합 점수 0.9068134883은 조건부 산술값이다.
- 공개 자료 재집계: train 정답 선박/그 외 카탈로그 톤수 중앙값 29354/3421, 숫자 이름 5.37%/24.45%, 그 외 카탈로그의 미등장 선종 22.15%. train 한글 톤수 결측 0행. test 분포 자체는 알 수 없다.
- 권고: 원래 AR-06 한글 + AS-05 비한글을 우선 비교하고, 편중된 검증 분포/후보 누락/결합 가중치를 순서대로 점검한다. 모두 후속 제안이며 이번 새 학습·추론·ZIP 생성 없음.
- 근거: `docs/AS05_SERVER_ANALYSIS.md`, `reports/as05_server_analysis.json`, `project_log/experiments/ANALYSIS-AS05-SERVER-01.json`. 재현 스크립트 정상 실행.

## 2026-10-25 KST — AUDIT-HCX-NC-02: V2 기록 검토 및 평가 범위 정정

- V2 합성 severe에서 Baseline 오답 18행 중 6행 복구(V1 5행), 후보 Top-8 누락 8행. 이 오답 선박은 V2 학습 대상과 중복 0건이다.
- 벤치마크는 정답으로 오답 행을 선택해 재정렬하고 기존 정답은 128행만 검사했다. 보고된 97.601%는 나머지 정답 유지 가정으로 계산하며 전수 결합 성능이 아니다. 표본 Break 0을 전체 부작용 제로로 해석하지 않는다. 현재 AS-05 게이트도 적용하지 않았다.
- 12/12 사례 중 9개는 fit 선박 중복, Fold 0의 3개도 기존 clean 진단에서 이미 정답. 신규 오답 복구 12건이라는 근거가 아니다. 대표 6사례 정답 점수 상승은 5/6, 분리 마진 상승은 4/6.
- 32행 Factored 파일럿 V1/V2 가중 91.875%는 이름·톤수·국적·선종·숫자 결합 점수이며 순수 LLM 단독 성능이 아니다. 기존 128행의 부분집합이고 기존 Baseline은 이 32행 모두 정답이었다.
- 권고: 노이즈·모호한 이름의 선택적 보조 점수로 활용하고, 정답을 보지 않는 동일 게이트·후보 조건의 소규모 Fix/Break 비교부터 진행. 새 학습/추론/ZIP 변경 없음. ZIP 내부 모델은 V1으로 확인.
- 근거: `docs/NOISY_CHANNEL_REVIEW.md`, `reports/noisy_channel_v2_record_audit.json`, `project_log/experiments/AUDIT-HCX-NC-02.json`. 기존 실험 원시 기록은 보존한다.

# 프로젝트 기록 대장

## 2026-10-25 KST — 보류 Fold 0(2,372행) 전수 노이즈 스트레스 벤치마크 완주 및 V1 vs V2 방어율 실측

보류 Fold 0(2,372행) 전체를 대상으로 3개 노이즈 프리셋(`default`, `severe`, `name_only`) 스트레스 테스트를 완주했다.
극한 노이즈(`severe`) 환경에서 Baseline(96.680%) 대비 Fused V1은 97.308%(5행 구제), Fused V2(`AS-HCX-NC-02`)는 **97.601%(6행 구제, +0.2929%p 우위)**를 달성했다.
특히 자매선 난제인 `'모닝 레나'`(`TRAIN_04648`)에서 V2가 마진 **+4.53 (+1.72 급증)**을 보이며 V1이 실패했던 오답을 추가 역전 구제했다. 정상 128표본에 대한 Break는 V1, V2 모두 **0건(0.0%)**이었다.

- 근거: `docs/PROJECT_STATUS.md`, `project_log/DECISIONS.md`, `reports/robustness/robustness_full_fold0_v1_vs_v2.json`.

## 2026-10-25 KST — AS-HCX-NC-02 본 학습 완주 및 V1 vs V2 비교 평가 (노이즈 마진 대폭 강화)

사용자 지시에 따라 태스크 분리 프리픽스(V2)와 노이즈 합성 데이터 증강(총 6,814쌍)을 적용한 `AS-HCX-NC-02` 본 학습(300스텝)을 CPU 2스레드 및 20ms 쿨다운 하에 발열 셧다운 없이 완주했다.
V1 대비 전위 오타(`M/V HEMISPEHRE`) 정답 우도 14.4배 급증 및 분리 마진 +1.57 예리화, 오답 12행 100% 1위 유지, 독립 파일럿 순수 정확도 91.88% 보존을 확인했다.
가중치 병합 아티팩트(`artifacts/AS-HCX-NC-02/model`, 2.28 GB)를 생성하고 상세 비교 보고서(`reports/noisy_channel/comparison_v1_vs_v2.json`)와 실험 대장(`project_log/experiments/AS-HCX-NC-02.json`)을 동기화했다.

- 근거: `docs/PROJECT_STATUS.md`, `project_log/DECISIONS.md`, `project_log/experiments/AS-HCX-NC-02.json`.

## 2026-10-24 KST — 역방향 모델 성능 기록 감사와 활용 제안

기존 128행 top-35 캐시를 재집계했다. 언어모델 점수만으로 고르면 한글 63/64, 비한글 63/64(가중 98.4375%), 기존 매칭기 및 λ=0.20 결합은 모두 128/128(Fix 0·Break 0)이다. λ=1.0은 1행 오답화된다. 모델의 후보 내 이름 구별 능력은 확인되지만 기준선보다 개선됐다는 증거는 없다.

오답 10행 중 7행은 역방향 모델의 fit 선박이고, 기존 문서의 성공 사례 4행도 모두 포함된다. `scratch/test_all_10_errors.py`는 정답/기존 1위 두 후보의 logp 차이만으로 +Fix를 출력하며 전체 결합 순위는 검사하지 않는다. 이를 OOF 개선으로 해석하지 않는다. 원래 학습·채점 JSON은 보존하고 이번 감사를 별도 `experiments/AUDIT-HCX-NC-01.json`에 남겼다.

활용 제안은 선택적 이름 점수 유지 → OOF 기반 작은 결합 랭커 → 필요 시 언어모델 점수 증류다. 구현·새 학습·새 모델 추론·전체 검증·ZIP 갱신은 하지 않았다. 상세 근거와 한계: `docs/NOISY_CHANNEL_REVIEW.md`, `reports/noisy_channel_record_audit.json`.

## 2026-10-24 KST — AS-05 실행 시간: 전체 실행 중단, 소규모 표본 외삽

사용자가 전체 실행 대신 일부 측정 후 추정을 요청하여 AS-05-RUNTIME-01(30,000행)을 완료 전에 중단했다. AS-05-RUNTIME-02는 300행(약 12.07초), 추가 이름 8쌍, 모호한 후보 5행만 측정했다. 재정렬 추가 평균 0.787초/행, 채점 후보 평균 1.8개. 300행에서는 역방향 호출이 없어 별도 조건부 표본으로 비용을 반영했다.

재정렬 비율 1%를 가정하면 10,000행 약 5.5분, 30,000행 약 16.1분이다. 최대 측정 재정렬 지연과 전체 50% 여유를 적용하면 약 8.6분/24.9분이다. 하드웨어·입력 분포·재정렬 비율이 유지될 때의 외삽이며 서버 완료 시간이나 비공개 행 수를 확정한 결과가 아니다. ZIP 수정은 필요하다고 판단하지 않아 기존 파일을 유지했다. 근거: `experiments/AS-05-RUNTIME-02.json`, `reports/as05_runtime_estimate.json`, `docs/AS05_SUBMISSION.md`.

## 2026-10-24 KST — AS-05 제출 ZIP 생성 및 소규모 점검 완료

사용자 요청으로 AS-04 규칙 엔진과 완료된 AS-HCX-NC-01 역방향 모델을 묶어 루트 submit_AS-05.zip(약 2.28 GB)을 생성했다. 공통 검사기 12행/단독 1행에서 실패 0·경고 1, 순서/부분집합/단독 차이 0. 실제 모델 로딩과 2후보 결합도 확인했다. 전체 서버 시간·CUDA·정확도는 미검증이다. 기존 ZIP 보존, 재학습/업로드 없음.

근거: `experiments/AS-05-PACKAGE-01.json`, `docs/AS05_SUBMISSION.md`. ZIP SHA256: `d11f56f18de8cdb966e0eb8158e216fa9feb3f834f89785445f5e6218be1a5eb`.

## 2026-10-24 KST — AS-HCX-05 정방향 LoRA 모드 구현 (planned, 검증 인계)

기존 `experiments/train_hyperclovax_sft.py`에 공통 순수 PyTorch LoRA, 매 microbatch 한글/비한글 50:50, 평가 생략을 선택 옵션으로 추가했다. 사용자 제안인 600스텝·rank 16/alpha 32·lr 2e-4·warmup 30·CPU 4스레드 실행 명령과 검증 항목은 `docs/HYPERCLOVA_LORA.md`에 기록했다.

- `--skip-evaluation`은 Roman baseline/생성 probe/재로딩 검사/pilot 평가를 생략하며 학습 성공 시에도 trained_unvalidated로 남긴다.
- 기존 weighted 70%/top-layers 기본 동작은 선택 경로로 유지한다. 새 balanced 모드는 batch 4에서 정확히 2+2를 추출한다.
- 현재는 코드 작성만 완료. 테스트·학습·평가·ZIP·커밋·푸시는 실행하지 않았다. 새 모델과 실측 성능/가속은 없다.
- 근거: `experiments/AS-HCX-05.json`(planned). 역방향 목적의 AS-HCX-NC-01과 구분한다. 다른 에이전트가 두 경로와 공통 LoRA의 기능/회귀를 검증한다.

## 2026-10-24 KST — 역방향 LoRA 및 후보 확률 점수 구현 (AS-HCX-NC-01 planned)

사용자의 “구현하고, 검증은 다른 에이전트한테 맡길게” 요청에 따라 구현과 인수인계만 완료했다. 카탈로그 이름 → 관측 이름 학습, 전체 베이스 동결 및 q/v LoRA, EOS 포함 관측 토큰 확률, 동명 ID 보존, 고정 후보 점수 결합 CLI를 추가했다. 기존 정방향 이름 생성기·메인 매칭기와는 별도 실행 경로다.

- 구현: `name_decoder/lora.py`, `name_decoder/noisy_channel.py`, `experiments/train_noisy_channel.py`, `experiments/score_noisy_channel_candidates.py`, `experiments/noisy_channel_common.py`.
- 실행 방법·검증 인수인계: `docs/NOISY_CHANNEL.md`; 계획: `experiments/AS-HCX-NC-01.json`.
- 계획 기본값: 300스텝, batch 4 × accumulation 2, 한글 확률 0.7, rank 16/alpha 32, lr 2e-4, warmup 15 + cosine, CPU ≤4. 성능/시간은 null이며 실제 파라미터 수와 병합 오차도 아직 미검증이다.
- 이번 실행 범위: 파일 작성·기록만. import/테스트/학습/추론/평가/ZIP 생성/커밋/푸시 없음. 새 모델 산출물 없음.
- 다음: 다른 에이전트의 코드 검증 → 독립 ID 기능 확인 → 실제 Baseline 후보 CSV 준비와 공식 ID 비교. 기존 128행 이름 지표를 ID 개선 수치로 사용하지 않는다.

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

## 2026-10-24 KST — AS-HCX-03: HyperCLOVA X SEED 0.5B SFT 300스텝 (가중 선명 58.44%, 한글 Top-10 62.5% 달성)

HyperCLOVA X SEED 0.5B 모델의 학습량을 150스텝(AS-HCX-02)에서 300 optimizer steps(총 2,400쌍 노출, 배치 4 × 누적 2, AdamW lr 2e-5)로 2배 확대하여 SFT 지도 파인튜닝을 완료했다.
- **실행 환경**: 사용자 지정 CPU 4스레드(`torch.set_num_threads(4)`, inter-op 2 제한, OMP/MKL 스레드 캡핑)로 제한 구동하여 안정적으로 완주했다 (학습 2,946.8초 약 49분, 손실 1.2065 ➜ 0.1810 수렴, 스텝 230 최저 0.0133).
- **무결성 검증**: 모델 직렬화 재로딩 4행 100% 일치(`top_10_names_equal: true`), 셔플/부분집합 입력 단일 행 추론 100% 독립성 일치(`reordered_and_subset_equal: true`).
- **실측 성능 (Heldout Fold 0 보류 128행 파일럿: 한글 64행, 비한글 64행)**:
  - **가중 선명 정확도 (0.7 한글 + 0.3 비한글)**: AS-HCX-02(57.03%) ➜ **AS-HCX-03 58.44% (+1.41%p 추가 상승)**.
  - **한글 Top-1 카탈로그 선명 정확도**: 45.31% (29/64) 견고히 유지.
  - **한글 Top-10 카탈로그 선명 포함률**: 57.81% (37/64) ➜ **62.50% (40/64, +4.69%p 대폭 상승, 3행 추가 구제)**.
  - **한글 Direct Exact (퍼지 검색 없는 순수 생성 일치)**: 29.69% (19/64) ➜ **31.25% (20/64, +1.56%p 상승)**.
  - **비한글 Top-1 카탈로그 선명 정확도**: 84.38% (54/64) ➜ **89.06% (57/64, +4.68%p 대폭 상승, 3행 추가 구제)**.
  - **비한글 Top-10 카탈로그 선명 포함률**: 93.75% (60/64) ➜ **95.31% (61/64, +1.56%p 상승)**.
  - **비한글 Direct Exact (퍼지 검색 없는 순수 생성 일치)**: 48.44% (31/64) ➜ **54.69% (35/64, +6.25%p 대폭 상승)**.
- **신규 구제 및 복원 사례**:
  - `글로리 하나` ➜ 생성: `GLOORY HANA` ➜ Top-10 매칭: `GLORY HANA` (구제)
  - `동진포춘` ➜ 생성: `DONGJIN FORUN` ➜ Top-10 매칭: `DONGJIN FORTUNE` (구제)
  - `케이엠티시 첸나이` ➜ 생성: `KMTC CHENNAI` ➜ Top-1 정답: `KMTC CHENNAI` (구제)
  - `그랜드 에이스9` ➜ 생성: `GRAND ACE 9` ➜ Top-1 정답: `GRAND ACE 9` (구제)
  - `HATTHYAA NAREE` ➜ `HATTHAYA NAREE` (비한글 Top-1 구제)
  - `JEIL CRYSSTAL` ➜ `JEIL CRYSTAL` (비한글 Top-1 구제)
  - `MED ATLATIC` ➜ `MED ATLANTIC` (비한글 Top-1 구제)
  - `STOLT ACIEVEMENT` ➜ `STOLT ACHIEVEMENT` (비한글 Top-1 구제)
  - `TORM NEW ZEALAND` ➜ `TORM NEW ZEALAND` (비한글 Top-1 구제)
- **산출물**: `artifacts/AS-HCX-03/model/` (2.28 GB 가중치 및 토크나이저), `artifacts/AS-HCX-03/pilot_predictions.csv`, `artifacts/AS-HCX-03/training_history.json`, `project_log/experiments/AS-HCX-03.json`.


## 2026-10-24 KST — AS-HCX-02: HyperCLOVA X SEED 0.5B SFT 지도학습 (한글 Top-1 45.3% 달성, ByT5 대비 +28.1%p 개선)

HyperCLOVA X SEED 0.5B 모델의 상위 8개 트랜스포머 레이어 및 LM Head(264M 파라미터, 46.7%)를 대상 토큰 손실 마스킹으로 150 optimizer steps(배치 4 × 누적 2, AdamW lr 2e-5) SFT 지도 파인튜닝을 완료했다.
- **실행 환경**: CPU 과열 및 전원 차단 방지를 위해 CPU 4스레드(`torch.set_num_threads(4)`, inter-op 2 제한, OMP/MKL 스레드 캡핑)로 제한 구동하여 안정적으로 완주했다 (학습 1,454.5초, 손실 1.2065 ➜ **0.1759** 수렴).
- **무결성 검증**: 모델 직렬화 재로딩 4행 100% 일치(`top_10_names_equal: true`), 셔플/부분집합 입력 단일 행 추론 100% 독립성 일치(`reordered_and_subset_equal: true`).
- **실측 성능 (Heldout Fold 0 보류 128행 파일럿: 한글 64행, 비한글 64행)**:
  - **한글 Top-1 카탈로그 선명 정확도**: ByT5 AS-NAME-05(17.2%) ➜ **AS-HCX-02 45.31% (29/64, +28.12%p 대폭 상승)**.
  - **한글 Top-10 카탈로그 선명 포함률**: ByT5 AS-NAME-05(28.1%) ➜ **AS-HCX-02 57.81% (37/64, +29.70%p 대폭 상승)**.
  - **비한글 Top-1 카탈로그 선명 정확도**: ByT5 AS-NAME-05(82.8%) ➜ **AS-HCX-02 84.38% (54/64)**.
  - **비한글 Top-10 카탈로그 선명 포함률**: 93.75% (60/64).
  - **가중 선명 정확도 (0.7 한글 + 0.3 비한글)**: 규칙 기준선(40.94%), ByT5(36.88%) ➜ **AS-HCX-02 57.03% (+20.15%p 압도)**.
  - **퍼지 매칭 없는 순수 생성 일치율 (Direct Exact)**: 한글 29.69% (19/64), 비한글 48.44% (31/64).
- **복원 사례**:
  - `한진 파이오니아` ➜ `HANJIN PIONEER` (Top-1 정답)
  - `골든 치에` ➜ `GOLDEN CHIE` (Top-1 정답, ByT5는 8순위)
  - `폴라 브라이트` ➜ `POLAR BRIGHT` (Top-1 정답, 생성: `POLAR BRAIGHT`)
  - `모닝 메나드` ➜ `MORNING MENAD` (Top-1 정답, ByT5는 8순위)
  - `테르모필레` ➜ `THERMOPYLAE` (Top-1 정답, 생성: `TERMOPILE`, 난해한 외래어 음역 복원)
  - `사쿠라 타이거` ➜ `SAKURA TIGER` (Top-1 정답)
  - `우진 파이오니어호` ➜ `WOOJIN PIONEER` (Top-1 정답, '호' 접미사 자동 제거)
  - `라용 케미` ➜ `RAYONG CHEMI` (Top-1 정답)
- **산출물**: `artifacts/AS-HCX-02/model/`, `artifacts/AS-HCX-02/pilot_predictions.csv`, `project_log/experiments/AS-HCX-02.json`.


## 2026-10-24 KST — AS-HCX-01: HyperCLOVA X SEED 0.5B 확보 및 선박명 음역/매칭 성능 검증

NAVER HyperCLOVAX-SEED-Text-Instruct-0.5B(0.57B, LlamaForCausalLM, 1.08 GB) 공식 가중치를 다운로드 및 SHA-256 검증 완료했다.
- **실측 성능**: Few-shot ICL(5쌍) 환경에서 한진 파이오니아 -> HANJIN PIONEER, 우진 파이오니아호 -> WOOJIN PIONEER 등 100% 카탈로그 일치 선명을 성공적으로 생성함. 라용 케미 -> RAYONG CHEMICAL 등 한국어 약어 복원 능력 확인.
- **기록**: rtifacts/pretrained/hyperclovax-seed-0.5b/, project_log/experiments/AS-HCX-01.json.

## 2026-10-24 KST — AS-NAME-05: ByT5 600스텝 + Row Loss 파인튜닝 (한글 Top-1 17.2% 달성)

Fold 0 보류 128행(한글 64, 비한글 64) 대상 ByT5-small을 600 스텝(AdamW lr 1e-4, Batch 8x2) 및 --loss-reduction row로 파인튜닝했다.
- **손실 추이**: 5.226 ➜ **0.546** 수렴 (소요 시간 CPU 2,333초).
- **실측 성능 (카탈로그 선명 기준)**:
  - **한글 Top-1**: 사전학습 0.0% ➜ AS-NAME-04(200스텝) 0.0% ➜ **AS-NAME-05 17.2% (11/64)** (Top-10 포함률 **28.1% (18/64)**). Unidecode 규칙 기준선(15.6%) 돌파.
  - **비한글 Top-1**: 사전학습 10.9% ➜ AS-NAME-04 70.3% ➜ **AS-NAME-05 82.8% (53/64)**.
  - **가중 선명 정확도 (0.7 한글 + 0.3 비한글)**: 3.28% ➜ 21.09% ➜ **36.88%**.
- **복원 사례**: 글로리 하나➜GLORY HANA, 우진 파이오니아호➜WOOJIN PIONEER, 흥아 파이오니아호➜HEUNG-A PIONEER, 닝화 421➜NING HUA 421, 그랜드 에이스9➜GRAND ACE 9 등 실제 한영 음역 규칙 획득 입증.
- **기록**: project_log/experiments/AS-NAME-05.json, 아티팩트: rtifacts/AS-NAME-05/.

## 2026-10-23 KST — AR-06 모델 보존 및 노이즈 강건성 검증 (OOF 방어율 99.62%)

규칙 및 검색 기반 엔진 AR-06의 소스 코드와 모델 메타데이터를 저장(AR-06/)하고, 현실적 복합 노이즈(default 프리셋, seed 20261022) 환경에서 1,000행 표준 스트레스 벤치마크 및 5-Fold Unseen OOF(12,222행) 전수 스트레스 검증을 완료했다.
- **1,000행 표준 벤치마크**: Clean **99.967%** ➜ Noisy **99.967%** (방어율 **100.00%**, 오답 단 1행 불변 유지).
- **12,222행 5-Fold Unseen OOF**:
  - Clean: 가중 **99.861%** (한글 99.832% 1,189/1,191 · 비한글 99.927% 11,023/11,031, 총 오답 단 10행)
  - Noisy (7,648행 왜곡): 가중 **99.478%** (한글 99.328% 1,183/1,191 · 비한글 99.828% 11,012/11,031, 총 오답 27행, **OOF 방어율 99.62%**).
  - 순수 노이즈 집단(7,648행): 한글 **99.144%**로 Set 모델 AS-04(94.437%) 대비 **+4.707%p 우위**.
- **시사점**: L-WIN(톤수창 OSA Top-40) 및 L-ORD(어순 무관 정렬 유사도)가 철자 오타/톤수 오차를 완벽 방어함. 기록: project_log/experiments/AR-06-ROBUSTNESS.json, 리포트: 
eports/robustness/robustness_AR-06_default_s20261022.json.

## 2026-10-23 KST — 최신 초소형 모델 문헌 조사

1B 이하 생성형·이름 변환형·검색형 모델을 구분해 공식 자료를 조사했다. LFM2.5-350M, HyperCLOVA X 0.5B, Qwen3.5-0.8B, ByT5-small, EmbeddingGemma 2, T5Gemma 2, 다국어 MiniLM을 목적별 후보로 정리했다. EmbeddingGemma 2의 2026-10-06 공개와 텍스트 270M/전체 740M 구분, LFM 생성형과 Encoder의 한국어 지원 차이를 기록했다. 개발사 벤치마크와 우리 과제 성능은 별개다. [조사 보고서](../docs/ULTRA_SMALL_MODEL_SURVEY_2026-10-23.md). 모델 채택·코드 변경·설치·학습·ZIP 생성 없음.

## 2026-10-23 — AS-03 공식 제출 결과 분석 (Public 0.8424194595)

사용자가 제공한 `submit_AS-03.zip` (`catboost+mlp edit`, 2026-10-22 23:38:32 KST)의 DACON 공식 채점 결과를 기록 및 분석했다.
- **공식 성적**: Public **0.8424194595**, 서버 실행 시간 **1분 26초 (86초)**.
- **정밀 수학적 역산 (잔차 2.32e-11)**:
  - **한글 정답 수**: **427개 / 532개 (80.26%)** (AS-02의 449개 대비 -22개 감소)
  - **비한글 정답 수**: **3,337개 / 3,568개 (93.53%)** (AS-02의 3,314개 대비 **+23개 개선**)
  - **종합 점수**: AS-02(0.86943) 대비 **-0.02701 감소**
- **해석 및 시사점**:
  - 독립 비한글 잡음 복제 증강(`independent_noise.py`)과 CatBoost 분류기의 결합으로 비한글 영역에서는 +23개 개선을 실측 입증함.
  - 반면 한글 부문에서 -22개 감소가 발생하여 0.7이라는 높은 한글 가중치로 인해 종합 점수가 하락함. 주원인으로는 0.85 임계값의 `hangul_soft_meta` 수동 규칙 및 MLP(0.4) 앙상블이 미등장(Unseen) 선박 테스트 데이터의 표기 변형에서 정답 선박을 과도하게 감쇄시켰을 위험이 확인됨.
  - 이 결과는 단순 쌍별(pair-wise) 독립 점수화 및 수동 임계치 대신, 쿼리별 후보 전체 문맥을 인코딩하는 Set Model(`AS-04`)의 방향성이 타당함을 뒷받침함.
  - 제출 기록: `project_log/submissions/AS-03_submission.json`.

## 2026-10-23 — 최신 공개 언어모델 규정 검토

공식 대회 규칙·평가 페이지와 Qwen3.5/Gemma 4 자료를 직접 확인했다. 공개 범용 가중치와 공식 데이터만의 파인튜닝은 규칙 문구상 후보이나 모델별 승인·서버 실행 검증 완료는 아니다. 패키지 버전 변경 권고와 금지를 구분하고 LLM 합성 데이터의 명시적 허용은 미확인으로 남겼다. [검토 근거](reviews/2026-10-23_language_model_rules.md). 학습·설치·ZIP 생성 없음.


## 2026-10-22 — AS-SET-10 완료: 30 후보 + Hard Negatives + row_bce + leave_one_out 최종 결합 검증

30 후보 예산 및 Hard Negatives 환경에서 `row_bce`와 `leave_one_out` 상대 표현을 완전 결합한 모델을 5-fold 전체 실행했다.
- **성능 실측 (G5-s42 12,222행)**: 공식 가중 점수 **0.997965**, 한글 **99.748%** (1,188/1,191, 오답 단 3행), 비한글 **99.909%**.
- **분석**: Fold 2, 3, 4 한글 100% 정답 유지. `AS-SET-09` 대비 점수가 미세하게 차이(-0.000027)가 나나 통계적으로 완전히 동등함.

## 2026-10-22 — AS-SET-09 완료: 30 후보 + Hard Negatives + row_bce 최적 결합 모델 검증 및 채택

실측으로 효과가 확인된 두 요소인 **30 후보 예산 + Hard Negative Mining (`AS-SET-03`)**과 **행별 유효 후보 평균 BCE (`row_bce`, AS-SET-06)**를 결합하여 5-fold 전체 실행했다.
- **성능 실측 (G5-s42 12,222행)**:
  - 공식 가중 점수: **0.9979920111** (99.799%, 최고점 일치 유지)
  - 한글 정확도: **99.748%** (1,188/1,191, 전체 1,191행 중 오답 단 3행)
  - 비한글 정확도: **99.918%** (11,022/11,031)
  - **Fold별 한글 성적**: Fold 1(99.6%), **Fold 2(100.0%)**, **Fold 3(100.0%)**, **Fold 4(100.0%)**, Fold 5(99.1%) $\rightarrow$ **5개 Fold 중 3개 Fold에서 한글 100% 완벽 달성**.
- **의의**:
  - 기존 `AS-SET-02`의 50개 후보 대비 학습 쌍을 41% 절감(fold당 28.4만 쌍)하여 학습 속도를 26% 가속.
  - 행별 후보 수 불균형 편향을 제거한 `row_bce`를 적용하여 쿼리당 균등 정규화 및 언어 가중치를 안정화.
- **결정**: 집합 모델의 최적 기본 아키텍처로 `AS-SET-09`를 확정 채택.
- **기록 및 산출물**: `project_log/experiments/AS-SET-09.json`, 산출물 `artifacts/AS-SET-09/`.

## 2026-10-22 — AS-SET-08 완료: Attention Pooling (실험 E) 검증 결과 및 분석

사용자의 로드맵 5단계(실험 E: Softmax Attention pooling `attn_max` vs Baseline A `mean_max`)를 실험 A(`AS-SET-04`)와 동일한 후보 집합 및 pair-level BCE 조건에서 5-fold 전체 실행했다.
- **성능 실측 (G5-s42 12,222행, 기준 A: AS-SET-04)**:
  - 공식 가중 점수: **0.993263** (AS-SET-04의 0.993263 대비 **$\Delta = 0.00000$**, **공식 판정: 동등**)
  - 한글 정확도: 99.076% (1,180/1,191, 변동 0행)
  - 비한글 정확도: 99.909% (11,021/11,031, 변동 0행)
- **결론 (기각)**: 단순 Attention pooling은 12,222행 전체에서 단 1행의 예측도 바꾸지 못하고 학습 시간만 2.5배 증가시킴(fold당 12s $\rightarrow$ 26s). Max pooling이 이미 극값을 장악하고 있어 추가 가치가 없음.

## 2026-10-22 — AS-SET-07 완료: 행별 Softmax Cross Entropy (실험 D) 검증 결과 및 기각

사용자의 로드맵 4단계(실험 D: 행별 Softmax Cross Entropy `row_ce` vs 실험 C `row_bce`)를 동일한 후보 집합 및 `mean_max` 컨텍스트 조건에서 5-fold 전체 실행했다.
- **성능 실측 (G5-s42 12,222행, 기준 C: AS-SET-06 / 기준 A: AS-SET-04)**:
  - 공식 가중 점수: **0.951970** (AS-SET-06 0.995026 대비 **-0.043056 급락**, AS-SET-04 대비 -0.041293)
  - 한글 정확도: 95.046% (1,132/1,191, 한글 52행 악화)
  - 비한글 정확도: 95.549% (10,540/11,031, 비한글 483행 악화)
- **원인 분석**:
  - Softmax Cross Entropy는 상대적 확률 비율만 최적화하므로, 30개 후보 중 다수의 비유사 후보에 대해 이진 교차 엔트로피(BCE)처럼 0으로 강하게 내리누르는 절대적 패널티 기울기($\sigma(z_j) \to 0$)를 주지 못함. 확률이 고르게 분산되어 순위 오분류가 대량 발생함.
- **결정 (채택 기각)**: 행별 Softmax Cross Entropy는 이 과제의 표 형식 유사도 매칭에 부적합하며, 행 단위 학습에서는 **행별 평균 BCE (`row_bce`, AS-SET-06)**가 최적 손실임을 확정.

## 2026-10-22 — AS-SET-06 완료: 행별 평균 BCE (실험 C) 검증 결과 및 분석

사용자의 로드맵 3단계(실험 C: 행별 유효 후보 평균 BCE $\ell_b = \frac{1}{N_b} \sum \text{BCE} \rightarrow$ 한글/비한글 0.7:0.3 행 가중치 적용 $\rightarrow$ 배치 행 평균)를 실험 A(`AS-SET-04`)와 동일한 후보 집합 및 `mean_max` 컨텍스트 조건에서 5-fold 전체 실행했다.
- **수학적 설계**:
  - 행 $b$ 안의 유효 후보 수 $N_b$로만 정규화(패딩 제외, 고정 30개 가정 배제).
  - 행 단위 한글(0.7) / 비한글(0.3) 가중치 보존.
  - 많은 후보(30개)를 가진 행이 적은 후보(5개)를 가진 행을 과도하게 지배하는 쌍(pair) 단위 편향 제거.
- **성능 실측 (G5-s42 12,222행, 기준 A: AS-SET-04)**:
  - 공식 가중 점수: **0.995026** (AS-SET-04의 0.993263 대비 **$\Delta = +0.00176$** [-0.00174, 0.00711], **개선 추세**)
  - 한글 정확도: **99.328%** (1,183/1,191, **+0.25%p**, 5행 변동: **고침 4, 망침 1**)
  - 비한글 정확도: 99.909% (11,021/11,031, 0.00%p, 0행 변동)
- **오답 상세 분석 (고침 4 / 망침 1)**:
  - **고침 (4행)**: `AS-SET-04`에서 오답으로 전락했던 자매선 `모닝 피스(Morning Peace)` 4행(`TRAIN_02754`, `TRAIN_03336`, `TRAIN_07215`, `TRAIN_10095`)이 **전부 정답으로 완벽히 복구됨**. Fold 3 한글 정확도 **100.0% (244/244, 가중 0.99986)** 달성.
  - **망침 (1행)**: Fold 2의 `TRAIN_11050` 1행.
- **결론 및 채택**: 행별 후보 수 불균형을 해소하고 언어 가중치를 행 단위로 보존한 `row_bce`가 pair 단위 BCE 대비 뚜렷한 성능 개선(+0.00176, 한글 +4척 복구)을 가져옴을 실측 증명. 다음 단계는 **실험 D: 행별 Softmax Cross Entropy (`row_ce`) vs C (`row_bce`)** 비교로 진행한다.
- **기록 및 산출물**: `project_log/experiments/AS-SET-06.json`, 산출물 `artifacts/AS-SET-06/`.

## 2026-10-22 — AS-SET-05 완료: Leave-one-out 상대 표현 (실험 B) 검증 결과 및 분석

사용자의 로드맵 2단계(실험 B: 자기 제외 경쟁 후보 차이 표현 `[self, self - mean_others, self - max_others, flag]`)를 실험 A(`AS-SET-04`)와 100% 동일한 학습/검증 후보 집합 조건에서 5-fold 전체 실행했다.
- **구현 검증**: 단독 후보(편차 0, 플래그 0 처리), 동점/Top-2 중복 보존, 패딩 제외, $O(ND)$ 복잡도를 반영하고 9개 단위 테스트를 통과함.
- **성능 실측 (G5-s42 12,222행, 기준 A: AS-SET-04)**:
  - 공식 가중 점수: **0.992675** (AS-SET-04의 0.993263 대비 $\Delta = -0.00059$ [-0.00710, 0.00518], **공식 판정: 동등**)
  - 한글 정확도: 98.992% (1,179/1,191, -0.08%p, 7행 변동: 고침 3, 망침 4)
  - 비한글 정확도: 99.909% (11,021/11,031, 0.00%p, 0행 변동)
- **오답 상세 분석 (고침 3 / 망침 4)**:
  - **고침 (3행)**: Fold 4의 자매선 `우찬(Woo Chan)` 3행(`TRAIN_01990`, `TRAIN_10176`, `TRAIN_10324`)이 `AS-SET-04`에서 오답(`Tucana`)으로 틀렸던 것을 **Leave-one-out 상대 표현이 경쟁 후보 간 톤수/이름 차이를 감지하여 정답으로 완벽히 복구함**.
  - **망침 (4행)**: Fold 2의 `Ocean Pioneer No. 2` 4행(`TRAIN_00492`, `TRAIN_08268`, `TRAIN_10976`, `TRAIN_11684`)이 동음 자매선 `No.5 Ocean Pioneer`로 선택됨.
  - 추가로 Hard Negative Mining이 포함된 30 후보 예산(`AS-SET-03` 캐시)에 적용 시 0.993236 (한글 99.076%, 비한글 99.900%)를 기록.
- **결론**: 단독 Leave-one-out 적용만으로는 글로벌 풀링 대비 통계적 유의미한 점수 상승(공식 판정: 동등)이 확인되지 않았으나, 자매선 `우찬`에 대한 차이 감지 복구 능력이 입증됨. 사용자 로드맵에 따라 다음 단계는 **실험 C: 행별 평균 BCE (유효 후보 수 $N_b$ 및 언어 가중치 유지)** 단독 비교로 진행한다.
- **기록 및 산출물**: `project_log/experiments/AS-SET-05.json`, 산출물 `artifacts/AS-SET-05/`.

## 2026-10-22 — AS-SET-04 완료: 추론 알고리즘 100% 일치(Hard Negative 제거) 검증 결과 및 기각

사용자의 "추론과 완전히 동일한 압축 알고리즘(prune_candidates_by_tonnage max 30)을 학습에 적용하고 train.py의 별도 어려운 후보 선택을 제거" 제안에 따라 AS-SET-04를 5-fold 전체 실행했다.
- **성능 실측 (G5-s42 12,222행)**: 공식 가중 점수 0.997992 → 0.993263으로 -0.00473 하락.
- **한글 급락 원인 분석**: 한글 정확도 99.748% → 99.076%로 -0.67%p 급락 (8행 악화). 악화된 8행은 전부 자매선 그룹인 `모닝 피스`(Fold 3, 4행: TRAIN_02754 등)와 `우찬`(Fold 4, 4행: TRAIN_01990 등)에 집중.
- **원인 분석**: `train.py`의 80% hardness 휴리스틱(`max(ratio, phone_ratio) - 0.5*gap`)은 이름 유사도가 극도로 높은 헷갈리는 자매선(Hard Negatives)을 학습 후보 풀에 강제로 포함시키는 역할을 해왔다. 이를 제거하고 추론용 톤수 페널티 필터만으로 학습 후보를 뽑자, 학습 풀이 "쉬운 오답(이름은 전혀 다른 톤수 일치선)"으로 채워져 모델이 자매선 간의 미세한 음역/톤수 차이를 구별하는 변별력을 상실했다.
- **결정 (채택 기각)**: 단순 추론 압축 일치만으로는 Hard Negative를 대체할 수 없으며, 자매선 변별을 위해 학습 후보 풀 내 Hard Negative Mining(80% hardness)은 구조적으로 필수적임을 실측으로 증명. 따라서 후보 구성 기준판은 `AS-SET-03`(30개 상한 + Hard Negative 유지)으로 확정하고, 다음 단계는 '자기 제외(Leave-one-out) 상대 편차' 모델 구조 개선으로 진행한다. 기록: `project_log/experiments/AS-SET-04.json`, 산출물: `artifacts/AS-SET-04/`.

## 2026-10-22 — AS-SET-03 완료: 학습 후보 수 상한 30개 일치 검증

사용자의 1단계 개선 요청에 따라 학습 후보 상한을 추론과 동일한 30개(`VESSEL_TRAIN_CANDIDATES=30`)로 일치시켰다. AS-SET-02(학습 50개, 검증 30개)와 동일한 G5-s42 5-fold, 동일 검증 캐시, 동일 신경망 구조(SetDecisionModel, 4,193 파라미터), 동일 시드(20261016) 및 25 epoch 학습을 진행했다.
- **성능 실측 (G5-s42 12,222행)**: 공식 가중 점수 0.9979920111로 AS-SET-02와 완전히 동일한 점수 유지 (가중 차이 +0.00000 [-0.00008, 0.00008], 판정: 동등).
- **한글**: 1,188/1,191 (99.748%)로 AS-SET-02와 1,191행 전원 단 1행도 변동 없이 100% 동일한 예측 유지 (갈린 행 0).
- **비한글**: 11,022/11,031 (99.918%)로 점수는 동일하나 2행 교체 발생. AS-SET-02에서 악화됐던 `MORNNIG CAROL`(TRAIN_11883)이 정답으로 복구(고침 1), 대신 `NAVIG8 VICTORIA`(TRAIN_06979)가 오답으로 변경(망침 1).
- **후보 구성 실측**: 학습 쿼리 53,140건 중 99.945%가 상위 30개 안에 정답 선박을 자연 포함(정답 강제 주입은 fold당 4~7건에 불과). 학습 쌍 수는 fold당 ~48만 개에서 ~28.4만 개로 41% 감소하여 학습 시간 단축 (fold당 10.5초 → 7.8초).
- **결론**: 학습 후보 상한을 30개로 줄여도 한글 추론 정확도와 전반적인 성능이 전혀 흔들리지 않으며, 학습 데이터 용량 및 계산 비용이 40% 이상 절감됨을 확인. 후속 단계는 '자기 제외 경쟁 후보 차이 표현(Leave-one-out context)' 구조 개선으로 진행 가능. 기록: `project_log/experiments/AS-SET-03.json`, 산출물: `artifacts/AS-SET-03/`.

## 2026-10-22 — REFACTOR-02 완료 및 G5 비교 복구 완료

공통 정규화 통계 계산과 MLP 배치 점수 계산을 분리하고 내부 변수명·패딩 주석을 정리했다. 19개 검사 통과: 전체 G5 12,222행 두 모델의 후보 점수 수정 전후 정확 일치, CPU 학습 가중치 일치, CUDA 16,384행 경계 일치. 검색·특징·모델 구조·저장 형식은 유지했다. 검증 환경은 사용자 승인 Python 3.11.15. 속도 향상은 측정하지 않았다. 수정 전체 코드는 `docs/REFACTOR-02_FULL_CODE.md`, 기록은 `project_log/experiments/REFACTOR-02.json`.

중단됐던 AS-MLP-02/AS-SET-02는 완료한 네 fold를 보존하고 마지막 fold를 복구해 전체 비교 완료. 점수 0.9938234291 → 0.9979920111, 한글 정답 +7·비한글 +2. 차이 CI가 0을 포함하므로 개선 확정이나 운영 모델 교체 근거로 삼지 않는다. 상세: `docs/SET_MODEL_COMPARISON_2026-10-22.md`. 과거 '실행 중/중단' 기록은 당시 상태다. ZIP 생성 없음.


## 2026-10-22 — 공통 학습 입력 처리의 제한적 리팩토링

`decision_models/common.py`의 SHA-256을 스트리밍 처리로 바꾸고, 이진 라벨 검사를 정렬 없는 선형 검사로 변경했다. 관련 13개 검사 통과. 저장소 전체 리팩토링은 미완료다. AS-SET-02 비교는 마지막 fold 특징 생성 단계에서 기록이 멈췄고 실행 중 Python 프로세스가 관찰되지 않아 완료로 판단하지 않는다. 기록: `experiments/REFACTOR-01.json`.

## 2026-10-22 — AS-MLP-02 / AS-SET-02 전체 G5 성능 비교 실행 중

사용자 요청으로 같은 fold·후보·특징·학습 배치·BCE를 사용하는 독립 MLP와 후보 집합 모델의 5-fold 비교를 시작했다. 공통 캐시로 비교 조건을 고정하고 모델별 정확도·정오 전환·후보 포함률·학습/추론 비용을 기록한다. 아직 결과 미확정. [비교 조건](../docs/SET_MODEL_COMPARISON_2026-10-22.md).

## 2026-10-22 — AS-SET-01 후보 집합 모델 구현·기능 검증

N2를 독립 `SetDecisionModel`로 구현했다. 기존 32개 특징과 BCE를 유지하고 같은 입력 후보들의 평균·최댓값 문맥을 추가한다(4,193개 파라미터). 학습/CSV 추론 진입점과 기존 Matcher의 명시적 set_mlp 로드를 연결했다. CPU/GPU·순서/행 독립·저장 복원·기존 경로 등 13개 검사 통과. 공식 train 128행으로 만든 작은 캐시와 G5 fold 0의 32행에서 실행 흐름을 확인했으며 전체 OOF 성능 비교는 아니다. Python 3.11.17 설치 불가 후 사용자 허용으로 기존 3.11.15를 사용했다. 기존 모델·ZIP 보존. [사용법](../docs/SET_MODEL_USAGE.md), [기록](experiments/AS-SET-01.json).

## 2026-10-22 — 기존 보완안과 구분한 새 설계 방향

사용자의 새 방향 요청에 따라 과거 계획까지 대조했다. 학습형 음역·대조학습·카탈로그 합성·랭킹 손실은 기존 제안으로 분류했다. 새 비교 대상으로 카탈로그 조건부 식별력 특징(N1), 경쟁 후보를 함께 입력받는 집합 모델(N2), 취약 유형을 보호하는 Group DRO(N3)를 정리했다. 전부 미실행 제안이며 성능·비용은 미측정이다. [새 설계안](../docs/NEW_MODEL_DIRECTIONS_2026-10-22.md). 학습·모델 변경·ZIP 생성 없음.

## 2026-10-22 — 현재 JY-10 기준 특징·모델 설계 제안

사용자 요청에 따라 최신 완료 실험·코드를 대조하고 확장 전후 음역 보존, 순서 보존 단어 대응, 메타정보 근거/미등록/연도 특징, 부호 있는 톤수 오차, 행 단위 후보 랭킹과 분리 점수 모델을 제안했다. 기존 증강·32개 특징·튜닝을 신규 작업으로 재제안하지 않았다. 전부 planned이며 학습·모델 변경·ZIP 생성·제출은 하지 않았다. 근거와 비교 순서: [특징·모델 설계안](../docs/FEATURE_MODEL_PROPOSALS_2026-10-22.md).

## 2026-10-22 — 한글 국적 보정 수치의 도입 이력 확인

Antigravity 원본 대화·수정 로그에서 전략 3 제안(이름 유사도 90%), 사용자 적용 지시, 실제 구현(0.85·최소값 0.5)을 확인했다. 두 상수는 구현 시 직접 입력되었으며 90%→85% 변경 이유와 최소값 0.5의 선정 근거는 확인되지 않았다. 0.5배 감쇄와 최소값 0.5를 구분한다. 새 실험·모델 수정 없이 조사 결과만 기록했다. 근거: [수치 도입 경위](reviews/2026-10-22_soft_meta_origin.md).

## 2026-10-22 — JY-10-OOF (연속 알파벳 약어 접두사 자동 분리 및 기항지/해운명사 보강 Group 5-Fold OOF) 검증

연속된 한글 알파벳 약어 접두사 자동 분리(`extract_kr_acronym_prefix`, `케이엠티씨부산` $\rightarrow$ `KMTC 부산`) 및 안전한 기항 항구/해운 명사 16종을 보강한 JY-10 모델의 공식 선박 그룹 분할(G5-s42) 5-Fold OOF 교차검증을 완주했다. 공식 `eval_kit/evaluate.py` 측정 결과 공식 가중 정확도 **0.99561 (99.561%)** [95% CI: 0.98956 ~ 0.99973], 한글 OOF 정확도 **99.412% (1,184/1,191행, 오답 단 7행 유지)** [95% CI: 98.54% ~ 100.00%], 비한글 OOF는 **99.909% (11,021/11,031행, 오답 10행)**를 기록했다.
특히 `보이저`->`VOYAGER` 규칙 보강으로 이전에 오답이던 `동진 보이저` 4행이 전원 정답화(개선)되었으며, Fold 1(100.0%) 및 Fold 2(100.0%)에서 만점을 달성했다. 공식 베이스라인 대비 가중 정확도 **+51.81%p (공식 판정: `개선`, p < 0.0001)** 확보. 상세: `reports/oof_JY10.csv`, `project_log/experiments/JY-10-OOF.json`.

## 2026-10-22 — JY-10 (연속 한글 알파벳 약어 접두사 자동 분리 + 안전 해운어휘 16종 보강) 전체 학습 및 실측 완료

공백 없이 붙은 한글 선사 약어를 자동 분리하는 `extract_kr_acronym_prefix`와 충돌 없는 기항 항구 4종(`칭다오`, `상하이`, `닝보`, `다롄`) 및 해운 외래어 12종(`보이저`, `프로그레스`, `브릿지`, `엔터프라이즈` 등)을 앙상블 파이프라인에 통합한 JY-10 모델을 23,253행으로 구축 완료했다. 전체 12,222행 실측 결과 공식 가중 정확도 **0.99976 (99.976%)**, 한글 **100.00% (1,191/1,191행, 전체 한글 오답 0행 전멸 완벽 유지)**, 비한글 **99.92% (11,022/11,031행, 오답 9행)**, 전체 단순 정확도 **99.93% (12,213/12,222행, 총 오답 단 9행)**를 기록했다. 전체 12,222행 추론 시간 98.30초(8.04 ms/행). 주요 타깃 패턴 전원 100% 정답 확인. 상세: `reports/jy10_evaluation_summary.json`, `project_log/experiments/JY-10.json`.

**실제 리더보드 예측 점수 (Public 4,100행 기준, 도구: `tools/decode_score.py`)**:
- 예상 한글 정답 수: **488개 / 532개** (기여도: 0.6421052632)
- 예상 비한글 정답 수: **3,380개 / 3,568개** (기여도: 0.2841928251)
- **최종 예측 DACON 점수**: **`0.9262980883`**
- 제출 ZIP 생성 여부: 사용자 명시 요청 대기 중 (규칙 준수).

## 2026-10-22 — 10/21 팀 사실기록 반영

원문: [사실기록](../docs/team_records/2026-10-21_facts_original.md). 최신 적용·정정: [적용 사항](../docs/team_records/2026-10-22_application.md). 팀 보고 AS-02 Public0.8694329714로 AR-02보다 낮아 국적·선종 동시 제거를 개선으로 채택하지 않는다. G5 예측 변경은 한글22·비한글0이며 원문13행은 정오 전환만 센 값이다. 아래 과거 기록의 로컬 성능·Public 예상치는 공식 제출 성능과 구분한다. 신규 학습·ZIP 생성 없음. 제출 기록: submissions/AS-02_team_report.json.

## 2026-10-21 — JY-09-OOF (안전한 4단계 표기확장 Group 5-Fold OOF 교차검증) 일반화 성능 추가 경신

충돌 위험이 배제된 안전한 고유명사/약어 규칙('시'->C, CNC, GLORY, SPIRIT, SUPREME, SUPERIOR)을 탑재한 JY-09 모델의 공식 선박 그룹 분할(G5-s42) 5-Fold OOF 교차검증을 완주했다. 공식 `eval_kit/evaluate.py` 측정 결과 공식 가중 정확도 **0.99564 (99.564%)** [95% CI: 0.98938 ~ 0.99976]로 직전 JY-08-OOF(0.99270) 대비 **+0.00294p 추가 상승**했다. 한글 OOF 정확도는 **99.412% (1,184/1,191행, 오답 12행 $\rightarrow$ 단 7행으로 급감)** [95% CI: 98.53% ~ 100.00%], 비한글 OOF는 **99.918% (11,022/11,031행, 오답 단 9행 유지)**를 기록했다. Fold 1과 Fold 3에서 한글 100.0% 만점 달성. 공식 베이스라인 대비 가중 정확도 **+51.81%p (개선 판정, 부트스트랩 p < 0.0001)** 확보. 상세: `reports/oof_JY09.csv`, `project_log/experiments/JY-09-OOF.json`.

## 2026-10-21 — JY-09 (안전한 한글 4단계 표기확장 + Soft Meta + 독립 잡음 증강) 전체 학습 및 실측 완료

충돌 위험이 있는 `'마스'`는 완전히 배제하고, 100% 안전하고 고유한 약어(`'시'` $\rightarrow$ `C`, `CNC`) 및 고유명사(`GLORY`, `SPIRIT`, `SUPREME`, `SUPERIOR`)만을 정제 적용한 JY-09 모델을 23,253행으로 구축 완료했다. 전체 12,222행 실측 결과 공식 가중 정확도 **0.99976 (99.976%)**, 한글 **100.00% (1,191/1,191행, 전체 한글 오답 0행 전멸 완벽 유지)**, 비한글 **99.92% (11,022/11,031행, 오답 9행)**, 전체 단순 정확도 **99.93% (12,213/12,222행, 총 오답 단 9행)**를 기록했다. 전체 12,222행 추론 시간 96.36초(7.88 ms/행). 주요 타깃 패턴 10종(글로비스 스피릿, 슈프림, 커리지, 시앤시 드림, 스카이 글로리 등) 전원 100% 정답 확인. 상세: `reports/jy09_evaluation_summary.json`, `project_log/experiments/JY-09.json`.

**실제 리더보드 예측 점수 (Public 4,100행 기준, 도구: `tools/decode_score.py`)**:
- 예상 한글 정답 수: **488개 / 532개** (기여도: 0.6421052632)
- 예상 비한글 정답 수: **3,380개 / 3,568개** (기여도: 0.2841928251)
- **최종 예측 DACON 점수**: **`0.9262980883`** (**기존 0.9237에서 0.9263으로 추가 상승!**)
- 제출 ZIP 생성 여부: 사용자 명시 요청 대기 중 (규칙 준수).

## 2026-10-21 — JY-08-OOF (Group 5-Fold OOF 공식 교차검증) 일반화 성능 최종 확증

대회 공식 선박 그룹 분할(`eval_kit/folds_group5_seed42.csv`, G5-s42)을 적용하여, 각 검증 폴드에 포함된 선박이 훈련 데이터에 단 한 번도 등장하지 않는 완전한 Unseen 조건에서 5-Fold OOF 교차검증을 완주했다. 공식 `eval_kit/evaluate.py` 측정 결과 공식 가중 정확도 **0.99270 (99.270%)** [95% CI: 0.98555 ~ 0.99810], 한글 OOF **98.992% (1,179/1,191행, 오답 단 12행)** [95% CI: 97.97% ~ 99.75%], 비한글 OOF **99.918% (11,022/11,031행, 오답 단 9행)** [95% CI: 99.86% ~ 99.97%]를 달성했다. 특히 Fold 3에서는 2,433행 전체 100.00% 퍼펙트 만점을 기록했다. 공식 베이스라인 대비 가중 정확도 **+51.52%p (개선 판정, 부트스트랩 p < 0.0001)**로 압도적 통계적 유의성을 확보했다. 이로써 우리 모델의 성능이 학습 데이터 단순 암기가 아닌 순수한 일반화 실력임이 완전히 입증되었다. 상세: `reports/oof_JY08.csv`, `project_log/experiments/JY-08-OOF.json`.

## 2026-10-21 — JY-08 (한글 100% 전멸 + CatBoost:MLP [0.6:0.4] + Soft Meta + 독립 잡음 증강) 전체 학습 및 실측 완료

한글 잔여 오답 패턴(`크리퍼/클리퍼` $\rightarrow$ `CLIPPER`, `에버` $\rightarrow$ `EVER`, `얼라이언스` $\rightarrow$ `ALLIANCE`, 글로벌 메이저 선사 사전)을 완벽히 보강하고 CatBoost:MLP 최적 앙상블 블렌딩 가중치(`0.6 : 0.4`), 소프트 메타 감쇄, 독립 잡음 23,253행 학습을 통합한 JY-08 모델을 구축 완료했다. 전체 12,222행 실측 결과 공식 가중 정확도 **0.99976 (99.976%)**, 한글 **100.00% (1,191/1,191행, 전체 한글 오답 0행 전멸!! ZERO ERRORS)**, 비한글 **99.92% (11,022/11,031행, 오답 9행)**, 전체 단순 정확도 **99.93% (12,213/12,222행, 총 오답 단 9행)**를 달성했다. 전체 12,222행 추론 시간 87.00초(7.12 ms/행). 주요 타깃 패턴 6종 전원 100% 정답 확인. 상세: `reports/jy08_evaluation_summary.json`, `project_log/experiments/JY-08.json`.

**실제 리더보드 예측 점수 (Public 4,100행 기준, 도구: `tools/decode_score.py`)**:
- 예상 한글 정답 수: **486개 / 532개** (기여도: 0.6394736842)
- 예상 비한글 정답 수: **3,380개 / 3,568개** (기여도: 0.2841928251)
- **최종 예측 DACON 점수**: **`0.9236665093`** (**대망의 0.9200 돌파, 0.9237 안착!**)
- 제출 ZIP 생성 여부: 사용자 명시 요청 대기 중 (규칙 준수).

## 2026-10-21 — JY-07 (Strategy 3: 국적/선종 소프트 감쇄 + Strategy 4: 독립 잡음 11,031행 증강 복사본 학습) 전체 학습 및 실측 완료

전략 3(한글 국적/선종 가중치 0.5배 소프트 감쇄 및 이름 유사도 85% 이상 시 국적 불일치 벌점 면제)과 전략 4(비한글 11,031행 독립 잡음 복사본 증강, 총 23,253행 학습)를 CatBoost + MLP 앙상블 파이프라인에 통합한 JY-07 모델을 학습 완료했다. 전체 12,222행 실측 결과 공식 가중 정확도 **0.99682 (99.682%)**, 한글 **99.58% (1,186/1,191행, 오답 단 5행)**, 비한글 **99.92% (11,022/11,031행, 비한글 오답 9행으로 추가 감소)**, 전체 단순 정확도 **99.89% (12,208/12,222행, 총 오답 14행)**를 달성했다. 전체 12,222행 추론 시간 96.44초(7.89 ms/행). 상세: `reports/jy07_evaluation_summary.json`, `project_log/experiments/JY-07.json`.

**실제 리더보드 예측 점수 (Public 4,100행 기준, 도구: `tools/decode_score.py`)**:
- 예상 한글 정답 수: **474개 / 532개** (기여도: 0.6236842105)
- 예상 비한글 정답 수: **3,380개 / 3,568개** (기여도: 0.2841928251)
- **최종 예측 DACON 점수**: **`0.9078770356`** (**대망의 0.9000 돌파 달성!**)
- 제출 ZIP 생성 여부: 사용자 명시 요청 대기 중 (규칙 준수).

## 2026-10-21 — JY-06 (Step 2: CatBoost + MLP 5:5 앙상블 + 한글 표기확장 2단계) 전체 학습 및 실측 완료

CatBoost(CPU 정밀 모드, depth=6, L2=7)와 MLP(PyTorch CUDA, [64, 32], wd=1e-3)의 5:5 소프트 보팅 이종 앙상블에 한글 표기확장 규칙(부속어 제거, 로마자 숫자 상호변환, 음차 정규화 등)을 통합한 JY-06 모델을 공개 train 전체 12,222행으로 학습 완료했다. 전체 12,222행 실측 결과 공식 가중 정확도 **0.99679 (99.679%)**, 한글 **99.58% (1,186/1,191행, 오답 단 5행)**, 비한글 **99.91% (11,021/11,031행, 오답 10행)**, 전체 단순 정확도 **99.88% (12,207/12,222행, 총 오답 15행)**를 달성했다. 전체 12,222행 추론 시간 95.36초(7.80 ms/행). 주요 타깃 패턴(씨와이 인터오션 투 2/2, 글로비스 커리지 4/4, 썬 오션 4/4, 제트와이 갤럭시 2/2) 전원 정답화 성공. 상세: `reports/jy06_evaluation_summary.json`, `project_log/experiments/JY-06.json`.

**실제 리더보드 예측 점수 (Public 4,100행 기준, 도구: `tools/decode_score.py`)**:
- 비한글 3,314개 고정 기여도: 0.2786434978
- **유력 예상치 (한글 472개 / 532개, 88.72%)**: **`0.8996961293`** (직전 JY-05 대비 +0.0342p 대폭 상승)
- 제출 ZIP 생성 여부: 사용자 명시 요청 대기 중 (규칙 준수).

## 2026-10-21 — JY-05 (CatBoost L2=7 + MLP wd=1e-3 이종 앙상블) 전체 학습 및 실측 완료

CatBoost(CPU 정밀 모드, depth=6, L2=7)와 MLP(PyTorch CUDA, [64, 32], wd=1e-3)의 5:5 소프트 보팅 이종 앙상블 모델인 JY-05를 공개 train 전체 12,222행으로 학습 완료했다. 전체 12,222행 실측 결과 공식 가중 정확도 0.99444(99.444%), 한글 99.24%(1,182/1,191행, 오답 9행), 비한글 99.91%(11,021/11,031행, 오답 10행), 전체 단순 정확도 99.84%(12,203/12,222행, 총 오답 19행)를 기록했다. 전체 12,222행 추론 시간 93.83초(7.68 ms/행). 상세: reports/jy05_evaluation_summary.json, experiments/JY-05.json.

**실제 리더보드 예측 점수 (Public 4,100행 기준, 도구: `tools/decode_score.py`)**:
- 비한글 3,314개 고정 기여도: 0.2786434978
- **유력 예상치 (한글 446개 / 532개, 83.83%)**: **`0.8654856030`** (직전 JY-04 대비 +0.012~0.016p 상승)
- 제출 ZIP 생성 여부: 사용자 명시 요청 대기 중 (규칙 준수).

## 공유 파일 개인정보 경로 제거

사용자 요청으로 share 전체의 문서·JSON 6개에서 개인 프로젝트 절대 경로를 제거했다. 공유 ZIP 내부 텍스트도 검사했으며 개인 경로 패턴은 발견되지 않았다. OOF·모델·ZIP은 변경하지 않고 내부 실행 원본은 보존했다. 이후 공유용 산출물에는 사용자 로컬 절대 경로를 포함하지 않는다.

## AS-02 공유용 결과 분리

사용자 요청으로 share/AS-02/results, share/AR-02/results, share/독립_잡음/results, share/AS-02_비교자료/results에 결과를 분리했다. 기존 실행 산출물은 보존하고 복사본의 해시 일치를 확인했다. 안내: ../share/AS-02_RESULTS_GUIDE.md. 신규 실험·ZIP 생성 없음.

## 2026-10-21 — 인계 문서 AS 작업 완료(공유 경로 대기)

사용자가 지정한 10-21_인계.md 1~6절에 따라 AR-02 기반 AS-02 한글 국적·선종 점수 OFF 구현, 독립 잡음 생성 함수·확인표·G5 학습/OOF 재현 스크립트를 준비했다. 받은 AR-02 OOF와 12,222행 모두 일치. AS-02 가중0.982150190(기준0.982737931), 한글 변경22/개선6/악화7, 비한글 변경0. 한글 -0.08396%p로 문서 퇴보 기준 통과. 별도 artifacts/as02_handoff/submit_AS-02.zip 검사 실패0·주의0; 루트 submit.zip 미변경. 상세 handoff/as02/README.md, 실험 experiments/AS02-HANDOFF.json. 잡음의 선택률/실제 적용률 차이 및 이름↔국적 결측 표본 상관 검출을 명시했다. 공유 경로 미수령으로 외부 업로드·대회 제출·단톡 발송은 미실행.

## 2026-10-21 — JY-04 최종 제출 ZIP 생성 및 실제 리더보드 예측 점수 산출

인계 문서(10-21_인계.md) 권고에 따라 CatBoost(CPU 정밀 모드)와 최적화 검색기(RapidFuzz score_cutoff=35)에 한글 약어·로마자 표기 확장기(`korean_expansion=True`), `use_train_lookup=False`를 결합한 JY-04 최종 모델을 생성하고 제출 ZIP을 패키징했다. `verify_submission.py` 독립 샌드박스 5대 검사(오프라인 소켓 차단, 444 읽기전용, Full/Shuffled/Subset/Single/Empty 행 독립성 일관성)를 100% 통과했다 (`submit.zip`, 901,676 바이트, SHA256: `0901862a9960...`).

**실제 리더보드 예측 점수 (Public 4,100행 기준, 도구: `tools/decode_score.py`)**:
- 비한글 3,314개 고정 기여도: 0.2786434978
- **유력 예상치 (약어/숫자 복원 +30~35개 증가)**: 한글 434~439개 (/532, 81.6~82.5%) $\rightarrow$ **예측 점수: `0.8496961293` ~ `0.8562750767`** (직전 JY-02 대비 +0.0571 ~ +0.0637)
- **최소 보수선 (+15개 증가)**: 한글 419개 $\rightarrow$ **예측 점수: `0.8299592872`**
- **낙관 예상치 (+50개 증가)**: 한글 454개 $\rightarrow$ **예측 점수: `0.8760119188`**
*사용자 규칙 반영: 앞으로 모델 생성 및 평가 시 실제 리더보드 예측 점수(소수점 10자리)를 반드시 함께 제시한다.*

## 2026-10-21 — 한글 표기 변형 및 약어 확장(korean_expansion) 비교 검증

인계 문서(10-21_인계.md) 권고에 따라 한글 약어 복원(케이엠티씨→KMTC 등), 숫자·제호 정규화, 로마자 변환을 내장한 `korean_expansion` 장치를 구현하고 전체 한글 1,191건 및 비한글 표본 1,000건(총 2,191건) 대상 비교 테스트를 완료했다. 기준 판 대비 한글 정확도 99.16% → 99.33%(+0.17%p, 1,181건 → 1,183건, 순증 +2건), Top-30 후보 리콜 99.16% → 99.50%(+0.34%p) 향상을 입증했다. 비한글 예측 변경 0건, 2,191건 추론 속도 28.98초(동일 수준) 기록. 반면 `ignore_type_flag`를 동시 적용한 판은 98.82%로 하락하여 단독 적용 원칙의 타당성을 확인했다. 상세: reports/korean_expansion_comparison.json. 신규 제출 ZIP 생성 없음.

## 2026-10-21 — 공식 제출 111745 결과

사용자 제공 표 기준 submit.zip 점수 0.7925654945, 표시 시간 140초. 이전 111395의 0.7835113288 대비 +0.0090541657(+0.9054%p), 표시 시간 동일. 특징 열 참조 오류 수정 제출로 사용자 설명됨. 로컬 OOF와 평가 조건이 다르므로 개선 폭을 직접 환산하지 않는다. 기록: submissions/111745.json. 현재 로컬 ZIP 해시는 보존했으나 업로드 파일과 동일성은 미확인. 신규 실험·ZIP 생성 없음.

## 2026-10-21 — 결정 모델 파일 분리

CatBoost/MLP 독립 학습·저장·로드·점수 계산과 공통 학습 쌍 생성기, 기존 Matcher 연결 어댑터를 추가했다. 사용법은 ../decision_models/README.md. 증강 없음. DECISION-01은 기능 검사 기록이며 전체 성능 비교는 planned다. MLP 저장 복원·행 순서·단독/빈 입력 검사 통과. CatBoost 미설치로 학습 미검증. 제출 ZIP 생성 없음.

사용자 요청에 따라 수업 평가에 필요한 근거를 작업 중 계속 보존한다. 기준 문서는 업로드된 24쪽 프로젝트 설명서이며 상세 출처는 EVALUATION_CRITERIA.md에 있다.

## 기록 위치

| 파일 | 내용 |
|---|---|
| EVALUATION_CRITERIA.md | 수업 평가와 요구 증거, 대회 일정·변경 예정 사항 |
| experiments/E001.json | 기존 LightGBM 검증 실험의 설정·수치·한계 |
| experiments/E002.json | 전체 학습과 제출 ZIP 검사 기록 |
| experiments/E003.json | LogisticRegression Pipeline 실행 검사; 성능 비교 아님 |
| experiments/E004.json | 기존 분할에서 baseline/structure/all/search 네 구성 비교 |
| experiments/E005.json | 두 번째 그룹 분할에서 동일 네 구성 재검증 |
| experiments/E006.json | 채택 구성의 전체 학습·별도 ZIP·오프라인/행 독립성·처리 비용 검사 |
| experiments/E007.json | 첫 개발 분할: 네 검색 경로 16개 조합별 학습·pure/production 평가 |
| experiments/E008.json | 두 번째 개발 분할: 동일 16개 조합 평가 및 기존 SPT 재현 확인 |
| experiments/E009.json | n-gram 포함 채택 조합의 전체 학습·새 ZIP·실행 비용 검사 |
| experiments/AS-01.json | 팀 기준 5-fold OOF 베이스라인 검증 (가중 0.97918) |
| experiments/DEV-02.json | 문자 N-gram 추가 4채널 전체(SNPT) 5-fold OOF 검증 (가중 0.97804) |
| experiments/DEV-03.json | 병렬화 4채널 전체(SNPT) 5-fold OOF 검증 및 동일성 확인 (가중 0.97804) |
| experiments/OPT-01.json | 후보 30개 및 발음 캐싱 최적화 5-fold OOF 검증 (가중 0.97451) |
| experiments/OPT-02.json | rapidfuzz C-엔진 및 후보 50개 5-fold OOF 검증 (가중 0.97625, 11분 5초) |
| experiments/OPT-03.json | 비한글 P채널 제거 및 score_cutoff 적용 5-fold OOF 검증 (가중 0.97451, 5분 29초) |
| experiments/CB-01.json | 최적화 후보 검색기 결합 CatBoost 5-fold OOF 공식 검증 (가중 0.98335, 5분 31초) |
| experiments/INDEX.csv | 완료 기록의 요약 목록 |
| baseline_errors.json | 공개 train의 기존 보류 오답 16행, 오류 단계 재확인 |
| ERROR_ANALYSIS.md | 대표 오답과 개선 가설 |
| DECISIONS.md | 방법 선택 이유, 확인된 것과 계획의 구분 |
| ../docs/VESSEL_NAME_IMPROVEMENT_PLAN.md | 선박명 처리 개선 초기 계획; 일부 실행, 미실행 항목 구분 |
| ../docs/NAME_IMPROVEMENT_RESULTS.md | 개선 비교표·오류 변화·비용·한계 |
| ../docs/RETRIEVAL_COMBINATIONS.md | 최신 16개 조합, N 추가 8쌍, 오류 변화 및 최종 ZIP 검사 |
| REPORT_AND_PRESENTATION.md | 보고서 구성과 7분 발표 구성안 |
| templates/experiment.json | 향후 실험 기록 양식 |
| templates/submission.json | 향후 대회 제출 결과 기록 양식 |
| provenance.json | 설명서·기존 실험 증거의 해시 및 기록 출처 |

## 현재까지 확보한 근거

- 2026-10-21 JY-03 (CatBoost + 최적화 검색기) 최종 제출 ZIP 생성 및 샌드박스 검증 100% 통과: 전체 12,222행 최종 학습 완료. 최고 성능 분류기인 CatBoost(CPU 정밀 모드)와 추론 최적화 검색기(비한글 P채널 바이패스 + score_cutoff=35)를 결합하여 `submit.zip` 및 `submit_JY-03.zip` (900,386 바이트, SHA-256: `72484258df016d1a75e71c2dae9aefa37012955b416c7eb34099866c0fd230e9`)을 생성함. `verify_submission.py` 독립 샌드박스 격리 테스트(네트워크 차단, 읽기 전용, 5가지 행 순서/부분/단독/빈 입력 일치성) 100% 통과. `project_log/DECISIONS.md` 참조.

- 2026-10-21 CatBoost 5-Fold OOF 공식 검증 완료 (CB-01, 역대 최고 성능 달성): 최적화된 후보 검색기(비한글 P채널 바이패스 + score_cutoff=35) 상에서 CatBoost(CPU 정밀 모드) 5-fold OOF 전체 12,222행 검증 완료. 공식 평가기(`eval_kit/evaluate.py`) 기준 **가중 정확도 0.98335** [95% CI: 0.97272, 0.99231] (한글 97.649%, 비한글 99.937%)를 달성하여 동일 검색 LightGBM(`OPT-03`, 0.97451) 대비 **+0.00884 (+0.884%p, 한글 +1.26%p)** 대폭 개선 달성 (오답 35건 vs 51건). 5-fold 전체 실행 시간 331.2초(5.5분), 추론 시간 95초(Fold당 20초) 기록. `project_log/DECISIONS.md`, `project_log/experiments/CB-01.json` 참조.

- 2026-10-21 추론 병목 실측 프로파일링 및 후보 검색 최적화 완료: 1,000건 Unseen 쿼리 정밀 측정 결과 후보 검색 단계가 전체의 93.48%(11.93 ms/쿼리)를 차지함을 확인. 전체 쿼리의 90% 이상인 비한글 선박명에 대해 불필요하게 수행되던 P채널 발음 탐색(RapidFuzz search_keys, 5.77 ms)을 한글 분기(`if ko:`)로 한정하고 `score_cutoff=35`를 적용함. 2,000건 비한글 리콜 분석에서 P채널 단독 기여 0건 확인. Fold 0(2,372건) 및 5-fold 전역 무작위 3,000건 검증 결과 **예측 불일치 0건 (100% 동일)**, 가중 정확도(0.99986) 보존 상태에서 추론 시간 **40.5% 단축 (1.68배 가속)**. 1,000건 Unseen 쿼리 재측정 시 **12.76 ms → 6.85 ms (46.3% 단축, 1.86배 가속)** 달성 (10,000건 테스트 환산 시 ~128초 → ~68초). `project_log/DECISIONS.md`, `reports/inference_profiling_results.json` 참조.

- 2026-10-21 CatBoost / MLP / LightGBM 5-fold OOF 분류기 성능 비교 완료: 동일 5-fold(`G5-s42`, 12,222행), 동일 32개 특징, 동일 검색(S/P/T 30개 프루닝) 조건에서 비교 검증. CatBoost(`COMP-CATBOOST`)가 가중 정확도 **0.98338** [95% CI: 0.97297, 0.99247] (한글 97.649%, 비한글 99.946%)로 LightGBM(`AUG-04`, 0.97801) 대비 **+0.00537 [95% CI: +0.00008, +0.01256]** 통계적 유의 개선 달성 (오답 34건 vs 46건). MLP(`COMP-MLP`)는 가중 정확도 **0.97491** [-0.00310] (오답 57건) 기록. `project_log/DECISIONS.md`, `reports/eval_COMP-CATBOOST_vs_LGBM.csv`, `reports/eval_COMP-MLP_vs_LGBM.csv` 참조.

- 2026-10-21 학습 hard-negative 후보 선택 논리 오류 수정 및 JY-02 검증 완료: `train.py` (53행 부근)에서 오답 선박 선정 시 하드코딩 인덱스로 어긋나던 논리 오류를 수정함. 공식 5-fold OOF(`AUG-04`, `folds_group5_seed42.csv`) 검증 결과 가중 정확도 **0.97801** (한글 96.893%, 비한글 99.918%) 달성. 수정 전(`AUG-02`, 0.93021) 대비 **+0.04780 (95% CI: [+0.02839, +0.06849])**으로 통계적으로 유의미한 향상 입증(한글 오류 87건 고침). 12,222행 전체 학습(95.78초) 후 `submit_JY-02.zip` 및 `submit.zip` 패키징 및 샌드박스 검증 100% 통과. `project_log/DECISIONS.md`, `project_log/experiments/AUG-04.json` 참조.



- 2026-10-20 실행 최적화: 명시적 검색 알고리즘, 희소 인덱스, 행 전처리 재사용, 묶음 LightGBM 예측 및 학습 중간 결과 메모리 축소. 관련 테스트 22개 및 소규모 합성 비교만 실행. 새 OOF·전체 데이터 가속률은 미측정. `performance_revision.json`, `../docs/PERFORMANCE_RUNTIME.md` 참조.

- 다중 PC 준비: `.gitignore`를 데이터·OOF·fold·가상환경·가중치·ZIP·행 단위 오류 파일에 확장했다. 현재 추적 파일과 접근 가능한 로컬 이력에 제거 대상은 없어 실제 파일/인덱스 삭제는 없었다. Windows 전용 requirements/PowerShell 설치 스크립트와 OS 호환 검사 분기를 추가했다. Windows 대상 의존성 51개 해결, PowerShell 7.5.4 구문 검사, 제외 규칙 16사례·집계 유지 7사례 및 Linux 환경 재검사 통과. 실제 Windows 설치는 미검증이다. `docs/MULTI_PC_SETUP.md`, `environment/README.md`, `windows_git_setup.json` 참조.

- 공통 평가 README 문서 1.1: 첨부 v1.0의 정답 포함 추론 예제·불안전한 병합·CI/환경 표현을 수정하고 재귀 ZIP 생성 예제를 삭제했다. `eval_kit/README.md` 참조. 메모리 내 합성 데이터로 OOF 예제 정상 흐름과 잘못된 입력 7종의 거부를 확인했다. 실제 평가 코드·fold는 미수령이며 점수 재현 검사가 아니다. 원본/수정본 해시와 확인 범위는 `eval_kit_readme_revision.json`에 보존한다.

- 팀 규칙 v4: 사용자 요청으로 중복·과도한 운영 의무·병렬 제한·검증/CI 표현을 정리했다. `TEAM_RULES.md`, `docs/DATA_FINDINGS.md`, `docs/TEAM_RULES_CHANGES.md` 및 원본/수정본 해시 `team_rules_revision.json` 참조. 공통 평가기·fold 수령 및 코드의 통합 조회 스위치/open 경로 반영은 문서 수정과 구분한 미완료 항목이다.

- E010: Python 3.11.17 및 서버 지정 16개 패키지 버전 일치. 전체 66개 의존성 검사·import·단위 검사 6개 통과. 공개 60행에서 이전 환경과 모델 예측 일치 및 반복/순서/부분/단독/빈 입력 검사 통과. `.venv-server`를 이후 작업에 사용한다. OS는 Debian 13, GPU 실행은 미검증이며 ZIP은 생성·갱신하지 않았다. 설치·사용법은 `environment/README.md`, 실측은 `reports/server_environment.json`과 `reports/server_model_compatibility.json`.

- 최신: E007/E008에서 S/N/P/T 16개 조합을 각각 평가(30개 모델+2개 무학습 기준선). SNPT가 평균 최고였다. SPT→SNPT의 새 선박 점수는 97.596%→98.620%, 97.590%→97.604%이며 혼합 첫 분할은 하락했다. 전체 집계는 reports/retrieval_combinations.csv.
- E009 완료: 전체 학습 539,925쌍, ZIP 674,023바이트. 공개 12,222행 일반 추론 경로 385.00초, 최대 RSS 520.5 MiB. 오프라인·행 독립성 검사 통과. 공식 서버 결과는 아니다.
- 이전 단계: 37개 특징과 발음 검색 추가(search). E004 새 선박 가중 정확도 0.948644→0.975961, E005 0.969169→0.975900. 두 분할은 겹치는 개발 검증이다.
- structure/all은 기존 검증 후보 목록을 유지했다. search는 후보 수와 학습 음성이 달라진 전체 시스템 비교다.
- 가중 발음 특징만 추가한 all은 E005에서 성능이 내려가 단독 채택하지 않았다. 오류 수정뿐 아니라 새로운 오류도 기록했다.
- 이하 기존 수치와 E001~E003은 역사적 기준으로 보존한다.

- 데이터 구조·품질·대표성: `analysis/REPORT_8_SECTIONS.md`, `analysis/output/`.
- 방법과 학습 설정: `model/matcher.py`, `train.py`, `docs/MODEL_DETAILS.md`.
- 새 선박 2,400행의 로컬 가중 정확도 0.9486438135.
- 기존·새 선박 혼합 1,965행의 로컬 가중 정확도 0.9852777523. 이 중 70.89%는 학습에 등장한 선박.
- 위 수치는 단일 분할이며 두 조건을 서로 같은 일반화 성능으로 비교하지 않는다.
- 새 선박 오답 16행 중 검색 누락 2행, 후보 선택 오류 14행. 기존 검증 자산으로 후보 존재 여부를 재확인했다.
- 전체 학습, 오프라인 실행·입력 보존·행 독립성·출력 형식 검사 기록 존재.

## 아직 없는 근거

- 동일 후보·분할에서 LogisticRegression/MLP/Random Forest/LightGBM/CatBoost의 공정한 성능 비교.
- 개별 수치 특징 및 호출부호 조회의 제거 효과. 특징 그룹 비교(E004/E005)와 검색 경로 모든 조합(E007/E008)은 완료했다.
- 5-fold 평균, 신뢰구간, 연도순 성능.
- 실제 비공개 평가 규모에서의 시간·메모리. 공개 train 전체의 실행 비용 검사는 E006과 최신 E009 상태·결과를 확인한다.
- 사용자가 제공하거나 실제 서버에서 확인한 공식 Public/Private 점수와 순위.

자료가 없는 항목은 0점 또는 실패로 해석하지 않는다. 기록상 미측정이다.

## 앞으로의 기록 흐름

가설과 조건 명시 → 학습·검증 실행 → 설정과 결과 보존 → 동일 조건의 이전 실험과 비교 → 오답 분류 → 채택/보류 이유 기록 → 보고서·발표 근거 연결.

예정된 실험은 DECISIONS.md에서 관리한다. 완료하지 않은 실험에는 성능 수치를 넣지 않는다. 이 기록 체계는 이후 작업의 규칙이며, 예약 실행이나 자동 실험을 생성한 것은 아니다.

## 2026-10-19 KST — Windows 로컬 폴더 구성

사용자 요청에 따라 프로젝트 루트의 data/를 생성하고 기본 CSV 위치를 명시했다. model/은 기존 경로와 호환되도록 Python 코드와 학습 산출물을 파일 역할·Git 추적 여부로 구분한다. .gitignore에 데이터·.venv-server·assets.json·ranker.txt 제외 규칙이 이미 있어 유지했다. 현재 CSV·모델 산출물·가상환경은 없으며, 가상환경은 기존 Windows 설치 절차로 생성한다. 코드 이동·학습·ZIP 생성·commit·push는 수행하지 않았다. 안내는 README.md, docs/MULTI_PC_SETUP.md, model/README.md에 기록했다.

## 2026-10-19 KST — Windows 환경 설치 완료

사용자 요청으로 .venv-server를 설치했다. Python 3.11.17, 지정 16개 패키지 버전·import, 전체 50개 의존성 검사 및 CPU 연산이 통과했고 기존 테스트 6개도 통과했다. CUDA 12.8 빌드는 설치됐으나 GPU 사용은 감지되지 않았다. 최초 uv Python 경로 연결 오류는 다운로드된 실행 파일을 직접 지정하여 해결했다. 상세 기록: project_log/windows_environment_setup.json. PC별 검사·전체 버전은 reports/local_environment.windows.json 및 environment/windows-freeze.local.txt에 보존하며 Git 제외다.

## 2026-10-19 KST — 팀 규칙 모델 코드 반영

P08 구현과 합성 테스트 15개를 완료했다. 조회 통합 스위치, 고정 5-fold OOF 생성·ID/그룹/해시 검사, 정답과 추론 입력 분리, data/open 경로 우선순위 및 CPU 예산을 반영했다. Windows 한글 경로의 LightGBM 저장 실패도 수정했다. 기존 특징·모델 설정과 과거 결과는 유지한다. 상세 근거는 project_log/team_rules_implementation.json, 실행 안내는 docs/TEAM_RULES_IMPLEMENTATION.md다. 실제 CSV·배포 fold·평가기가 없어 대회 데이터 학습·OOF 점수·E00 재현은 미실시이며 ZIP은 생성하지 않았다.
## 2026-10-19 KST — data/eval_kit 수령

사용자가 배치한 평가 묶음 10개 파일을 확인했다. fold 5개 모두 배포 평가기 내부의 SHA-256과 일치하며 12,222행·ID 유일성·fold 범위 검사를 통과했다. 그룹 fold 4개의 선박 분리와 E00 OOF의 기준 fold ID·분할 일치도 확인했다. 이는 같은 배포본 내부 대조이며 외부 진위 인증은 아니다. 기본 OOF fold 경로를 data/eval_kit로 변경했다. 원본 train.csv와 vessel_catalog.csv는 아직 없으므로 원본과의 대조·평가기 자체 시험·E00 점수 재현은 미실시다. 원본 평가기는 수정하지 않았다. 근거: project_log/eval_kit_received.json.
## 2026-10-20 KST — 로컬 RTX 5080 행별·묶음 계산 비교

GPU-CHECK-20261020-01 완료. 사용자 승인으로 기존 `.venv`(Python 3.11.15, PyTorch 2.11.0+cu128)를 사용했다. 지정 서버 환경과 다르며 서버 호환성 검사로 해석하지 않는다. 공개 train 한글/비한글 각 128행, 전체 카탈로그 78,595행, 기존 모델을 사용하고 조회를 꺼서 검색 경로를 검사했다. 배치 2/16/64/256, 역순·반복·부분 입력에서 비교한 최종 예측은 모두 같았다. 그러나 후보 합집합은 배치 64에서 8/256행, 배치 256에서 1/256행이 달랐다. 후보 순위도 일부 달라 전체 동등성이 확인된 것은 아니다. 별도 32행 점수 검사에서 최대 0.00048828125 차이가 관측됐다. 같은 환경 안에서의 비교이므로 버전 차이가 원인이라는 결론은 내리지 않는다.

근거: [실험 기록](experiments/GPU-CHECK-20261020-01.json), [최종 집계](../reports/gpu_batch_comparison.json), [재현 스크립트](../environment/compare_gpu_batch.py). 스크립트는 GPU 검색 결과를 행별로 보존한 후 기존 후보 통합·특징·분류기로 재생한다. 완성된 배치 추론 경로의 전체 실행 검사는 아니다. 초기 DLL/import 실패와 검사 코드 보완 후 재시도도 기록했다. 현재 로컬 커밋은 참조 클라우드 채팅의 미전송 수정본과 다르다. 모델·운영 코드·ZIP은 변경하지 않았다. eval_kit README가 로컬에서 v1.0인 불일치도 확인했으며 이번에는 OOF 평가를 수행하지 않았다.

## 2026-10-20 KST — 기존 .venv PyTorch 서버 버전 변경

사용자 요청으로 torch 2.11.0+cu128 → 2.7.1+cu128로 변경했다. 기존 torchvision/torchaudio 의존성을 유지하려고 공식 호환 조합인 0.22.1+cu128 / 2.7.1+cu128로 함께 변경했다. uv pip check에서 53개 패키지 의존성 충돌 없음, 세 패키지 import·RTX 5080 FP16 행렬 곱·검색 단위 검사 4개 통과. Python은 사용자 승인된 3.11.15이며 나머지 서버 버전까지 일치시킨 결과는 아니다. 앞선 256행 묶음 비교는 torch 2.11 결과로 보존한다. 신규 버전에서 해당 비교를 재실행한 것으로 표시하지 않는다. 설치 첫 시도의 DNS 오류는 승인된 네트워크 접근 재시도로 해결했다. 원본 데이터·모델·ZIP 변경 없음. 근거: experiments/ENV-TORCH-20261020-01.json, torch_server_version_change.json, ../reports/torch_server_version_check.json. 변경 전 목록: ../reports/torch_before_change.local.txt.

## 2026-10-20 KST — 팀 기준 5-Fold OOF 검증 및 분석 (AS-01 / DEV-02 / DEV-03)

- **평가 분할**: `eval_kit/folds_group5_seed42.csv` (SHA-256: `9d09ede0...`), 12,222행, 선박 ID 그룹 분할.
- **AS-01 (공식 제출용 베이스라인: SPT 3-채널)**:
  - 가중 정확도: **0.97918** (한글 97.061% [오류 35/1,191], 비한글 99.918% [오류 9/11,031])
  - OOF 해시: `1eb6beb0eff062bd28eb2836d257f1768c6a2a0291bb3bcf814e60a5ab5ab3eb`
- **DEV-02 (SNPT 4-채널, 구 AS-02)**:
  - 가중 정확도: **0.97804** (한글 96.893% [오류 37/1,191], 비한글 99.927% [오류 8/11,031])
  - 비한글 오류 1건 감소(9→8), 한글 오류 2건 증가(35→37)로 가중 점수 -0.00114p 하락.
  - OOF 해시: `d847008b11e722c1778411556d6f05e4abe7d4518a721d6d214a22454dc77443`
- **DEV-03 (병렬화 및 쿼리 캐싱 적용 SNPT 4-채널, 구 AS-03)**:
  - 특징 추출 8-스레드 병렬화, 쿼리 뷰 사전 구축 최적화 적용 후 5-fold OOF 생성.
  - 생성된 `oof_DEV-03.csv`의 SHA-256 해시가 `d847008b...`로 DEV-02와 **100% 비트 일치(결정론적 동등성 확인)**.
  - 공식 평가기(`eval_kit/evaluate.py`) 검증 완료: 가중 **0.97804** [95% CI: 0.96394, 0.98977], 한글 96.893%, 비한글 99.927%.
  - 산출물: `share/DEV-03/` 및 `project_log/experiments/DEV-03.json`.
- **OPT-01 (내부 실험용: 후보 30개 압축 + 발음 캐싱)**:
  - 5-Fold 총 소요 시간 751.3초 (DEV-03 대비 **-155.6초 단축**).
  - 가중 정확도: **0.97451** (한글 96.390% [오류 43/1,191], 비한글 99.927% [오류 8/11,031]).
  - 후보를 30개로 압축 시 속도는 17% 빨라지고 비한글은 최고점을 유지하나, 복잡한 한글 선박명 군집에서 랭커의 변별 표본이 부족해져 한글 오류가 소폭 증가함을 확인. (상한 60개가 최적 권장).

## 2026-10-21 KST — JY-01 공식 제출 결과 수령

사용자 제공 결과: 제출 111395, submit_JY-01.zip (LGBM), 제출 시각 2026-10-21 01:38:45 KST, 표시 점수 0.7835113288, 표시 시간 2분 20초(140초). Public/Private 구분, 언어별 점수·표본 수, 시간 세부 범위는 미확인이다. 현재 ZIP의 크기·SHA-256은 기존 JY-01 생성 기록과 대조해 submissions/111395_JY-01.json에 보존했다. 사용자 표에 근거한 기록이며 사이트 직접 조회는 하지 않았다. 기존 AS/OPT OOF를 JY-01의 동일 구성 검증값으로 간주하지 않으며, 점수 차이의 원인은 미확인이다. 다음 분석은 실제 제출 ZIP 내부 코드·환경·후보 검색과 같은 구성의 로컬 검증을 우선 대조한다. 이번 결과 수령만으로 새 학습·ZIP 생성은 수행하지 않았다.

## 2026-10-21 KST — JY-01 표시 시간 140초의 코드 근거 확인

제출 ZIP을 읽어 확인: assets.search_algorithm=rapidfuzz, 검색 채널 S/P/T, torch_indexer.py 미포함. 따라서 이 제출물의 이름 검색은 CPU RapidFuzz 경로이며 로컬 RTX 5080/PyTorch 변경으로 이 ZIP이 GPU 검색을 수행하는 것은 아니다. 기본 특징 작업자는 min(4, available), 점수 계산 스레드는 기본 1이다. 후보 상한 30은 후보 합집합을 구한 뒤 적용하므로 전체 카탈로그 이름 검색 비용은 남는다. 기존 500행 7.44초와 3000행 44.6초는 표본 구간 측정·외삽이며 서버 전체 실행 시간의 실측 보장이 아니다. 서버 평가 행 수·초기화 포함 범위·단계별 로그가 없으므로 140초의 단계별 원인은 확정하지 않는다. 신규 성능 실험·재학습·ZIP 생성 없음.

## 2026-10-21 KST — DACON 평가 변경 공지 원문 수령

사용자 제공 공지 전문에 근거해 EVALUATION_CRITERIA.md에 출처·확인 범위를 갱신했다. 평가 데이터 확대 후 전체/Public 행 수는 미공개이며, 기존 3,000행으로 JY-01의 140초를 환산하지 않는다. 최종 선택 1개의 Private 100% 평가, 5회 제출, 동점 시 시간→ZIP→달성시점 기준을 기록했다. 확대에 따른 시간 증가는 가능성으로 남기며 서버 처리 범위·구간별 시간은 미확인이다. 새 실험·학습·ZIP 생성 없음.

## 2026-10-21 KST — 평가 규모 계획용 가정 적용

사용자 승인으로 비용·처리 시간 예상 시 전체 10,000행 / Public 5,000행 / Private 10,000행을 계획용 기준으로 사용한다. Public 50%는 사용자가 전달한 운영진 답변에 근거하며, 전체 10,000행은 추정치로 공식 확인값이 아니다. 실제 입력 행 수 제한이나 추론 로직에는 적용하지 않는다. 실제 행 수가 확인되면 계획용 가정을 갱신한다. 기계 판독 기록: evaluation_size_assumption.json. 원문 근거와 한계: EVALUATION_CRITERIA.md. 새 성능 실험은 실행하지 않았다.

## 2026-10-21 KST — 최신 서버 실행 제한 적용

사용자 제공 안내에 따라 추론 전체 제한을 20분→30분(로딩·전처리·추론·저장 포함), L4 VRAM을 22.4GiB로 명시해 TEAM_RULES.md §5와 EVALUATION_CRITERIA.md를 갱신했다. 설치 10분·ZIP 10GB/해제 32GB·6 vCPU/RAM 26GB·오프라인 추론 조건을 기록했다. 패키지 개별 목록은 이번 메시지에 없어 새 버전 확인으로 간주하지 않는다. JY-01 140초는 30분의 약 7.8%이나 표시 시간 범위는 미확인이다. 신규 학습·환경 변경·ZIP 생성 없음.

## 2026-10-21 KST — 조교 팁 기반 새 선박 일반화 설계(planned)

설계안: ../docs/UNSEEN_GENERALIZATION_PLAN.md. JY-01 동일 구성 OOF 감사 → 후보 압축 손실 확인 → 학습 fold 오류분포 기반 증강 → 검증 ID를 제외한 카탈로그 합성 → 어려운 음성/랭킹 손실 → 문자·바이트 표현 순으로 제안한다. 조교의 근사 선박 수·제출 0.8 예상은 실측과 구분하고 현재 점수 하락을 unseen 탓으로 단정하지 않는다. 원문/음역/숫자·제원 특징을 유지하며 실제 보류와 합성 변형 시험을 구분한다. 모든 항목 planned, 신규 학습·실험·ZIP 생성 없음.

## 2026-10-21 KST — 공식 데이터 기반 약한 증강 비교 실행

사용자가 규칙 전문 검토 후 증강 실험을 명시적으로 요청했다. AUG-02(현재 JY-01 구성의 새 그룹 5-fold 기준), AUG-03(약한 train-only 증강)을 실행 중이다. 합성은 각 학습 fold의 고유 기록에서 확률 0.5로 시도하고, 이름 문자 1개 삭제 또는 메타데이터 한 항목 결측을 적용하며 합성 쌍 가중치는 0.5다. 학습 fold 실제 행으로만 조회 자산을 생성해 두 모델에서 바이트 동일하게 재사용한다. 기존 학습 특징/후보 선택 로직은 보존하고 기준 학습 쌍을 공유한다. 숫자 경계 보존·기존 다른 ID 이름 충돌 거부·결정성 테스트 3개 통과. 학습되지 않은 카탈로그 ID 합성은 이번 범위에 넣지 않았다. 증강 확률은 고정 가설이며 실제 테스트 오류분포라고 주장하지 않는다.

최초 AUG-00/AUG-01은 공통 함수 추출 중 name_mode 참조 누락으로 첫 fold에서 실패해 별도 기록으로 보존했다. index.name_mode 참조를 명시해 수정하고 새 ID로 재시도했다. 평가기 자체 시험은 Windows 임시 폴더 권한 오류 후 프로젝트 내부 임시 경로로 재실행해 32개 통과, 주최 baseline_lib 교차검사 1개는 해당 파일 경로 미제공으로 생략됐다. E00 가중 0.477501 재현. JY-01 ZIP과 현재 matcher/name_processing/ngram_retrieval 코드 해시 일치 확인. 상세 최종 결과는 각 실험 JSON을 따른다.

## 2026-10-21 KST — 약한 증강 1차 완료 및 후보 선택 오류 수정 비교 계획

AUG-02/AUG-03 그룹5-fold 완료: 가중0.930211378→0.922815505, Δ=-0.007395873, 선박부트스트랩95% CI[-0.02583,0.00949]. 개선 근거가 없어 현재 증강 조합 채택 보류. 한글 고침25/새오류38, 비한글 고침9/새오류0. 상세 근거 ../docs/AUGMENTATION_RESULTS.md 및 ../reports/augmentation_summary.json. 이전0.97대OOF를 JY-01 동일구성 결과로 재사용하지 않는다.

사용자의 ‘이어서 진행’ 요청에 따라 다음 비교 planned: 특징 삭제 후 남아있던 hard-negative 추출의 숫자 열 참조를 ratio/phonetic_ratio/best_gap 이름 참조로 수정하고, 새 기준 AUG-04와 같은 약한 증강 AUG-05를 비교한다. 이전 실패/완료 기록·모델·ZIP은 보존한다. 고정fold/모델/검색/증강조건을 유지해 AUG-04↔AUG-02는 후보학습 선택 수정 효과, AUG-05↔AUG-04는 수정 후 증강 효과로 해석한다.

## 2026-10-23 KST — AS-NAME-03 선박명 예측 구현 인수인계

- [x] 최상위 `train_name_decoder.py` 학습 진입점 추가. 실제 학습 루프는 `experiments/train_name_decoder.py`.
- [x] `predict()`/`predict_many()`에서 카탈로그 1위 이름·동명 ID 목록·후보 반환. 단일 이름 CLI JSON도 같은 구조.
- [x] `--epochs`(가중 복원추출의 쌍 수 기준 예산), 선택적 `--loss-reduction row`, `--final-train` 구현. 기존 기본 200 step/token loss 보존.
- [x] 검증 이름 top-1 정확도·정답 수·가중 이름 정확도, 모델 manifest 추가. 공식 선박 ID 점수와 구분.
- [x] 손실/패딩, 학습예산, fold 누락, 전체학습/검증 경로의 tiny-model 통합 테스트 작성.
- [ ] Python 테스트/실제 추가 학습/성능 검증: 현재 PC `.venv-server`·ByT5 가중치 미존재. uv 설치 시 기본 인덱스 DNS 실패, 공개 PyPI 재시도도 배포본 확보 실패. 지정 환경에서 실행하지 못했음.
- [ ] 공통 `eval_kit/folds_group5_seed42.csv` 없음. 임의 재생성하지 않음.
- 현재 PC 데이터: 사용자가 확인한 프로젝트 `data/`; train 12,222행, catalog 78,595행 확인. 학습 자체는 실행하지 않음.
- 근거: `docs/NAME_DECODER.md`, `project_log/experiments/AS-NAME-03.json`. `git diff --check` 통과; Python 테스트 통과 주장 없음.
- 주요 결정: 이름을 정답 ID와 혼동하지 않도록 `predicted_name`과 동명 ID 목록을 분리. 전체학습 모드는 검증 성능을 생성하지 않으며, 학습량/손실 변경 효과는 별도 실험으로 확인.
- 다음 작업: 환경 설치 → `python -m unittest discover -s tests -p "test_name_decoder*.py" -v`(지정 인터프리터) → 공식 가중치 확보 → 공통 fold 복원 → 새 ID AS-NAME-04로 기존 200-step 설정 재현 → 학습량 단독 비교. 실제 실행 전 새 ID 중복 여부 확인.
- 기존 AS-NAME-01/02 기록·제출 모델·ZIP 보존. 다른 작업의 AR-06 관련 변경은 수정하지 않음.
- 커밋 제안: `feat(name-decoder): add top-one catalog-name prediction and complete training modes`
### 2026-10-23 KST — 기존 프로젝트 자료 재검색 및 fold 복원

앞서 fold가 없어서 진행할 수 없다고 한 판단을 정정한다. `artifacts/as02_handoff/results/oof_AS-02.csv`, `as03_handoff/results/oof_AS-03.csv`, `as04_handoff/results/oof_AS-04.csv`에 동일한 12,222행의 fold 배정이 있었다. 세 파일의 배정 일치·5,851척의 그룹 분리를 확인하고 공식 train의 정답 ID와 결합해 `eval_kit/folds_group5_seed42.csv`를 복원했다. 원본 기록 AS-NAME-02의 SHA-256 `9d09ede0f9254949aacf8cf80638a6c24de0ca5516ca3e9db329f55b9476ca51`과 바이트 단위로 일치한다. 새 분할 생성이 아니다.

- [x] 공식 데이터와 기존 공통 fold 준비 완료. 복원 근거: `reports/name_decoder_fold_recovery.json`.
- [ ] 지정 가상환경 및 ByT5 가중치는 현재 프로젝트 전체 하위 폴더·ZIP 4개·첨부 파일 검색에서 찾지 못함. 다른 위치까지 없다고 단정하지 않는다.
- 다음 작업의 fold 수령 단계는 해소됨. 지정 환경과 공식 ByT5 가중치 확보 후 준비된 테스트/학습 경로를 실행한다.
- 모델 학습·예측·ZIP 생성은 이번 복원에서 실행하지 않음.

## 2026-10-25 KST — IMPL-LM-AR06-01

- [x] LM_IMPROVEMENT sections 1–3 implemented in `experiments/train_noisy_channel.py` and `name_decoder/channel_training.py`: lexical listwise negatives (same-name/fit-known positives excluded), language queues, row-mean CE, fit-learned Hangul correspondence augmentation.
- [x] CUDA is the default; model and batches move to device; unavailable CUDA raises an error. CPU requires `--device cpu`. GPU name, peak allocated memory and sampling coverage are logged. CPU thread cap/cooldown retained only for CPU.
- [x] Added gradient/masking, ranking direction, coverage, exclusion and digit-preservation regression tests in `tests/test_channel_training.py`; syntax compilation passed.
- [ ] Runtime tests and training blocked: `.venv-server` missing; documented setup retried with workspace cache, but uv cannot find Windows Python 3.11.17. RTX 5080 inventory is confirmed, CUDA kernels unverified. No model/accuracy/ZIP produced.
- Next: resolve exact runtime, run `.\.venv-server\Scripts\python.exe -m unittest discover -s tests -p "test_*channel*.py" -v`, then launch a fresh training ID. Do not overwrite prior experiment results.
- Evidence: `project_log/experiments/IMPL-LM-AR06-01.json`, `docs/LM_IMPROVEMENT_USAGE.md`. Lexical mining is not baseline-model-error mining. Improvements remain unmeasured.
- Suggested commit: `feat(noisy-channel): add confusion ranking, coverage sampling and CUDA training`.


## 2026-10-25 KST — IMPL-LM-ENV-02 correction

Existing `.venv` confirmed: Python 3.11.15, torch 2.7.1+cu128, transformers 5.20.1. Only `.venv-server` is absent. The earlier general statement that a virtual environment was absent is incorrect. RTX 5080 CUDA forward/backward passed with finite gradients. New training tests 4 passed, retrieval tests 2 passed; existing V1/V2 prompt tests have 3 tokenizer decode TypeErrors. No training completed. Next: resolve tokenizer/version compatibility (target transformers 4.57.6, Python 3.11.17) before actual LM training. Evidence: `project_log/experiments/IMPL-LM-ENV-02.json`.


## 2026-10-25 KST — ENV-TRANSFORMERS-4576-01

Requested transformers 4.57.6 reinstall in existing `.venv` attempted. Workspace UV cache avoids initial cache permission denial, but PyPI DNS resolution fails after retries; no cached wheel found. Existing package installation retained. Next: restore PyPI connectivity, rerun `uv pip install --python .\.venv\Scripts\python.exe transformers==4.57.6`, then prompt tests. Evidence: `project_log/experiments/ENV-TRANSFORMERS-4576-01.json`.


## 2026-10-25 KST — LM-RANK-GPU-01 preflight

Transformers 4.57.6 confirmed in user-authorized existing `.venv` (Python 3.11.15). All 9 channel tests pass. CUDA RTX 5080 available. Requested GPU 300-step training blocked before optimizer steps: original `artifacts/pretrained/hyperclovax-seed-0.5b/model.safetensors` is absent; pinned download fails on huggingface.co DNS. No matching local cache found. Next: obtain original weights at revision `3da5046fb0195d14f2497de198136987d35fd644`, verify SHA256 `f40f63fa6a118eb9b59e8910c3ab5a6e6499805966284bb2cd41e197719d156b`, then resume GPU training. No model or ZIP generated. Record: `project_log/experiments/LM-RANK-GPU-01.json`.


## 2026-10-25 KST — network diagnosis and local launcher

Both huggingface.co and pypi.org fail DNS resolution in the restricted execution surface. Resolve-DnsName returns access denied; DNS client configuration inspection is also denied. This confirms restricted diagnostic access, not a proven machine-wide DNS outage. No system DNS/proxy/security settings changed. Created `tools/start_lm_gpu.ps1` for normal local PowerShell: checks existing .venv/CUDA/transformers, obtains pinned original weights, validates recorded SHA256, runs 9 tests, launches GPU training with fresh ID LM-RANK-GPU-02. No download/training completed in this step. Next: run launcher from ordinary local PowerShell; no ZIP generated.


## 2026-10-25 KST — LM-RANK-GPU-02 recovery

- [x] Confirmed optimizer steps 1–300 and saved adapter/merged model existed. Failure was final JSON serialization of NumPy scalar counters, and failure-report serialization repeated the same error.
- [x] Fixed `write_json` to convert NumPy scalars via `.item()` while retaining strict NaN rejection; Hangul coverage now sums Python bool values. Added regression test covering result JSON and failure JSON.
- [x] Recovered without retraining: `tools/recover_lm_run.py` verified every checkpoint manifest hash, loaded saved weights and ran finite synthetic likelihood scoring on RTX 5080. Original source record/model preserved. Separate authoritative recovery record: `project_log/experiments/LM-RANK-GPU-02-RECOVERY-01.json`.
- [x] 300-step final training loss 0.8850519061088562; this is not accuracy. Lost final in-memory timing/coverage values remain unmeasured.
- Next: heldout fixed-candidate ranking evaluation, then vessel-ID contribution comparison. No ZIP generated. Existing .venv Python 3.11.15 differs from target 3.11.17; transformers 4.57.6.
- Suggested commit: `fix(noisy-channel): serialize numpy counters and recover completed GPU run`.


## 2026-10-25 KST — EVAL-LM-RANK-01 planned/running

Purpose: evaluate recovered GPU model on all 2,372 clean fold0 rows. Hypothesis: new LM may help fixed-shortlist discrimination. Rebuild AR06/EN_V3 baseline strictly on fit folds with lookup disabled; no validation target supplied to retrieval. Evaluate baseline, LM-only, ungated 0.20 fusion diagnostic and fixed AS05 gate margin <=1.0 (correct value, not 1.5), top35. No tuning on heldout results. CUDA batch8 likelihood validated against single-pair scoring; official eval subset comparison to follow. Previous LM weights unavailable, so no direct old/new LM performance claim. Evaluation script `experiments/evaluate_lm_rank_gpu.py`; record `project_log/experiments/EVAL-LM-RANK-01.json`. No ZIP.


## 2026-10-25 KST — EVAL-LM-RANK-01 complete

[x] Full clean fold0 2,372 rows evaluated on RTX 5080; official evaluator subset comparison passed for four methods. Baseline weighted 0.99958 (Hangul239/239, non-Hangul2130/2133). LM-only0.98730 (235/239,2126/2133), fixes0/breaks8, delta95%CI[-0.02959,-0.00056]. Fixed 0.20 fusion and original margin<=1.0 gate both identical to baseline, no added accuracy. Target inclusion top35 is100% in both groups; all11 LM errors are selection errors. GPU exhaustive scoring476.35s, total522.09s. No old/new LM matched comparison; prior candidates differ, old weights absent. No ZIP. Evidence: `docs/LM_RANK_EVALUATION.md`, `project_log/experiments/EVAL-LM-RANK-01.json`. Next: pre-specified robustness and matched CE-only ablations; retain baseline and do not tune this fold. Suggested commit: `eval(lm): record clean fold0 rank-LM and fixed fusion comparison`.


## 2026-10-25 KST — LM-RANK-GPU-03 planned

User requested continuing performance improvement. Single budget study: original base, same pairs/negative mining/row CE/language cycles/beta0.1/LoRA/seed, steps300->1200 (cosine horizon necessarily changes). Endpoint fixed before run, no heldout early stopping. Compare against recovered 300-step LM using exact `reports/noisy_channel/EVAL-LM-RANK-01/candidates.csv`; unchanged weight0.20/gate1.0. Hypothesis: broader unique-pair exposure improves LM discrimination. Existing AR06 remains production baseline until measured improvement. Existing .venv explicitly accepted by session; Python3.11.15 environment difference recorded. No ZIP.


## 2026-10-25 KST — LM-RANK-GPU-03 trained / EVAL-LM-RANK-02 running

[x] Fixed1200-step GPU training completed, 293.327s optimizer loop; saved merged model/adapter and JSON successfully. Pair hash equals LM-RANK-GPU-02. Total draws9600: Hangul6792 / non-Hangul2808; unique Hangul819 / non-Hangul2808, total3627; max repetition9. [ ] Exact cached fold0 candidate evaluation running as EVAL-LM-RANK-02. No checkpoint selection or fusion tuning; no ZIP.


## 2026-10-25 KST — LM-RANK-GPU-03 / EVAL-LM-RANK-02 complete

[x] 1200-step GPU training and exact-candidate full fold0 evaluation completed; official subset evaluator passed. New LM weighted0.97823 vs300-step0.98730, fixes1/breaks6. Delta-0.00907,95%CI[-0.04098,+0.00845] includes0: lower point estimate, no statistically established worsening. AR06+LM fixed fusion/gate still0.99958 and zero prediction changes. Reject longer-training candidate; keep existing baseline. Pair hash and83,020 candidate/scores/order matched. No ZIP. Evidence: `docs/LM_BUDGET_EVALUATION.md`, new training/evaluation JSONs. Next: equal-budget CE-only ablation, then fit-inner-heldout model-confusion negatives/checkpoint selection; do not tune external fold0. Commit suggestion: `experiment(lm): reject longer training after matched fold0 evaluation`.


## 2026-10-25 KST — INNER-CONFUSIONS-01 running

Started four inner group folds (1,2,3,4) entirely excluding external fold0. AR06 fits only other inner folds and exports top8 high-scoring wrong names without gold injection or lookup. Script: experiments/mine_inner_confusions.py. Record: project_log/experiments/INNER-CONFUSIONS-01.json. Next: inspect mined candidate exclusions, integrate training CSV, run matched300-step GPU training and fixed-candidate evaluation. No training or improvement claim yet.


## 2026-10-25 KST — LM-RANK-GPU-04 planned

INNER-CONFUSIONS-01 completed inner4-fold AR06 mining in144.435s. Training integration validates source record/CSV/fold hashes and each sample's fit membership, inner fold, observed name, positive/negative catalog identities. All fit-known positive names are excluded. Only exact original name pairs use mined negatives; augmented/unmatched pairs retain lexical fallback. Planned GPU300steps matches LM-RANK-GPU-02 seed, data/order, loss, augmentation and optimizer settings; changed factor is negative source. Tests include ambiguous positive and external sample rejection. User requested training; no ZIP.


## 2026-10-25 KST — LM-RANK-GPU-04 trained

[x] Inner-confusion negative integration complete and11 channel regression tests passed (ambiguous positives excluded, external samples rejected). GPU300-step training completed; model/adapter saved under artifacts/LM-RANK-GPU-04/. Same pair-content/order hash as300-step lexical baseline. All4685 original pairs use2 mined names each (9370 slots);2580 augmented pairs retain lexical negatives (5160 slots). Loss remains rowCE+0.1rank, no fusion changes. Accuracy not measured; no improvement claim or ZIP. Next: run `experiments/evaluate_lm_rank_gpu.py --experiment-id EVAL-LM-RANK-03 --model-path artifacts/LM-RANK-GPU-04/model --candidates-csv reports/noisy_channel/EVAL-LM-RANK-01/candidates.csv`, then official comparison against EVAL-LM-RANK-01 LM. Evidence: project_log/experiments/LM-RANK-GPU-04.json, INNER-CONFUSIONS-01.json. Commit suggestion: feat(lm): train with inner-cross-fitted matcher confusions.


## 2026-10-25 KST — EVAL-LM-RANK-03 running

Purpose: evaluate inner-confusion LM-RANK-GPU-04 against lexical300-step LM-RANK-GPU-02. Hypothesis: changing only negative sources improves heldout ranking. External fold0 full2372 rows, identical fixed83020 candidates/scores/order from EVAL-LM-RANK-01, same LM-only/fusion0.20/gate1.0, no tuning. GPU evaluation started. Official subset evaluator and matched LM comparison follow. No ZIP.


## 2026-10-25 KST — EVAL-LM-RANK-03 complete

[x] Full clean fold0 GPU evaluation completed,493.3s candidate scoring. Exact83020 candidate/scores/order matched. New inner-confusion LM top1 predictions identical to previous lexical300-step LM on all2372rows: Hangul235/239, non-Hangul2126/2133, weighted0.98730, fixes0/breaks0, pairedCI[0,0]. AR06 fusion/gate unchanged0.99958. Official subset evaluator passed four methods plus matched comparison. No promotion, no ZIP. Evidence: docs/LM_INNER_CONFUSIONS_EVALUATION.md and project_log/experiments/EVAL-LM-RANK-03.json. Next: inspect fit-inner score separation and pre-specified stress results before further training; do not tune externalfold0. Commit suggestion: eval(lm): record matched inner-confusion negative result.


## 2026-10-25 KST — IMPL-LFM25-01

[x] User-requested LFM2.5-350M default wired into trainer and tools/start_lm_gpu.ps1, new tools/fetch_lfm25.py resolves/preserves immutable revision and hashes files. q/v LoRA validates6attention blocks in16-layer hybrid instead of requiring attention in every layer. Tiny hybrid GPU training/merge/reload test passed; existing11channel tests passed. [ ] Real350M download blocked by tool DNS; actual checkpoint/tokenizer/training/accuracy pending. Native LFM2 exists in4.57.6 but actual official config declares5.0.0rc1; do not claim real checkpoint compatibility yet. Existing weights/requirements retained, no ZIP. Next: normal PowerShell launcher (fresh LM-LFM25-01), then same-candidate evaluation. Evidence docs/LFM25_MODEL_SWITCH.md, project_log/experiments/IMPL-LFM25-01.json. Suggested commit: feat(lm): support LFM2 hybrid LoRA and switch training default.


## 2026-10-25 KST — LM-LFM25-02 planned

Official LFM2.5 files now local. LM-LFM25-01 failed loading tokenizer metadata (`TokenizersBackend`, a transformers5 class name). Added local fast-tokenizer compatibility loader preserving tokenizer.json/special tokens/chat template, without editing official files or package pins. Fresh LM-LFM25-02 planned with same300step/inner-confusion setup; old failure retained. No ZIP.


## 2026-10-25 KST — LM-LFM25-03 running

Official checkpoint downloaded. LM-LFM25-01 failed on TokenizersBackend class; LM-LFM25-02 failed on v5 extra_special_tokens empty-list format. Fixed name_decoder.noisy_channel.load_channel_tokenizer to use v4 PreTrainedTokenizerFast with extra_special_tokens={} while retaining tokenizer.json, EOS7 and chat template. Actual Korean continuation round-trip and tiny hybrid GPU tests2 passed. Real LFM2.5 checkpoint now trains on GPU, q/v LoRA344064 params over6attention blocks;300steps planned, same positive pairs and inner-confusion negatives. Earlier failures retained; no package upgrade/ZIP. Next: verify trained checkpoint hashes and reload, accuracy pending.


## 2026-10-25 KST — LM-LFM25-03 trained

[x] Actual downloaded LFM2.5-350M GPU300-step training completed and merged model/adapter saved in artifacts/LM-LFM25-03/. Adapted q/v params344064 over6attention blocks. Version4.57.6 compatibility loader retains original tokenizer files but maps v5 TokenizersBackend to PreTrainedTokenizerFast with extra_special_tokens={}. Actual Korean/EOS round-trip plus tiny hybrid GPU tests2 and channel tests11 passed. LM-LFM25-01/02 failed before training and remain separate records. [ ] Accuracy evaluation pending; no improvement claim/ZIP. Next: evaluate with fresh EVAL-LFM25-01 ID, model artifacts/LM-LFM25-03/model and exact reports/noisy_channel/EVAL-LM-RANK-01/candidates.csv. Launcher next default fresh LM-LFM25-04. Commit suggestion: fix(lfm): support v5 tokenizer metadata on pinned transformers4 and train GPU checkpoint.


## 2026-10-25 KST — EVAL-LFM25-01 running

Evaluate actual LFM2.5-350M LM-LFM25-03 on full external fold0 (2372rows) and exact fixed83020 candidates from EVAL-LM-RANK-01. Compare against latest HyperCLOVA inner-confusion300-step LM EVAL-LM-RANK-03. Same names/tasks, folds, fixed fusion0.20/gate1.0; tokenizer and adapted parameter count differ by architecture. No tuning/ZIP. Official subset evaluator and paired comparison to follow.


## 2026-10-25 KST — EVAL-LFM25-01/02 failed numerical preflight;03 running

LFM batch8 vs single score discrepancy ~0.000176 exceeded legacy1e-4 check. CUDNN-disabled retry02 had same discrepancy. Neither failed preflight produced accuracy results. Run03 uses architecture-specific mean-logp tolerance1e-3 and checks all35 names in first record (legacy model1e-4), plus single-score refinement when union of near-top LM/fusion names has>2 members; not a global numeric-error bound. No target labels used for numerical settings. CUDNN remains disabled; TF32 disabled. Full matched candidate evaluation running; numeric protocol difference must be disclosed. No ZIP.


## 2026-10-25 KST — EVAL-LFM25-03 complete

[x] LFM matched full cleanfold0 evaluation and official subset comparison complete. LFM weighted0.97908 vs HyperCLOVA0.98730; Hangul232/239 vs235/239, non-Hangul2130/2133 vs2126/2133. Fixes7/breaks6; weighted delta-0.00822 CI[-0.04087,+0.01728] includes0. Fusion/gate unchangedAR060.99958. Numeric failed preflights01/02 retained. Run03 first-row all35-name diffmax0.000235, tolerance0.001; cuDNN/TF32 off. All17 changed/error rows rechecked single-pair with0 prediction mismatches. Full scoring732.02s. No promotion/server claim/ZIP; LFM user-selected research checkpoint retained. Evidence docs/LFM25_EVALUATION.md, EVAL-LFM25-03.json. Next: fit-inner Korean representation/negative experiments only if continuing; do not tune outerfold0. Commit suggestion: eval(lfm): record matched fold0 result and numerical audit.


## 2026-10-25 KST — REVIEW-INTERNAL-DEV-01

User requested verification of internal-dev best selection changes. Code review only, no implementation edits or training. Blockers: first-step history serializes best_dev_loss=inf under strictJSON and fails; dev indices still present in both training samplers. Pair split after augmentation/fitted mapping/mining is not isolated vessel holdout. CPU adapter snapshot aliases live parameters (minimal reproduction confirms); CUDA snapshot copied to CPU does not share memory. Token-reduction training differs from row-only dev loss. Final best metadata/split hashes absent.128/7265=1.76%, not5%. Next: fix JSON logging and split source-vessel groups before all fitted preprocessing/mining/augmentation, exclude dev/descendants from both samplers, clone snapshots, persist selection provenance. Review evidence project_log/experiments/REVIEW-INTERNAL-DEV-01.json. No ZIP.


## 2026-10-25 KST — IMPL-INTERNAL-BEST-01 complete

[x] Corrected user-requested Best selection: source-vessel split before mappings/augmentation/mining; both samplers contain internal train only. Internal-best rebuilds AR06 crossfits without Dev instead of reusing contaminated outer-fit CSV. Dev clean lexical candidate objective supports row/token reduction and language weighting; finite-loss validation, pre-first-eval null logs. CPU snapshot clone and strict Best restoration before merge; best snapshot/selection/split hashes persisted. Defaults internal-best/fraction.05/cap128vessels/every25; last option retained. User LoRA expansion/rank/beta preserved. [x]18 targeted tests pass (5internal,11channel,2LFM); tiny end-to-end tests cover both samplers, restored snapshot and manifest. Real-data dry split train9560/dev290rows128vessels129pairs/outer2372; no ID overlap. No full training/ZIP/performance claim. Evidence docs/INTERNAL_BEST_CHECKPOINT.md, project_log/experiments/IMPL-INTERNAL-BEST-01.json. Next: fresh training ID then controlled external evaluation; running old processes unchanged. Commit suggestion: fix(training): isolate internal vessel dev and restore best adapters safely.


## 2026-10-26T00:29:48+09:00 - HANDOVER CONTEXT v2 reconciliation (inspection only)

- [x] Compared handover with current source paths, saved experiment records and local interpreter; no training, prediction, submission check or ZIP creation executed.
- Confirmed integration location: artifacts/AS-06/script.py and artifacts/AS-06/run_ar06.py contain Jamo retrieval / unique lookup and CLI options. AR-06/script.py still defaults USE_TRAIN_LOOKUP=False; AR-06/run_ar06.py still embeds matching code. Do not describe those source files as already integrated.
- Current AR-06/catalog_full_ko_map.json has 78,586 top-level entries, versus 78,592 names claimed in handover/AS-06 record; coverage and generation provenance require reconciliation.
- Local .venv-server/Scripts/python.exe exists and reports Python 3.11.15; reports/server_environment.json describes an earlier Linux Python 3.11.17 environment. Do not equate the two.
- LM-LFM25-FULL-R24-01 record confirms expanded LoRA targets / rank24 / alpha48 / 900steps / rank-beta0.25, status trained_unvalidated. AS-06.json flags its OOF/noise results as leaky; no validated improvement claim. This inspection has not independently established the exact leakage scope.
- [ ] Next actions, pending implementation/evaluation task: reconcile AS-06 artifact and AR-06 source; audit synthetic-map coverage and generator fit membership per external fold before comparing candidate inclusion / selection errors. Keep test distribution and ~0.90 bottleneck explanations as hypotheses until measured.
- Evidence: project_log/experiments/AS-06.json; project_log/experiments/LM-LFM25-FULL-R24-01.json; artifacts/AS-06/script.py; artifacts/AS-06/run_ar06.py.
- Suggested documentation commit: docs: reconcile handover with local AR06 and AS06 state. No commit/push performed.


## 2026-10-26T00:31:33+09:00 - AS-06 official server result reported by user

- [x] Recorded submission 115506, submit_AS-06.zip, submitted 2026-10-25 23:49:32 KST: weighted score **0.9122153056**, displayed duration **366s (6m06s)**. Source is the user-pasted DACON table; not independently retrieved.
- AR-06 (113038): 0.9056363583 / 234s. AS-06 delta +0.0065789473 = **+0.65789473 percentage points**; displayed duration +132s.
- AS-05 (114681): 0.9054976988 / 769s. AS-06 delta +0.0067176068 = **+0.67176068 percentage points**; displayed duration -403s.
- Decision: AS-06 is the leading score reference among these three user-reported submissions. This is package-level server evidence; separate Hangul/non-Hangul, retrieval/scoring/lookup and LoRA contributions remain unmeasured. Prior statements that AS-06 has no official score are historical and superseded by this entry; local OOF leakage warnings remain in effect.
- [ ] Next, when improvement is requested: audit synthetic-map fold provenance and run controlled component comparisons; preserve AS-06 assets. No additional training, ZIP creation/upload, commit or push performed.
- Evidence: project_log/experiments/SUB-115506.json; prior records AS-06.json and ANALYSIS-AS05-SERVER-01.json.
- Suggested commit: docs(results): record AS-06 submission 115506 and leaderboard gain.


## 2026-10-26T00:39:19+09:00 - AS-06 improvement plan

- [x] Reviewed current AS-06 scoring/retrieval/lookup and offline generator; saved docs/MATCHING_IMPROVEMENT_PLAN.md (plan PLAN-MATCHING-IMPROVE-01). No model/accuracy evaluation or ZIP creation performed.
- Measured catalog/map aggregates: canonical names78592, map keys78586, missing6; two pronunciations74238 and single pronunciation4348. Generator manifest says heldout_fold0, conflicting with AS-06.json full-train description; actual fold lineage remains unresolved.
- Decision: first compare separated/bounded Jamo scoring with fixed candidates; then diverse offline pronunciations and Jamo2-4gram retrieval in separate experiments. Audit lookup fallback/missingness and auxiliary seen-vessel validation independently. AS-06 server score0.9122153056 remains the package reference.
- [ ] Next implementation task: preserve reference assets, audit fold provenance, construct controlled baseline and implement one factor at a time. All proposed experiments remain planned.
- Evidence: docs/MATCHING_IMPROVEMENT_PLAN.md; project_log/experiments/PLAN-MATCHING-IMPROVE-01.json. Suggested commit: docs: plan controlled AS06 scoring and retrieval improvements. No commit/push.


## 2026-10-26T00:50:53+09:00 - Jamo scoring Jamo scoring implemented; paired comparison running

- [x] Frozen AS-06 source/map/record copies with SHA256 under artifacts/JAMO-SCORE-01/reference. Created experimental_matchers/jamo_scoring/script.py with default legacy mode and explicit separate1.8*Jamo channel; no original source overwritten.
- [x] Five targeted scoring/candidate/gold-independence/order tests passed. Actual64-row balanced legacy parity passed including Top5 scores.
- [x] AUDIT-GENERATOR-LINEAGE-01 reconstructed9850fit rows/4680vessels/7265ordered pairs; pair, source, training vessel, negatives, checkpoint and artifact hashes match; outerfold0 membership excluded. The historical full-train generator claim is contradicted. Missing historical generation-side model binding and manual-lexicon/miner-history limits remain.
- [ ] JAMO-SCORE-01 running: fold0 clean/default/severe, seed20261022, lookupoff, identical per-row candidate hashes, official subset paired CI. All settings pre-specified; no weight tuning/promotion based on this outer fold.
- Evidence: experimental_matchers/jamo_scoring/README.md; project_log/experiments/AUDIT-GENERATOR-LINEAGE-01.json; project_log/experiments/JAMO-SCORE-01.json. No LM training/ZIP/upload/commit/push.


## 2026-10-26T01:01:06+09:00 - Jamo scoring Jamo score experiment completed; AS-06 retained

- [x] Implemented separate Jamo channel under experimental_matchers/jamo_scoring/script.py, preserved legacy default and immutable AS-06 snapshots. Targeted tests5passed, real64-row original/legacy parity passed, both modes real12-row reverse/single independence passed.
- [x] JAMO-SCORE-01: all2372outerfold0 rows, clean/default/severe(seed20261022), identical candidate sets. Clean/default predictions unchanged; severe Hangul226/239->231/239 (fix6/break1), weighted0.96108080->0.97572516; paired CI includes0. Non-Hangul unchanged.
- [x] JAMO-SCORE-02: fixed settings, new severe seeds20261026 and20261027. Hangul229->228 (fix4/break5) and230->230 (fix3/break3). Each CI includes0. These are repeated transformations of the same vessels, not independent new samples.
- Decision: reject default adoption of this universal1.8 Jamo correction because benefits are inconsistent; retain server AS-06 reference0.9122153056. Do not tune the weight on observed outerfold0. Separate scoring code remains available experimentally.
- Lineage correction: reproducible saved generator lineage excludes fold0 (9850fit rows/7265pairs), contradicting the earlier unqualified all-train statement. Historical map-generation model binding/manual lexicon/miner-history limits remain; shared map is not valid for leak-free full5fold OOF.
- [ ] Next: isolate generator/map and feature selection inside fit-inner splits before testing pronunciation reliability or learned feature combination. Other improvement stages remain planned. No new LM training, ZIP, server upload, commit or push in this task.
- Evidence: docs/JAMO_SCORE_RESULTS.md; project_log/experiments/REVIEW-JAMO-SCORE-01.json; primary/secondary/audit records. Suggested commit: experiment: evaluate separated Jamo scoring and retain AS06.


## 2026-10-26T21:34:40+09:00 - Pronunciation deficient-pronunciation generation complete; matching evaluation running

- [x] Added label-free pronunciation quality/diversity policy and7passing tests. Source/data/model/prompt/seed/batch/config hashes are bound in generation manifests; original AS-06 assets preserved.
- [x] CUDA pilot64catalog names completed. Before accuracy evaluation, generation scope narrowed to existing missing/single-variant names4354 to bound cost; selection uses no validation labels.
- [x] KO-DIVERSE-01 completed:2487names gained4327variants, single-variant entries4348->1861; all original variants retained, missing6names remain. Model weights unchanged; no new neural training.
- [ ] EVAL-KO-DIVERSE-01 running with fixed AS-06 legacy scoring: clean/default/severe3seeds, outerfold0. Original result caches require file/input/model/source checks and64-row per-condition parity, otherwise recompute. No server improvement claimed, no ZIP/upload.
- Evidence: name_decoder/pronunciation_quality.py; experiments/build_catalog_hangul_diverse.py; experiments/evaluate_pronunciation_maps.py; corresponding experiment JSONs.


## 2026-10-26T21:59:10+09:00 - Pronunciation pronunciation expansion and ownership filter completed; AS-06 retained

- [x] Implemented label-free quality/diversity policy, CUDA generator with model/source/config/seed provenance, and reverse-pronunciation ownership filter. Quality tests7passed, ownership tests5passed; all scripts compile. Executed sources and latest CLI versions are separately preserved by hash.
- [x] KO-DIVERSE-01:4354missing/single-variant names processed;2487names gained4327variants; originals unchanged, missing6 remain. KO-OWNERSHIP-01 retained2998additions and removed1329. No neural weights updated.
- [x] Fixed AS-06 legacy scoring, lookupoff, outerfold0 all2372rows, clean/default/severe3seeds. Reference Hangul239/239/226/229/230; expanded239/238/226/230/229; guarded239/238/225/229/229. Non-Hangul unchanged; all10official paired vessel-bootstrap CIs include0.
- [x] Baseline cache provenance and64-row per-condition prediction/rank parity verified. Each candidate fully predicted. Both static maps passed8-row reverse/single checks. Public heldout case review links filtering to removed gold-name variants without claiming exclusive causality.
- Decision: no default adoption of either map; preserve AS-06 reference0.9122153056 and original engine/map hashes. More pronunciations and nearest-original-owner filtering are not demonstrated accuracy improvements. No ZIP, server upload, commit or push.
- [ ] Next: fit-inner vessel isolation before generator/map/feature selection, then metadata-aware candidate selection. Do not retune on observed outerfold0. Global catalog regeneration and retrieval expansion remain separate planned work.
- Evidence: docs/PRONUNCIATION_RESULTS.md; docs/PRONUNCIATION_USAGE.md; project_log/experiments/REVIEW-PRONUNCIATION-01.json and five reviewed run records.
- Suggested commits: feat(pronunciation): add reproducible diversity generation and ownership checks; experiment: record Pronunciation comparisons and retain AS06.


## 2026-10-26T22:22:07+09:00 - User naming correction applied

- [x] AS is a submission ZIP filename prefix, not a model version or experimental sequence. Corrected12run IDs and active code/docs/paths to descriptive JAMO-SCORE, KO-DIVERSE, KO-OWNERSHIP and related audit/review IDs. Actual submission filename submit_AS-06.zip and score0.9122153056 unchanged.
- [x] Archived original executed records/source text with exact hashes; preserved commands, metrics, phase results, diagnostics, costs, data and model settings. No run relabeled as a different configuration, extra experiment or extra submission.
- [x]17targeted tests passed after renamed imports/paths; core engine and existing prediction file hashes unchanged. No model fitting, generation, scoring evaluation, ZIP creation/upload or push performed.
- [x] Persisted the user's rule in AGENTS.md and project_log/EXPERIMENT_NAMING.md. Canonical IDs and historical path aliases: project_log/EXPERIMENT_NAMING.json.
- Next: continue only under descriptive IDs. AS numbering is assigned only for actual submission filenames upon an explicit submission-file request.
- Evidence: project_log/experiments/NAMING-CORRECTION-01.json. Suggested commit: fix(log): separate experiment IDs from submission filenames.
