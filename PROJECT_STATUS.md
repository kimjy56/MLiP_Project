# PROJECT_STATUS — 프로젝트 진행 상태 및 인수인계 대장

최종 갱신 시각: 2026-10-23 18:10 (KST)
현재 마일스톤: **Matcher 로직 정밀화 및 matcher.md 문서화 완료**

---

## 1. 프로젝트 핵심 정보 (Quick Reference)

- **과제**: DACON 236771 항만 입출항 기록 기반 선박 식별 매칭 (레코드 링키지)
- **공식 평가 산식**: `0.7 × 한글 정확도 + 0.3 × 비한글 정확도`
- **검증 기준**: 선박 그룹 5-fold seed 42 (`eval_kit/folds_group5_seed42.csv`, 총 12,222행)
- **개발 환경**:
  - Python: 3.11.15 (Windows, RTX 5080, CUDA 12.8)
  - 핵심 패키지: PyTorch 2.7.1+cu128, RapidFuzz 3.14.3, Pandas 2.0.3, NumPy 1.26.4
- **제출 규칙**: 제출 ZIP(`submit.zip`)은 **사용자의 명시적 요청 시에만 생성**.

---

## 2. 현재 진행 상황 및 완료된 작업

### [완료된 마일스톤 및 실험]
- [x] **Matcher 전면 문서화 (`matcher.md`, 2026-10-23)**: 2단계 캐스케이드(S/P/T/N 후보 검색 + 결정 모델 랭킹), 입력 전처리 파이프라인, 한글 외래어 확장, 도메인 게이팅 규칙, 20개 특징 벡터 수식, 런타임 제어 환경변수 등 Matcher 전체 아키텍처 상세 명세 완료.
- [x] **Matcher 국적/선종 점수 정밀화 및 오답 복구**:
  - `model/matcher.py`: Flat 1.0 이진화 문제 해소. 지배적 전이($p \ge 0.30$) 및 대한민국은 1.0점, 미세 전이($0.02 \le p < 0.30$)는 비례 확률 $p$ 그대로 가산하여 주 후보와의 변별력 보존.
  - 건조연도 감쇄(`year <= 2019`) 시 이름 유사도 게이팅(`sim >= 0.70`) 강제하여 엉뚱한 선박의 점수 무임승차 차단.
  - 문자열 타입 가드 보강(`_similarity_cdist`)으로 노이즈 데이터의 NaN/비문자열 C++ 에러 원천 차단.
  - 대표 오답 `할사이언`(TRAIN_01409), `레지슬레이터`(TRAIN_11050) 정답 복구 확인.
  - 1,000행 고속 노이즈 스트레스 검증: Clean 98.538% ➜ Noisy 94.900% (노이즈 방어율 96.31%, 비한글 99.667% 방어).
- [x] **초소형 모델 조사 (2026-10-23)**: LFM2.5-350M·HyperCLOVA X 0.5B·Qwen3.5-0.8B와 이름 변환/검색 모델의 공식 성능 자료·라이선스 검토 완료. [조사 근거](docs/ULTRA_SMALL_MODEL_SURVEY_2026-10-23.md). 문헌 조사이며 모델 성능 실험·코드 변경·설치·ZIP 생성은 하지 않음.
- [x] **REFACTOR-01 & REFACTOR-02**: 결정 모델 입력 전처리 및 배치 추론 분리 (19개 단위 테스트 통과, 12,222행 비트 단위 동일성 검증)
- [x] **REFACTOR-03**: Matcher 구조 개선 및 train_oof 해시 검증 완전 제거
  - `model/matcher.py`: Scorer 클래스(`CatBoostScorer`, `EnsembleScorer`) 및 `load_decision_model` 팩토리 분리, 핫패스 정적 도우미 함수 추출로 오버헤드 제거
  - `train_oof.py`: 해시 함수(`_sha256`, `sha256`) 및 `--folds-sha256` 인자 완전 삭제, 동적 모델 아티팩트 수집 적용
  - `tests/test_matcher_refactor.py`: 5개 신규 단위 테스트 추가 및 전체 54개 테스트 전건 통과 (5.60초)
- [x] **AS-SET-02**: 후보 집합 문맥 모델(`CandidateSetNetwork`, 4,193 파라미터) 5-fold 전체 OOF 평가 완료
  - 공식 가중 점수: **0.997992** (한글 99.748%, 비한글 99.918%)
- [x] **AS-SET-03**: 학습 후보 상한을 추론과 동일한 30개로 축소 (Hard Negatives 80% 유지)
  - 점수 0.997992 유지, 학습 쌍 수 41% 절감(48만 $\rightarrow$ 28.4만 쌍), `MORNNIG CAROL` 정답 복구
- [x] **AS-SET-04 (로드맵 A)**: 추론 동일 자연 후보 추출(`prune_candidates_by_tonnage`, max 30) 검증
  - Hard Negative 제거 시 자매선(`모닝 피스`, `우찬`) 구별 실패로 가중 점수 0.993263으로 급락 확인 $\rightarrow$ **기각**
- [x] **AS-SET-05 (로드맵 B)**: Leave-one-out 상대 표현(`[self, self - mean_others, self - max_others, flag]`)
  - 자매선 `우찬` 3행 정답 복구 확인, 공식 가중 점수 0.992675 (판정: 동등)
- [x] **AS-SET-06 (로드맵 C)**: 행별 유효 후보 평균 BCE (`row_bce`) 검증
  - 가중 점수 **0.995026 (+0.001763 반등)**, 한글 +0.25%p, 자매선 `모닝 피스` 4행 전원 정답 복구 $\rightarrow$ **채택**
- [x] **AS-SET-07 (로드맵 D)**: 행별 Softmax Cross Entropy (`row_ce`)
  - 점수 0.951970으로 급락 (비유사 오답 억제 기울기 부재) $\rightarrow$ **기각**
- [x] **AS-SET-08 (로드맵 E)**: Softmax Attention Pooling (`attn_max`)
  - 12,222행 변동 0행, 학습 시간 2.5배 지연 $\rightarrow$ **기각**
- [x] **AS-SET-09**: 검증된 최적 구성 결합 (30 예산 + Hard Negatives + `row_bce`)
  - 가중 점수: **0.9979920111** (최고점 완벽 유지)
  - 한글 정확도: **99.748%** (1,188 / 1,191, 오답 단 3행)
  - **Fold별 한글 성적**: Fold 1(99.6%), **Fold 2(100.0%)**, **Fold 3(100.0%)**, **Fold 4(100.0%)**, Fold 5(99.1%) $\rightarrow$ **5개 중 3개 Fold 한글 100% 만점 달성**
- [x] **AS-SET-10**: `row_bce` + `leave_one_out` 결합 검증
  - 가중 점수 0.997965, 한글 99.748% 100% 동일 일치
- [x] **JY-11 (앙상블 국적/선종 호환성 검증)**: CatBoost 0.6 + MLP 0.4 5-Fold OOF 재학습 및 검증 완료
  - 공식 가중 점수: **0.994466** (한글 99.244%, 비한글 99.918%)
  - 결과: 비한글 1행 추가 복구(오답 10행 $\rightarrow$ 9행, +0.01%p), 한글 2행 변동(오답 7행 $\rightarrow$ 9행)으로 AS-03(0.99561) 대비 $\Delta=-0.00115$.
  - 시사점: 선종 1:N 이진 호환성 매칭이 비한글 노이즈를 구제했으나, 세부 차종 미세 확률 희석으로 일부 음차 선박(`할사이언`, `레지슬레이터`)에서 경합 발생. **AS-04 (Set Decision Model, 0.99799, 한글 오답 3행)의 구조적 우위 확증.**

---

## 3. 진행 중인 상태 및 다음 예정 작업 (Next Actions)

- [x] **제출 패키징 및 검증**:
  - `AS-03` (JY-10 기반): `submit_AS-03.zip`, 가중 0.99561 (+0.01346 개선)
  - `AS-04` (AS-SET-09 기반): `submit_AS-04.zip`, `submit.zip`, 가중 **0.99799 (역대 최고점)**, 한글 99.748% (오답 단 3행), `check_submission.py` 실패 0/주의 0 100% 통과
- [x] **일반화 및 강건성 검증 (Robustness Stress Test 완료)**:
  - **AS-03 vs AS-04 5-Fold Unseen OOF 스트레스 테스트 (12,222행 전수 조사)**:
    - Clean 환경: AS-04(0.99799) > AS-03(0.99561) (+0.00238 우위, 오답 5행 감소)
    - Noisy 환경(오타/톤수오차/결측 7,648행 주입): AS-04(0.97463) > AS-03(0.97426) (+0.037%p 우위 유지)
    - 순수 노이즈 집단(7,648행): 한글 정확도 94.437%로 양 모델 동일 방어, 비한글 결측에서 CatBoost가 미세 강건
  - [ ] **1. 보조 그룹 시드 재현성 검증**: `eval_kit/aux/folds_group5_seed{0,1,2}.csv` 3개 독립 분할에서 점수 유지력 확인
  - [ ] **3. 미등장 선박 분리 검증**: `eval_kit/aux/folds_row5_seed42.csv` 기반 등장/미등장 선박 식별력 비교
- [ ] **앙상블 및 최종 추론 파이프라인 통합**:
  - `SetDecisionModel(AS-04)`과 `CatBoost(AS-03)` 앙상블 간의 상호보완성 실측 (노이즈 방어율 극대화)

---

## 4. 주요 결정 사항 (Decisions & Rationale)

- **초소형 모델은 역할별 실험 후보로만 유지 (2026-10-23)**: 일반 지식·지시 수행·검색 벤치마크를 선박 매칭 순위로 합치지 않는다. 생성형 효율은 LFM2.5-350M, 한국어는 HyperCLOVA X 0.5B, 표기 변형은 ByT5-small, 최신 검색형은 EmbeddingGemma 2를 우선 비교할 근거가 있다. 기존 파이프라인에 채택하는 결정은 미실행이며, 실제 성능·서버 호환성 검증이 남아 있다.

1. **Hard Negative Mining (80%) 유지**:
   - 추론용 톤수 필터만으로 학습 후보를 구성하면 자매선(`모닝 피스`, `우찬`)을 구분할 수 없음. 학습 풀에는 이름/발음 유사도가 높은 하드 네거티브가 필수적임.
2. **손실 함수: 행별 유효 후보 평균 BCE (`row_bce`) 채택**:
   - 쌍(pair) 단위 BCE는 후보가 30개인 쿼리가 5개인 쿼리보다 6배 많은 그래디언트를 차지하는 편향이 발생함.
   - 각 행의 유효 후보 수($N_b$)로 나눈 뒤 한글(0.7)/비한글(0.3) 언어 가중치를 부여하는 `row_bce`가 Fold 3 자매선 전원을 복구하며 명확한 개선을 입증함.
3. **Softmax Cross Entropy 및 단순 Attention Pooling 배제**:
   - Softmax CE는 표 형식의 비유사 오답에 대한 패널티가 희석되어 순위 역전 대량 발생(0.9519로 급락).
   - Attention Pooling은 기존 Max pooling 대비 추가 정보 이득 없이 연산 시간만 2.5배 낭비함.

---

## 5. 다음 에이전트 인수인계 가이드 (Handoff Guide)

- **초소형 모델 조사 인수인계**: 현재 요청 범위의 조사는 완료. 후속 구현 요청 시 `docs/ULTRA_SMALL_MODEL_SURVEY_2026-10-23.md`에서 목적별 후보를 고르고, 공식 fold 학습·라이선스 고지·로컬 번들 및 서버 로딩 검증부터 진행한다. 학습/추론 시간과 정확도는 아직 미측정이다. 문서 변경만의 커밋 제안: `docs: survey ultra-small models for vessel matching`. 기존 `model/matcher.py` 수정은 이번 조사와 무관하므로 함께 커밋하지 않는다.

- **핵심 소스코드**:
  - 모델 정의: `decision_models/set_model.py` (`CandidateSetNetwork`, `SetDecisionModel`)
  - 어댑터: `decision_models/matcher_adapter.py` (`DecisionMatcher`)
  - 최신 실험 스크립트: `experiments/compare_set_loss.py`
  - 단위 테스트: `tests/test_set_model.py` (10개 테스트 전체 통과 확인 상태)
- **최신 모델 및 산출물**:
  - 최적 모델 가중치 및 캐시: `artifacts/AS-SET-09/fold_{0..4}/set/`
  - OOF 예측 파일: `artifacts/AS-SET-09/oof_AS-SET-09.csv`
- **주의사항**:
  - `submit.zip`은 사용자가 명시적으로 지시하기 전까지 절대로 자동 생성하지 말 것.
  - 모든 검증은 고정된 `eval_kit/folds_group5_seed42.csv` 및 `eval_kit/evaluate.py` 산식을 따를 것.
