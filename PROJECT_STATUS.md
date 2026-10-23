# PROJECT_STATUS — 프로젝트 진행 상태 및 인수인계 대장

최종 갱신 시각: 2026-10-23 11:07 (KST)  
현재 마일스톤: **코드베이스 리팩터링 및 원격 동기화 완료**

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
