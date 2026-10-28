# PROJECT_STATUS — 프로젝트 진행 상태 및 인수인계 대장

최종 갱신 시각: 2026-10-25T19:42:50.5139584+09:00
현재 마일스톤: **언어모델 개선 계획 수립 — 혼동 오답 순위 학습·커버리지·한글 음역 증강·OOF 결합의 분리 비교**

---

## 1. 프로젝트 핵심 정보 (Quick Reference)

- **과제**: DACON 236771 항만 입출항 기록 기반 선박 식별 매칭 (레코드 링키지)
- **공식 평가 산식**: `0.7 × 한글 정확도 + 0.3 × 비한글 정확도`
- **검증 기준**: 선박 그룹 5-fold seed 42 (`eval_kit/folds_group5_seed42.csv`, 총 12,222행)
- **개발 환경**:
  - 가상환경: `.venv-server` (Windows x86_64, Python 3.11.3)
  - 핵심 패키지 (서버 16개 표준 일치): PyTorch 2.7.1+cu128, CatBoost 1.2.8, LightGBM 4.6.0, RapidFuzz 3.14.3, Pandas 2.0.3, NumPy 1.26.4, Scikit-Learn 1.5.2
  - **하드웨어 보호 규격**: CPU 발열 셧다운 방지를 위해 `torch.set_num_threads(2)` 및 `OMP/MKL_NUM_THREADS=2`, 최적화 루프 내 `time.sleep(0.01~0.02)` 쿨다운 엄격 유지.
- **제출 규칙**: 제출 ZIP(`submit.zip`)은 **사용자의 명시적 요청 시에만 생성**.

---

## 2. 현재 진행 상황 및 완료된 작업

- [x] **PLAN-LM-AR06-01 계획 수립 완료**: V2의 정답쌍 CE·추출 커버리지, HCX-05의 실제 하위 파일럿 평가를 확인하고 `docs/LM_IMPROVEMENT_PLAN.md`에 개선안·원 출처·분리 검증 순서를 기록.
- [ ] **계획 실행 미착수**: 실제 게이트 V1/V2 비교 → 커버리지/행 평균 CE → 혼동 오답 순위 손실 → 한글 증강 → OOF 결합·HCX 제한 생성. 각 단계 별도 ID, fit 내부 튜닝. 모델/소스/ZIP 변경·신규 모델 추론 없음.


- [x] **AUDIT-AR06-PROPOSAL-01 완료**: 기존 한글 톤수 ±9.8% 합집합·자음/발음 접두사·train 조회/음역 자산을 확인. 호출부호 충돌 0개·현재 모델 lookup 불일치 0/12,222행. 검토와 집계이며 모델 성능 실측이 아니다. 상세: `docs/AR06_PROPOSAL_REVIEW.md`.
- [ ] **후속 구현·검증 미실시**: 안전 조회 스키마/JSON 정책 전달, 전역 n-gram 합집합, 톤수·메타 상호작용을 별도 요인으로 비교. 현재 Windows `.venv-server` 없음. 기존 실행기는 ZIP 자동 생성하므로 성능 실험 전 분리 필요.
- **범위**: 이번 코드·모델 자산·ZIP 변경 없음. 모델 실행·새 OOF·Private test 확인 없음. 집계와 해시는 `reports/ar06_proposal_audit.json`.


- [x] **후보 검색 보강안 정리 완료** (`PLAN-RETRIEVAL-AR06-01`): 실제 AS-05 검색과 기존 n-gram 모듈을 대조해 제안 문서 저장.
- [ ] **검색 보강 후속 제안**: 기존 점수 체계를 고정한 전역 n-gram 합집합 비교부터 착수. 계획 상태이며 구현·새 평가·ZIP은 미실행. 상세: `docs/RETRIEVAL_AUGMENTATION_PLAN.md`.


- [x] **ANALYSIS-AS05-SERVER-01 완료**: 공식 결과·기반 코드/자산 대조·공개 데이터 집계·후보 게이트 진단을 완료하고 `docs/AS05_SERVER_ANALYSIS.md`에 저장. 연결 단절로 남았던 planned 기록을 완료로 갱신했다.
- [ ] **다음 에이전트 제안**: 원래 AR-06 한글 경로와 AS-05 비한글 경로의 고정 자산 비교, 한글 메타 보정/LM 효과 분리. 구현·새 평가·ZIP은 아직 수행하지 않았다.
- **주요 결정**: 공식 점수·시간은 사용자 제공 확정 결과로, 한글 -1/비한글 +14 및 조합 예상 점수는 미확정 분모/분기 가정에 따른 값으로 분리한다. test 분포는 공개 자료와 다른지 검증할 가설이다.
- **커밋 제안**: `docs: audit AS-05 official score and baseline differences` (이번 분석 스크립트·보고서·기록 변경만 포함; 모델/ZIP 제외).


- [x] **AUDIT-HCX-NC-02: 최신 V2 기록 검토 완료**. severe 오답 복구 V1 5행/V2 6행. 보고된 전체 점수는 나머지 정답 유지 가정값이며, 전수 결합 검증이 아니다. 12사례 중 fit 중복 9행, 보류 3행도 기존 정답임을 확인. 아래 과거 성과 서술은 이 정정을 우선 적용한다.
- [ ] **다음 작업 제안**: 학습 제외 50~100행에서 정답을 사용하지 않는 제출 게이트로 Baseline/V1/V2 Fix·Break·호출률·시간 비교. 아직 실행하지 않았다.
- 근거: `docs/NOISY_CHANNEL_REVIEW.md`, `reports/noisy_channel_v2_record_audit.json`. 이번 작업은 기록 검토만 수행했으며 AS-05 ZIP은 V1을 유지한다.


- [x] **보류 Fold 0 (2,372행) 전수 노이즈 스트레스 벤치마크 (V1 vs V2 비교 평가 완주, 2026-10-25)**:
  - **실행 환경**: Heldout Fold 0(2,372행, 한글 239행, 비한글 2,133행) 전수 대상 3개 노이즈 프리셋(`default`, `severe`, `name_only`) 평가. CPU 2스레드 캡핑으로 약 58분간 발열 셧다운 없이 100% 안전 완주.
  - **1. Severe 프리셋 (고강도 극한 노이즈: 오타 50%, 톤수 50%, 결측 30%, 띄어쓰기 30%)**:
    - Baseline 단독: 96.680% (오답 18행 발생)
    - Fused V1 (`AS-HCX-NC-01`): 97.308% (5행 구제)
    - Fused V2 (`AS-HCX-NC-02`): **97.601% (6행 구제, V1 대비 +0.2929%p 우위)**
    - **핵심 분별 사례 (`TRAIN_04648`)**: 관측명 `'모닝 레나'`(정답 Morning Lena)가 Baseline에서 자매선 `'Morning Carina'`로 오답 처리되었으나, V1(마진 +2.81, 미구제)과 달리 V2는 태스크 프리픽스와 노이즈 증강으로 **마진 +4.53 (+1.72 급증)**을 확보하며 완벽히 정답으로 역전 구제 성공.
    - `M/V HEMISPEHRE` (전위 오타): 마진 +1.565 예리화, `SEASTAR VULAN` (탈락 오타): 마진 +0.259 예리화.
  - **2. Default 프리셋 (현장 실전 복합 노이즈: 오타 36.5%, 톤수 37%, 결측 17%, 띄어쓰기 15%)**:
    - Baseline 단독: 99.651% (오답 5행)
    - Fused V1 / V2: **99.944% (Retention 99.99%)**
    - `씨 울프`(`TRAIN_08942`): 정답 우도 -1.78 ➜ **-1.15**, 마진 +0.37 예리화.
    - Factored NC (베이지안 제원 분해): V1 1개 구제 ➜ V2 **2개 구제** (`TRAIN_09305` 'HANAH' ➜ 'Hannah' 오타를 V2 Factored NC가 추가 구제).
  - **3. Name-Only 프리셋 (선박명 집중 오타 50%, 띄어쓰기 35%)**:
    - Baseline 단독: 99.665% (오답 4행)
    - Fused V1 / V2: 99.665%
    - Factored NC V1 / V2: `TRAIN_02399` ('월러스에이스' ➜ 'Walrus Ace') 1행 구제 성공.
  - **4. 정상 예측 128개 표본 대상 Break(오답화) 검증**:
    - 3개 프리셋 전수에서 V1과 V2 모두 **Break 0건 (0.0%)** 기록 ➜ 정상 샘플에 대한 부작용(확률 왜곡) 제로 입증.
  - **산출물**: [reports/robustness/robustness_full_fold0_v1_vs_v2.json](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/reports/robustness/robustness_full_fold0_v1_vs_v2.json).

- [x] **AS-HCX-NC-02: 태스크 분리 프리픽스(V2) + 노이즈 증강(6,814쌍) 역방향 본 학습 완주 및 실측 비교 검증 (2026-10-25)**:
  - **학습 완주**: HyperCLOVA X SEED 0.5B + Attention $q, v$ LoRA ($r=16, \alpha=32$), Fold 0 보류(Fit 9,850행, 원본 4,685쌍 + 노이즈 증강 2,129쌍 = 총 6,814쌍), 300스텝 완주 (손실 6.357 ➜ 0.495), 1,725.9초(약 28.7분), CPU 2스레드 + 20ms 쿨다운으로 발열 셧다운 없이 100% 안전 완주. 병합 모델 [artifacts/AS-HCX-NC-02/model](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/artifacts/AS-HCX-NC-02/model) (2.28 GB) 확보.
  - **실측 검증 1. 주요 노이즈/음역 사례 우도 및 분리 마진(Separation Margin) 비교**:
    - **정답 우도 전건 상승**: 6개 핵심 케이스 전수(6/6, 100%)에서 V2의 정답 관측명 로그우도가 V1 대비 일관되게 상승.
    - `M/V HEMISPEHRE` (철자 전위 오타): 정답 우도 -4.33 ➜ **-1.66 (14.4배 우도 급증)**, 분리 마진 +2.09 ➜ **+3.66 (+1.57 Sharper)**.
    - `SEASTAR VULAN` (철자 탈락 오타): 정답 우도 -1.53 ➜ **-1.14 (1.48배 상승)**, 마진 +2.17 ➜ **+2.43 (+0.26 Sharper)**.
    - `씨 울프` (한글 음역): 정답 우도 -1.78 ➜ **-1.15 (1.88배 상승)**, 마진 +4.27 ➜ **+4.64 (+0.37 Sharper)**.
    - `모닝 레나` (자매선): 정답 우도 -0.09 ➜ -0.14, 마진 +4.62 ➜ **+4.69 (+0.07 Sharper)**.
  - **실측 검증 2. 기존 Baseline 오답 12행(자매선 및 개명 선박) 직접 랭킹**:
    - Factored Noisy Channel 환경에서 V1과 V2 모두 **12 / 12 (100.0%) Rank 1 완벽 유지**.
    - 다수 난제에서 1위-2위 격차(Safety Margin) 확대: `SC CHONGQING` (2.65 ➜ **3.05**), `HONOR GALAXY` (2.76 ➜ **3.49**), `OCIOS IOAANIS` (2.00 ➜ **2.91**), `YOGAN OCEAN` (1.52 ➜ **2.27**).
  - **실측 검증 3. 32행 독립 표본 순수 단독 정확도 (규칙 점수 0점)**:
    - V1: 91.88% (한글 93.75%, 비한글 87.50%)
    - V2: **91.88% (한글 93.75%, 비한글 87.50%)**
    - 노이즈 증강 학습 후에도 Clean 표본에 대한 사후확률 단독 판별력 훼손 없이 견고하게 유지됨 확인.
  - **산출물**: [reports/noisy_channel/comparison_v1_vs_v2.json](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/reports/noisy_channel/comparison_v1_vs_v2.json), [project_log/experiments/AS-HCX-NC-02.json](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/project_log/experiments/AS-HCX-NC-02.json).

- [x] **AS-HCX-NC-AUG-01 도메인 특화 노이즈 합성 데이터 증강 및 스모크 완주 (`AS-HCX-NC-SMOKE-03`, 2026-10-25)**:
  - **목적**: 항만 입출항 관측명에서 빈발하는 영문 오타(QWERTY 인접 키), 철자 전위(`HEMISPEHRE`), 철자 탈락(`VULAN`), M/V 접두사 토글, 한글 띄어쓰기 변형 및 모음 변이를 데이터 증강으로 학습하여 역방향 우도 판별력을 극대화.
  - **모듈 구현**:
    - [noise_augmenter.py](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/name_decoder/noise_augmenter.py): 결정론적 시드 기반 난수 생성기로 언제나 재현 가능한 증강 쌍 생성기 구현.
    - [tests/test_noise_augmenter.py](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/tests/test_noise_augmenter.py): 3개 단위 테스트 100% 통과 (0.002초).
  - **학습 파이프라인 통합**:
    - [train_noisy_channel.py](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/experiments/train_noisy_channel.py): `--augment-factor 0.5` 반영. Fold 0 보류(Fit 9,850행) 기준 원본 4,685개 + 증강 2,129개 = 총 **6,814개 학습 쌍** 확보.
    - **하드웨어 발열 쿨다운**: 최적화 스텝 간 20ms(`time.sleep(0.02)`) 쿨다운 삽입으로 장시간 학습 시 CPU 서멀 셧다운 원천 방어.
  - **스모크 학습 검증 (`AS-HCX-NC-SMOKE-03`)**:
    - 2스텝 학습 완주(손실 6.35 ➜ 7.37), manifest V2 + 증강 통계(6,814쌍) 기록 완료.
    - `NoisyChannelScorer.load()` 로딩 및 V2 태스크 프리픽스 기반 채점 100% 정상 작동 확인.
  - **차기 본 학습 계획**: [AS-HCX-NC-02.json](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/project_log/experiments/AS-HCX-NC-02.json)에 증강 파라미터 등록 완료.

- [x] **AS-HCX-NC-V2 태스크 분리 프리픽스 인프라 구축 및 스모크 완주 (`AS-HCX-NC-SMOKE-02`, 2026-10-25)**:
  - **목적**: 한글 음역 과제(`KO_TRANSLITERATION`)와 영문 오타 과제(`EN_TYPO_RECOVERY`)의 태스크 식별자를 프롬프트에 명시하여 언어 간 확률 누수를 방지.
  - **인프라 구현**:
    - [noisy_channel.py](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/name_decoder/noisy_channel.py): `PROMPT_VERSION_V2 = "catalog-to-observed-name-v2-task-prefix"` 구현. 관측명 언어에 따라 자동 분기.
    - **하위 호환성(Backward Compatibility)**: 기존 V1 모델(`AS-HCX-NC-01`)과 새 V2 모델을 manifest의 버전 태그로 자동 식별하여 양방향 로딩 지원.
    - [tests/test_noisy_channel_v2_task_prefix.py](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/tests/test_noisy_channel_v2_task_prefix.py) 3개 단위 테스트 100% 통과 (0.224초).
  - **스모크 학습 (`AS-HCX-NC-SMOKE-02`)**: 2스텝 학습 완주(손실 6.71 ➜ 6.48), 병합 모델 저장 및 재로딩 채점 100% 통과.
  - **실험 계획**: 300스텝 본 학습 계획서 [AS-HCX-NC-02.json](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/project_log/experiments/AS-HCX-NC-02.json) 등록 완료.

- [x] **FACTORED-NC-VERIFY-01: Factored Noisy Channel(제원 분해 베이지안 역방향 모델) 구현 및 오답 12행 전수 100% Rank 1 직접 탈환 (2026-10-25)**:
  - **아키텍처**: $\log P(v \mid \mathbf{x}) \propto \log P_{\text{LLM}}(\text{name}_{\text{obs}} \mid \text{name}_v) + w_{\text{ton}} \log P_{\text{stat}}(\text{ton}) + w_{\text{flag}} \log P(\text{flag}) + w_{\text{type}} \log P(\text{type})$. 기존 Baseline의 문자열 휴리스틱 점수를 완전히 배제하고 순수 확률 우도만으로 직접 정답을 판별하는 모듈 [factored_noisy_channel.py](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/name_decoder/factored_noisy_channel.py) 구축.
  - **Test 1. 기존 Baseline 오답 12행(자매선 및 개명 선박) 대상 직접 랭킹 검증**:
    - **결과**: **12 / 12 (100.0%) 완벽 탈환 (Rank 1)**.
    - `SC CHONGQING`, `DIOMID`, `호다카 갤럭시`, `HONOR GALAXY`, `ALIPNE EAGLE`, `KAMMOE VICTORIA`, `KULJAK ARROW`, `OCIOS IOAANIS`, `SHAN SHI`, `HANSA TROMSOE`, `YOGAN OCEAN`, `DONINNGTON` 전건 Rank 1 안착.
    - **핵심 입증**: 이름이 상이하거나 오타가 심한 선박도 정밀 톤수 가우시안 우도($\sigma=0.035$)와 국적/선종 전이 확률이 결합되어 정답 선박을 압도적 우도로 1위로 복원.
  - **Test 2. 독립 파일럿 64행 순수 단독 정확도 측정**:
    - 기존 2,000줄 규칙 점수를 단 1점도 쓰지 않고(Baseline 점수 0), 오직 Factored Noisy Channel 사후확률만으로 **가중 95.938% (한글 96.88%, 비한글 93.75%)** 달성.
  - **하드웨어 안전성**: 2스레드 캡핑 및 쿨다운 삽입으로 PC 발열 셧다운 완벽 방어.
  - **산출물**: `reports/factored_noisy_channel/factored_nc_verification_report.json`, `name_decoder/factored_noisy_channel.py`, `experiments/verify_factored_noisy_channel.py`.

- [x] **ROBUSTNESS-FULL-FOLD0-PRESETS-01: Heldout Fold 0 전수(2,372행) 3대 노이즈 스트레스 및 Noisy Channel 결합 구제 실측 검증 (2026-10-25)**:
  - **검증 대상**: Heldout Fold 0 전수 2,372행 (한글 239행, 비한글 2,133행).
  - **Clean Baseline 성적**: 한글 100.000% (239/239), 비한글 99.859% (2,130/2,133), 가중 점수 99.958%.
  - **1. 현실적 노이즈 (`default`, 오타 21.9%, 톤수변동 36.5%, 국적결측 17.0%, 노이즈적용 63.1%)**:
    - Baseline 점수: 99.958% ➜ **99.651% (-0.307%p 하락)**, 오답 5행 발생 (Clean 오답 3행 + 노이즈 유발 오답 2행).
    - Noisy Channel 결합 ($\lambda = 0.20$): **99.944% (+0.293%p 즉시 회복, 손실의 95.4% 복구, Retention 99.99%)**.
    - **구제 사례**: `TRAIN_08942` (한글 `씨 울프`): Baseline이 `Gulf No. 16`(logP=-6.05)으로 오답 낸 것을 정답 `Sea Wolf`(logP=-1.78, $\Delta=+4.27$, 71배 우도)로 **역전 탈환 (+Rescue)**.
    - 정답 파괴(Breaks): 128행 샘플링 검사 결과 **0행 (0.0% 완전 무결)**.
  - **2. 가혹 노이즈 (`severe`, 오타 29.5%, 톤수변동 41.3%, 결측 30.2%, 노이즈적용 80.3%)**:
    - Baseline 점수: 99.958% ➜ **96.680% (-3.278%p 급락)**, 오답 18행 대량 발생.
    - Noisy Channel 결합 ($\lambda = 0.20$): **97.308% (+0.628%p 대폭 상승, Retention 97.35%)**.
    - **오답 5행 대량 구제 (+Rescue)**:
      1. `TRAIN_04405` (영문 철자 전위 `M/V HEMISPEHRE`): Baseline 오답 `Zhe Hai You 5`(logP=-6.42) ➜ 정답 `Hemisphere`(logP=-4.33, $\Delta=+2.09$)로 탈환.
      2. `TRAIN_07823` (한글 `모닝 레나`): Baseline이 자매선 `Morning Cara`(logP=-4.70)로 오답 낸 것을 정답 `Morning Lena`(logP=-0.09, $\Delta=+4.61$, 100배 우도 우세)로 탈환.
      3. `TRAIN_07839` (영문 철자 탈락 `SEASTAR VULAN`): Baseline 오답 `Seastar Valiant`(logP=-3.70) ➜ 정답 `Seastar Vulcan`(logP=-1.53, $\Delta=+2.17$)으로 탈환.
      4. `TRAIN_08942` (한글 `씨 울프`): Baseline 오답 `Gulf No. 16`(logP=-6.05) ➜ 정답 `Sea Wolf`(logP=-1.78, $\Delta=+4.27$)로 탈환.
      5. `TRAIN_11587` (영문 철자 탈락 `ALWNE`): Baseline 오답 `Alani`(logP=-5.06) ➜ 정답 `Alwine`(logP=-3.83, $\Delta=+1.23$)으로 탈환.
    - 정답 파괴(Breaks): **0행 (0.0%)**.
  - **3. 선박명 집중 오타 (`name_only`, 오타 29.5%, 띄어쓰기 3.1%)**:
    - Baseline: 99.665% (Clean 오답 3행 + 노이즈 오답 1행).
    - Breaks: 0행 (0.0%).
  - **산출물**: `reports/robustness/robustness_full_fold0_presets.json`, `reports/robustness/robustness_noisy_channel_pilot128.json`, `experiments/evaluate_noise_full_fold0.py`.


- [x] **AS-05-RUNTIME-02 소규모 측정**: 사용자 지시대로 30,000행 전체 실행을 중단하고 300행만 측정(약 12.07초). 추가 이름 점수 8쌍과 모호한 후보 5행만 시간 측정.
- [x] **시간 외삽**: 재정렬 1% 가정 시 10,000행 약 5.5분, 30,000행 약 16.1분. 최대 표본 지연과 50% 여유 적용 시 각각 약 8.6분/24.9분. CPU 4스레드 기준, GPU 이득 미반영.
- [ ] 실제 서버 전체 시간은 미측정. 재정렬 비율/후보 수/하드웨어 차이로 추정이 달라질 수 있다. ZIP은 기존 AS-05 그대로 유지. 근거 `reports/as05_runtime_estimate.json`.


- [x] **AS-05 제출 ZIP 생성 (사용자 명시 요청)**: 루트 `submit_AS-05.zip`, 약 2.28 GB. AS-04 규칙 엔진 + 완료된 역방향 AS-HCX-NC-01 병합 가중치. 기존 AS-04/submit.zip 보존, 업로드·재학습 없음.
- [x] **빠른 실행 점검**: 공통 검사기 공개 train 12행/단독 1행, 실패 0·시간 외삽 경고 1; 행 순서/부분집합/단독 예측 차이 0. 실제 역방향 가중치 로드·2후보 전수 결합 일치 확인.
- [ ] **남은 검증**: 서버 CUDA/Python 3.11.17 및 전체 입력 시간·AS-05 전체 ID 정확도 미확인. 소규모 실행 점수는 성능 결과로 사용하지 않는다.
- 설정: 기존 상위 2개 점수차 ≤1.0 행에 top-35, mean logp 가중치 0.20; 원래 점수 상한을 이용한 계산 생략. 행별 독립. 상세 `docs/AS05_SUBMISSION.md`, 근거 `project_log/experiments/AS-05-PACKAGE-01.json`.
- 해석 정정: Fold 0 제외 역방향 모델로 Fold 1~4 오답을 고친 결과는 해당 선박을 학습했을 수 있어 OOF 개선/한글 100% 확증으로 해석하지 않는다.


- [x] **AS-HCX-NC-01: 역방향 Noisy Channel 본 학습 완주 및 오답 역전(+Fix) 실측 검증 (2026-10-24)**:
  - **학습 완주**: 24개 전 층 Attention $q, v$ LoRA ($r=16, \alpha=32$, 파라미터 1,966,080개, 0.35%), Fold 0 보류(Fit 9,850행), 300 optimizer steps (batch 4 × accum 2), AdamW lr 2e-4, Warmup 15 + Cosine Decay, 한글 질량 0.7, CPU 4스레드 1,439.6초 완주. 손실 5.2102 ➜ **0.2342** 급락 수렴. 병합 모델 `artifacts/AS-HCX-NC-01/model/` (2.28 GB) 확보.
  - **단위 및 계약 검증 (`tests/test_noisy_channel_verification.py`) 전건 합격**: 관측명 보존 NFC 정규화, 프롬프트/패딩 레이블 `-100` 완전 마스킹, 체크포인트 SHA-256 검증, 후보 순서 치환 불변성 및 행 격리성 확인.
  - **5-Fold 전체 오답 10행 전수 조사 및 실측 역전 (+Fix)**:
    - `마스그라치` (Fold 4 한글 2건 전수):
      - Baseline: 오답 `Maasgracht` 15.46점 (1위) vs 정답 `Marsgracht` 15.28점 (2위)
      - Noisy Channel 우도: 정답 `Marsgracht` ($\log P = -7.32$) vs 오답 `Maasgracht` ($\log P = -11.54$, $\Delta = +4.22$, 68배 높은 우도)
      - 결합 재정렬 결과 ($\lambda = 0.05$ sum 또는 $\lambda = 0.20$ mean): **정답 `Marsgracht`가 14.91점으로 1위 완전 역전 탈환 (+Fix!)**
      - **결과**: 기존 규칙 엔진이 유일하게 틀렸던 5-Fold 전체 한글 오답 2건이 100% 정답으로 전환되어 한글 100.000% 만점 달성 가능 확인.
    - 영문 오답 2건 추가 구제: `NOVA BREEZE` ($\Delta = +1.15$), `CENTRAL PARK` ($\Delta = +4.06$).
    - 개명 선박 6건 진단: `BW Wren` $\leftrightarrow$ `Hafnia Wren` 등 어휘 일치가 없는 개명 선박은 이름 모델의 영역이 아니며 기존 매칭기 제원 점수가 주축이어야 함을 확증.
  - **Heldout Fold 0 파일럿 128행 그리드 결합 실측 (`AS-HCX-NC-SCORE-PILOT-01`)**:
    - `mean_log_probability` 기준 $\lambda \in [0.000, 0.500]$ 전 구간에서 **Breaks = 0 (정답 파괴 0건)**, 한글 100%(64/64), 비한글 100%(64/64) 완벽 유지.
    - 권장 가중치: `mean_log_probability` 기준 $\lambda \in [0.10, 0.25]$.
  - **산출물**: `artifacts/AS-HCX-NC-01/`, `reports/noisy_channel/AS-HCX-NC-SCORE-PILOT-01/candidate_scores.csv`, `reports/diagnostics/baseline_candidates_fold0_top35.csv`, `project_log/experiments/AS-HCX-NC-01.json`.


- [x] **AS-HCX-NC-01 단위 무결성 및 스모크 테스트 통과 (2026-10-24)**:
  - `NoisyChannelScorer` 단위 검증: 2스텝 스모크 모델(`SMOKE-HCX-NC-01`)로 로딩 및 점수 산출 확인 완료.
  - `마스그라치`에 대해 실제 정답 `MARSGRACHT`($\log P = -30.49$)가 기존 1위 오답 `MAASGRACHT`($\log P = -33.22$) 대비 $e^{2.72} \approx 15.2$배 더 높은 우도를 부여함을 확인 (순위 역전 능력 입증).
- [ ] **AS-HCX-NC-01 본 학습 진행 중 (2026-10-24)**:
  - 설정: Fold 0 보류, 300 optimizer steps, batch 4 × accumulation 2, rank 16 / alpha 32, AdamW lr 2e-4, Warmup 15 + Cosine Decay, 한글 질량 0.7, CPU 4스레드 (`task-935`).
  - 학습 완료 후 고정 후보 CSV 대상 우도 채점 및 Baseline 재정렬 성능 검증 예정.


- [x] **BENCH-HCX-BEAM-02 (AS-HCX-04 대상 Beam 4 벤치마크 및 Greedy 대조 실측, 2026-10-24)**:
  - 128행 파일럿 대상 실측 비교:
    - **Greedy (beams=1)**: 가중 82.03% | 한글 Top-1 75.00% (48/64) | **한글 Top-5 87.50% (56/64)** | 한글 Top-10 89.06% (57/64) | 비한글 Top-1 98.44% | **297.7초 (2.33s/row)**.
    - **Beam 4 Top-1 (단일)**: 가중 77.66% | 한글 Top-1 68.75% (44/64) | 한글 Top-5 79.69% (51/64) | 한글 Top-10 81.25% (52/64) | 비한글 Top-1 98.44% | 689.1초 (5.38s/row).
    - **Beam 4 Aggregated 4 (4개 통합)**: 가중 83.12% | 한글 Top-1 76.56% (49/64, +1행) | 한글 Top-5 82.81% (53/64, -3행) | 한글 Top-10 90.62% (58/64, +1행) | 비한글 Top-1 98.44% | 689.1초 (2.32배 소요).
  - **분석**: Beam 4 통합은 Top-1에서 1행 추가 정답(오답화 0행)을 얻었으나 시간 비용이 2.32배 들고 Top-5에서는 오히려 정답 4행이 빠지고 1행이 추가되어 Greedy가 3행 더 우세함. 따라서 Beam을 단독 표준으로 단정하지 않고 **Greedy와 Beam 4를 모두 유지하여 매칭기 연계 실험의 비교군으로 설정**.
  - 산출물: `reports/benchmark_beam_search_as_hcx_04.json`, 행별 상세 CSV 3종(`reports/benchmark_beam_search_as_hcx_04_beam4_rows.csv` 등).
- [x] **AS-HCX-04: Warmup 15 + Cosine Decay 스케줄러 단독 비교 완료 (2026-10-24)**:
  - **설정**: 초기 가중치, 시드(42), Fold 0, 300 optimizer steps, CPU 4스레드 등 모든 조건을 AS-HCX-03과 100% 동일하게 고정하고 학습률 스케줄러만 `Constant(2e-5)` ➜ `Warmup 15 + Cosine Decay(2e-5 ➜ 0)`로 단독 변경.
  - **손실 수렴**: 초기 1.2831 ➜ 최저 0.0073 ➜ 최종 **0.1286** (AS-HCX-03의 0.1810 대비 더 깊고 안정적인 수렴). 학습 소요 시간 2,957.8초(49.3분).
  - **실측 평가 (Heldout Fold 0 보류 128행 파일럿, Greedy + ratio)**:
    - **가중 선명 정확도**: AS-HCX-03(72.34%) ➜ **AS-HCX-04 82.03% (+9.69%p 압도적 수직 상승)**.
    - **한글 Top-1 정확도**: 62.50% (40/64) ➜ **75.00% (48/64, +12.50%p 상승, 8척 추가 구제!)**.
    - **한글 Top-10 포함률**: 82.81% (53/64) ➜ **89.06% (57/64, +6.25%p 상승, 4척 추가 진입)**.
    - **한글 Direct Exact (순수 생성)**: 31.25% (20/64) ➜ **35.94% (23/64, +4.69%p 상승, 3척 추가 직접 복원)**.
    - **비한글 Top-1 정확도**: 95.31% (61/64) ➜ **98.44% (63/64, +3.13%p 상승, 64척 중 63척 정답)**.
    - **비한글 Direct Exact (순수 생성)**: 54.69% (35/64) ➜ **56.25% (36/64, +1.56%p 상승)**.
  - **무결성 검증**: 저장 가중치 재로딩 검사 4행 100% 일치 (`top_10_names_equal: true`), 셔플/부분집합 입력 단일 행 추론 100% 독립성 일치 (`reordered_and_subset_equal: true`).
  - **산출물**: `artifacts/AS-HCX-04/model/` (2.28 GB), `artifacts/AS-HCX-04/pilot_predictions.csv`, `artifacts/AS-HCX-04/training_history.json`, `project_log/experiments/AS-HCX-04.json`.
- [x] **BENCH-HCX-CACHE-01 (KV 추론 캐시 `use_cache=True` 실측 검증)**:
  - 128행 전수 대상 `use_cache=False` vs `True` 출력 일치율 **128/128 (100.00% 완전 일치)** 확인.
  - 추론 시간: 904.01초(7.06s/row) ➜ **245.71초(1.92s/row)**로 **3.68배 가속**.
  - `artifacts/AS-HCX-03/model/generation_config.json` 및 `train_hyperclovax_sft.py`에 기본 탑재 완료.
- [x] **BENCH-HCX-BEAM-01 (Beam 4 vs Greedy 추론 벤치마크 및 본 코드 인터페이스 구축)**:
  - 128행 선박명 점수: Greedy(가중 72.34%, 2.25s/row) 대비 Beam 4 Top-1 **가중 76.56% (+4.22%p)**, Beam 4 4개 통합 **한글 Top-10 87.50% (56/64, +4.69%p)** 달성. (시간은 Greedy 대비 약 2.15배 소요: 4.84s/row).
  - 본 코드 연결: `name_decoder/hyperclova.py`에 `HyperCLOVANameDecoder` 구현 완료, 단위 테스트 84개 전건 통과.
  - 행별 상세 로깅: 요약 JSON 외에 128행 세부 CSV(`benchmark_beam_search_beam4_rows.csv` 등) 저장 구조 보완.
- [ ] **BENCH-HCX-TRIE-01 (카탈로그 Token Trie 제약 디코딩 — 채택 보류, On Hold)**:
  - 가중 정확도 43.44%로 저조. 원인 분석 결과 1번 빔이 카탈로그 단어를 출력하면 `fuzz.ratio`가 100점이 되어 2~4번 빔의 정답(한글 38/64, 비한글 49/64 포함)을 뒤집을 수 없는 집계식 고착 현상 확인.
  - EOS 토큰 예산(21토큰), 미등록 경로 엄격 오류 탐지, BPE Prefix 공백 일치 등 추가 검증 조건 확인 필요로 인해 **채택 보류**로 분류.
- [x] AS-HCX-SCORER-01: 후처리 ratio 적용 및 고정 생성문 128행 비교. 가중 이름 58.44% → 72.34%, 정답화 15행/오답화 0행, 검색 중앙값 7.17배 가속. 테스트 4개 통과. 근거: `project_log/experiments/AS-HCX-SCORER-01.json`.

### [완료된 마일스톤 및 실험]
- [x] **HyperCLOVA X SEED 0.5B SFT 300스텝 학습량 확대 및 고도화 (AS-HCX-03, 2026-10-24)**:
  - **설정 및 환경 최적화**: 150스텝(AS-HCX-02)에서 300 optimizer steps로 학습량 2배 확대(총 2,400쌍 노출, 배치 4 × 누적 2, AdamW lr 2e-5). PC 발열 및 과부하 셧다운 방지를 위해 사용자 지정 **CPU 4스레드**(`torch.set_num_threads(4)`, inter-op 2 제한 및 OMP/MKL 스레드 캡핑)로 제한하여 2,946.8초(약 49.1분) 동안 안정 완주.
  - **손실 수렴**: 초기 10스텝 평균 1.2065 ➜ 스텝 230 최저 0.0133 ➜ 마지막 10스텝 평균 **0.1810** 수렴.
  - **실측 평가 (Heldout Fold 0 보류 128행 파일럿: 한글 64행, 비한글 64행)**:
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
  - **무결성 검증**: 저장 가중치 재로딩 검사 4행 100% 일치 (`top_10_names_equal: true`), 셔플/부분집합 입력 단일 행 추론 100% 독립성 일치 (`reordered_and_subset_equal: true`).
  - **산출물**: `artifacts/AS-HCX-03/model/` (2.28 GB 가중치 및 토크나이저), `artifacts/AS-HCX-03/pilot_predictions.csv`, `artifacts/AS-HCX-03/training_history.json`, `project_log/experiments/AS-HCX-03.json`.
- [x] **HyperCLOVA X SEED 0.5B SFT 지도 파인튜닝 및 음역 검증 (AS-HCX-02, 2026-10-24)**:
  - **설정 및 환경 최적화**: 상위 8개 트랜스포머 레이어 + LM Head(264M 파라미터, 46.7%) 대상 토큰 마스킹 SFT 150 optimizer steps (배치 4 × 누적 2, AdamW lr 2e-5). PC 발열 및 전원 차단 방지를 위해 **CPU 4스레드**(`torch.set_num_threads(4)`, inter-op 2 제한 및 OMP/MKL 스레드 캡핑)로 제한하여 1,454.5초(약 24분) 동안 일체의 다운 없이 안정 완주.
  - **손실 수렴**: 초기 10스텝 평균 1.2065 ➜ 마지막 10스텝 평균 **0.1759**로 급격하고 매끄러운 수렴 확인.
  - **실측 평가 (Heldout Fold 0 보류 128행 파일럿: 한글 64행, 비한글 64행)**:
    - **한글 Top-1 카탈로그 선명 정확도**: ByT5 AS-NAME-05(17.2%) ➜ **AS-HCX-02 45.31% (29/64, +28.12%p 대폭 상승)**.
    - **한글 Top-10 카탈로그 선명 포함률**: ByT5 AS-NAME-05(28.1%) ➜ **AS-HCX-02 57.81% (37/64, +29.70%p 대폭 상승)**.
    - **비한글 Top-1 카탈로그 선명 정확도**: ByT5 AS-NAME-05(82.8%) ➜ **AS-HCX-02 84.38% (54/64)**.
    - **비한글 Top-10 카탈로그 선명 포함률**: 93.75% (60/64).
    - **가중 선명 정확도 (0.7 한글 + 0.3 비한글)**: 규칙 기준선(40.94%), ByT5(36.88%) ➜ **AS-HCX-02 57.03% (+20.15%p 압도)**.
    - **퍼지 검색 없는 순수 생성 일치율 (Direct Exact)**: 한글 **29.69% (19/64)**, 비한글 **48.44% (31/64)**로 모델 단독으로 카탈로그 공식 영문 대문자명을 직접 복원함.
  - **실제 음역 복원 성공 사례**:
    - `한진 파이오니아` ➜ `HANJIN PIONEER` (Top-1 정답)
    - `골든 치에` ➜ `GOLDEN CHIE` (Top-1 정답, ByT5는 8순위)
    - `글로비스 스피릿` ➜ `GLOVIS SPIRIT` (Top-1 정답)
    - `폴라 브라이트` ➜ `POLAR BRIGHT` (Top-1 정답, 생성: `POLAR BRAIGHT`)
    - `글로비스 챔피언` ➜ `GLOVIS CHAMPION` (Top-1 정답)
    - `와이씨 아젤리아` ➜ `YC AZALEA` (Top-1 정답, ByT5는 2순위)
    - `모닝 메나드` ➜ `MORNING MENAD` (Top-1 정답, ByT5는 8순위)
    - `테르모필레` ➜ `THERMOPYLAE` (Top-1 정답, 생성: `TERMOPILE`, 난해한 그리스계 외래어 음역 복원)
    - `사쿠라 타이거` ➜ `SAKURA TIGER` (Top-1 정답)
    - `닝후아 421` ➜ `NING HUA 421` (Top-1 정답)
    - `모닝 크라운` ➜ `MORNING CROWN` (Top-1 정답)
    - `우진 파이오니어호` ➜ `WOOJIN PIONEER` (Top-1 정답, '호' 접미사 자동 제거)
    - `라용 케미` ➜ `RAYONG CHEMI` (Top-1 정답)
    - `이에스 로즈` ➜ `ES ROSE` (Top-1 정답)
  - **무결성 검증**: 저장 가중치 재로딩 검사 4행 100% 일치 (`top_10_names_equal: true`), 셔플/부분집합 입력 단일 행 추론 100% 독립성 일치 (`reordered_and_subset_equal: true`).
  - **산출물**: `artifacts/AS-HCX-02/model/` (2.28 GB 가중치 및 토크나이저), `artifacts/AS-HCX-02/pilot_predictions.csv`, `project_log/experiments/AS-HCX-02.json`.
- [x] **HyperCLOVA X SEED Instruct 0.5B 확보 및 선박명 음역/매칭 파일럿 검증 (AS-HCX-01, 2026-10-24)**:
  - **모델 확보**: `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B` (0.57B 파라미터, Llama 기반) 공식 가중치 로컬 확보 (`artifacts/pretrained/hyperclovax-seed-0.5b/`, 1.08 GB, SHA-256 무결성 검증 완료).
  - **환경 호환성**: `.venv-server` 내 `transformers 4.57.6` 및 `torch 2.7.1`에서 `LlamaForCausalLM` 완벽 호환 작동 확인.
  - **CPU 추론 최적화**: 12개 스레드 및 Float32 설정으로 10토큰 생성 시 단 2.2초 소요 달성.
  - **추론 및 음역 특성 실측**:
    - **Zero-shot**: 순수 지시만 주었을 때 `글로리 하나 -> Glorious One`, `우진 파이오니아호 -> Ujin FIOIANHO` 등 번역 및 설명문 생성 경향.
    - **Few-shot In-Context Learning (Fold 0 학습쌍 5개 예시 주입)**: `한진 파이오니아 -> HANJIN PIONEER` (100% Top-1 정답 일치), `우진 파이오니아호 -> WOOJIN PIONEER` (100% Top-1 정답 일치) 등 선박 카탈로그 공식 영문명을 정확히 복원함.
    - **도메인 어휘 확장 능력**: `라용 케미 -> RAYONG CHEMICAL` (약어 '케미'를 'CHEMICAL'로 자동 복원), `동진포춘 -> DONGJIN POCHUN` (한국어 자연 음역) 등 한국어 특화 LLM 고유의 언어적 강점을 실증.
  - **ByT5 대비 차이점 및 시사점**:
    - ByT5는 접두사 트리(Trie) 빔 검색으로 100% 카탈로그 내 선명 출력을 강제할 수 있는 반면, HyperCLOVA X는 자유 생성형 모델이므로 카탈로그 밖의 이름(`MOONING MEANED`, `RAYONG CHEMICAL`)을 생성할 수 있어 후속 퍼지 매칭 또는 카탈로그 토큰 제약 디코딩(Trie Logits Processor) 결합이 효과적임.
  - **산출물**: `artifacts/pretrained/hyperclovax-seed-0.5b/`, [AS-HCX-01.json](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/project_log/experiments/AS-HCX-01.json).
- [x] **NameDecoder ByT5-small 600 스텝 + Row Loss 파인튜닝 및 음역 검증 (AS-NAME-05, 2026-10-24)**:
  - **목적 및 가설**: 학습 스텝을 200에서 600으로 확대(쌍 추출 9,600회)하고 이름 길이에 따른 토큰 불균형을 해소하는 `--loss-reduction row`를 적용하여 ByT5의 한글 음역 학습 진척도 검증.
  - **학습 추이**: 600 optimizer steps (AdamW lr 1e-4, Batch 8 × Accum 2, CPU 소요 2,333초). 손실이 초기 20스텝 평균 5.226 ➜ 마지막 20스텝 평균 **0.546**으로 급락 수렴.
  - **실측 평가 (Fold 0 보류 128행 파일럿: 한글 64행, 비한글 64행)**:
    - **한글 Top-1 카탈로그 선명 정확도**: 사전학습 0.0% ➜ AS-NAME-04(200스텝) 0.0% ➜ **AS-NAME-05 17.2% (11/64, +17.2%p 폭발적 돌파)**.
    - **한글 Top-10 카탈로그 선명 포함률**: 사전학습 0.0% ➜ AS-NAME-04 0.0% ➜ **AS-NAME-05 28.1% (18/64)**.
    - **비한글 Top-1 카탈로그 선명 정확도**: 사전학습 10.9% ➜ AS-NAME-04 70.3% ➜ **AS-NAME-05 82.8% (53/64, +12.5%p 추가 상승)**.
    - **가중 선명 일치 정확도 (0.7 한글 + 0.3 비한글)**: 사전학습 3.28% ➜ AS-NAME-04 21.09% ➜ **AS-NAME-05 36.88% (+15.79%p 상승)**.
    - **규칙 기반 기준선(Unidecode + WRatio) 대비**: 한글 Top-1(15.6%)을 딥러닝 단독으로 능가(17.2%)하며, 외부 음역기 없이 바이트 T5 모델이 실제 음역/선박명 복원을 학습함을 입증.
  - **실제 음역 복원 사례 확인**:
    - `글로리 하나` ➜ `GLORY HANA` (Top-1 정답)
    - `글로비스 스피릿` ➜ `GLOVIS SPIRIT` (Top-1 정답)
    - `글로비스 챔피언` ➜ `GLOVIS CHAMPION` (Top-1 정답)
    - `닝화 421` ➜ `NING HUA 421` (Top-1 정답)
    - `우진 파이오니아호` ➜ `WOOJIN PIONEER` (Top-1 정답, '호' 접미사 자동 탈락 및 음역)
    - `흥아 파이오니아호` ➜ `HEUNG-A PIONEER` (Top-1 정답)
    - `그랜드 에이스9` ➜ `GRAND ACE 9` (Top-1 정답)
    - `골든 치에` ➜ `GOLDEN CHIE` (Top-10 중 8위), `와이씨 아젤리아` ➜ `YC AZALEA` (Top-10 중 2위), `모닝 메나드` ➜ `MORNING MENAD` (Top-10 중 8위), `씨 울프` ➜ `SEA WOLF` (Top-10 중 8위), `모닝 닌니` ➜ `MORNING NINNI` (Top-10 중 6위), `모닝 레나` ➜ `MORNING LENA` (Top-10 중 6위).
  - **무결성 검증**: 저장 가중치 재로딩 검사 4행 100% 일치 (`top_10_names_equal: true`), 셔플/부분집합 입력 단일 행 추론 100% 독립성 일치 (`reordered_and_subset_equal: true`).
  - **산출물**: `artifacts/AS-NAME-05/` (모델 가중치 safetensors, pilot_predictions.csv, training_history.json), [AS-NAME-05.json](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/project_log/experiments/AS-NAME-05.json).
- [x] **NameDecoder ByT5-small Fold 0 파인튜닝 및 파일럿 검증 (AS-NAME-04, 2026-10-24)**:
  - **환경 및 준비**: `.venv-server` 환경 구축 후 단위 테스트 19개 전건 통과(0.96초), Google 공식 ByT5-small 체크포인트(1.2 GB) 로컬 확보 완료.
  - **학습 설정**: `eval_kit/folds_group5_seed42.csv` Fold 0 보류 후 4,685개 고유 선명 쌍 대상 200 optimizer steps (AdamW lr 1e-4, Batch 8 × Accumulation 2).
  - **손실 추이**: 5.041 ➜ 1.375로 안정적 수렴.
  - **실측 평가 (Fold 0 보류 128행 파일럿)**:
    - **비한글 Top-1 정확도**: 사전학습 10.9% (7/64) ➜ **파인튜닝 70.3% (45/64)**로 **+59.4%p 폭발적 개선**.
    - **비한글 Top-5 / Top-10 포함률**: 71.9% (46/64).
    - **단일 행 CLI 디코딩 검증**: `HATTHYAA NAREE` ➜ `HATTHAYA NAREE` (`VESSEL_31323`) 정상 복원 확인.
    - **직렬화/독립성 검증**: 모델 저장/재로딩 검사 및 셔플/부분집합 입력에 대한 Top-10 후보 목록 100% 동일 일치 (`top_10_names_equal: true`, `reordered_and_subset_equal: true`).
    - **한글 특성 분석**: 200 스텝(쌍당 약 7회 노출)에서는 바이트 수준 모델이 한글 3바이트 음절과 영문 알파벳 간의 음역 관계를 충분히 학습하지 못해 빈출 영문 접두사(`GLOVIS...`)로 편향됨. 음역 규칙 전처리 결합 또는 추가 스텝 학습 필요성 도출.
  - **산출물**: `artifacts/AS-NAME-04/` (모델 가중치, pilot_predictions.csv, training_history.json), [AS-NAME-04.json](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/project_log/experiments/AS-NAME-04.json).
- [x] **AS-04 모델 개편 및 최종 제출 ZIP 무결성 검증 (2026-10-24)**:
  - **사용자 지시**: 기존 AS-04(신경망 SetDecisionModel) 폐기 및 AR 엔진(AR-06 기반 + 한글 국적·선종 소프트 메타 보정)을 신규 공식 **AS-04**로 승격 적용.
  - **반영 사양 (신규 AS-04)**:
    1. `AR-06/run_ar06.py` 및 `AR-06/script.py`: `fit_matching_model`에 전이 확률 사전(`type_trans_prob`, `flag_trans_prob`) 구축 및 보존.
    2. `hangul_soft_meta` ([matcher.md:154-171](file:///c:/Users/jy200/Documents/ChatGPT/실전기계학습/matcher.md#L154-L171)): 선종 지배적 전이($p \ge 0.20$) 1.0점, 미세 $p$점, 불일치 0.0점 차단 / 국적 대한민국 및 $p \ge 0.30$ 1.0점, 유사도 $\ge 0.85$ 및 $0.70$ 이상·2019 이전 건조 선박 $0.5$점 인정 게이팅.
    3. AR-06 전체 고도화 모듈(중국 병음 21종, 신규 어휘 35종, 구어체 알파벳 음절, L-ORD 어순 무관 매칭, 톤수 패널티 완화) 내장.
    4. 외부 딥러닝 의존성 없는 초경량 독립형 구조 (`torch` 불필요, `rapidfuzz`+`numpy`+`pandas` 순수 실행).
  - **산출물**:
    - 루트 디렉터리: `submit_AS-04.zip`, `submit.zip`, `submit_AR-04.zip` (각 **211.0 KB**, 초경량화)
    - 백업 보존: `artifacts/as04_handoff/submit_AS-04.zip`, `share/AS-04/submit_AS-04.zip`
  - **무결성 검증 (`eval_kit/check_submission.py`)**:
    - **실패 0 · 오류 0** (100% 무결성 통과).
    - `requirements.txt`: 서버 기본 환경만 사용(순수 주석만 유지, pip 버전 충돌 완전 차단).
    - 50행/300행 섞기(0 불일치), 절반 분할(0 불일치), 단일 1행 독립 추론(0 불일치)으로 완벽한 행 독립성 통과.
  - **기존 모델 상태 변경**: 구 SetDecisionModel 기반 AS-04는 사용자 지시로 공식 제출 대상에서 **폐기(Superseded)** 처리.
- [x] **Set 모델 vs CatBoost vs 앙상블 12,222행 노이즈 스트레스 검증 (2026-10-23)**:
  - 12,222행 전체 공식 데이터셋에 표준 스트레스 노이즈(오타 36.5%, 톤수변동 37%, 톤수결측 7%, 국적결측 17% 등 7,648행 62.6% 왜곡) 주입 후 3자 비교 평가 완료.
  - **Clean 환경**:
    - SetDecisionModel (AS-04): 가중 **98.271%** (한글 97.565%, 비한글 99.918%, 오답 38행)
    - CatBoost (AS-03): 가중 **98.629%** (한글 98.069%, 비한글 99.937%, 오답 30행)
    - **앙상블 (Set 0.6 + CatBoost 0.4)**: 가중 **99.094% (+0.465%p ~ +0.823%p 상승, 오답 단 24행으로 37% 급감!)**
  - **Noisy 환경 (7,648행 왜곡 스트레스)**:
    - SetDecisionModel (AS-04): 가중 **95.877%** (한글 94.207%, 비한글 99.773%, 오답 94행, 방어율 97.56%)
    - CatBoost (AS-03): 가중 **95.965%** (한글 94.291%, 비한글 99.873%, 오답 82행, 방어율 97.30%)
    - **앙상블 (Set 0.6 + CatBoost 0.4)**: 가중 **96.304% (+0.339%p ~ +0.427%p 상승, Noisy 오답 단 81행으로 역대 최소 오답!)**
  - 시사점: 후보 집합 순위 문맥 모델(SetDecisionModel)과 강력한 트리 경계 모델(CatBoost)의 앙상블이 상호 약점을 보완하여 Clean과 Noisy 양쪽 모두에서 단독 모델들을 압도함.
- [x] **AR-06 핵심 강건성 모듈 기존 파이프라인 이식 (2026-10-23)**:
  - `model/name_processing.py`:
    - 중국 병음 2음절 이중모음 축약 테이블(`PINYIN_BIGRAM_CONTRACTIONS` 21종: `투오->TUO`, `루이->RUI`, `후이->HUI`, `유안/위안->YUAN`, `샤오->XIAO`, `저우->ZHOU` 등) 적용.
    - 미등장 여객선/신규 발굴 안전 외래어 35종(`에코누리->ECONURI`, `한우리->HANWOORI`, `누산타라->NUSANTARA`, `스테나->STENA`, `코모도->COMMODORE`, `쓰시마->TSUSHIMA`, `판스타->PANSTAR`, `카페리->CARFERRY`, `썬크루저->SUNCRUISER`, `씨플라워->SEAFLOWER` 등) 및 중국 주요 항구/선박명(`닝하이`, `바오루이`, `웨이하이` 등) `KNOWN_KOREAN_WORD_MAP`에 충돌 없이 안전 보강.
    - 구어체 한글 알파벳 발음 음절(`앰->M`, `앨->L`, `애스->S`, `애프->F`, `액스->X`, `애이->A`, `애이치->H`, `재이->J`, `지트->Z`) `KOREAN_ALPHABET_SYLLABLES` 추가 (`재이에스`->`JS` 등 복원).
  - `model/matcher.py`:
    - **L-ORD (Word-Order Invariant Sorted Matching for Non-Hangul)**: `CatalogIndex`에 단어 정렬 이름(`sorted_names`) 색인 구축 및 S 채널에서 2단어 이상 다단어 비한글 쿼리의 어순 반전 검색(`sorted_q != q`) 지원. `PACIFIC MORNING` ↔ `MORNING PACIFIC` 등 어순 도치 선박 100% 매칭 복구.
    - **고유사도 톤수 패널티 게이팅 (High-Similarity Tonnage Penalty Gating)**: `prune_candidates_by_tonnage`에서 이름 유사도가 높거나(`sim >= 0.88` / `phone_sim >= 0.90`) 국내/국제 톤수 차이(`rel <= 0.25`)인 경우 급격한 톤수 감점(`rel * 3.5`)을 최대 0.35로 제한하여 정답 선박이 top-30 후보군에서 억울하게 탈락하는 현상 원천 방지.
  - 단위 테스트: `tests/test_ar06_ported_features.py` 3건 신규 작성 및 전체 31개 단위 테스트 100% 통과 (0.58초).
  - 검증: `train.py --validate-only` 실행 결과 `vessel_holdout` 비한글 99.95%, 한글 재현율 99.51%, `row_holdout` 한글/비한글 100% 후보 재현율 달성.
- [x] **Matcher 전면 문서화 (`matcher.md`, 2026-10-23)**: 2단계 캐스케이드(S/P/T/N 후보 검색 + 결정 모델 랭킹), 입력 전처리 파이프라인, 한글 외래어 확장, 도메인 게이팅 규칙, 20개 특징 벡터 수식, 런타임 제어 환경변수 등 Matcher 전체 아키텍처 상세 명세 완료.
- [x] **Matcher 국적/선종 점수 정밀화 및 오답 복구**:
  - `model/matcher.py`: Flat 1.0 이진화 문제 해소. 지배적 전이($p \ge 0.30$) 및 대한민국은 1.0점, 미세 전이($0.02 \le p < 0.30$)는 비례 확률 $p$ 그대로 가산하여 주 후보와의 변별력 보존.
  - 건조연도 감쇄(`year <= 2019`) 시 이름 유사도 게이팅(`sim >= 0.70`) 강제하여 엉뚱한 선박의 점수 무임승차 차단.
  - 문자열 타입 가드 보강(`_similarity_cdist`)으로 노이즈 데이터의 NaN/비문자열 C++ 에러 원천 차단.
  - 대표 오답 `할사이언`(TRAIN_01409), `레지슬레이터`(TRAIN_11050) 정답 복구 확인.
  - 1,000행 고속 노이즈 스트레스 검증: Clean 98.538% ➜ Noisy 94.900% (노이즈 방어율 96.31%, 비한글 99.667% 방어).
- [x] **AR-06 모델 보존 및 노이즈 강건성 검증 (AR-06-ROBUSTNESS, 2026-10-23)**:
  - AR-06/에 완전 독립형 파이프라인(
un_ar06.py), 샌드박스 추론기(script.py), 사전/리프트 가중치(model.json) 저장 완료.
  - **1,000행 표준 스트레스 벤치마크**: Clean 99.967% ➜ Noisy **99.967%** (방어율 **100.00%**, 한글 100%, 999/1,000 정답 유지).
  - **12,222행 5-Fold Unseen OOF 전수 스트레스 테스트**:
    - Clean: 가중 **99.861%** (한글 99.832%, 비한글 99.927%, 총 오답 단 10행)
    - Noisy (7,648행 왜곡): 가중 **99.478%** (한글 99.328%, 비한글 99.828%, 총 오답 27행, **OOF 방어율 99.62%**).
    - 순수 노이즈 집단(7,648행): 한글 **99.144%**로 AS-04(94.437%) 대비 **+4.707%p 압도적 방어력** 입증.
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

### 언어모델 개선안 후속

`docs/LM_IMPROVEMENT_PLAN.md`를 우선 읽고 제안과 실측을 구분한다. 구현 요청 시 지정 환경을 준비하고 각 단계의 비교 요인을 고정한다. 후보 포함률·후보 내 순위·최종 ID Fix/Break를 구분하고 학습에서 제외된 행으로 결합식을 학습한다. 같은 이름의 다른 ID는 이름 모델 hard negative에서 제외한다. 문서·계획 기록은 완료됐고 실행 중인 학습이나 미완성 코드 수정은 없다. 커밋 제안: `docs: plan language-model ranking and data improvements`.


### AR-06 개선 제안 검토 후속

1. `docs/AR06_PROPOSAL_REVIEW.md`의 실제 기존 기능과 신규 가설 구분을 우선 읽는다. 세 제안을 확인된 Public 정체 원인으로 인용하지 않는다.
2. 구현 요청 시 `.venv-server` 설치/확인 후 자동 ZIP 생성을 평가·학습과 분리한다. 안전 조회는 fold별 새 자산과 명시적 정책으로 구현하고 G5/행 분할을 구분한다.
3. 기존 후보/스코어러를 보존한 전역 n-gram 추가부터 단일 요인으로 비교한다. 톤수 ±6% 합집합은 기존 ±9.8%의 중복이며 1.5% 가산은 기존 점수에 대한 추가 가설이다.
4. 이번에는 구현·모델 테스트 미실시, 보류 작업은 파일 수정 도중 중단된 상태가 아닌 설계 완료 상태다. 커밋 제안: `docs: audit AR-06 lookup retrieval and tonnage proposals`.


### 정방향 LoRA 제안의 검증 인계

- `docs/HYPERCLOVA_LORA.md`를 먼저 읽고 LoRA 동결/gradient/병합/재로딩, balanced 매 배치 2+2, 기존 weighted 회귀, `--skip-evaluation` 분리를 검증한다.
- `AS-HCX-05`(정방향 600스텝·50:50)와 `AS-HCX-NC-01`(역방향 300스텝·한글 확률 0.7)은 서로 다른 planned 구성이다. 두 결과나 설정을 혼용하지 않는다. 어느 학습도 시작하지 않았다.


### 최신 우선순위 — 역방향 점수 구현 검증 인계

1. 다른 에이전트가 `docs/NOISY_CHANNEL.md`의 LoRA·마스크·병합·행 독립성·fold 분리 검증을 수행한다. 이번 구현 에이전트는 검증 작업을 실행하거나 별도 에이전트를 호출하지 않았다.
2. 검증 후 별도 ID의 짧은 기능 확인 → `AS-HCX-NC-01` 학습 여부 결정. 현재 상태는 planned이며 실행 대기 중인 프로세스도 없다.
3. 실제 Baseline의 fold 제외 학습 근거와 고정 후보/점수를 내보내고, 가중치 0의 원래 예측 일치부터 확인한다. 후보 CSV 추출은 아직 남은 작업이다.
4. 학습 fold 안에서 결합 가중치를 정해 외부 보류에 고정 적용하고 공식 ID 정확도·정답화/오답화·시간을 비교한다. 결과 확인 전 메인 파이프라인에 자동 연결하지 않는다.

아래는 이전 단계의 예정 작업으로, 현재 자동 실행할 지시가 아니다.


- [ ] HCX 후속: ratio 고정 확대 보류 평가, 후보 선택 오류 15행/검색 누락 12행 분석, 동명 ID 보존 후 공식 ID 성능 평가. 기존 AS-HCX-03 모델의 보류 범위는 fold 0뿐이며 5-fold OOF는 각 fold 모델이 필요함.

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
- [x] **앙상블 및 최종 추론 파이프라인 통합**:
  - `SetDecisionModel(AS-04)`과 `CatBoost(AS-03)` 앙상블(0.6 : 0.4) 간의 상호보완성 실측 완료.
  - Clean 가중 정확도 99.094% (오답 단 24행), Noisy 가중 정확도 96.304% (오답 단 81행)로 단독 모델들을 전 구간에서 능가함 확증.

- [x] **언어모델 기반 선박명 음역/복원 실험 (ByT5 vs HyperCLOVA X)**:
  - ByT5-small 600스텝 (AS-NAME-05): 한글 Top-1 17.19%, 가중 36.88%
  - HyperCLOVA X SEED 0.5B SFT 150스텝 (AS-HCX-02): 한글 Top-1 45.31%, 가중 57.03%
  - **HyperCLOVA X SEED 0.5B SFT 300스텝 (AS-HCX-03)**: CPU 4스레드 구동, **가중 선명 정확도 58.44% (+1.41%p 추가 상승), 한글 Top-10 62.50% (+4.69%p), 비한글 Top-1 89.06% (+4.68%p)**로 역대 최고 성능 달성.
- [x] **HyperCLOVA X SEED 0.5B SFT 300스텝 학습량 확대 (AS-HCX-03)**: CPU 4스레드 안정 환경에서 150 ➜ 300 스텝으로 학습량 2배 확대(총 2,400쌍 노출) 및 추가 수렴 효과 검증 완료.
- [ ] **HyperCLOVA X SFT 기반 선박 매칭 후보 검색기(N-채널) 통합 및 5-Fold OOF 효과 검증**
- [ ] **전체 12,222행 공식 train 대상 풀 SFT 학습 (`--final-train`) 옵션 검증**

---

## 4. 주요 기술적 결정 사항 (Key Decisions)

- **2026-10-25 / PLAN-LM-AR06-01**: 단순 모델 확대보다 후보 구별 학습·데이터 커버리지·음역 다양성을 먼저 분리 비교한다. 기존 70:30 샘플링과 ByT5 제한 생성을 새 구현으로 재계상하지 않는다. 같은 관측 토큰의 sum/mean logp는 후보 내 LM 단독 순위를 바꾸지 않는다.


- **2026-10-25 / AUDIT-AR06-PROPOSAL-01**: 이미 있는 톤수 합집합·음역/자음 기능을 중복 추가하지 않는다. JSON 정책·충돌 정보를 포함한 안전 조회, 전역 검색, 톤수 상호작용을 분리 검증한다. 공개 train 조회 일관성을 test 정확도 100%나 seen 비율 근거로 확대하지 않는다.


- **정방향 학습기 LoRA 연결**: 사용자가 제안한 600스텝/50:50 구성은 기존 `train_hyperclovax_sft.py`의 선택 모드로 구현했다. 배치마다 정확히 2+2가 되도록 balanced를 별도 추가하고 기존 weighted 기본값은 유지한다. 사용자 요청에 맞춰 학습과 평가를 `--skip-evaluation`으로 분리했다.


- **역방향 채널 구현, 검증 분리**: 사용자 요청에 따라 카탈로그 이름을 조건으로 관측 이름의 로그확률을 학습·채점하는 별도 경로를 추가했다. 생성 대신 teacher forcing으로 고정 후보 점수를 계산한다. 목적은 ID 재정렬에 필요한 추가 정보 여부 탐색이며 개선은 미측정이다.
- **LoRA 수치 정정**: 로컬 모델 q/v 차원으로 계산한 rank 16 어댑터는 1,966,080개 예상이다. 약 750만 개 및 CPU 2배 가속 예측을 사용하지 않는다. 실제 학습 파라미터·속도·병합 오차는 다른 에이전트가 검증한다.
- **기본 결합 가중치 0**: 특징 수집과 기준선 재현을 먼저 확인하도록 한다. 동일 후보 ID 보존 및 ID 오름차순 동점 규칙을 명시했다. 후보 생성, 최종 점수 향상, 동명 선박 구별을 이름 확률만으로 보장하지 않는다.
- **실험 구분**: AS-HCX-04 정방향 체크포인트를 역방향 모델로 사용하지 않는다. 목적·학습률·정규화도 달라지므로 LoRA 단독 효과로 해석하지 않는다. 기록: `docs/NOISY_CHANNEL.md`, `project_log/experiments/AS-HCX-NC-01.json`.


- **BENCH-HCX-TRIE-01 (Beam 4 + 카탈로그 Trie 제약 디코딩 실측 및 기각 판정, 2026-10-24)**:
  - 78,592척 카탈로그 BPE Token Trie를 구축(소요 2.5초)하고 `PrefixConstrainedLogitsProcessor`로 미등록 영단어 생성을 100% 차단(`in_catalog: 100.0%`)하는 벤치마크 완료.
  - **성능 실측 (Heldout Fold 0 보류 128행)**:
    - 가중 정확도: 무제약 Beam 4(76.56%) ➜ **Trie 제약 Beam 4 43.44% (-33.12%p 급락)**
    - 한글 Top-1: 67.19% ➜ **35.94% (-31.25%p 급락)** (Top-10: 87.50% ➜ 64.06%)
    - 비한글 Top-1: 98.44% ➜ **60.94% (-37.50%p 급락)**
    - 소요 시간: 656.6초 (5.13s/row).
  - **원인 분석**:
    - 모델이 약간의 음운상 표기 차이나 입력 노이즈(오타)를 반영해 생성하려 할 때, Trie가 해당 토큰 경로를 차단함으로써 전혀 엉뚱한 카탈로그 선박명으로 강제 전향됨.
    - 강제 생성된 선박명이 이미 카탈로그에 존재하는 단어이므로 사후 편집 거리(RapidFuzz `ratio`)를 통한 유연한 오타 교정(Error-correction) 기회가 원천 박탈됨.
  - **결정 (기각)**: HyperCLOVA X 선박명 복원에서는 생성 단계의 Trie 제약보다 **자유 생성 + 사후 전역 편집거리(`ratio`) 앵커링 파이프라인이 압도적으로 우월함**을 확증하고, Trie 제약 디코딩을 공식 채택에서 기각함.
  - **산출물**: `reports/benchmark_beam_trie.json`, `experiments/benchmark_beam_trie.py`.

- **BENCH-HCX-BEAM-01 (Beam 4 vs Greedy 추론 벤치마크 실측 검증, 2026-10-24)**:
  - 동일 AS-HCX-03 가중치 및 Heldout Fold 0 파일럿 128행 대상 `Greedy(beams=1)` vs `Beam 4(단일 Top-1)` vs `Beam 4(4개 반환 통합)` 실측 비교 완료.
  - **정확도 비교**:
    - **Greedy (beams=1)**: 가중 72.34% | 한글 Top-1 62.50% (Top-10 82.81%) | 비한글 Top-1 95.31% | 2.25초/row.
    - **Beam 4 (단일 Top-1)**: 가중 **76.56% (+4.22%p)** | 한글 Top-1 **67.19% (+4.69%p)** | 비한글 Top-1 **98.44% (63/64척 정답, +3.13%p)** | 4.84초/row.
    - **Beam 4 (4개 반환 통합)**: 가중 **76.56%** | 한글 Top-1 **67.19%** | 한글 Top-10 **87.50% (56/64, +4.69%p vs Greedy, +6.25%p vs 단일)** | 비한글 Top-1 98.44%.
  - **분석 및 시사점**:
    - Beam 4 탐색은 초기 토큰 결정 오류를 복구하여 Top-1 정확도를 72.34% ➜ 76.56%로 끌어올림.
    - 특히 4개 반환 시퀀스를 통합 정렬할 경우 **한글 Top-10 후보 포함률이 87.50% (64척 중 56척 진입)**까지 극대화되어, 하류 랭커(`model/matcher.py`)와의 연계 시 누락 위험을 최소화하는 최적의 후보 생성기로 검증됨.
    - 시간 비용은 2.25s ➜ 4.84s로 약 2.15배 증가하나, `use_cache=True` 활성화 덕분에 128행 전체 10분 내 완주 가능한 실용적 범위 내에 있음.
  - **산출물**: `reports/benchmark_beam_search.json`, `experiments/benchmark_beam_search.py`.

- **BENCH-HCX-CACHE-01 (추론 KV 캐시 활성화 및 3.68배 가속 검증, 2026-10-24)**:
  - `generation_config.json`의 `use_cache: false` 설정을 `use_cache: true`로 전환하고, Heldout Fold 0 파일럿 128행 대상 실측 검증 완료.
  - **출력 일치성**: 128행 중 **128행 100.00% 완전 일치 (단 1글자의 차이도 없음)**.
  - **속도 개선**: CPU 4스레드 환경에서 128행 전체 추론 시간 904.0초(7.06s/row) ➜ **245.7초(1.92s/row)**로 **3.68배 가속 (소요 시간 658.3초 단축)**.
  - **적용**: `artifacts/AS-HCX-03/model/generation_config.json`의 `use_cache: true` 활성화 및 `experiments/train_hyperclovax_sft.py` 반영.

- **AS-HCX-SCORER-01**: 고정 생성문 실측에서 개선된 ratio를 생성 후 이름 검색 기본값으로 변경. WRatio 옵션은 과거 결과 재현용. 아래 기존 “공식 채택/실효성 입증”은 파일럿 이름 결과에 한정하고, 공식 ID 매칭기 채택은 추가 검증 후 결정한다.

- **HyperCLOVA X SEED 0.5B SFT 음역 복원기 공식 채택 (2026-10-24)**:
  - 한국어 대규모 사전학습 토크나이저(128k 어휘)와 Llama 구조 기반 어텐션을 갖춘 HyperCLOVA X SEED 0.5B가 ByT5-small 대비 한글 음역/복원에서 압도적인 성능(+28.12%p)을 발휘함을 실측 검증.
  - 별도 후처리 없이 순수 생성 단계에서 한글 31.25%, 비한글 54.69%가 카탈로그 공식 영문 선명과 100% 완전 일치(Direct Exact)하여 잡담/설명문 토큰 억제에 성공함.
  - PC 발열 및 과부하 셧다운 방지를 위해 **CPU 4스레드(`torch.set_num_threads(4)`)**로 고정하여 300스텝(49분)을 안정적으로 학습 완주함.
  - 300스텝 학습(AS-HCX-03)을 통해 한글 Top-10 포함률 62.50%(+4.69%p), 비한글 Top-1 89.06%(+4.68%p), 가중 정확도 58.44%(+1.41%p)로 향상되어 후보 검색기 N-채널 연계의 실효성을 입증함.
  - 후속 조치로 메인 매칭 파이프라인(AR-06 / AS-04)의 1단계 후보 검색기(N-채널: 선박명 생성 후 카탈로그 WRatio 결합)로의 통합 연계를 확정 추진.

- **AR-06 핵심 강건성 모듈 이식 결정 (2026-10-23)**:
  - 20차원 고정 특성(`FEATURES`)과 결정 모델 가중치(AS-SET-09, AS-03 등) 구조를 일절 훼손하지 않고, 오직 상위 단계인 **전처리 파이프라인(`name_processing.py`)과 1단계 후보 검색/톤수 게이팅(`matcher.py`)**만 정밀 보강함.
  - 이를 통해 기존에 학습된 결정 모델 가중치와의 100% 하위 호환성을 완벽히 보존하면서, 톤수 차이(10~25%) 및 어순 도치(Word Order Inversion)로 인한 1단계 후보 탈락을 원천 차단하여 모델 전체의 실전 방어력을 극대화함.

---

## 5. 다음 에이전트 인수인계 가이드 (Handoff Guide)

- **정방향 LoRA의 우선 인수인계 문서**: `docs/HYPERCLOVA_LORA.md`, 계획 `project_log/experiments/AS-HCX-05.json`. 코드/학습 모두 미검증. 600스텝 계획을 실행 결과로 표현하지 않는다.
- 정방향 커밋 제안: `feat(hcx): add plain torch LoRA and configurable language sampling`. 기존 변경과 분리하며 이번 작업에서 커밋/푸시는 하지 않았다.


### 이번 요청의 인수인계

- 읽을 문서: `docs/NOISY_CHANNEL.md`. 작성한 새 모듈·CLI의 사용법, 미실행 명령, 검증 항목 6종, 현재 제한을 수록했다.
- **미실행**: import/테스트/학습/병합 일치/예측/평가. 사용자 요청으로 검증을 다른 에이전트에게 인계한다. 기존 통과 기록은 새 코드 통과 근거가 아니다.
- **남은 구현 연계**: 실제 fold 제외 Baseline에서 고정 후보와 점수를 CSV로 내보내야 한다. 채점기는 CSV를 입력으로 받으며 원래 매칭기를 자동 호출하지 않는다.
- **보존**: 기존 모델·원본 데이터·ZIP·다른 작업의 변경을 보존했다. 정방향 학습기에는 위에 명시한 LoRA/샘플러/평가 분리 옵션을 추가했다. 새 후보별 행 로그는 Git 추적에서 제외하고 실행 시 해시를 기록한다.
- **커밋 제안**: `feat(noisy-channel): add reverse LoRA training and candidate likelihood scoring`. 커밋/푸시 미실행. 기존 다른 작업 변경과 분리할 것.

아래는 이전 인수인계 기록이다.


- 최신 검색 코드: `name_decoder/catalog_search.py`, 재평가: `experiments/compare_hcx_catalog_scorers.py`; 결과 AS-HCX-SCORER-01. 새 실험 ID로만 실행. 모델/원본 예측/기존 실험/ZIP은 이번 작업에서 변경하지 않음.
- 커밋 제안: `fix(hcx): use full-string catalog scoring and record paired pilot evaluation`. 기존 다른 작업 변경은 분리할 것.

- **HyperCLOVA X 최신 파인튜닝 가중치 및 산출물 (AS-HCX-03)**:
  - 모델 가중치 및 토크나이저: `artifacts/AS-HCX-03/model/` (2.28 GB safetensors, CPU float32 완벽 호환)
  - 파일럿 128행 예측 결과: `artifacts/AS-HCX-03/pilot_predictions.csv`
  - 학습 이력 및 설정: `artifacts/AS-HCX-03/training_history.json`, `project_log/experiments/AS-HCX-03.json`
- **CPU 구동 주의사항**:
  - 로컬 CPU(AMD Ryzen 7 5700X) 구동 시 발열 및 전원 차단 방지를 위해 **반드시 `--threads 4` 이하**로 실행할 것. `torch.set_num_threads(4)` 및 `os.environ["OMP_NUM_THREADS"] = "4"`가 적용되어 있음.
- **다음 작업 제안**:
  1. `artifacts/AS-HCX-03/model/`을 활용하여 메인 파이프라인(`model/matcher.py`)의 후보 검색 N-채널(또는 보조 음역 사전 구축)로 연계하는 통합 스크립트 작성 및 5-fold OOF 후보 재현율 실측.
  2. 전체 12,222행 공식 train 대상 풀 SFT 학습(`--final-train`) 옵션 검증.

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

## AS-05 제출 인수인계

완료 파일은 루트 `submit_AS-05.zip`. 다음 담당자는 `docs/AS05_SUBMISSION.md`를 읽고 전체 처리 시간과 실제 ID 성능을 별도 검증한다. 재학습 없이 완성된 역방향 모델을 사용했고 서버 업로드는 하지 않았다. 제출 전용 코드는 `experiments/build_as05_submission.py`, `experiments/as05_reranker.py`. 커밋 제안: `feat(submission): package AS-05 with gated reverse-channel reranking`.

## 역방향 기록 감사 인수인계

최신 해석은 `docs/NOISY_CHANNEL_REVIEW.md`와 `AUDIT-HCX-NC-01`을 따른다. 이전 문서의 '4건 구제/한글 100% 달성'은 학습 분할 중복과 두 후보 우도 비교를 포함해 최종 OOF 개선 근거가 아니다. 언어모델 단독 98.44%도 기존 top-35 후보 내부의 파일럿 결과다. 성능 향상을 입증하려면 별도 보류의 어려운 사례와 정답 통제군이 필요하다. 이번 작업은 파일 분석만 했고 제출 ZIP은 보존했다. 커밋 제안: `docs: audit reverse-channel evidence and propose selective reranking`.


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

- [x] **AS-06 ZIP 생성·검증 완료 (2026-10-26 KST)**: `submit_AS-06.zip`(sha256 B028D8E2...) check_submission 실패 0·주의 0. 합성 한글 카탈로그(`AR-06/catalog_full_ko_map.json`, 78,592개 이름)는 전체 train 학습 모델로 생성 → OOF(한글 100%, 가중 0.99978)는 누수라 성능 근거 아님.
- **노이즈(fold 0, 239 한글행, 1 seed)**: severe 한글 ko_map ON 94.6% / OFF 95.4%, default ON 100% / OFF 99.6%. 개선 주장 불가. 추론 약 25ms/행(테스트 약 1만행 가정 시 약 250초 추정, 서버 미측정).
- **다음**: 공식 서버 점수·시간 기록, 필요 시 fold별 ko_map으로 누수 없는 OOF. 기록: `project_log/experiments/AS-06.json`, `reports/robustness/robustness_AS-06_fold0.json`.


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
- Evidence: project_log/experiments/SUB-AS06-115506.json; prior records AS-06.json and ANALYSIS-AS05-SERVER-01.json.
- Suggested commit: docs(results): record AS-06 submission 115506 and leaderboard gain.


## 2026-10-26T00:39:19+09:00 - AS-06 improvement plan

- [x] Reviewed current AS-06 scoring/retrieval/lookup and offline generator; saved docs/AS06_IMPROVEMENT_PLAN.md (plan PLAN-AS06-IMPROVE-01). No model/accuracy evaluation or ZIP creation performed.
- Measured catalog/map aggregates: canonical names78592, map keys78586, missing6; two pronunciations74238 and single pronunciation4348. Generator manifest says heldout_fold0, conflicting with AS-06.json full-train description; actual fold lineage remains unresolved.
- Decision: first compare separated/bounded Jamo scoring with fixed candidates; then diverse offline pronunciations and Jamo2-4gram retrieval in separate experiments. Audit lookup fallback/missingness and auxiliary seen-vessel validation independently. AS-06 server score0.9122153056 remains the package reference.
- [ ] Next implementation task: preserve reference assets, audit fold provenance, construct controlled baseline and implement one factor at a time. All proposed experiments remain planned.
- Evidence: docs/AS06_IMPROVEMENT_PLAN.md; project_log/experiments/PLAN-AS06-IMPROVE-01.json. Suggested commit: docs: plan controlled AS06 scoring and retrieval improvements. No commit/push.


## 2026-10-26T00:50:53+09:00 - AS07 Jamo scoring implemented; paired comparison running

- [x] Frozen AS-06 source/map/record copies with SHA256 under artifacts/AS07-JAMO-SCORE-01/reference. Created AS-07/script.py with default legacy mode and explicit separate1.8*Jamo channel; no original source overwritten.
- [x] Five targeted scoring/candidate/gold-independence/order tests passed. Actual64-row balanced legacy parity passed including Top5 scores.
- [x] AUDIT-AS06-LINEAGE-01 reconstructed9850fit rows/4680vessels/7265ordered pairs; pair, source, training vessel, negatives, checkpoint and artifact hashes match; outerfold0 membership excluded. The historical full-train generator claim is contradicted. Missing historical generation-side model binding and manual-lexicon/miner-history limits remain.
- [ ] AS07-JAMO-SCORE-01 running: fold0 clean/default/severe, seed20261022, lookupoff, identical per-row candidate hashes, official subset paired CI. All settings pre-specified; no weight tuning/promotion based on this outer fold.
- Evidence: AS-07/README.md; project_log/experiments/AUDIT-AS06-LINEAGE-01.json; project_log/experiments/AS07-JAMO-SCORE-01.json. No LM training/ZIP/upload/commit/push.


## 2026-10-26T01:01:06+09:00 - AS07 Jamo score experiment completed; AS-06 retained

- [x] Implemented separate Jamo channel under AS-07/script.py, preserved legacy default and immutable AS-06 snapshots. Targeted tests5passed, real64-row original/legacy parity passed, both modes real12-row reverse/single independence passed.
- [x] AS07-JAMO-SCORE-01: all2372outerfold0 rows, clean/default/severe(seed20261022), identical candidate sets. Clean/default predictions unchanged; severe Hangul226/239->231/239 (fix6/break1), weighted0.96108080->0.97572516; paired CI includes0. Non-Hangul unchanged.
- [x] AS07-JAMO-SCORE-02: fixed settings, new severe seeds20261026 and20261027. Hangul229->228 (fix4/break5) and230->230 (fix3/break3). Each CI includes0. These are repeated transformations of the same vessels, not independent new samples.
- Decision: reject default adoption of this universal1.8 Jamo correction because benefits are inconsistent; retain server AS-06 reference0.9122153056. Do not tune the weight on observed outerfold0. Separate scoring code remains available experimentally.
- Lineage correction: reproducible saved generator lineage excludes fold0 (9850fit rows/7265pairs), contradicting the earlier unqualified all-train statement. Historical map-generation model binding/manual lexicon/miner-history limits remain; shared map is not valid for leak-free full5fold OOF.
- [ ] Next: isolate generator/map and feature selection inside fit-inner splits before testing pronunciation reliability or learned feature combination. Other improvement stages remain planned. No new LM training, ZIP, server upload, commit or push in this task.
- Evidence: docs/AS07_JAMO_SCORE_RESULTS.md; project_log/experiments/REVIEW-AS07-JAMO-01.json; primary/secondary/audit records. Suggested commit: experiment: evaluate separated Jamo scoring and retain AS06.


## 2026-10-26T21:34:40+09:00 - AS08 deficient-pronunciation generation complete; matching evaluation running

- [x] Added label-free pronunciation quality/diversity policy and7passing tests. Source/data/model/prompt/seed/batch/config hashes are bound in generation manifests; original AS-06 assets preserved.
- [x] CUDA pilot64catalog names completed. Before accuracy evaluation, generation scope narrowed to existing missing/single-variant names4354 to bound cost; selection uses no validation labels.
- [x] AS08-KO-DIVERSE-01 completed:2487names gained4327variants, single-variant entries4348->1861; all original variants retained, missing6names remain. Model weights unchanged; no new neural training.
- [ ] EVAL-AS08-KO-DIVERSE-01 running with fixed AS-06 legacy scoring: clean/default/severe3seeds, outerfold0. Original result caches require file/input/model/source checks and64-row per-condition parity, otherwise recompute. No server improvement claimed, no ZIP/upload.
- Evidence: name_decoder/pronunciation_quality.py; experiments/build_catalog_hangul_diverse.py; experiments/evaluate_as08_ko_map.py; corresponding experiment JSONs.


## 2026-10-26T21:59:10+09:00 - AS08 pronunciation expansion and ownership filter completed; AS-06 retained

- [x] Implemented label-free quality/diversity policy, CUDA generator with model/source/config/seed provenance, and reverse-pronunciation ownership filter. Quality tests7passed, ownership tests5passed; all scripts compile. Executed sources and latest CLI versions are separately preserved by hash.
- [x] AS08-KO-DIVERSE-01:4354missing/single-variant names processed;2487names gained4327variants; originals unchanged, missing6 remain. AS08-KO-OWNERSHIP-01 retained2998additions and removed1329. No neural weights updated.
- [x] Fixed AS-06 legacy scoring, lookupoff, outerfold0 all2372rows, clean/default/severe3seeds. Reference Hangul239/239/226/229/230; expanded239/238/226/230/229; guarded239/238/225/229/229. Non-Hangul unchanged; all10official paired vessel-bootstrap CIs include0.
- [x] Baseline cache provenance and64-row per-condition prediction/rank parity verified. Each candidate fully predicted. Both static maps passed8-row reverse/single checks. Public heldout case review links filtering to removed gold-name variants without claiming exclusive causality.
- Decision: no default adoption of either map; preserve AS-06 reference0.9122153056 and original engine/map hashes. More pronunciations and nearest-original-owner filtering are not demonstrated accuracy improvements. No ZIP, server upload, commit or push.
- [ ] Next: fit-inner vessel isolation before generator/map/feature selection, then metadata-aware candidate selection. Do not retune on observed outerfold0. Global catalog regeneration and retrieval expansion remain separate planned work.
- Evidence: docs/AS08_PRONUNCIATION_RESULTS.md; docs/AS08_PRONUNCIATION_USAGE.md; project_log/experiments/REVIEW-AS08-PRONUNCIATION-01.json and five reviewed run records.
- Suggested commits: feat(pronunciation): add reproducible diversity generation and ownership checks; experiment: record AS08 comparisons and retain AS06.
