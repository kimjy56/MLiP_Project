# 역방향 선박명 채널: 구현과 검증 인수인계

이 문서는 **구현만 완료하고 학습·검증은 다른 에이전트가 수행**하도록 한 사용자 요청에 따른 인수인계다. 새 코드의 import, 단위 테스트, 학습, 병합 일치 검사, 성능 비교는 아직 실행하지 않았다. 학습 계획 `AS-HCX-NC-01`의 상태는 `planned`이며, 새 체크포인트나 실측 성능은 없다.

## 목적과 범위

공식 카탈로그 이름을 조건으로 실제 관측 이름의 확률을 학습한다.

`catalog_name -> observed_name`

후보 ID마다 `log P(observed_name | catalog_name)`를 계산해 기존 매칭 점수에 더할 수 있도록 한다. 초기 버전의 조건은 **이름만**이다. 톤수·국적·선종은 이 모델 입력에 포함하지 않고 기존 매칭기가 처리한다. 동명 선박 ID는 같은 언어모델 점수를 받으므로 이름 점수만으로 구별되지 않는다.

역방향 채점은 정방향 `HyperCLOVANameDecoder` 및 메인 매칭기와 분리되어 있다. 기존 정방향 학습기에 추가한 LoRA 선택 모드는 별도 [HYPERCLOVA_LORA.md](HYPERCLOVA_LORA.md)에 기록했다. 비교용 고정 후보 CSV와 원래 점수를 입력으로 받는 별도 실험 경로다. 새로운 외부 데이터, PEFT 설치, 제출 ZIP 생성은 없다.

| 파일 | 역할 |
|---|---|
| `name_decoder/lora.py` | 전체 베이스 동결, 모든 층 q/v LoRA 주입, 어댑터 상태 추출, 한 번만 병합 |
| `name_decoder/noisy_channel.py` | 역방향 학습쌍·프롬프트·마스크, 관측 토큰 로그확률, 모든 후보 ID 보존 |
| `experiments/noisy_channel_common.py` | CPU 제한, KST 기록, 입력·코드 해시, 새 실험 ID 및 실패 기록 |
| `experiments/train_noisy_channel.py` | Fold 제외 학습, 학습 이력·어댑터·병합 체크포인트 저장 |
| `experiments/score_noisy_channel_candidates.py` | 공개 보류 행의 고정 후보 점수 계산·결합·CSV 저장 |

## 학습 설정과 해석

기본값은 CPU float32, 최대 4스레드, seed 42, Fold 0 제외, 300 optimizer steps, batch 4 × accumulation 2, rank 16, alpha 32, AdamW lr 2e-4, warmup 15 + cosine, weight decay 0.01, gradient clip 1.0이다. 모두 CLI로 노출된 값은 변경 가능하며, 학습률은 검증할 초기 설정이지 최적값이 아니다.

샘플러는 중복 제거한 학습쌍을 한글/비한글로 나누고, **한글 총 확률 질량 0.7**로 복원추출한다. 배치마다 정확히 70%를 강제하지 않는다. 기본 총 2,400회 추출 중 한글의 기대 횟수는 1,680회이며, 실제 횟수와 샘플 인덱스 스트림 SHA256은 실행 후 기록한다. `--hangul-mass 0.5`로 50:50을 선택할 수 있다. 이는 그룹별 추출 확률이지 토큰 손실 비중을 보장하는 설정이 아니다.

로컬 원본 모델의 24개 층에서 q projection은 `(2048, 1024)`, v projection은 `(1024, 1024)`다. rank 16 q/v LoRA 예상 학습 파라미터 수는 `24 × 16 × ((2048+1024)+(1024+1024)) = 1,966,080`개다. 앞선 제안의 약 750만 개는 이 적용 범위와 맞지 않는다. 코드가 실행 시 실제 모듈·학습 파라미터 수를 기록하도록 했다. 이 산술값은 학습 실행 결과가 아니다.

학습 파라미터 감소가 CPU 학습 시간 감소를 보장하지 않는다. 전체 층으로 기울기를 전파하는 비용도 존재한다. 2배 가속이나 40~50분 완주를 예측 결과처럼 취급하지 않는다. 병합 후 어댑터 연산 경로는 제거되지만, 부동소수점 출력 일치·속도는 별도 확인한다.

AS-HCX-04와 비교하면 학습 방향, LoRA 적용 범위, 학습률, 관측 이름 정규화 등이 달라진다. 따라서 **LoRA 단독 효과 실험이 아니다**. 개선 귀속이 필요하면 역방향 목적·샘플러·데이터·스텝 등을 고정한 별도 대조군을 작성한다.

## 토큰·데이터 계약

- 공식 `read_frames`의 ID 일치 및 선박 그룹 분리 검사를 재사용한다. 새 학습쌍은 반환된 `fit_rows`에서만 만든다. 기존 배포 fold가 없으면 중단하며 새 fold를 생성하지 않는다.
- 카탈로그 이름은 기존 `canonical_name`(NFC, 대문자, 공백 정리)을 사용한다. 관측 이름은 NFC와 앞뒤 공백 제거만 하며 대소문자·내부 띄어쓰기·숫자·문장부호를 유지한다.
- 정규화한 `(catalog_name, observed_name)` 쌍을 중복 제거하고 두 필드로 정렬한다. 같은 관측 이름이 서로 다른 정답 이름에 연결되면 둘 다 남긴다.
- tokenizer의 chat template에 카탈로그 이름을 JSON 필드로 넣는다. 프롬프트와 관측 문자열은 **각각** 토큰화한 후 연결하고 EOS를 하나 붙인다. 이 경계 규칙은 학습·채점에서 동일하다.
- 프롬프트와 오른쪽 패딩의 label은 `-100`. 관측 토큰과 EOS만 학습 손실·확률 합산에 포함한다. causal shift를 적용해 직전 위치 logits로 다음 토큰을 채점한다.
- 이름 안의 특수 토큰, 빈 관측 이름, 길이 초과는 오류다. `--max-length`(기본 256)를 넘는 문자열을 조용히 자르지 않는다.
- `log_probability`는 EOS 포함 합계, `mean_log_probability`는 이를 관측 토큰 수로 나눈 값이다. **같은 관측 행에서는 후보마다 목표 토큰 수가 같으므로 sum/mean 선택만으로 언어모델 단독 순위가 바뀌지 않는다.** 기존 점수와 결합할 때 행별 배율은 달라진다.
- 후보별 batch size 1, eval 모드, 자유 생성 없음. 같은 행의 동명 후보만 계산을 공유하며 다른 행을 참조하지 않는다.
- `NoisyChannelScorer.load()`는 역방향 manifest, 전처리·프롬프트 버전, EOS, 저장 파일 해시를 요구한다. 정방향 AS-HCX-04 체크포인트를 대신 넣을 수 없다.

## 실행 방법 — 검증 에이전트가 실행

현재 저장소 루트에서 실행한다. 아래는 **아직 실행하지 않은 명령**이다. 기존 `.venv-server`를 사용하며 추가 설치 명령은 없다.

Windows PowerShell, 기본 학습 계획:

```powershell
.\.venv-server\Scripts\python.exe -X utf8 experiments\train_noisy_channel.py --experiment-id AS-HCX-NC-01 --heldout-fold 0 --threads 4
```

Linux에서는 같은 인자에 `.venv-server/bin/python experiments/train_noisy_channel.py`를 사용한다. 기본 경로는 스크립트가 있는 저장소의 `data/`, `eval_kit/folds_group5_seed42.csv`, `artifacts/pretrained/hyperclovax-seed-0.5b`다. 다른 배치 위치에서는 `--data-dir`, `--folds-file`, `--pretrained`에 실제 경로를 지정한다.

검증 담당자가 먼저 짧은 실행 확인을 선택할 경우 별도 ID와 warmup을 사용한다. 이것은 성능 비교 실험이 아니다.

```powershell
.\.venv-server\Scripts\python.exe -X utf8 experiments\train_noisy_channel.py --experiment-id AS-HCX-NC-SMOKE-01 --steps 2 --warmup-steps 0 --batch-size 1 --accumulation 1 --threads 4
```

완료 시 `artifacts/<ID>/`에 다음 파일을 저장하도록 구현했다.

- `model/`: 병합 모델, tokenizer, `noisy_channel_manifest.json`.
- `adapter.safetensors`, `adapter_config.json`: 원본 베이스 해시와 어댑터 설정. 어댑터 단독 resume CLI는 제공하지 않으며 추론은 병합 모델로 수행한다.
- `training_history.json`: optimizer step별 손실, 실제 사용 LR, clipping 전 gradient norm.
- `training_vessel_ids.json`, `run_record.json`: 학습 선박 목록과 실행 근거.

실험 기록은 `project_log/experiments/<ID>.json`에 저장하며 학습 성공 상태도 `trained_unvalidated`다. 예정 상태의 동일 ID만 시작할 수 있고, 출력 폴더가 이미 있거나 기존 기록이 실행된 상태이면 덮어쓰지 않는다. 실패·중단 상태와 부분 산출물을 보존하고 재실행은 새 ID를 사용한다. 강제 프로세스 종료처럼 예외 처리조차 실행할 수 없는 종료는 `running`에 남을 수 있어 인수인계 때 정리해야 한다.

## 고정 후보 채점 계약

후보 CSV는 별도 Baseline 실행에서 내보내야 한다. 이 구현은 후보 추출기 자체를 추가하지 않는다. 필수 열:

| 열 | 의미 |
|---|---|
| `sample_id` | 공식 공개 train의 지정 보류 fold 행 |
| `candidate_vessel_id` | 기존 검색기가 뽑은 카탈로그 ID |
| `baseline_score` | 클수록 좋은 기존 매칭 점수 |

각 행의 후보 ID 쌍은 중복이 없어야 한다. 모든 후보가 보존되며, 정답 ID를 보고 후보를 추가하거나 변경하지 않는다. 후보가 수천 개이면 확률 계산 비용도 크게 증가한다. 임의로 Top-30으로 자르면 기존 진단의 gold rank 31 사례가 탈락할 수 있으므로, 후보 축소를 택할 때에는 별도 후보 포함률 비교가 필요하다.

다음 명령의 `baseline_candidates_fold0.csv`는 **입력 계약을 설명하기 위한 아직 생성하지 않은 파일명**이다. 기존 진단 JSON은 실제 후보를 만든 Baseline의 기록일 때만 사용할 수 있다. 다른 실행으로 후보를 만들었다면 그 기록 경로로 교체한다.

```powershell
.\.venv-server\Scripts\python.exe -X utf8 experiments\score_noisy_channel_candidates.py --experiment-id AS-HCX-NC-SCORE-01 --model-path artifacts\AS-HCX-NC-01\model --candidates-csv reports\diagnostics\baseline_candidates_fold0.csv --baseline-record reports\diagnostics\fold0_baseline_diagnostics.json --heldout-fold 0 --channel-weight 0 --normalization mean --threads 4
```

기본값은 보류 fold 전 행을 요구한다. 일부 공개 보류 행의 기능 검사에는 `--allow-subset`을 명시해야 하며 결과에 subset 범위를 남긴다. 비공개 평가 입력·개별 예측을 저장하는 용도로 이 CLI를 사용하지 않는다.

결합식은 `baseline_score + channel_weight * channel_logp`이며, 이는 확률 보정이 완료된 공식 posterior를 뜻하지 않는다. 기본 가중치 0은 언어모델 특징을 저장하되 **공급된 점수의 순위**를 유지하기 위한 설정이다. 동점 규칙은 ID 오름차순이므로 원래 Baseline의 동점 처리와도 대조해야 한다. 가중치는 학습 fold 내부 검증으로 정하고 외부 보류 평가 전에 고정한다. Fold 0 오답 3건만 보고 가중치를 맞추지 않는다.

출력은 `reports/noisy_channel/<ID>/candidate_scores.csv`, `predictions.csv`, `run_record.json`이다. 후보 점수에는 원래 점수, 확률 합계·평균, 목표 토큰 수, 원래 순위, 결합 점수·순위가 포함된다. 예측 CSV는 `sample_id,baseline_vessel_id,pred_vessel_id,fold`다. 정답·정확도는 채점 코드가 읽어 붙이지 않는다. 입력 후보와 Baseline 기록의 해시를 남기지만 **Baseline이 정말 학습 fold만으로 fit했는지는 별도 검토 사항**이다.

## 검증 완료 및 실측 결과 (2026-10-24 KST)

### 1. 학습 및 단위 무결성 통과
- **학습 완주 (`AS-HCX-NC-01`)**: 300 optimizer steps, batch 4 × accum 2, rank 16 / alpha 32, lr 2e-4, CPU 4스레드 1,439.6초 완주. 손실 5.2102 ➜ **0.2342** 급락 수렴. 병합 모델 `artifacts/AS-HCX-NC-01/model/` (2.28 GB) 저장 완료.
- **단위 검증 (`tests/test_noisy_channel_verification.py`)**:
  - `observed_normalization`: 대소문자, 내부 공백, 숫자, 문장부호 100% 보존 및 NFC 확인.
  - `encode_channel_example` & `collate_channel_examples`: 프롬프트와 패딩 레이블 `-100` 완전 마스킹 확인.
  - `NoisyChannelScorer.load()`: manifest 해시 100% 일치 로딩 통과.
  - `score_candidates()`: 후보 순서 치환 불변성 및 행 격리성 통과.

### 2. 5-Fold 전체 오답 10행 전수 진단 및 순위 역전(+Fix) 실측
- **`마스그라치` (Fold 4 한글 2건 전수)**:
  - Baseline: `Maasgracht` (15.46점, 1위) vs `Marsgracht` (15.28점, 2위)
  - Noisy Channel 우도: `Marsgracht` ($\log P = -7.32$) vs `Maasgracht` ($\log P = -11.54$, $\Delta = +4.22$, 68배 높은 우도)
  - 결합 재정렬 결과 ($\lambda = 0.05$ sum 또는 $\lambda = 0.20$ mean): **정답 `Marsgracht`가 14.91점으로 1위 역전 탈환 (+Fix!)**
  - **결과**: 규칙 엔진이 유일하게 틀렸던 5-Fold 전체 한글 오답 2건이 100% 정답으로 전환됨.
- **영문 오답 2건 추가 구제 확인**:
  - `NOVA BREEZE` (Fold 2): 정답 `Trammo Cornell` ($\log P = -31.31$) vs 오답 `Green One` ($\log P = -32.46$, $\Delta = +1.15$)
  - `CENTRAL PARK` (Fold 4): 정답 `Caifu Chao` ($\log P = -25.70$) vs 오답 `Toubkal Park` ($\log P = -29.76$, $\Delta = +4.06$)

### 3. Heldout Fold 0 파일럿 128행 그리드 결합 평가 (`AS-HCX-NC-SCORE-PILOT-01`)
- **Baseline 후보군**: `reports/diagnostics/baseline_candidates_fold0_top35.csv` (83,020쌍, predict_rows 100% 비트 일치, Top-35 정답 포함률 100.000%).
- **그리드 실측 결과 (`scratch/evaluate_fusion_grid.py`)**:
  - $\lambda \in [0.000, 0.500]$ (`mean_log_probability`): 한글 100% (64/64), 비한글 100% (64/64), **Breaks = 0 (정답 파괴 0건)**.
  - $\lambda \in [0.000, 0.150]$ (`log_probability`): 한글 100% (64/64), 비한글 100% (64/64), **Breaks = 0 (정답 파괴 0건)**.
- **안전 권장 가중치**:
  - `mean_log_probability` 기준 $\lambda \in [0.10, 0.25]$ (기존 정답 0 파괴 유지 + `마스그라치` 역전 성공).

### 4. Heldout Fold 0 전수(2,372행) 노이즈 스트레스 및 결합 구제 실측 검증 (`ROBUSTNESS-FULL-FOLD0-PRESETS-01`)
- **실행 일시**: 2026-10-25 KST (`experiments/evaluate_noise_full_fold0.py`)
- **검증 대상**: Heldout Fold 0 전체 2,372행 (한글 239, 비한글 2,133), Clean Baseline 99.958% (한글 100.000%, 비한글 99.859%).
- **1. 현실적 종합 노이즈 (`default`, 오타 21.9%, 톤수변동 36.5%, 국적결측 17.0%, 노이즈적용 63.1%)**:
  - Baseline 점수: 99.958% ➜ **99.651% (-0.307%p 하락)**, 오답 5행 발생 (Clean 오답 3행 + 노이즈 유발 오답 2행).
  - Noisy Channel 결합 ($\lambda = 0.20$): **99.944% (+0.293%p 즉시 회복, 손실의 95.4%를 즉시 복구, Retention 99.99%)**.
  - **구제 사례**: `TRAIN_08942` (한글 `씨 울프`): Baseline이 `Gulf No. 16`(logP=-6.05)으로 오답 낸 것을 정답 `Sea Wolf`(logP=-1.78, $\Delta=+4.27$, 71배 우도 우세)로 역전 탈환 (+Rescue).
  - 정답 파괴(Breaks): 128행 샘플링 검사 결과 **0행 (0.0% 완전 무결)**.
- **2. 가혹 노이즈 (`severe`, 오타 29.5%, 톤수변동 41.3%, 결측 30.2%, 노이즈적용 80.3%)**:
  - Baseline 점수: 99.958% ➜ **96.680% (-3.278%p 급락)**, 오답 18행 대량 발생.
  - Noisy Channel 결합 ($\lambda = 0.20$): **97.308% (+0.628%p 대폭 상승, Retention 97.35%)**.
  - **오답 5행 대량 구제 (+Rescue)**:
    1. `TRAIN_04405` (영문 철자 전위 `M/V HEMISPEHRE`): Baseline 오답 `Zhe Hai You 5` ➜ 정답 `Hemisphere` ($\Delta=+2.09$)로 탈환.
    2. `TRAIN_07823` (한글 `모닝 레나`): Baseline 오답 자매선 `Morning Cara` ➜ 정답 `Morning Lena` ($\Delta=+4.61$, 100배 우도 우세)로 탈환.
    3. `TRAIN_07839` (영문 철자 탈락 `SEASTAR VULAN`): Baseline 오답 자매선 `Seastar Valiant` ➜ 정답 `Seastar Vulcan` ($\Delta=+2.17$)으로 탈환.
    4. `TRAIN_08942` (한글 `씨 울프`): Baseline 오답 `Gulf No. 16` ➜ 정답 `Sea Wolf` ($\Delta=+4.27$)로 탈환.
    5. `TRAIN_11587` (영문 철자 탈락 `ALWNE`): Baseline 오답 `Alani` ➜ 정답 `Alwine` ($\Delta=+1.23$)으로 탈환.
  - 정답 파괴(Breaks): **0행 (0.0%)**.
- **3. 결론**:
  - Baseline의 유사도/n-gram 매칭은 철자 전위/탈락 시 자매선이나 다른 선박으로 오답을 내기 쉬우나, **Noisy Channel 역방향 모델 $P(\text{noisy} \mid \text{catalog})$**은 오타 발생 확률을 직접 모델링하므로 오답을 드라마틱하게 역전 구제함.
  - 깨끗한 데이터에서 기존 정답 훼손(Breaks) 0.0%를 유지하면서, 노이즈 환경에서 방어율과 회복력을 실측으로 확증함.

기록 위치: `reports/robustness/robustness_full_fold0_presets.json`, `project_log/experiments/AS-HCX-NC-01.json`, `project_log/DECISIONS.md`, `docs/PROJECT_STATUS.md`.
