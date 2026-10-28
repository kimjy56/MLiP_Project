# SPEC-DPR-03 구현·실험 기록

사용자 요청에 따라 DPR-SYMPHON-03을 기존 제출 자산과 분리해 구현하고 실제 실험을 진행한다. 작성 기준은 2026-10-28 KST이며, 아래 실행 상태·수치는 완료된 산출물로 갱신한다. 제출 ZIP은 생성하지 않는다.

## 적용 범위와 해석

- Stage A 합성 10만 건과 Stage A checkpoint 재사용을 제거한다. `--epochs-pretrain`은 0만 허용하며 원본 `docuracy/symphonym-v8` snapshot에서 초기화한다.
- 실제 모듈명 `encoder.conv_block`까지 신규 모듈 그룹에 포함한다. backbone LR 3e-5, 신규 모듈 LR 1.5e-4, AdamW weight decay 1e-4, 8epoch cosine/eta_min 5e-6을 적용한다.
- epoch를 0부터 셀 때 0·4 시작 전에 fit 쿼리만 사용해 hard negative를 마이닝한다. 동일 쿼리의 fit 양성 이름은 모두 negative에서 제외한다.
- 복합 점수는 사용자 수식 그대로 `0.45*R35 + 0.30*R10 + 0.25*R1 + 10*MRR`다. 입력 지표는 0~1 비율이며 MRR의 영향이 큰 수식이다. 전 epoch의 최대값을 유지하고 각 epoch도 별도 보존한다.
- 사용자가 외부 fold 0의 매 epoch 점수로 checkpoint를 선택하도록 명시했으므로 이번에는 그 요청을 적용한다. 따라서 결과는 **외부 fold를 모델 선택에 사용한 개발 점수**다. 독립 최종 검증·누수 없는 모델 선택·서버 점수 예측값으로 표현하지 않는다.
- 이전 코드를 확인하면 best 값은 학습 루프 밖에서 한 번 초기화되며 재마이닝 시 초기화하는 코드는 없다. 이전 선택 기준은 Recall@35 우선, 동점 시 MRR였다. 이번 복합 기준이 Recall@1 최고 epoch와 항상 일치한다는 보장도 없다.
- Stage A가 망각의 원인이었다는 진단은 가설이다. 학습률·초기화·epoch·마이닝·선택 기준을 함께 바꾸므로 이번 차이를 Stage A 단독 효과로 귀속하지 않는다.

## 추론 변경

`phonetic_encoder/model.py`는 primary/secondary/tail을 합쳐 `encode_batch()`를 한 번 호출한다. 단일 행에서는 encoder forward 1회다. catalog 생성도 동일 방식으로 묶는다. 추론 API의 raw/rom 모드와 빈 입력을 유지한다.

`phonetic_encoder/hybrid_matcher.py`의 `DenseRowResult`는 한 행의 전체 catalog 내적 점수·상위 K·정규화 통계를 전달한다. Stage-1과 Stage-2가 이 객체를 공유하며 객체를 다음 테스트 행으로 보관하지 않는다. 실제 호출 계측 검사에서 한 행당 쿼리 인코딩 1회·전체 catalog 내적 1회를 확인한다.

기존 hybrid 코드는 매 행에서 tail 쿼리를 다시 인코딩하고 후보 catalog tail도 재인코딩했다. 그 결과의 `dense_tail_sim`은 현재 Stage-2에서 읽지 않으므로 이 추가 계산과 해당 미사용 출력을 제거했다. 현재 scoring에 쓰는 4개 Dense 특징은 유지한다.

Dense 검색은 한글 행에서 60개다. Dense-only 후보뿐 아니라 lexical union 후보도 같은 행의 원시 내적 점수에서 특징을 받는다. `dense_rank_recip`은 Dense 상위 K 안에서만 역순위를 주며 범위 밖은 0이다. 범위 밖 후보에 임의의 정확한 전체 순위를 붙이지 않는다. 동점은 vessel ID 오름차순으로 정한다.

기존 이름 점수에는 독립적인 `sim_best` 항이 없고 4개 유사도의 가중합이 있다. 이를 보존하면서 다음과 같이 사용자 수식을 연결했다.

```text
sim_best_lexical = max(sim_alpha, sim_phon, sim_skel, sim_tok)
sim_dense_cal = clip((dense_sim - 0.35)/0.60, 0, 1)
sim_best_final = max(sim_best_lexical, 0.92*sim_dense_cal)
name_score += 11.5*(sim_best_final - sim_best_lexical)
```

11.5는 기존 이름 유사도 계수 3.8+2.8+2.2+2.7의 합이다. 이름 점수의 상승분으로 연결하고 메타 조건의 대표 유사도에도 반영한다. affine 변환이 통계적으로 보정된 확률이라는 주장은 하지 않는다.

generic tail 규칙은 선박명·sample ID 하드코딩 없이 구현한다. 기본 점수 상위 10개에서 같은 첫 어절과 tail 유사도 0.8 이상인 경쟁 후보가 존재할 때 tail 0.45 미만 후보를 1.5 감점한다. 영문 희귀 tail은 catalog에서 10척 이하, 서로 다른 첫 어절, tail 완전 일치, 유효 총톤수 상대 오차 0.2% 미만이면 2.5 가점한다. 조건 충족은 실제 개명을 확인했다는 뜻이 아니다. 알려진 Glovis/WREN 사례는 개발 진단으로만 사용한다.

## 평가 조건

수동 `IDEA_B_WORD_VARS`는 기본적으로 빈 dict다. 학습 fold에서 fit한 음역·선종/국적 대응표는 유지하고 이를 사전 0개 모델이라고 부르지 않는다.

| 조건 | 수동 사전 | 기존 생성 catalog map | fit 기반 정답 조회 |
|---|---|---|---|
| true_unseen_clean / default / severe | OFF | OFF | OFF |
| full_system_clean / default / severe | OFF | ON | ON |

full_system의 기존 map은 현재 자산의 계보 한계를 유지한다. 이 조건의 clean은 노이즈를 주입하지 않았다는 뜻이며 새 사전으로 생성했다는 뜻이 아니다. EXAONE 자산은 읽지 않는다.

각 조건에서 네 구성을 비교한다.

1. `reference02`: 기존 DPR02 모델·임베딩과 보존한 이전 scoring 소스. 쿼리 인코딩은 현재 가속 구현을 사용하므로 시간이 과거 서버 실행 시간과 같지 않다.
2. `retrieval03`: DPR03, Dense K=60, 직접 이름 점수·tail 규칙 OFF.
3. `direct03`: 같은 DPR03/K=60에서 직접 이름 점수 ON.
4. `complete03`: 같은 조건에서 tail 규칙까지 ON.

고정 K Dense Recall@1/5/10/35/60과 가변 lexical+Dense 합집합 포함률을 구분한다. 평가·epoch 선택 모두 catalog 이름을 합치지 않고 선박 ID 완전 일치로 측정한다. MRR는 전체 catalog 순위이고 과거 코드의 truncated 지표는 별도의 `mrr@35`로 남긴다. 이전 이름 중복 제거 평가와 정의 차이를 기록한다.

공통 평가기에는 공개 train 보류 예측만 `--subset`으로 전달하고 선박 단위 paired bootstrap CI를 보존한다. severe의 실제 노이즈 적용률은 실행 결과로 기록하며 80.27%를 강제로 맞추지 않는다. 노이즈에서도 언어 집단은 원래 공개 train 이름 기준으로 고정한다. 비공개 테스트는 읽거나 결과를 기록하지 않는다.

## 실행·환경

현재 Windows `.venv-server`는 Python 3.11.15이며 지정 3.11.17과 다르다. torch는 2.7.1+cu128, GPU는 RTX 5080이다. 지정 패키지 5개의 불일치·누락을 `reports/dpr03_environment.json`에 기록했다. 환경을 임의 변경하지 않았고 Ubuntu/L4 서버 재현으로 표시하지 않는다.

pytest는 설치되어 있지 않아 첫 pytest 명령은 실패했다. 같은 함수형 테스트와 unittest 테스트를 표준 라이브러리 `FunctionTestCase`로 실행할 수 있는 runner를 제공했다. 기능 검사는 실제 base 로딩, 저장·복원, 행 독립성, one-forward/one-dot, fold 거부, generic scoring 조건을 확인한다.

PowerShell, 프로젝트 루트:

```powershell
.\.venv-server\Scripts\python.exe -B -X utf8 tests/run_dpr_tests.py
.\.venv-server\Scripts\python.exe -B -X utf8 experiments/train_bi_encoder.py --experiment-id DPR-SYMPHON-03 --epochs-pretrain 0 --epochs-finetune 8 --batch-size 128 --fold 0
.\.venv-server\Scripts\python.exe -B -X utf8 experiments/export_catalog_embeddings.py --checkpoint artifacts/DPR-SYMPHON-03/fold_0_encoder.pt --output-npy artifacts/DPR-SYMPHON-03/catalog_embeds_fp16.npy
.\.venv-server\Scripts\python.exe -B -X utf8 experiments/evaluate_bi_encoder_oof.py --experiment-id DPR-SYMPHON-03 --checkpoint artifacts/DPR-SYMPHON-03/fold_0_encoder.pt --catalog-embeds artifacts/DPR-SYMPHON-03/catalog_embeds_fp16.npy --fold 0 --output-json reports/dpr_symphon_03_evaluation.json
.\.venv-server\Scripts\python.exe -B -X utf8 experiments/evaluate_dpr_robustness.py --checkpoint artifacts/DPR-SYMPHON-03/fold_0_encoder.pt --catalog-embeds artifacts/DPR-SYMPHON-03/catalog_embeds_fp16.npy --output-json reports/matching_improvements/DPR-HYBRID-03/dpr_hybrid_robustness_benchmark.json
.\.venv-server\Scripts\python.exe -B -X utf8 experiments/benchmark_dpr03_latency.py
.\.venv-server\Scripts\python.exe -B -X utf8 tools/verify_dpr03.py
.\.venv-server\Scripts\python.exe -B -X utf8 experiments/analyze_dpr03_results.py
```

이는 최초 실행 명령이다. 실행된 ID·완료 출력을 재사용하거나 덮어쓰지 않는다. export/evaluation은 각각 별도 ID를 쓴다. 실제 성공한 standalone 재시도에는 `--evaluation-id EVAL-DPR-SYMPHON-03-RECHECK-02`를 사용했다. 하이브리드 실행은 `--presets severe`를 지정해 clean/severe를 측정했으며 default noise는 이번에 실행하지 않았다. Windows path·환경·실행 코드는 해당 기록의 실제 식별자를 참조한다.

## 결과와 상태

8epoch 학습·전수 임베딩 추출·단독 검색·하이브리드 비교·행 독립성·자산 검증을 완료했다. 복합 기준 최고는 epoch 5이고 마지막 epoch 8은 한글 Recall@1 21.76%로 하락했다. 선택한 epoch 5 가중치를 보존했으며 이후 재학습·하이퍼파라미터 튜닝은 하지 않았다.

| 항목 | DPR03 실측 | 요청 목표·판정 |
|---|---:|---|
| 한글 Dense Recall@1 | 97/239 = 40.5858% | 68% 이상: 미달 |
| 한글 Dense Recall@35 | 213/239 = 89.1213% | 96% 이상: 미달 |
| 한글 Dense Recall@60 | 214/239 = 89.5397% | 97.5% 이상: 미달 |
| 한글 전체 catalog MRR | 0.509896 | 과거 truncated MRR와 구분 |
| 단일 행 encoder forward | 3회 → 1회 | 달성 |
| 단일 행 CPU 인코딩 평균 | 8.443ms → 3.383ms, 2.496배 | 고정 3배 보장 불가 |
| Dense 검색 CPU 평균 | 30.864ms(@35) → 12.920ms(@60), 2.389배 | 로컬 32행/6threads, 모델 로딩 제외 |
| 서버 전체 시간 2~3분 | 미측정 | 새 제출·서버 측정 없음 |

정밀도 검사 첫 실행은 실제 모델의 cuDNN TF32로 배치/단독 오차가 약 2.8e-5여서 실패했다. IEEE FP32 연산으로 변경한 수치 preflight에서 GPU 1.19e-7, CPU 1.49e-7을 확인했다. 재평가 첫 재시도는 numpy bool의 JSON 직렬화 오류로 중단되어 별도 실패 기록으로 보존했고, native bool로 고친 재시도 02가 1e-6 기준을 통과했다. 허용오차를 완화하지 않았다. 모델 가중치는 바꾸지 않았다.

최종 기능 검사는 **18개 통과**다. 앞선 17개·7개 실행과 중복 합산하지 않는다. 정밀도·속도 근거는 `reports/dpr03_numerical_preflight.json`, `reports/dpr03_latency.json`, `EVAL-DPR-SYMPHON-03-RECHECK-02` 기록이다.

### 하이브리드 실측

공통 fold 0 2,372행, 한글239/비한글2,133행이다. severe 실제 노이즈율은 80.2698%다.

| 조건 | 기존 DPR02 가중 정확도 | DPR03 전체 구성 | 차이(%p) | 가중 차이 95% CI |
|---|---:|---:|---:|---|
| 사전 OFF clean | 99.664919% | 99.664919% | 0.000000 | [0, 0] |
| 사전 ON clean | 99.957806% | 99.664919% | -0.292887 | [-0.933, 0.000]%p |
| 사전 OFF severe | 98.744064% | 96.108080% | -2.635983 | [-4.709, -0.827]%p |
| 사전 ON severe | 98.744064% | 97.865403% | -0.878661 | [-3.105, +1.459]%p |

사전 OFF clean은 네 구성의 예측이 같았다. 로컬 사전 OFF 점수 약92%가 리더보드와 일치한다는 명세의 전제는 재현되지 않았다. 절대 정확도 목표95.5%는 넘지만 기존 기준선도 이미99.6649%이므로 개선이라고 해석하지 않는다. 사전 OFF severe에서는 paired CI가0보다 낮아 해당 개발 조건의 악화 근거가 있다. 다른 CI의0 포함은 통계적 동등성의 입증이 아니다.

같은 DPR03/K=60 조건에서 이름 직결만 켜면 severe 사전 OFF 한글 정답이 **230→225**, tail 규칙까지 켜면 **226**이다. 직결은 고침0/망침5, tail은 고침1/망침0이었다. 공통 평가기의 직결 paired 가중 차이는 **-0.01464, 95% CI[-0.03105,-0.00281]**이며, tail 차이는 **+0.00293, CI[0,0.00925]**다. 두 비교의 후보 생성 설정은 동일하고 변경 플래그가 점수 계산 이후에만 작용한다. 개별 후보 집합 해시를 별도 저장하지는 않았다.

사전 ON severe에서는 한글 정답이 기존235, 검색03 234, 직결231, 전체232였다. 직결 공식이 모든 잡음·모델 분포에서 도움이 된다는 주장은 이 결과로 지지되지 않는다. 고정 스케일 변환이나 알려진 사례를 보고 다시 튜닝하지 않았다.

### 행 독립성·보존·비용

- 공개 보류300행(한글150/비한글150): reverse300, subset150, 단독3행, repeat300에서 **예측 ID 불일치0**, 빈 입력 통과. 별도 실제 모델 임베딩 검사도1e-6 통과.
- 선택 checkpoint의 tensor가 보관한 epoch5와 모두 같고, global composite 최대값을 확인했다. 기존 모델·임베딩·ZIP 등35개 자산의 SHA256도 원본과 일치한다.
- 가중치는 FP32 5,523,013bytes, catalog 임베딩은 FP16 20,120,448bytes다. 학습 전체324.775초는 데이터/base 로딩·초기 평가·마이닝·8epoch·epoch별 평가·저장을 포함한다. 임베딩 export 전체12.110초, encoding 구간11.136초다.
- 4조건×4구성 benchmark 전체949.736초는 모든 모델/index 로딩·반복 평가·공통 평가기·저장을 포함한다. 한 번의 최종 추론 시간으로 부르지 않는다. 사전 OFF clean의 실제 검색·점수 구간은 가속 인코더를 공통 사용한 reference02 48.171초, DPR03 전체60.057초였다. 인코더 부분 가속이 새 scoring을 포함한 전체 matching 가속을 보장하지 않는다.
- 공개 보류의 조건별 오답23건을 분석했다. 동일 행이 조건에 따라 중복될 수 있다. 확인한 범주는 후보 선택 오류22건/검색 누락1건이며 원인 가설은 미확인으로 남겼다. 비공개 입력·예측은 읽거나 저장하지 않았다.

### 결정과 산출물

**DPR03 checkpoint와 고정 Dense 직결 규칙은 성능 개선 구성으로 채택하지 않는다.** AS-09의 기존 제출 파일·자산·서버 기준선을 유지한다. Packing/한 행 점수 재사용은 별도의 속도 근거가 있는 구현이며, 이미 제출된 AS-09를 재포장한 것은 아니다. 추가 학습·가중치 조정·서버 제출은 수행하지 않는다.

주요 산출물:

- `artifacts/DPR-SYMPHON-03/fold_0_encoder.pt`, `.manifest.json`, `training_history.json`, `catalog_embeds_fp16.npy`, `.json`
- `reports/dpr_symphon_03_evaluation.json`, `reports/dpr03_latency.json`, `reports/dpr03_final_tests.txt`
- `reports/matching_improvements/DPR-HYBRID-03/dpr_hybrid_robustness_benchmark.json`
- `artifacts/DPR-HYBRID-03/oof_*.csv`, `official_*.txt` — 공개 train 보류의 내부 자료
- `artifacts/VERIFY-DPR03-01/verification.json`
- `artifacts/ANALYZE-DPR03-01/summary.json`, `public_heldout_error_cases.json`, `official_component_*.txt`

행 독립성 자체 검사는 구현·실제 보류 행에서 수행한다. 제출 ZIP을 생성하지 않았으므로 `eval_kit/check_submission.py`의 새 제출물 PASS를 주장하지 않는다. AS-09의 0.9201100425/497초는 이전 제출 기준선으로 유지한다.

근거: `project_log/experiments/DPR-SYMPHON-03.json`, `artifacts/DPR-SYMPHON-03/training_history.json`, `reports/dpr_symphon_03_training.txt`, `reports/dpr03_unit_tests.txt`, `reports/dpr03_integration_tests.txt`.
