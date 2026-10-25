# HyperCLOVA X LoRA 구현 인수인계 — AS-HCX-05

사용자가 제안한 정방향 선박명 SFT의 LoRA 모드를 `experiments/train_hyperclovax_sft.py`에 추가했다. **구현만 했으며 import·테스트·학습·검증은 실행하지 않았다.** `AS-HCX-05`는 planned이고, 새 가중치나 성능 결과는 없다. 검증은 사용자가 지정하는 다른 에이전트가 맡는다.

## 추가된 기능

- `--train-mode lora`: 공통 `name_decoder/lora.py`를 이용해 전체 베이스(embedding/LM head 포함)를 동결하고, 모든 층의 `q_proj`, `v_proj`에 어댑터를 적용한다. PEFT 설치는 필요 없다.
- `--lora-rank 16 --lora-alpha 32`: A는 Kaiming 초기화, B는 0 초기화. 어댑터만 학습하고 저장 후 원래 Linear 가중치에 병합한다.
- `--sampling-mode balanced --hangul-mass 0.5`: 각 microbatch에 정확히 한글/비한글을 절반씩 복원추출한다. batch 4이면 2쌍+2쌍이며 배치 크기는 짝수여야 한다. seed가 고정된 별도 generator로 배치 안 순서도 섞는다.
- `--skip-evaluation`: Roman baseline, 생성 probe, 저장 후 재로딩 검사, 행 독립성 검사, pilot 평가를 모두 건너뛴다. 학습·저장만 완료하면 `trained_unvalidated`, 모든 평가 지표는 null로 기록한다.
- 원본 데이터/분할/베이스/코드 해시, 파라미터 수, 추출 인덱스 해시·언어별 실제 추출 횟수, 학습 이력, 어댑터와 병합 모델 해시를 기록한다. 기존 완료 기록/출력 디렉터리 덮어쓰기는 거절하고 실패·중단을 기록한다.

기존 기본값은 `top-layers`, 150스텝, 한글 70% 확률의 weighted sampler이며 기존 평가 경로도 유지한다. 아래 제안 설정은 명령으로 명시한다. 평가를 생략하지 않으면 기존 pilot 평가가 실행되므로 검증을 분리할 때 `--skip-evaluation`을 사용한다.

## 사용자 제안 설정의 실행 명령 — 미실행

저장소 루트의 Windows PowerShell:

```powershell
.\.venv-server\Scripts\python.exe -X utf8 experiments\train_hyperclovax_sft.py --experiment-id AS-HCX-05 --train-mode lora --lora-rank 16 --lora-alpha 32 --steps 600 --batch-size 4 --accumulation 2 --sampling-mode balanced --hangul-mass 0.5 --learning-rate 2e-4 --lr-scheduler cosine --warmup-steps 30 --threads 4 --catalog-scorer ratio --skip-evaluation
```

Linux에서는 같은 인자에 `.venv-server/bin/python experiments/train_hyperclovax_sft.py`를 사용한다. 경로가 다르면 `--pretrained-model`, `--data-dir`, `--folds`로 실제 로컬 경로를 지정한다. Fold 0 제외 학습이 기본이며 새 fold를 임의 생성하지 않는다. AS-HCX-04를 이어 학습하는 설정이 아니라 원본 로컬 pretrained 모델에서 시작한다.

검증 에이전트가 기능 확인용 짧은 실행을 할 때는 다른 ID, `--steps 2 --warmup-steps 0`을 사용한다. 이것을 성능 비교 결과로 표시하지 않는다. 실제 학습 이후에만 `artifacts/AS-HCX-05/adapter.safetensors`, `adapter_config.json`, `model/`, `training_history.json`이 생긴다. 최종 추론 인터페이스는 병합된 `model/`을 기존 `HyperCLOVANameDecoder`로 읽는 방식이다. 이 호환성 역시 아직 실행 확인하지 않았다.

## 수치 해석과 검증 항목

로컬 모델의 q/v projection 차원으로 계산한 rank 16 어댑터는 **1,966,080개**다. 약 750만 개 추정은 이 구성에 맞지 않는다. 실행 시 실제 개수와 예상 개수를 대조한다. 600스텝 × batch 4 × accumulation 2의 balanced 설정은 총 4,800쌍, 한글 2,400쌍 추출이다. 실제 완료한 스텝과 로그를 함께 확인한다.

기존 구현도 한글 확률 70%로 추출했으므로, 단순 원본 한글 비율로 과거 학습 노출을 추정하면 안 된다. 학습 시간 2배 단축, 40~50분 완주, 정확도 상승은 아직 근거가 없다. LoRA는 전체 층 역전파가 필요하므로 파라미터 수 감소만으로 시간을 예측하지 않는다.

AS-HCX-04와 비교하면 LoRA 외에도 스텝·학습률·샘플러·warmup이 바뀐다. **제안 구성 전체의 비교이며 LoRA 단독 효과로 해석하지 않는다.** 같은 평가 행·ratio·추론 설정을 고정하고, 이름 정확도와 최종 선박 ID 정확도를 분리한다.

검증 담당자가 우선 확인할 사항:

1. LoRA B=0의 초기 출력, 모든 베이스 동결, A/B gradient, q/v 48개 적용과 파라미터 수, 병합 전후 logits 허용 오차, 반복 병합 방지, 저장·로드 후 기존 디코더 호환성.
2. balanced의 매 배치 2+2 보장, 총 노출 수, seed/추출 해시 재현, 기존 weighted 기본 경로의 회귀 여부.
3. 기존 SFT의 prompt/target 토큰 경계와 마스크, EOS 처리, 학습 fold 분리, CPU 4스레드 이하 유지.
4. `--skip-evaluation`에서 evaluate/generate/reload가 호출되지 않고, 평가 지표가 null이며 결과 파일을 날조하지 않는지 확인.
5. 새 ID 강제·실패 기록·해시와 저장 용량을 확인. 기존 생성 probe는 병합 후 저장/로드 검사이므로 **병합 전후 수치 일치 검사를 대체하지 않는다**.
6. 이후 같은 128행 이름 pilot 및 필요 시 전체 fold ID 매칭을 별도 기록. 5-fold에는 fold별 별도 모델 필요. 단일 파일럿 개선을 공식 서버 개선으로 표현하지 않는다.

역방향 `catalog_name -> observed_name` 점수 실험은 별도 `AS-HCX-NC-01`이며 [NOISY_CHANNEL.md](NOISY_CHANNEL.md)에 인수인계했다. 두 학습 목적과 체크포인트를 혼용하지 않는다. 이 작업에서 제출 ZIP·커밋·푸시는 실행하지 않았다.

커밋 제안: `feat(hcx): add plain torch LoRA and configurable language sampling`.
