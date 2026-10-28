# LFM 변환표 출처·학습 코드와 AS-09 인코더 전달

이 묶음은 코드와 직렬화 변환 결과다. 새 학습·정확도 평가·제출 ZIP 생성은 수행하지 않았다.

## 질문에 대한 답

1. `catalog_full_ko_map.json`은 기록상 **로컬 공개 모델 `LiquidAI/LFM2.5-350M` (Instruct)**에 LoRA 추가 학습한 `LM-LFM25-FULL-R24-01/model`을 사용했다. 생성 코드 `experiments/build_catalog_full_hangul.py`는 `local_files_only=True`로 모델을 읽는다. 이 생성 경로에는 ChatGPT·번역 API 호출이 없다. 공식 영문 카탈로그명에서 첫 토큰 상위 2개 분기 후 greedy 생성하고, 한글·숫자·공백으로 후처리한다. 중복 제거와 빈 결과 때문에 이름마다 반드시 2개가 남는 것은 아니다. 현재 변환표는 78,586개 이름을 담고 있다.
2. **train 실제 한글 이름은 생성 모델의 추가 학습에 사용했다.** 학습의 정답 쌍은 `target_vessel_id`로 찾은 영문 카탈로그명 → train의 관측 이름이다. 생성 스크립트 자체는 train CSV를 읽어 정답 이름을 맵에 덧붙이지 않는다. 다만 기존 출력 JSON을 읽고 없는 키만 생성하므로, 임의 병합 이력이 전혀 없었다는 것을 코드만으로 입증할 수는 없다. 과거 감사는 fold 0을 제외한 **9,850행**, 원본 고유 쌍 4,685개와 증강 후 7,265쌍(한글 819쌍)을 확인했다. 이름에 FULL이 있어도 LFM을 12,222행 전부로 학습했다는 뜻이 아니다. 이 맵을 모든 fold에 공통 사용한 5-fold OOF를 누수 없는 검증으로 표시하면 안 된다.
3. `full_encoder.safetensors`는 **AS-09의 `DPR-SYMPHON-02` 음운 Bi-Encoder**를 변환한 것이다. LFM 생성 모델과 별개이며, 토크나이저 사전·차원·dtype 설정은 safetensors 메타데이터와 `full_encoder.config.json`에 보존했다. LFM 자체의 병합 가중치는 원래부터 `artifacts/LM-LFM25-FULL-R24-01/model/model.safetensors`다(이 코드 묶음에는 포함하지 않음).

## 출처의 한계

- 기존 AS-06 기록이 LFM 생성 모델 경로를 명시하고 있으며 현재 맵 해시와 일치한다. 당시 생성 시점의 모델/시드 매니페스트는 없어, 이번에 전체 생성을 재현해 출처를 독립적으로 증명한 것은 아니다.
- `phonetic_encoder/`는 실제 AS-09 패키지에 들어 있는 소스의 사본이다. 현재 저장소의 DPR-03 코드와 섞지 않았다.
- `experiments/train_bi_encoder.py`는 `artifacts/IMPL-DPR-SYMPHON-03/before/`에 남은 DPR-02 계열 학습 스냅샷이다. 학습 기록의 소스 해시와 다르므로 **당시 실행 파일의 바이트 동일 복구본은 아니다**. Stage A 합성 데이터/초기 체크포인트 유무와 학습 횟수도 최종 가중치 재현에 영향을 준다.
- LFM 학습 파일은 현재 소스다. 원본 학습 기록과 일치하는 파일과 달라진 파일을 `source_manifest.json`에 구분했다. 현재 코드에는 내부 검증 선택 기능이 추가됐다. 아래 예시는 과거 마지막-step 선택에 대응하도록 `--checkpoint-selection last`를 지정한다.
- 원본 데이터·fold·행 단위 혼동 후보 CSV·OOF는 동봉하지 않았다. 원본 LFM 실행에는 `INNER-CONFUSIONS-01`의 학습 fold 내부에서 만든 음성 후보도 사용했다. 이 자산을 생략하면 동일 학습 조건이 아니다. `experiments/mine_inner_confusions.py`는 당시 생성 방식을 보여 주는 참고 소스이며 실행 시 고정 ID를 사용한다.

## 파일

| 파일 | 용도 |
|---|---|
| `full_encoder.safetensors` | AS-09 음운 인코더 가중치와 설정 메타데이터 |
| `full_encoder.config.json` | 토크나이저 사전·출력 차원·FP16 설정 |
| `tools/export_encoder_safetensors.py` | 원본 PT → safetensors 변환, 엄격 로더 |
| `experiments/build_catalog_full_hangul.py` | LFM 영문 카탈로그 → 한글 변환표 생성 |
| `experiments/train_noisy_channel.py` | LFM LoRA 추가 학습 |
| `name_decoder/` | LFM 학습·전처리·증강·손실 구현 |
| `experiments/train_bi_encoder.py` | 보존된 DPR-02 계열 학습 코드 |
| `phonetic_encoder/` | AS-09 실제 모델·토크나이저·손실·데이터셋 소스 |
| `experiments/build_synthetic_phonetic.py` | Bi-Encoder Stage A 합성 데이터 생성 참고 코드 |
| `source_manifest.json` | 원본 위치·해시·과거 기록과의 일치 여부 |
| `verification.json` | 변환 후 가중치·설정·출력 검증 결과 |

## 사용 예시 (Windows PowerShell)

코드 묶음을 푼 폴더를 작업 디렉터리로 사용한다. Python은 프로젝트의 `.venv-server\Scripts\python.exe`를 사용한다. 아래 경로는 **현재 사용자의 데이터 경로를 가정한 것이 아니며**, 실제 파일 위치를 넣어야 한다. 환경 설치 기준은 `environment/windows-requirements.txt`이고 Linux는 `environment/server-requirements.txt`다. 이번 변환 실행 환경은 Python 3.11.15 / torch 2.7.1+cu128 / safetensors 0.8.0으로 목표 Python 3.11.17 / safetensors 0.6.2와 차이가 있다.

Python에서 safetensors 로드(묶음 루트에서 실행):

```python
from tools.export_encoder_safetensors import load_encoder_safetensors

model = load_encoder_safetensors("full_encoder.safetensors", device="cpu").float()
vectors = model.encode_queries(["테스트 선박 1"])
assert vectors.shape == (1, 128)
```

기존 `CharPhoneticBiEncoder.load_checkpoint()`는 PT용이므로 safetensors 경로를 그대로 넘기면 안 된다. 동봉한 로더를 사용한다. 기존 제출 추론 코드의 자동 형식 전환까지 구현한 묶음은 아니다.

LFM 학습 설정 예시(실행하지 않음):

```powershell
$python = '<프로젝트>\.venv-server\Scripts\python.exe'
& $python experiments/train_noisy_channel.py `
  --experiment-id LFM-HANDOFF-RETRAIN-01 `
  --pretrained '<로컬_LFM2.5-350M_폴더>' `
  --data-dir '<공식_train과_catalog가_있는_폴더>' `
  --folds-file '<배포된_folds_group5_seed42.csv>' --heldout-fold 0 `
  --checkpoint-selection last --device cuda `
  --steps 900 --warmup-steps 45 --rank 24 --alpha 48 `
  --lora-targets q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3 `
  --rank-beta 0.25 `
  --confusions-csv '<학습_fold_내부_confusions.csv>' `
  --confusions-record '<대응_혼동후보_실험_JSON>'
```

변환표 생성 예시(실행하지 않음):

```powershell
& $python experiments/build_catalog_full_hangul.py `
  --model-dir '<추가학습된_LFM의_model_폴더>' `
  --catalog-csv '<실제_vessel_catalog.csv>' `
  --output '<새_catalog_ko_map.json>' `
  --batch-size 128 --num-candidates 2 --max-new-tokens 20
```

기존 출력은 이어서 읽으므로 처음 생성 시 새 출력 경로를 사용한다.

Bi-Encoder 전체 train 학습 경로의 예시(실행하지 않음, 최종 가중치의 완전 재현 보장 아님):

```powershell
& $python experiments/train_bi_encoder.py `
  --experiment-id DPR-HANDOFF-RETRAIN-01 `
  --train-csv '<실제_train.csv>' --catalog-csv '<실제_vessel_catalog.csv>' `
  --folds-csv '<배포된_folds_group5_seed42.csv>' `
  --fold -1 --epochs-pretrain 0 --epochs-finetune 12 --batch-size 128 `
  --output-dir artifacts/DPR-HANDOFF-RETRAIN-01
```

이 스냅샷은 Hugging Face 로컬 캐시의 `docuracy/symphonym-v8`를 찾는다. 출력 폴더에 `pretrain_synthetic_encoder.pt`가 있으면 그것을 초기값으로 사용하고, 없으면 Symphonym에서 시작한다. Stage A를 사용하려면 합성 입력과 초기값 계보를 따로 기록해야 한다. `--fold -1`은 전체 학습 경로이며 보류 성능을 측정하지 않는다. fold별 코드의 과거 early-stopping은 외부 보류를 사용하므로 독립 최종 테스트로 해석하지 않는다.

## 검증 범위

정확한 수치와 SHA256은 `verification.json` 참조. 변환은 가중치 값·dtype·키·토크나이저 설정을 유지한다. 공개 성능 점수·후보 포함률·서버 실행 시간은 이번 작업에서 측정하지 않았다. 코드 묶음 ZIP은 전달 편의를 위한 것이며 DACON 제출물 구조가 아니다.
