# AS-09 Bi-Encoder 전달

`DPR-SYMPHON-02` 음운 Bi-Encoder의 가중치·설정·로더·학습 코드 묶음이다. 기반 모델은 `docuracy/symphonym-v8`이며 출력은 128차원이다.

## 구성

- `full_encoder.safetensors`: 기존 AS-09 PT에서 변환한 FP16 가중치 61개. 토크나이저 사전과 설정도 메타데이터에 포함한다.
- `full_encoder.config.json`: 토크나이저 사전·출력 차원·dtype 설정의 JSON 사본.
- `phonetic_encoder/`: 실제 AS-09 패키지의 모델·토크나이저·데이터셋·손실·검색 유틸리티.
- `experiments/train_bi_encoder.py`: 보존된 DPR-02 계열 학습 스냅샷.
- `experiments/build_synthetic_phonetic.py`: Bi-Encoder Stage A 합성 데이터 생성 참고 코드.
- `tools/export_encoder_safetensors.py`: PT 변환기 및 safetensors 로더.
- `source_manifest.json`, `verification.json`: 파일 해시·출처와 검증 기록.

## 사용

ZIP을 푼 폴더를 작업 디렉터리로 사용한다. 기존 프로젝트의 `.venv-server\Scripts\python.exe`로 실행한다(Linux는 `.venv-server/bin/python`). 필요한 패키지는 `requirements.txt`에 기재했다.

```python
from tools.export_encoder_safetensors import load_encoder_safetensors

model = load_encoder_safetensors("full_encoder.safetensors", device="cpu").float()
vectors = model.encode_queries(["테스트 선박 1"])
assert vectors.shape == (1, 128)
```

기존 `CharPhoneticBiEncoder.load_checkpoint()`는 PT 전용이므로 safetensors에는 위 로더를 사용한다.

학습 CLI 확인(Windows PowerShell):

```powershell
& '<프로젝트>\.venv-server\Scripts\python.exe' experiments/train_bi_encoder.py --help
```

학습 코드는 공식 train·카탈로그·배포 fold CSV의 실제 경로를 각각 `--train-csv`, `--catalog-csv`, `--folds-csv`로 받는다. 원본 데이터와 fold는 동봉하지 않았다. 로컬 Hugging Face 캐시에 `docuracy/symphonym-v8` 원본이 필요하다. `--fold -1`은 전체 train 학습이고 `--fold 0`은 fold 0 보류 경로다. 새 출력 폴더와 실험 ID를 지정해야 기존 자산을 보존할 수 있다.

## 재현 범위

- 학습 스냅샷은 과거 학습 기록의 소스 해시와 다르므로 당시 실행 파일의 바이트 동일 복구본은 아니다. 초기 `pretrain_synthetic_encoder.pt` 유무·합성 데이터·학습 횟수에 따라 재학습 결과가 달라진다.
- 이 코드가 fold 보류를 사용해 체크포인트를 선택한 과거 결과는 개발 선택 결과이며 독립 최종 테스트가 아니다.
- 가중치는 앞선 변환 검증에서 텐서61개·설정 정확 일치 및 합성 한글/영문 각각3개 임베딩 차이0을 확인했다. 이번 작업은 파일 선별·해시·격리 로드 검사이며 새 학습이나 성능 평가는 아니다.
- 앞선 변환 환경은 Python3.11.15 / torch2.7.1+cu128 / safetensors0.8.0이다. 목표 Python3.11.17 / safetensors0.6.2와의 차이를 유지한다. 의존성 설치나 변경은 수행하지 않았다.

ZIP은 코드 전달용이며 대회 제출물 구조가 아니다.
