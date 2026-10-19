# 평가 서버 Python·라이브러리 환경

사용자 제공 서버 사양(Python 3.11.17)과 16개 패키지 버전을 맞추는 로컬 환경이다.
`server-requirements.txt`는 Linux x86_64 / CPython 3.11용이며 PyTorch의 공식 CUDA 12.8 wheel을 지정한다.
Windows x86_64는 `windows-requirements.txt`와 `setup_windows.ps1`을 사용한다.
루트의 `requirements.txt`는 대회 제출용으로, 서버 기본 패키지를 재설치하지 않도록 유지한다.

## Windows PC에서 설치

프로젝트 폴더에서 PowerShell을 연다. uv가 없으면 먼저 `winget install --id astral-sh.uv -e`로 설치하고 터미널을 다시 연다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\environment\setup_windows.ps1
.\.venv-server\Scripts\python.exe --version
```

`-ExecutionPolicy Bypass`는 해당 프로세스에만 적용하며 시스템 실행 정책을 영구 변경하지 않는다.
스크립트는 Python 3.11.17과 지정 16개 패키지를 설치·검사하고 기존 환경의 버전이 다르면 중단한다.
CUDA 빌드도 설치하지만 NVIDIA GPU/드라이버가 없으면 GPU 연산 검증은 하지 않는다. Windows x86_64 전용이며 ARM64용이 아니다.
PyTorch wheel만 약 3.27GB이므로 다운로드·압축 해제·캐시를 위한 충분한 디스크 공간이 필요하다.
데이터 파일 경로는 설치에 필요하지 않으며 CSV를 읽거나 학습·ZIP 생성을 하지 않는다.

가상환경을 활성화하지 않아도 위처럼 Python 전체 경로로 실행할 수 있다. 활성화할 경우:

| 셸 | 명령 |
|---|---|
| PowerShell | `.\.venv-server\Scripts\Activate.ps1` |
| CMD | `.venv-server\Scripts\activate.bat` |

패키지 확인 결과는 `reports/local_environment.windows.json`, 전체 설치 버전은
`environment/windows-freeze.local.txt`에 저장한다. 두 파일은 PC마다 달라 Git에서 제외한다.
기존 `server-freeze.txt`는 Linux 설치 기록이므로 Windows에서 사용하지 않는다.
다른 Windows PC에서 동일한 전체 버전이 필요하면 설치 기록을 승인된 내부 공간에서 전달하고
`uv pip install --python .venv-server -r environment/windows-freeze.local.txt`로 설치한다.

Windows 대상 의존성 해결·PowerShell 구문 확인과 실제 Windows 설치/실행 검사는 구분한다.
현재 작업공간은 Linux이므로 사용자 PC 설치 성공 여부는 로컬 검사 결과로 확인해야 한다.

## 이 작업공간에서 사용

```bash
cd /workspace/MLiP_Project
source environment/activate.sh
python --version
python environment/check_environment.py --output reports/server_environment.json
```

비대화형 명령은 `/workspace/MLiP_Project/.venv-server/bin/python`을 직접 사용한다.
기존 `/workspace/vessel_matching/.venv`는 이전 실험 재현용 Python 3.12 환경으로 보존한다.
시스템 Python을 교체하지 않는다. Windows PC의 환경은 이 작업으로 변경되지 않는다.

## 새 Linux 환경에 재설치

uv가 설치된 상태에서 다음을 실행한다. 환경이 이미 있다면 검사 결과와 사용 여부를 먼저 확인한다.

```bash
bash environment/setup_server.sh
```

클라우드의 Python·캐시 경로를 작업공간으로 제한하려면 위 명령에
`UV_PYTHON_INSTALL_DIR=/workspace/python-runtimes UV_CACHE_DIR=/workspace/.cache/uv`를 지정한다.
직접 의존성은 `server-requirements.txt`, 설치된 전체 의존성은 `server-freeze.txt`에 기록한다.
전체 버전을 재현하려면 해당 파일로 `uv pip install --python .venv-server/bin/python -r environment/server-freeze.txt`를 실행한다.

## 기존 모델 호환성 검사 — ZIP 생성 없음

```bash
.venv-server/bin/python -m unittest discover -s tests -v
.venv-server/bin/python environment/check_model_compatibility.py \
  --bundle /workspace/vessel_matching/ngram_submission \
  --data-dir /workspace/vessel_matching/data \
  --reference-python /workspace/vessel_matching/.venv/bin/python \
  --output reports/server_model_compatibility.json
```

공개 train에서 한글 30행·비한글 30행을 골라 일부 호출부호를 미등록 값으로 바꾸고,
기존 환경과 새 환경의 예측을 비교한다. 반복·순서 변경·절반·단독·빈 입력도 확인한다.
행 단위 데이터와 예측은 임시 디렉터리에서만 처리하고 보고서에는 집계만 기록한다.
모델을 다시 학습하거나 ZIP을 만들지 않는다. `verify_submission.py`는 ZIP을 생성하므로
환경 확인 목적만으로 실행하지 않는다.

## 검증 범위

- 실제 설치·import 결과: `reports/server_environment.json`.
- 기존 모델 실행 검사: `reports/server_model_compatibility.json`.
- 설치 과정과 실패·재시도, 기존 ZIP 보존 근거: `project_log/experiments/E010.json`.
- 호스트 OS는 Debian 13으로 서버의 Ubuntu 22.04와 다르다.
- CUDA용 패키지 설치와 실제 GPU 사용 가능 여부는 다르다. GPU가 없으면 CUDA 추론은 미검증이다.
- 일부 공개 입력의 일치가 모든 입력의 예측 일치나 공식 서버 실행 시간까지 보장하지 않는다.
