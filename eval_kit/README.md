# eval_kit — 팀 공통 평가 묶음 (DACON 236771)

문서 버전 1.1 · 2026-10-19 · [TEAM_RULES v4](../TEAM_RULES.md) 반영.

같은 fold와 평가 산식으로 **팀 공통 순위 비교**를 하기 위한 안내다. 별도 개발·보조 실험은 가능하지만 기준 점수와 구분한다. 이번 수정은 README에 한정하며 평가기·검사기 버전이 변경된 것은 아니다.

현재 이 작업공간에는 첨부 README만 반영했다. 아래 평가 코드·fold·E00 OOF는 별도로 배포받아야 하며, 존재·동작·점수를 이번 문서 수정에서 검증한 것은 아니다.

## 1. 배포받을 파일

| 파일 | 용도 |
|---|---|
| `evaluate.py` | 공통 평가기. 버그가 의심되면 보고하고 임의 수정하지 않는다 |
| `selftest_evaluate.py` | 평가기 자체 검사. 최초 도입 및 평가기·실행 환경 변경 시 수행 |
| `check_submission.py` | 이미 생성한 제출 ZIP 검사. 실제 검사 범위는 코드와 버전으로 확인 |
| `folds_group5_seed42.csv` | 팀 공통 비교 기준인 선박 그룹 5-fold |
| `oof_E00_baseline.csv` | 기준 fold에서 주최 baseline을 학습한 것으로 보고된 기준 OOF |
| `aux/folds_group5_seed{0,1,2}.csv` | 최종 후보 재확인용 보조 그룹 분할 |
| `aux/folds_row5_seed42.csv` | 기존 선박과 사전 조회 효과를 확인하는 보조 행 분할 |

fold 파일은 `sample_id, target_vessel_id, fold` 형식이며 fold는 0~4다. 파일을 그대로 읽고 해시를 기록한다. 같은 seed로 다른 알고리즘을 실행해도 같은 분할이 되지 않으므로 재생성하지 않는다. 숫자 fold 라벨의 일치율만으로 두 분할의 유사성을 단정하지 않는다.

## 2. 최초 도입·변경 시 확인

다음 명령은 eval_kit 디렉터리에서 실행한다. `PATH/train.csv`는 실제 파일 경로로 바꾼다. 옵션·출력 문구는 배포본의 `--help`와 대조한다.

```bash
python selftest_evaluate.py --train "PATH/train.csv"
python evaluate.py --oof oof_E00_baseline.csv --train "PATH/train.csv"
```

- v1.0 README가 보고한 E00 기대값은 가중 0.47750, 한글 26.196%, 비한글 98.042%다. 배포본 코드·데이터·fold·OOF 해시와 함께 재현한다.
- 자체 검사 통과, 입력 해시, 실행 환경과 종료 코드를 기록한다. 버전·환경 변경이 없다면 매 실험마다 전체 자체 검사를 반복할 필요는 없다.
- 해시가 다르면 파일 경로·배포 버전·인코딩·수정 여부를 먼저 확인한다. 원본 손상·변경이 확인되면 공식 파일을 다시 받는다. 엑셀로 다시 저장하면 바이트·자료형이 달라질 수 있다.
- 평가기 원문은 numpy·pandas를 의존성으로 안내한다. 확인하지 않은 Python/pandas 버전 조합 전체의 호환성을 보장하지 않는다. 이 프로젝트의 기준 환경은 §7을 따른다.
- Windows 한글 출력 문제가 있으면 `python -X utf8 evaluate.py ...`를 사용할 수 있다.

## 3. OOF 만들기

OOF는 각 행을 그 행이 포함되지 않은 학습 집합으로 학습한 모델로 예측한 결과다. **기준 그룹 fold에서는 같은 선박의 모든 행도 학습에서 제외된다.** 보조 행 분할 OOF는 같은 선박의 다른 행이 학습에 있을 수 있으므로 구분한다.

아래는 구현 틀이다. `my_fit`과 `my_predict`는 모델별로 구현해야 한다. `my_predict`는 입력 순서에 대응하는 ID 목록을 반환한다. 이 예제를 작성했다고 실제 OOF를 생성하거나 성능을 검증한 것은 아니다.

```python
import pandas as pd

train = pd.read_csv(
    "train.csv", encoding="utf-8-sig",
    dtype={"sample_id": str, "target_vessel_id": str},
)
folds = pd.read_csv(
    "folds_group5_seed42.csv", encoding="utf-8-sig",
    dtype={"sample_id": str, "target_vessel_id": str, "fold": int},
)
catalog = pd.read_csv(
    "vessel_catalog.csv", encoding="utf-8-sig", dtype={"vessel_id": str},
)

if len(train) != 12222 or len(folds) != len(train):
    raise ValueError("기준 train/fold 행 수를 확인하세요")
for frame in (train, folds):
    if frame[["sample_id", "target_vessel_id"]].isna().any().any():
        raise ValueError("ID에 결측값이 있습니다")
    if not frame["sample_id"].is_unique:
        raise ValueError("sample_id가 중복됩니다")
if set(train["sample_id"]) != set(folds["sample_id"]):
    raise ValueError("train과 fold의 sample_id 집합이 다릅니다")
if folds["fold"].isna().any() or set(folds["fold"]) != set(range(5)):
    raise ValueError("fold는 0~4여야 합니다")
if not folds.groupby("target_vessel_id")["fold"].nunique().eq(1).all():
    raise ValueError("같은 선박이 여러 fold에 있습니다")
if catalog["vessel_id"].isna().any() or not catalog["vessel_id"].is_unique:
    raise ValueError("카탈로그 ID가 결측 또는 중복입니다")
catalog_ids = set(catalog["vessel_id"])
if not set(train["target_vessel_id"]).issubset(catalog_ids):
    raise ValueError("카탈로그에 없는 정답 ID가 있습니다")

# inner join의 조용한 행 누락이나 중복 병합을 허용하지 않는다.
joined = train.merge(
    folds.rename(columns={"target_vessel_id": "fold_target_vessel_id"}),
    on="sample_id", how="left", validate="one_to_one", indicator=True,
)
if len(joined) != len(train) or not joined["_merge"].eq("both").all():
    raise ValueError("fold 병합에 실패했습니다")
if joined["fold"].isna().any():
    raise ValueError("fold가 누락됐습니다")
if not joined["target_vessel_id"].eq(joined["fold_target_vessel_id"]).all():
    raise ValueError("train과 fold의 정답 ID가 다릅니다")
joined = joined.drop(columns=["_merge", "fold_target_vessel_id"])

parts = []
for k in range(5):
    fit = joined.loc[joined["fold"] != k].drop(columns="fold").copy()
    val = joined.loc[joined["fold"] == k].copy()
    # 학습 사전·특징·튜닝은 fit만 사용한다. 필요하면 fit 내부 검증을 둔다.
    model = my_fit(fit, catalog)
    # 정답과 fold 메타데이터를 추론 함수에 전달하지 않는다.
    queries = val.drop(columns=["target_vessel_id", "fold"]).copy()
    pred = list(my_predict(model, queries, catalog))
    if len(pred) != len(queries):
        raise ValueError("예측 행 수가 입력과 다릅니다")
    if any(not isinstance(v, str) or v not in catalog_ids for v in pred):
        raise ValueError("카탈로그에 없는 예측 ID 또는 잘못된 자료형입니다")
    parts.append(pd.DataFrame({
        "sample_id": queries["sample_id"].tolist(),
        "fold": [k] * len(queries), "pred_vessel_id": pred,
    }))

oof = pd.concat(parts, ignore_index=True)
if len(oof) != len(train) or not oof["sample_id"].is_unique:
    raise ValueError("OOF 행이 누락 또는 중복됐습니다")
if set(oof["sample_id"]) != set(train["sample_id"]):
    raise ValueError("OOF의 sample_id 집합이 다릅니다")
oof.to_csv("oof_O-01.csv", index=False, encoding="utf-8")
```

| 컬럼 | 사용 | 의미 |
|---|---|---|
| `sample_id` | 필수 | 기준 train 전체 행, 누락·중복 없음 |
| `pred_vessel_id` | 필수 | 예측 ID. 평가 시 앞뒤 공백도 임의 보정하지 않음 |
| `fold` | 팀 저장 기준 | 예측 모델의 외부 검증 fold. 평가기에서 대조 |
| `gold_rank` | 선택 | 확정한 후보 순위를 정답과 사후 대조한 진단값 |

`gold_rank`가 필요하면 **정답을 보지 않고 확정한 후보 목록·점수·동점 처리 결과**를 먼저 보관한 뒤, 별도 진단 단계에서 정답을 결합한다. 진단 때문에 후보를 추가하거나 예측을 고치지 않는다. v1.0 문서의 1=1위, 0=후보 없음, 맞힌 사전 조회=1 규약과 조회 오류 처리·최종 순위 정의는 실제 평가기와 대조한다. 정답 순위는 추론 특징이 아니다.

이 검사들은 병합·그룹 분리를 확인하는 방어 장치다. 공식 파일 해시 검사나 모델 내부 누수 검토를 대신하지 않는다.

## 4. 평가와 결과 해석

아래 옵션은 원문에 기재된 사용법이다. 실제 배포된 평가기가 지원하는지 확인하고 사용한다. 예시 명령에는 실측하지 않은 시간·ZIP 크기를 넣지 않았다.

```bash
# 기본 평가
python evaluate.py --oof oof_O-02.csv --train "PATH/train.csv"

# 같은 행의 짝지은 차이: O-02 minus O-01
python evaluate.py --oof oof_O-02.csv --train "PATH/train.csv" --compare oof_O-01.csv

# 팀원별 집계 기록
python evaluate.py --oof oof_O-02.csv --train "PATH/train.csv" --compare oof_O-01.csv --id O-02 --note "숫자 표기 정규화" --leaderboard leaderboard_O.csv

# 구간별 정확도
python evaluate.py --oof oof_O-02.csv --train "PATH/train.csv" --slices

# 보조 행 분할: 기준 그룹 점수와 구분
python evaluate.py --oof oof_O-02_row.csv --train "PATH/train.csv" --folds aux/folds_row5_seed42.csv

# 보조 부분집합 진단
python evaluate.py --oof oof_O-02_numvar.csv --train "PATH/train.csv" --subset --compare oof_O-02.csv
```

- **가중 정확도:** 0.7 × 한글 + 0.3 × 비한글. 부분집합에 한 언어가 없을 때 처리 방식은 평가기를 확인하며, 임의로 0점이나 완전한 공식 점수로 표시하지 않는다.
- **불확실성:** 원문은 언어별 선박 단위 부트스트랩 B=2000, seed=0을 설명한다. 실제 구현·버전·그룹 처리와 재현 결과를 확인한다.
- **짝지은 차이의 CI:** 하한 > 0이면 해당 검증에서 개선 근거, 상한 < 0이면 악화 근거, 0 포함이면 **차이가 명확하지 않음**이다. 0 포함이 동등성의 증명은 아니다. 평가기가 `동등`이라고 출력해도 원본 결과는 보존하고 보고서에서 의미를 설명한다.
- 한글 1~2%p 같은 고정 문턱이나 고정 표준오차로 판정하지 않는다. 실제 OOF·선박 구조·표본 수를 기준으로 해석한다. 차이가 불명확할 때 비용·강건성으로 실무 선택할 수 있다.
- 여러 구성을 선택한 최고 OOF 점수에는 선택 편향이 있다. CI만으로 편향이 사라지지 않으므로 보조 검증으로 재확인한다.
- **갈린 행:** 기존 오답을 맞힌 행과 기존 정답을 틀린 행을 각각 보고한다. 반복 행 수와 독립 선박 사례 수를 구분한다.
- `--subset`은 같은 표본의 보조 진단이며 전체 OOF 순위에 합치지 않는다. 변형 규칙·표본 선택·원본 입력과의 차이를 기록한다. 변형된 검증 행이 학습에 들어가지 않도록 한다.
- 시간은 학습/예측, 로딩·인덱스·저장 포함 여부를 구분해 기록한다. ZIP은 요청해 생성한 경우에만 실측 크기를 적는다. 미측정은 미측정으로 남기며 도구의 기본 0을 실제 비용으로 해석하지 않는다.
- 입력 오류의 종료 코드·메시지는 배포본에 따른다(원문은 코드 2로 안내). 오류를 무시한 점수를 팀 순위에 올리지 않는다.

## 5. 요청하여 만든 제출 ZIP의 검사

**ZIP은 사용자가 명시적으로 생성·갱신을 요청할 때만 만든다.** 일반 실험과 후보 공유에는 OOF·코드·설정·집계 결과로 충분하다. README의 검사 안내는 ZIP 생성이나 대회 제출을 실행하라는 지시가 아니다.

요청하여 생성한 기존 ZIP은 다음처럼 검사한다.

```bash
python check_submission.py --zip submit_AS-01.zip --train "PATH/train.csv"
```

원문이 설명한 검사 범위는 ZIP 구조, 기본 패키지 중복 설치, 코드 내 의심 문자열, 공개 train 300행 실행, 입력 보존, 출력 형식, 순서·부분·단독 입력 일치 및 시간 추정이다. 이번에는 검사기 코드가 첨부되지 않았으므로 그 기능이 실제 구현돼 있다고 확인한 것은 아니다.

- `실패 0`은 **해당 버전 검사기의 검사 항목 통과**를 뜻한다. 공식 서버 성공, 모든 입력의 행 독립성, 전체 규정 준수를 보장하지 않는다. `올려도 된다`라는 출력도 자동 제출 승인으로 해석하지 않는다.
- 코드에서 네트워크 관련 문자열을 찾는 정적 검사와 실제 인터넷 차단 실행은 다르다. 차단·입력 읽기전용·`data/` 및 팀 호환성용 `open/` 처리·반복 실행을 실제로 검사했는지 기록한다.
- 공개 train에서 뽑은 입력은 전체 학습 모델이 이미 본 데이터다. 정확도 비교에 쓰지 않는다. 호출부호 조회가 대부분을 처리할 수 있으므로 일반 후보 검색 경로의 비용은 별도로 검사하고 별칭 유지 여부도 기록한다.
- 300행으로 계산한 3,000/30,000행 시간은 **외삽 추정**이다. 인덱스 초기화·행별 후보 수·캐시·메모리·병렬화 차이를 포함한 실측과 구분한다. 이를 비공개 테스트 크기나 공식 서버 시간으로 단정하지 않는다.
- ZIP 안의 코드를 실행하므로 검토한 코드와 자산만 검사한다. 호출하는 모듈과 의존성도 검토 범위에 포함한다.
- ZIP 생성 요청 시에는 필요한 `script.py`, `requirements.txt`, `model/` 내부 파일을 **명시적 목록**으로 선택한다. 전체 작업 폴더 재귀 압축 예제는 삭제했다. 원본 데이터·로그·OOF·보고서·캐시를 포함하지 않고 기존 ZIP을 무단 덮어쓰지 않는다.
- 압축 도구 자체를 일괄 금지하지 않는다. 실제 ZIP 항목 경로가 `/` 구분자를 사용하며 상위 폴더 없이 요구 구조를 따르는지 검사한다.

## 6. 기록·공유·누수 방지

- OOF는 행 단위 예측 결과이므로 팀 내부 승인된 공간에서만 관리한다. 공개 저장소·AI 채팅에 올리지 않는다.
- leaderboard는 통상 **실험별 집계 결과**다. 행 단위 파일로 분류하지 않는다. 공유 범위는 팀 정책을 따르며 원본 예시·식별정보·비공개 평가 정보가 메모에 들어가지 않도록 한다.
- 동기화 충돌을 줄이려면 `leaderboard_<접두>.csv`로 관리하고 리더가 합친다. AI가 공유 폴더 업로드나 메시지 전송을 자동 수행하는 허가는 아니다.
- 검증 정답을 학습·사전·정규화·특징 선택·개별 예측 보정에 넣지 않는다. **예측 확정 후** 평가·오류 분석·gold_rank 진단에는 사용한다.
- 튜닝·early stopping은 학습 fold 안의 내부 검증으로 수행한다. 완성된 구성의 OOF 비교는 개발 선택이며 독립 최종 성능으로 표시하지 않는다.
- 팀 공통 순위는 기준 fold `G5-s42`의 전체 OOF로 비교한다. 보조 fold·행 분할·변형·기존 두 개발 분할 결과는 별도로 기록한다. 한 fold만으로 모델을 고르지 않는다.
- 원본 데이터·외부 API·AI 업로드는 TEAM_RULES §6·§7을 따른다. 과거 점수·실험 ID·완료 기록은 덮어쓰지 않는다.

## 7. Python·라이브러리 환경

목표는 **Python 3.11.17**과 서버 지정 패키지 버전이다. Python 3.11.x를 선택했다고 3.11.17이 되는 것은 아니다. 버전을 직접 확인한다.

이 프로젝트의 Linux 설치 목록·전체 버전 고정 파일·검사 방법은 [environment/README.md](../environment/README.md)에 있다. 16개 직접 패키지는 [server-requirements.txt](../environment/server-requirements.txt)에서 확인한다. 이 목록의 PyTorch wheel은 Linux x86_64/CPython 3.11용이므로 Windows·macOS에 그대로 설치하지 않는다.

현재 작업공간에서 이미 만든 환경 사용(Linux):

```bash
source /workspace/MLiP_Project/environment/activate.sh
python --version
```

다른 머신에서 uv를 이용해 정확한 Python 버전의 가상환경을 새로 만들 때:

```text
uv python install 3.11.17 --no-bin
uv venv --python 3.11.17 .venv311
```

활성화 명령은 셸별로 다르다.

| 환경 | 활성화 명령 |
|---|---|
| Linux/macOS bash·zsh | `source .venv311/bin/activate` |
| Windows PowerShell | `.\.venv311\Scripts\Activate.ps1` |
| Windows CMD | `.venv311\Scripts\activate.bat` |

평가기의 최소 의존성만 설치하는 예시(전체 모델 환경 검증이 아님):

```text
uv pip install --python .venv311 numpy==1.26.4 pandas==2.0.3
python -c "import sys,pandas,numpy; assert sys.version_info[:3] == (3,11,17); print(sys.version.split()[0], pandas.__version__, numpy.__version__)"
```

모델 실행에는 해당 플랫폼에 맞는 추가 패키지·빌드가 필요하다. 서버 목표 PyTorch는 2.7.1+cu128, CUDA 빌드는 12.8이다. macOS·GPU 없는 머신을 L4/CUDA 검증 환경으로 표시하지 않는다. 제출용 requirements.txt와 로컬 설치용 파일은 분리한다.

이 작업공간의 E010은 Python 3.11.17·지정 16개 패키지 버전/import, 기존 모델의 공개 60행 호환성을 확인했다. **호스트는 Debian 13이며 GPU 연산은 미검증**이다. 이는 아직 받지 않은 eval_kit의 자체 시험/E00 재현을 통과했다는 의미가 아니다.

## 8. 문서 변경 이력과 확인 범위

- **문서 1.1 (2026-10-19):** 팀 룰 v4 반영. OOF 입력에서 정답 분리, ID·병합·그룹 검사, CI 해석 수정, 요청 시 ZIP 생성, 재귀 압축 예제 삭제, 추정 시간/검사 통과 범위 명확화, 집계·행 단위 공유 구분, 정확한 Python 및 셸별 환경 안내로 정리. 평가기·검사기는 수정하지 않았다.
- **원문 1.0 (2026-10-19) 보고:** 자체 시험 33개 통과 환경은 Python 3.11.9/pandas 2.0.3/numpy 1.26.4 및 Python 3.14/pandas 3.0.3/numpy 2.3.5. E00 0.47750, CI [0.44598, 0.51008]. 일부 정상·비정상 ZIP 검사 결과도 보고했다. 이는 원문 작성자의 보고를 보존한 것이며 이번 작업에서 재현한 결과가 아니다. **3.11.9는 서버의 3.11.17과 같은 버전이 아니다.**

현재 README 예제는 구현 안내다. 평가 코드·fold·OOF를 배포받으면 버전·해시·CLI와 실제 기능을 대조하고 검사 결과를 별도 기록한다.
