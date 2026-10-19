# 프로젝트 기록 대장

사용자 요청에 따라 수업 평가에 필요한 근거를 작업 중 계속 보존한다. 기준 문서는 업로드된 24쪽 프로젝트 설명서이며 상세 출처는 EVALUATION_CRITERIA.md에 있다.

## 기록 위치

| 파일 | 내용 |
|---|---|
| EVALUATION_CRITERIA.md | 수업 평가와 요구 증거, 대회 일정·변경 예정 사항 |
| experiments/E001.json | 기존 LightGBM 검증 실험의 설정·수치·한계 |
| experiments/E002.json | 전체 학습과 제출 ZIP 검사 기록 |
| experiments/E003.json | LogisticRegression Pipeline 실행 검사; 성능 비교 아님 |
| experiments/E004.json | 기존 분할에서 baseline/structure/all/search 네 구성 비교 |
| experiments/E005.json | 두 번째 그룹 분할에서 동일 네 구성 재검증 |
| experiments/E006.json | 채택 구성의 전체 학습·별도 ZIP·오프라인/행 독립성·처리 비용 검사 |
| experiments/E007.json | 첫 개발 분할: 네 검색 경로 16개 조합별 학습·pure/production 평가 |
| experiments/E008.json | 두 번째 개발 분할: 동일 16개 조합 평가 및 기존 SPT 재현 확인 |
| experiments/E009.json | n-gram 포함 채택 조합의 전체 학습·새 ZIP·실행 비용 검사 |
| experiments/INDEX.csv | 완료 기록의 요약 목록 |
| baseline_errors.json | 공개 train의 기존 보류 오답 16행, 오류 단계 재확인 |
| ERROR_ANALYSIS.md | 대표 오답과 개선 가설 |
| DECISIONS.md | 방법 선택 이유, 확인된 것과 계획의 구분 |
| ../docs/VESSEL_NAME_IMPROVEMENT_PLAN.md | 선박명 처리 개선 초기 계획; 일부 실행, 미실행 항목 구분 |
| ../docs/NAME_IMPROVEMENT_RESULTS.md | 개선 비교표·오류 변화·비용·한계 |
| ../docs/RETRIEVAL_COMBINATIONS.md | 최신 16개 조합, N 추가 8쌍, 오류 변화 및 최종 ZIP 검사 |
| REPORT_AND_PRESENTATION.md | 보고서 구성과 7분 발표 구성안 |
| templates/experiment.json | 향후 실험 기록 양식 |
| templates/submission.json | 향후 대회 제출 결과 기록 양식 |
| provenance.json | 설명서·기존 실험 증거의 해시 및 기록 출처 |

## 현재까지 확보한 근거

- 다중 PC 준비: `.gitignore`를 데이터·OOF·fold·가상환경·가중치·ZIP·행 단위 오류 파일에 확장했다. 현재 추적 파일과 접근 가능한 로컬 이력에 제거 대상은 없어 실제 파일/인덱스 삭제는 없었다. Windows 전용 requirements/PowerShell 설치 스크립트와 OS 호환 검사 분기를 추가했다. Windows 대상 의존성 51개 해결, PowerShell 7.5.4 구문 검사, 제외 규칙 16사례·집계 유지 7사례 및 Linux 환경 재검사 통과. 실제 Windows 설치는 미검증이다. `docs/MULTI_PC_SETUP.md`, `environment/README.md`, `windows_git_setup.json` 참조.

- 공통 평가 README 문서 1.1: 첨부 v1.0의 정답 포함 추론 예제·불안전한 병합·CI/환경 표현을 수정하고 재귀 ZIP 생성 예제를 삭제했다. `eval_kit/README.md` 참조. 메모리 내 합성 데이터로 OOF 예제 정상 흐름과 잘못된 입력 7종의 거부를 확인했다. 실제 평가 코드·fold는 미수령이며 점수 재현 검사가 아니다. 원본/수정본 해시와 확인 범위는 `eval_kit_readme_revision.json`에 보존한다.

- 팀 규칙 v4: 사용자 요청으로 중복·과도한 운영 의무·병렬 제한·검증/CI 표현을 정리했다. `TEAM_RULES.md`, `docs/DATA_FINDINGS.md`, `docs/TEAM_RULES_CHANGES.md` 및 원본/수정본 해시 `team_rules_revision.json` 참조. 공통 평가기·fold 수령 및 코드의 통합 조회 스위치/open 경로 반영은 문서 수정과 구분한 미완료 항목이다.

- E010: Python 3.11.17 및 서버 지정 16개 패키지 버전 일치. 전체 66개 의존성 검사·import·단위 검사 6개 통과. 공개 60행에서 이전 환경과 모델 예측 일치 및 반복/순서/부분/단독/빈 입력 검사 통과. `.venv-server`를 이후 작업에 사용한다. OS는 Debian 13, GPU 실행은 미검증이며 ZIP은 생성·갱신하지 않았다. 설치·사용법은 `environment/README.md`, 실측은 `reports/server_environment.json`과 `reports/server_model_compatibility.json`.

- 최신: E007/E008에서 S/N/P/T 16개 조합을 각각 평가(30개 모델+2개 무학습 기준선). SNPT가 평균 최고였다. SPT→SNPT의 새 선박 점수는 97.596%→98.620%, 97.590%→97.604%이며 혼합 첫 분할은 하락했다. 전체 집계는 reports/retrieval_combinations.csv.
- E009 완료: 전체 학습 539,925쌍, ZIP 674,023바이트. 공개 12,222행 일반 추론 경로 385.00초, 최대 RSS 520.5 MiB. 오프라인·행 독립성 검사 통과. 공식 서버 결과는 아니다.
- 이전 단계: 37개 특징과 발음 검색 추가(search). E004 새 선박 가중 정확도 0.948644→0.975961, E005 0.969169→0.975900. 두 분할은 겹치는 개발 검증이다.
- structure/all은 기존 검증 후보 목록을 유지했다. search는 후보 수와 학습 음성이 달라진 전체 시스템 비교다.
- 가중 발음 특징만 추가한 all은 E005에서 성능이 내려가 단독 채택하지 않았다. 오류 수정뿐 아니라 새로운 오류도 기록했다.
- 이하 기존 수치와 E001~E003은 역사적 기준으로 보존한다.

- 데이터 구조·품질·대표성: `analysis/REPORT_8_SECTIONS.md`, `analysis/output/`.
- 방법과 학습 설정: `model/matcher.py`, `train.py`, `docs/MODEL_DETAILS.md`.
- 새 선박 2,400행의 로컬 가중 정확도 0.9486438135.
- 기존·새 선박 혼합 1,965행의 로컬 가중 정확도 0.9852777523. 이 중 70.89%는 학습에 등장한 선박.
- 위 수치는 단일 분할이며 두 조건을 서로 같은 일반화 성능으로 비교하지 않는다.
- 새 선박 오답 16행 중 검색 누락 2행, 후보 선택 오류 14행. 기존 검증 자산으로 후보 존재 여부를 재확인했다.
- 전체 학습, 오프라인 실행·입력 보존·행 독립성·출력 형식 검사 기록 존재.

## 아직 없는 근거

- 동일 후보·분할에서 LogisticRegression/MLP/Random Forest/LightGBM/CatBoost의 공정한 성능 비교.
- 개별 수치 특징 및 호출부호 조회의 제거 효과. 특징 그룹 비교(E004/E005)와 검색 경로 모든 조합(E007/E008)은 완료했다.
- 5-fold 평균, 신뢰구간, 연도순 성능.
- 실제 비공개 평가 규모에서의 시간·메모리. 공개 train 전체의 실행 비용 검사는 E006과 최신 E009 상태·결과를 확인한다.
- 사용자가 제공하거나 실제 서버에서 확인한 공식 Public/Private 점수와 순위.

자료가 없는 항목은 0점 또는 실패로 해석하지 않는다. 기록상 미측정이다.

## 앞으로의 기록 흐름

가설과 조건 명시 → 학습·검증 실행 → 설정과 결과 보존 → 동일 조건의 이전 실험과 비교 → 오답 분류 → 채택/보류 이유 기록 → 보고서·발표 근거 연결.

예정된 실험은 DECISIONS.md에서 관리한다. 완료하지 않은 실험에는 성능 수치를 넣지 않는다. 이 기록 체계는 이후 작업의 규칙이며, 예약 실행이나 자동 실험을 생성한 것은 아니다.

## 2026-10-19 KST — Windows 로컬 폴더 구성

사용자 요청에 따라 프로젝트 루트의 data/를 생성하고 기본 CSV 위치를 명시했다. model/은 기존 경로와 호환되도록 Python 코드와 학습 산출물을 파일 역할·Git 추적 여부로 구분한다. .gitignore에 데이터·.venv-server·assets.json·ranker.txt 제외 규칙이 이미 있어 유지했다. 현재 CSV·모델 산출물·가상환경은 없으며, 가상환경은 기존 Windows 설치 절차로 생성한다. 코드 이동·학습·ZIP 생성·commit·push는 수행하지 않았다. 안내는 README.md, docs/MULTI_PC_SETUP.md, model/README.md에 기록했다.

## 2026-10-19 KST — Windows 환경 설치 완료

사용자 요청으로 .venv-server를 설치했다. Python 3.11.17, 지정 16개 패키지 버전·import, 전체 50개 의존성 검사 및 CPU 연산이 통과했고 기존 테스트 6개도 통과했다. CUDA 12.8 빌드는 설치됐으나 GPU 사용은 감지되지 않았다. 최초 uv Python 경로 연결 오류는 다운로드된 실행 파일을 직접 지정하여 해결했다. 상세 기록: project_log/windows_environment_setup.json. PC별 검사·전체 버전은 reports/local_environment.windows.json 및 environment/windows-freeze.local.txt에 보존하며 Git 제외다.
