# AS-09 공식 제출 결과와 확인 범위

수정 갱신: 2026-10-28T22:09:58+09:00 — 사용자 제공 수정 AS-08 결과 `117597`을 반영했다. **AS-09의 최신 AS-08 대비 차이는 +1.18421053%p / 표시 시간+170초**다. 앞선 +1.97368421%p는 최초 AS-08(117501) 대비 값이다. 기존 완료 JSON과 이전 검증 수치는 당시 알려진 제출에 대한 기록으로 보존한다. 근거: [SUB-117597](../project_log/experiments/SUB-117597.json), [VERIFY-AS08-FIXED-RESULT-01](../project_log/experiments/VERIFY-AS08-FIXED-RESULT-01.json).

기록·검증 시각: 2026-10-28T22:06:06+09:00. 검증 ID: `VERIFY-AS09-RESULT-01`. 제출 `117799`의 기존 결과 기록 `SUB-117799`를 대조한 것으로 새 제출·성능 실험이 아니다.

사용자 제공 표: `submit_AS-09.zip`, 제출 `2026-10-28 00:54:00 KST`, 공식 표시 점수 **0.9201100425**, 표시 시간 **497초(8분 17초)**. 사이트 직접 조회는 하지 않았다. 현재 저장된 아래 제출 중 가장 높은 표시 점수다. 최종 Private 점수·순위·통계적 유의성을 뜻하지 않는다.

| 제출 | 표시 점수 | AS-09 점수 차이 | 표시 시간 | AS-09 시간 차이 |
|---|---:|---:|---:|---:|
| AS-06 | 0.9122153056 | +0.0078947369 (+0.78947369%p) | 366초 | +131초 |
| AS-07 | 0.9122153056 | +0.0078947369 (+0.78947369%p) | 372초 | +125초 |
| AS-08 최초 (117501) | 0.9003732004 | +0.0197368421 (+1.97368421%p) | 236초 | +261초 |
| AS-08 오류 수정 (117597) | 0.9082679372 | +0.0118421053 (+1.18421053%p) | 327초 | +170초 |
| AS-09 | **0.9201100425** | 기준 | **497초** | 기준 |

## 실제 로컬 ZIP 설정

- SHA256: `3c89f0d60d8c14b369b7f4846951cbce53acc39652f2359f72b29466000b0325`; 22,798,425 bytes. 기존 AS-09 패키지 기록 및 SUB-117799 로컬 해시와 일치하며 ZIP CRC 검사를 통과했다. 업로드된 서버 원본 해시는 미확인이다.
- `dense_mode=hybrid`, `dense_weight=2.5`. `loss.py`를 포함한 `phonetic_encoder` 소스 7개가 동봉됐다. 실제 entry point에서 Dense import/초기화 실패는 RuntimeError로 중단한다. 같은 바이트의 ZIP이 서버에서 실행되었다면 과거의 조용한 fallback 설명은 해당하지 않는다. 서버 로그는 미수령이다.
- Stage-1은 기존 문자열 후보와 한글 입력의 Dense Top-35 후보를 합친다. Stage-2는 기존 문자열/숫자/톤수/메타 점수에 Dense 점수를 더한다. Dense가 문자열 점수를 대체하는 구성은 아니다.
- Dense 디바이스는 entry point에 `cpu`로 지정되어 있다. 표시 시간만으로 CPU 점수 계산이 병목이라고 단정하거나 GPU 효과를 산정하지 않는다.
- `IDEA_B_WORD_VARS`는 빈 사전(0개)이다. 반면 `use_full_ko_map=True`이고 기존 `catalog_full_ko_map` **78,586개가 ON**이다. `use_train_lookup_default=True`, 호출부호 조회5,851개·이름 조회5,857개·학습 단어466개도 유지된다.
- EXAONE-only 신규 사전을 검증한 결과가 아니다. 신규 EXAONE 생성/학습의 차단 기록과 별개다. map을 고신뢰도라고 부를 별도 품질 측정은 이번 작업에 없다.

## 해석과 기존 기록의 정정

AS-09를 현재 **관측된 서버 점수 기준선**으로 유지한다. Encoder 구조·패키징·전처리·한글 map 설정 등이 함께 바뀌었으므로 개선 전부를 Dense나 특정 학습 기법에 배분할 수 없다. 한글/비한글 정확도·표본 수·seen/unseen·후보 포함률·Fix/Break·CI는 미측정(null)이다.

기존 완료 JSON은 덮어쓰지 않았다. `SUB-117799.json`, `AS-09.json`과 당일 문서에 있는 다음 표현은 이 검증의 범위로 한정한다.

1. 서버 Dense 구동의 직접 확인: 정적 로컬 ZIP 근거와 동일 바이트 실행 가정에 따른 해석이다. 서버 로그에 의한 직접 확인이 아니다.
2. `submitted_zip_hash_verified=true`: 확인된 것은 **로컬 패키지 기록 간 해시 일치**다. 서버 업로드 바이트 일치 검증은 미완료다.
3. 1,200초 제한·703초 여유: 현재 프로젝트에 기록된 최신 사용자 제공 안내는 **1,800초(30분)**이다. 표의 표시 시간과 단계별 측정 범위가 같다고 확인하지 못했으므로 실측 여유 시간으로 보고하지 않는다.
4. DPR-SYMPHON-02의 INDEX `5-fold group OOF standalone`: 연결된 기록은 fold0/2,372행이므로 이번 결과를 전수5-fold OOF로 제시하지 않는다.
5. 이전 로컬 제출 검사와 정확도는 과거 기록이며 이번에 재실행하지 않았다.

TEAM_RULES v4를 확인했다. eval_kit README 실제 버전1.0과 사용자 기준1.1의 차이는 유지하며1.1 적용 완료로 표시하지 않는다. 현재 Windows `.venv-server` Python3.11.15는 목표3.11.17과 다르다. `reports/server_environment.json`은 과거 Linux 환경이다.

## 다음 작업

개선 실험을 별도로 진행할 때 같은 fold·후보·lookup 조건에서 Dense OFF/후보만/점수까지를 분리하고, 한글 map ON/OFF를 별도 비교한다. 다른 설정이 고정된 실측 비교 전에는 Dense의 기여량을 주장하지 않는다. 이 결과 확인만으로 새 학습·튜닝·ZIP 생성을 시작하지 않았다.

근거: [검증 JSON](../reports/submission_results/VERIFY-AS09-RESULT-01/verification.json), [기존 서버 결과](../project_log/experiments/SUB-117799.json), [검증 작업 기록](../project_log/experiments/VERIFY-AS09-RESULT-01.json).

권장 커밋: `docs(results): verify AS09 score, package settings and evidence limits`
