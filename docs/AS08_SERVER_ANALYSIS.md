# AS-08 서버 결과 분석과 점수 회복 계획

## 2026-10-28T22:09:58+09:00 — 수정 AS-08(117597) 결과 반영 / VERIFY-AS08-FIXED-RESULT-01

- [x] 사용자 정정: **오류 수정본 AS-08**은 제출 `117597`, `2026-10-27 23:17:35 KST`, **0.9082679372**, **327초(5분27초)**다. 최초 AS-08 `117501`(0.9003732004/236초)은 별도 제출로 보존한다. 기존 SUB-117597 기록과 일치하며 중복 등록하지 않았다.
- [x] 수정본−최초 AS-08: **+0.0078947368(+0.78947368%p), 표시 시간+91초**. 수정본−AS-06/07: **−0.0039473684(−0.39473684%p)**.
- [x] **AS-09−수정 AS-08: +0.0118421053(+1.18421053%p), 표시 시간+170초**. 앞선 +1.97368421%p/+261초 비교는 최초 AS-08에 대한 값으로만 유효하다. 향후 최신 AS-08 기준 비교에는 수정본을 쓴다.
- **주요 결정/해석**: AS-09는 관측된 최고 서버 점수 기준선을 유지한다. 오류 수정 후 점수 회복은 확인되지만 전체 상승분을 Dense 단독 효과로 귀속하거나 잔여 격차를 사전 제거 때문으로 확정하지 않는다. 기존 SUB-117597의 인과 확정 표현은 서버 로그·대응 ZIP 변경 범위가 확인되기 전까지 가설로 한정한다. 현재 로컬 ZIP과 서버 업로드 바이트 일치는 미확인이다.
- **범위/제약**: 기록·소수 연산·로컬 ZIP 설정 확인만 수행. 새 학습/모델 추론/ZIP 수정/업로드/커밋/push 없음. 언어별 정확도·표본 수·구성별 기여는 미측정이다. TEAM_RULES v4 확인, eval_kit 실제1.0/요청1.1 차이와 Python3.11.15/목표3.11.17 차이를 유지한다.
- [ ] **다음 작업**: 별도 개선 요청 시 같은 조건의 구성별 비교로 기여를 분리한다. 이번 결과 반영은 완료했다.
- **근거**: [수정 제출 결과](../project_log/experiments/SUB-117597.json), [검증 기록](../project_log/experiments/VERIFY-AS08-FIXED-RESULT-01.json), [갱신 비교표](AS09_SERVER_RESULT.md). 권장 커밋: `docs(results): distinguish corrected AS08 submission and revise AS09 comparisons`.

---

아래 분석의 최초 AS-08은 제출117501을 가리킨다. 수정본117597의 점수로 기존 분석/해시를 소급 치환하지 않는다.

## 결론

AS-08은 AS-06/07 대비 서버 표시 가중 점수가 **-0.0118421052(-1.18421052%p)**다. 이번 감사의 핵심 발견은 **AS-07과 AS-08 ZIP 모두 Dense 패키지가 요구하는 loss.py를 포함하지 않아, ZIP 단독 격리 import에서 Dense 클래스가 비활성화된다는 것**이다.

이는 소스/실험 패키지의 설계상 Dense 하이브리드와 실제 제출 ZIP의 독립 실행을 구분해야 한다는 뜻이다. 서버 실행 로그는 없으므로 실제 서버에서 어느 경로가 실행됐는지까지 확정하지 않는다. 이번에는 모델 로딩·추론·학습·튜닝·ZIP 생성 없이 정적 검사와 import-only 재현만 수행했다.

## 1. 공식 서버 결과와 해석 범위

| 제출 | 제출 ID | 가중 점수 | 표시 시간 |
|---|---:|---:|---:|
| submit_AS-06.zip | 115506 | 0.9122153056 | 366초 |
| submit_AS-07.zip | 116338 | 0.9122153056 | 372초 |
| submit_AS-08.zip | 117501 | 0.9003732004 | 236초 |

사용자가 전달한 공식 서버 결과이며 사이트 직접 조회는 하지 않았다. AS-08은 AS-07보다 136초 짧지만, 표시 시간의 세부 포함 범위·입력 수·반복성은 미확인이다. 이 시간 차이를 Dense 속도 향상의 증거로 쓰지 않는다.

언어별 정확도/표본 수·후보 포함률·seen/unseen·Fix/Break·신뢰구간은 미측정이다. 총점만으로 특정 한글 오답 수나 테스트 분포를 역산해 확정하지 않는다. AS-06/07을 더 높은 서버 점수 기준으로 보존한다.

## 2. 실제 ZIP에서 확인한 결함

AS-07/08 모두 다음 조건이 성립한다.

1. ZIP 안 model/phonetic_encoder/__init__.py가 tokenizer/model 외에 .loss의 두 학습용 손실 클래스를 import한다.
2. ZIP에는 model/phonetic_encoder/loss.py가 없다.
3. script.py는 두 import 경로에서 ImportError를 잡은 뒤 HybridDenseLexicalMatcher=None으로 처리한다.
4. main은 클래스가 없으면 dense_matcher를 만들지 않고 dense_mode를 none으로 바꾼다. model.json의 dense_mode=hybrid만으로 활성화를 보장하지 않는다.
5. 로딩 과정의 예외도 조용히 처리하므로 출력 CSV가 생성돼도 의도한 Dense가 사용됐다는 증거가 되지 않는다.

격리 검사는 지정 Windows 인터프리터를 -I -B로 실행하고, ZIP의 Python 소스만 검증된 scratch 하위 임시 폴더에 복사했다. 저장소 루트가 import 경로에 없는지 확인했다. 네트워크와 torch.load를 차단했으며 학습/입력 데이터/가중치는 임시 환경에 복사하지 않았다.

| import-only 조건 | AS-07 | AS-08 |
|---|---|---|
| 원본 ZIP 소스 | model.phonetic_encoder.loss 없음, Dense 클래스 unavailable | 같은 실패 |
| 임시 복사본에 loss.py만 추가 | import 성공, Dense 클래스 available | import 성공, Dense 클래스 available |
| 임시 복사본의 __init__를 추론용 두 import로 제한 | import 성공, Dense 클래스 available | import 성공, Dense 클래스 available |
| 실제 모델 로드 횟수 | 0 | 0 |

두 수정은 임시 복사본에서만 시험했다. 원본 소스·제출 ZIP은 수정하지 않았다. 이 결과는 import 결함의 원인을 확인한 것이며, 모델 로드/실제 예측/서버 점수 회복까지 검증한 것은 아니다.

재현 코드: [analyze_as08_server_result.py](../experiments/analyze_as08_server_result.py).
측정 근거: [archive_audit.json](../reports/as08_server_analysis/archive_audit.json).

## 3. 패키지 변화와 가장 유력한 설명

| 요소 | AS-06 | AS-07 | AS-08 |
|---|---|---|---|
| 기존 공통 lookup·학습 대응표·메타 자산 | 기준 | 17개 기존 model 키 값 동일 | 한글 map 제외 기존 공통 값 동일 |
| 수동 IDEA_B_WORD_VARS | 299개 | 299개 | 0개 |
| 합성 한글 catalog map | 78,586개 | 78,586개 | 0개, 전역 스위치도 OFF |
| lattice_word_similarity | 존재 | 존재 | 정의/선택 경로 제거 |
| Dense 설계 | 없음 | Top-35 합집합 + Dense 가산 | 멀티뷰 표현 + 같은 결합 가중치 |
| ZIP 단독 Dense import | 해당 없음 | 실패 재현 | 실패 재현 |

AS-06→07은 기존 17개 모델 자산 값과 전역 상수가 같고, 주요 함수 차이는 main/predict_rows의 Dense 연결이다. 따라서 Dense가 활성화되지 않았을 때 AS-06과 비슷한 동작·동일 총점이 나온다는 설명은 코드 및 서버 관측과 일치한다. 모든 입력 예측의 동일성이나 실제 서버의 fallback 실행을 증명한 것은 아니다.

AS-07→08에서는 저장 인코더 full_encoder.pt와 hybrid_matcher.py의 바이트 해시가 같다. 바뀐 것은 tokenizer/model 표현 코드, catalog 임베딩, 수동 사전·map·lattice 등이다. AS-08을 더 큰/새로 학습된 신경모델의 성능 저하로 해석하지 않는다.

**현재 가장 유력한 가설**은 Dense 대체 경로가 동작하지 않는 상태에서 기존 한글 보완 경로까지 제거돼 점수가 하락했다는 것이다. map·사전·lattice 중 어느 것이 얼마나 기여했는지는 분리 실측하지 않았으므로 미확인이다. Dense 미작동만으로 AS-07→08의 점수 차이를 설명할 수 없고, 함께 제거된 장치의 영향을 봐야 한다.

## 4. 로컬 결과가 서버 개선을 보장하지 못한 이유

### 4.1 실행 대상이 다름

로컬 DPR 평가 스크립트는 저장소의 완전한 phonetic_encoder 패키지를 직접 import한다. ZIP은 필요한 loss.py가 빠졌다. 따라서 로컬에서 실제 Dense를 평가한 결과를 독립 ZIP의 동일 기능 실행 근거로 전용할 수 없다.

공통 제출 검사도 CSV의 행 수·형식·행 독립성을 통과할 수 있지만 Dense 활성화·호출 횟수를 확인하지 않는다. 이전 완료 기록의 checker 통과 주장을 이번에 다시 실행해 확인한 것은 아니다. 이번 발견은 “checker가 통과하면 의도한 모든 구성요소가 활성화됐다”는 해석을 부정한다.

### 4.2 멀티뷰 Top-1 상승과 후보 포함률은 다름

기존 저장 보고서의 같은 fold0 표본을 비교하면 다음과 같다. 이번에 새 추론을 실행한 수치가 아니다.

| Dense 단독 검색 지표 | 기존 표현 | 멀티뷰 표현 |
|---|---:|---:|
| 한글 Top-1 | 138/239 (57.74%) | 147/239 (61.51%) |
| 한글 Recall@35 | 227/239 (94.98%) | 226/239 (94.56%) |
| 비한글 Top-1 | 2014/2133 (94.42%) | 1922/2133 (90.11%) |
| 비한글 Recall@35 | 2103/2133 (98.59%) | 2091/2133 (98.03%) |

Stage-1은 정답을 후보에 넣는 것이 중요하다. 한글 Top-1 상승만으로 후보 검색 개선 전체를 주장할 수 없다. 현재 DPR 엔진의 Dense 추가는 한글 경로에 제한되므로 비한글 단독 검색 하락을 이번 서버 손실의 직접 원인으로 연결하지 않는다.

두 보고서에는 코드/임베딩의 완전한 hash 결합과 당시 소스 보존이 충분하지 않아, 이 표는 저장 결과 비교 수준의 증거다. 같은 분할·가중치 경로라도 모든 조건이 통제됐다는 증명이나 독립 최종 테스트 결과가 아니다.

### 4.3 검증 선택·캐시의 한계

현재 train_bi_encoder.py는 외부 보류 fold의 한글 Recall@35로 best checkpoint를 선택한다. 팀 기준의 fit 내부 검증 선택과 다르며, 해당 소스 경로로 학습한 모델의 같은 외부 fold 결과를 완전히 손대지 않은 독립 성적으로 표현하면 안 된다. 저장 checkpoint가 정확히 어느 실행 소스로 선택됐는지 완전한 provenance는 추가 확인 대상이다.

evaluate_dpr_robustness.py의 baseline cache는 조건 이름을 중심으로 재사용하며 source/data/model/input-order 해시 검증이 없다. 현재 소스와 캐시의 baseline이 같은지 재검증해야 한다. 캐시 적중 시간을 0.001초로 기록하는 값도 실제 baseline 추론 시간으로 비교하지 않는다.

사전 제거의 기존 보고서는 실제 Dense가 활성화된 로컬 fold0 clean/severe에서 결과가 같았다. 이를 Dense가 비활성화된 ZIP에서도 안전하게 제거된다는 근거로 사용하면 안 된다. synthetic 한글 map의 fold 계보·수동 사전 생성 이력에 관한 기존 감사의 한계도 유지한다.

## 5. 점수 향상 우선순위

예상 점수 상승 폭은 모두 미측정이다. 아래는 planned 제안이며 구현·학습·성능 비교를 이번에 시작하지 않았다.

| 순서 | 제안 | 이유 | 고정할 조건·판정 |
|---|---|---|---|
| P0 | 추론 패키지에서 학습용 import 분리 또는 loss.py 포함; Dense 필수 모드에서는 import/로드 실패 시 명시적 오류 | 빠진 의존성부터 해결해야 의도한 실험을 할 수 있음 | 새로운 ZIP을 만들지 않고 uncompressed sandbox에서 필요한 파일 목록·isolated import·저장 checkpoint load 확인. 미활성 상태의 조용한 성공 금지 |
| P1 | AS-06 기능을 유지한 기준으로 Dense 후보 추가부터 단독 비교 | 사전/map/lattice 제거를 Dense 추가와 분리 | 기존 후보를 보존하고 Dense Top-35만 합집합. 기존 scorer 유지. 그 다음에만 Dense 가산을 별도 실험 |
| P2 | Dense 점수의 조건부 적용 | 현재 cosine·순위·z-score·margin의 고정 가산은 불확실한 후보도 크게 끌어올릴 수 있음 | 기존 이름 점수/후보 간 margin/톤수·숫자 충돌/Dense 일치를 fit-inner에서 결정. 강한 기존 매칭의 Break와 복구 Fix를 함께 확인 |
| P3 | 기본 음소 표현을 보존하고 acronym/initial/semantic 뷰를 보조 검색 채널로 분리 | 모든 추가 뷰를 0.8로 더하면 원래 후보가 밀릴 수 있음 | 기본 Top-K 유지 + 보조 후보 제한 추가. base-only/aux-on을 고정 후보·동일 fold로 비교. RRF/합집합 비교는 별도 요인 |
| P4 | 검증 환경과 계보 정비 후 학습 손실/음성 표본 개선 | 기존 로컬 near-100%가 서버 오류를 대표하지 않음 | best 선택은 fit-inner, 외부 fold는 한 번의 고정 평가. 실제 어려운 오답·숫자/자매선박 충돌은 fit 내부에서만 mining |

P0의 추천 구현은 추론 번들 전용 __init__에 tokenizer/model만 노출하고, 학습 코드가 필요할 때 loss를 직접 import하는 방식이다. 또는 기존 API를 유지하려면 누락된 loss.py를 번들에 포함한다. 어느 쪽이든 실제 모델 load와 checkpoint/embedding/catalog 순서 hash 검사를 후속으로 통과해야 한다. import 복구만으로 개선 완료라고 하지 않는다.

실행 로그는 Dense 활성 상태·처리한 한글 행 수·Dense 호출 횟수·후보 추가 집계·로드된 자산 hash만 저장한다. 비공개 입력/개별 예측은 저장하지 않는다. 논리적으로 Dense를 사용할 수 있는 공개 한글 probe에서 호출 횟수가 양수인지 검사해야 하며, lookup으로 우회된 행은 별도로 세어야 한다.

## 6. 통제 실험 계획

| 제안 ID | 상태 | 변경 요소 | 목적 |
|---|---|---|---|
| VERIFY-DENSE-BUNDLE-01 | planned | 의존성 복구 및 startup/load 검사 | Dense 실행 경로 확인, 정확도 개선 주장은 하지 않음 |
| RETRIEVAL-DENSE-ONLY-01 | planned | 기존 후보 ∪ Dense 후보, 기존 점수 유지 | 후보 누락 복구와 새 Break 분리 |
| RANK-DENSE-GATED-01 | planned | 앞 조건 고정 + Dense 가산/게이트 | 후보 추가와 점수 효과 분리 |
| ABLATE-KO-SUPPORT-01 | planned | map·수동 사전·lattice를 각각 하나씩 제거 | 제거할 보완 장치의 실제 기여 확인 |
| VIEW-AUX-RETRIEVAL-01 | planned | 기본 표현 보존 + 보조 뷰 검색 | Top-1과 후보 포함률의 상충 확인 |

위 ID는 제안명이며 실행/완료 실험으로 등록하지 않았다. 기계 판독 계획은 PLAN-SERVER-RECOVERY-01이다. 새 AS 제출 번호를 선할당하지 않는다.

검증은 두 범위를 분리한다.

- 제출 자산의 실행·호환성 진단: 기존 AS-06/07/08 자산을 공개 train에서 확인할 수 있으나 학습에 사용된 자산의 정확도는 일반화 성능이 아니다.
- 성능 비교: 기존 배포 fold를 읽고 각 fold의 fit에서만 lookup·대응표·특징 선택·튜닝을 수행한다. map/generator도 fold별 계보가 필요하다. 공유된 하나의 map을 5-fold 모두에 사용해 누수 없는 OOF라고 표시하지 않는다. 기존 fold0 지도는 계보가 확인된 범위의 개발 진단에서만 사용할 수 있고 수동 사전 이력 한계도 표시한다.

전체 OOF와 fold별 한글/비한글 정확도, 공식 0.7/0.3 점수, 표본 수, 후보 포함률, 검색 누락/선택 오류/조회 오류, Fix/Break와 선박 단위 paired CI를 기록한다. 가변 합집합의 포함률을 Recall@35로 부르지 않는다. 보조 행 분할에서는 seen/unseen을 나눠 lookup 효과를 보고 테스트의 seen 비율로 일반화하지 않는다. clean/default/severe는 개발 진단이며 비공개 test 분포의 재현이라고 주장하지 않는다.

## 7. 재현과 한계

PowerShell, 저장소 루트:

    .\.venv-server\Scripts\python.exe -B -X utf8 experiments/analyze_as08_server_result.py --output reports/as08_server_analysis/archive_audit.json

이 명령은 기존 ZIP을 읽고 임시 소스만 import하며 새 제출 ZIP을 만들지 않는다. 모델 로딩/추론도 차단한다. 최초 시도는 기본 sandbox temp 디렉터리 권한 오류로 중단됐고, 확인된 저장소 scratch 하위 경로로 한정해 재실행했다. 기존 파일 삭제·권한 변경으로 우회하지 않았다.

현재 지정 인터프리터는 Python3.11.15이며 목표3.11.17과 다르다. 5개 패키지 불일치/누락은 앞선 REFACTOR-RETRIEVAL-03 환경 검사에 기록돼 있다. TEAM_RULES v4를 적용하되 eval_kit README 실제1.0/사용자 기준1.1 차이도 유지한다. 서버 환경을 동일하게 재현했다고 주장하지 않는다.

이번 감사는 archive dependency 문제와 local/archive 차이를 확인했다. 실제 서버 Dense 활성 로그, 비공개 분포, 한글/비한글 오류의 기여, 패키징 복구 후 점수 개선은 미확인이다. 기존 기록은 수정하지 않고 후속 해석의 근거만 추가했다.

근거: [archive 감사](../reports/as08_server_analysis/archive_audit.json), [기존 평가·코드 검토](../reports/as08_server_analysis/evidence_review.json), [실행 기록](../project_log/experiments/ANALYSIS-AS08-SERVER-01.json), [복구 계획](../project_log/experiments/PLAN-SERVER-RECOVERY-01.json).


## 8. 사전 두 개를 꺼도 남아 있는 매칭 기능 (AUDIT-AS08-REMAINING-01)

AS-08의 수동 사전 제거는 IDEA_B_WORD_VARS 299개를 뜻하며 학습된 모든 음역표 제거를 뜻하지 않는다. 실제 ZIP model.json에는 호출부호 유일 조회 **5851개**, 이름 유일 조회 **5857개**, 학습 단어 대응 **466개**, 학습 음절 대응 **271개**가 남아 있다. top2/top3는 같은 key의 대안들이므로 별도 어휘 수로 합산하지 않는다. use_train_lookup_default=True이다.

호출부호 조회는 톤수 호환12%, 이름 조회는6% 조건으로 사용할 수 있다. 조회 실패 뒤에도 학습 단어/음절 후보, 알파벳 음역·로마자·발음 규칙, EN_V3 문자열 검색, 숫자·톤수·선종·국적 점수가 작동한다. map가 없어 합성 한글 자모 경로는 비지만 영어 이름에 대한 음역/발음 비교는 남는다.

공식 가중치 때문에 비한글만 잘 맞혀0.9가 된다는 설명은 성립하지 않는다. 비한글100%를 상한으로 두어도 한글 정확도는 최소 (0.9003732004-0.3)/0.7 = 약 **85.7676%**여야 한다. 이는 산술적 하한이며 실제 한글 정확도/각 경로의 기여를 측정한 값이 아니다. 조회 항목 개수도 비공개 test의 조회 적중 수/비율을 의미하지 않는다.

근거: [잔존 경로 정적 감사](../project_log/experiments/AUDIT-AS08-REMAINING-01.json). 이번 추가는 설명을 위한 정적 확인이며 새 모델 실행/성능 실험이 아니다.
