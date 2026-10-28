# AS-06 이후 개선 계획

계획 ID: PLAN-AS06-IMPROVE-01. 작성일: 2026-10-26 KST.
범위: 현재 코드·기록 검토 및 카탈로그/합성 사전 집계. 신규 학습·추론·성능 평가·ZIP 생성은 실행하지 않았다. 아래 변경은 모두 planned다.

## 기준

사용자 제공 제출 115506: AS-06 점수 0.9122153056, 표시 시간 366초. AR-06보다 +0.65789473%p. 이는 전체 패키지의 서버 점수 상승 근거다. 합성 발음·자모 검색·점수·조회 각각의 기여와 한글/비한글 득실은 미측정이다.
실제 통합본은 artifacts/AS-06/script.py 및 run_ar06.py다. 기존 AS-06 자산을 보존하고 새 실험 경로에서 비교한다. 현재 .venv-server Python은 3.11.15이며 목표 3.11.17과 구분한다.

## 확인한 사실

| 항목 | 현재 상태 | 해석의 한계 |
|---|---|---|
| 카탈로그 | 78,595행 / 원문 고유 이름 78,594 / canonical 고유 이름 78,592 | 행과 정규화 이름을 구분 |
| 합성 사전 | 78,586키, 누락 6개, 추가 키 0개 | 누락 보완의 큰 점수 이득은 보장할 수 없음 |
| 발음 개수 | 2개: 74,238개 이름 / 1개: 4,348개 이름 | 단일 발음의 원인은 아직 미분류 |
| 생성 | 첫 token Top-K 분기 후 각각 greedy, 기본 최대 새 token 20 | 전체 문장 확률 기준 Top-2가 아님 |
| 점수 | 자모 유사도를 alpha/phon/skel/tok 및 마지막 lattice에 반영 | 상관된 신호의 반복 반영이 실제 오류를 만드는지 검증 필요 |
| 검색 | 정규화 자모 3-gram 누적, posting 2,500 초과 생략, 상위 250척 | 길이·빈번한 부분음·오타의 영향 미측정 |
| 조회 | unique dict가 비면 다수결 dict로 fallback, 톤수 결측/ID 미발견은 호환성 True | Safe Lookup의 실제 보장 범위 점검 필요 |
| 생성기 분할 | manifest와 LM 기록은 heldout_fold=0 | AS-06.json의 전체 train 학습 설명과 충돌; 실제 fit 범위 감사 필요 |

사전 SHA256: e5a3ed8ce2b8eedfb499f09307e04ccd178992d838091a7e091cddc0c6fc06a4.
집계 출력 뒤 부가 소스 출력 범위를 넘겨 IndexError로 종료했다. 집계는 완료됐으며 원본 데이터·코드 변경은 없었다. 성능 실험으로 분류하지 않는다.

## 1. 자모 점수 중복 반영 완화 — 우선 비교

가설: 비슷하게 생성된 다른 선박의 발음에 여러 점수 항목이 동시에 올라 선택 오류가 발생할 수 있다.

- 기존 로마자 alpha/phon/skel/tok 특징을 덮어쓰기 전 값으로 보존하고 자모 특징을 분리한다.
- 후보 집합·합성 사전·검색·조회 조건을 고정한 채 기존 AS-06과 ‘기존 이름 점수 + 하나의 자모 보정’을 비교한다.
- 원자모/정규화 자모 차이, 이름 길이, 숫자, GT/IGT, 동일 발음 후보 수, 1·2위 차이를 신뢰도 특징으로 검토한다.
- 보정 크기와 적용 조건은 학습 fold 내부에서만 선택한다. 한글·비한글 Fix/Break와 동일 후보 내 정답 선택률을 기록한다.
- 실패하면 기존 AS-06을 유지한다. 낮은 개발 비용을 이유로 첫 실험으로 제안하며, 효과가 이미 확인됐다는 의미는 아니다.

## 2. 합성 한글 발음 다양성·유효성 개선

가설: 같은 시작 token 뒤의 다른 음역을 추가하면 미등장 한글 선박의 검색 누락을 복구할 수 있다.

- 누락 6개와 단일 발음 4,348개부터 점검한다. 길이 제한 도달, EOS 미종료, 한글 필터 탈락, 중복, 숫자 변화는 별도 집계한다. 숫자는 로마 숫자·서수 해석을 고려한다.
- 고정 seed의 제한된 sampling 또는 뒤쪽 token에서도 분기하는 탐색을 비교하고, 자모 기준 중복 제거 후 최대 4개 발음을 남기는 구성을 검토한다.
- 생성 확률과 발음 간 거리로 품질·다양성을 제한한다. 저품질 발음을 많이 추가하면 충돌과 오답도 늘 수 있다.
- 모델·prompt·검색·점수식을 고정하고 생성 방식과 생성량의 효과를 분리한다. 재개 생성 때 모델/prompt가 섞이지 않도록 manifest를 기록한다.
- 측정: 고유 자모 발음 수, 사전 커버리지, unseen 후보 포함률, 최종 Fix/Break, 충돌, 자산 용량, 인덱스/검색/점수 시간.
- 생성은 오프라인이다. 서버 LLM 호출은 없어도 자산 확대에 따른 로딩·검색 비용은 발생한다.

## 3. 자모 2~4-gram 길이 정규화 검색 추가

가설: 첫 글자 오류·한 음절 탈락·띄어쓰기 차이에서 현재 3-gram 경로가 놓친 후보를 복구한다.

- 정규화 자모의 2~4-gram TF-IDF cosine 검색을 기존 검색과 합집합으로 추가한다. 동점은 vessel ID로 처리한다.
- 첫 비교는 사전·점수식을 고정한다. 필요하면 원자모·영문 발음 골격 채널을 이후 별도 실험한다.
- 해당 행의 이름 길이·유사도·점수차·톤수 결측으로만 확장 여부를 결정한다. 테스트 전체 통계·처리 순서·남은 시간은 사용하지 않는다.
- 톤수 ±10%를 전역 검색의 필수 필터로 쓰지 않는다. GT/IGT 차이·큰 오차 사례를 남긴다.
- 측정: 합집합 가변 후보 포함률, 같은 고정 K의 Recall@K, 새 경로만 복구한 수, 최종 Fix/Break, 후보 수 평균/P95, 단계별 시간.

## 4. 조회 오류와 조회 누락 분리

가설: 안전 조건을 명확히 하고 확인 불가 사례를 후보 경쟁에 남기면 잘못된 즉시 반환을 줄일 수 있다.

- unique 사전이 존재하지만 비어 있는 경우 다수결 사전으로 fallback하지 않는 변경을 독립 비교한다.
- 톤수 결측·카탈로그 ID 미발견은 검증 통과와 구분한다. 즉시 반환 대신 후보 추가/보너스를 비교한다.
- 호출부호 일치·이름 불일치만으로 차단하지 않는다. 개명 선박을 놓칠 수 있어 fit 기록의 GT/IGT·연도·선종 및 다른 근거의 충돌을 함께 검토한다.
- 12%/6% 기준을 일괄 완화하지 않고 탈락한 참 매칭과 통과한 오조회를 먼저 집계한다.
- 보조 행 분할에서 사전을 fit 행에만 만들고 seen/unseen을 나눠 측정한다. Group 5-Fold만으로 seen 조회 효과를 평가했다고 주장하지 않는다.

## 후속: 가벼운 후보 선택 학습

검색 누락보다 후보 선택 오류가 충분히 확인될 때 LightGBM LambdaRank 등으로 위 특징을 결합한다. query는 입출항 기록 한 행, 항목은 후보 ID다. 기존 로마자·자모·숫자·GT/IGT·메타데이터·조회 충돌을 독립 특징으로 보존한다.
후보·특징 생성도 학습 fold 내에서 cross-fit한다. 외부 fold 정답을 후보에 주입하지 않는다. 후보에 정답이 없는 행은 검색 실패로 별도 기록한다. 한글 선박 382척의 작은 표본과 반복행을 고려해 자유도를 제한한다.
추가 LM 학습은 발음 다양성 부족의 근거를 확인한 뒤 진행한다. LM loss 감소를 검색/ID 성능 향상으로 간주하지 않는다. 온라인 LLM 재정렬 재도입은 비용 대비 이득이 아직 입증되지 않아 우선순위가 낮다.

## 검증 순서

1. 기존 AS-06 자산·코드·사전 해시를 보존한다. 원본 ZIP을 재생성하지 않는다.
2. 외부 fold별 생성기·음역 사전·증강·혼동 후보의 fit 범위를 감사한다. heldout=0 생성기 하나를 전체 5-fold에 재사용한 결과는 누수 없는 전체 OOF가 아니다. manifest만으로 fold0의 모든 누수 가능성이 배제됐다고 단정하지 않는다.
3. 누수 통제 baseline을 구성하고 변경을 하나씩 비교한다. 점수 변경은 고정 후보, 검색 변경은 고정 점수, 생성 변경은 고정 모델로 분리한다.
4. Clean Group 5-Fold는 퇴보 검사로 사용한다. 미등장 선박·숫자·소형선·톤수 결측/혼용·이름 변형 스트레스는 별도 보고한다. 공개 train 보류를 독립 최종 테스트라 부르지 않는다.
5. 한글·비한글 정확도, 공식 가중 점수, 표본 수, 후보 포함률, Fix/Break, seen/unseen, 단계별 시간, 자산 용량을 기록한다. 동일 조건의 선박 단위 CI·fold별 방향·행 독립성을 함께 확인한다.
6. 단일 fold/severe 결과나 Public 점수만 반복 선택하지 않는다. 향후 서버 점수 0.92/0.93 도달 여부와 언어별 개선량은 현재 추정하지 않는다.

## 기술 참고와 근거

공식 문서는 기능 존재의 근거이며 본 프로젝트 효과의 증거는 아니다. 설치 버전 호환성은 별도 검사한다.

- Transformers 생성 전략: https://huggingface.co/docs/transformers/v4.57.1/generation_strategies
- scikit-learn 1.5.2 TF-IDF: https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- LightGBM 4.6.0 목표 함수: https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html
- 내부: SUB-AS06-115506.json, AS-06.json, LM-LFM25-FULL-R24-01.json, docs/AS05_SERVER_ANALYSIS.md 및 현재 AS-06 소스.

## 2026-10-26T01:01:06+09:00 - Execution update

First scoring experiment and two secondary noise seeds completed. Default adoption rejected; AS-06 retained. See AS07_JAMO_SCORE_RESULTS.md and REVIEW-AS07-JAMO-01.json. Other stages remain planned. The generator all-train claim is corrected by AUDIT-AS06-LINEAGE-01; source snapshots and fold limitations are documented in the results.


## 2026-10-26T21:59:10+09:00 - Pronunciation stage execution update

Deficient-only diverse generation and optional-variant ownership filtering completed. Both candidate maps are not promoted after the five-condition fold0 comparison. Global regeneration was not performed. See AS08_PRONUNCIATION_RESULTS.md and REVIEW-AS08-PRONUNCIATION-01.json; all original AS-06 assets retained.
