# 자모 점수 분리 실험

이 폴더는 AS-06을 보존하면서 점수 합성만 비교하는 실험 엔진이다. 제출 패키지가 아니며 model 자산·ZIP을 만들지 않는다.

- `legacy` (기본값): 기존 AS-06 자모 점수 반영 방식을 보존한다.
- `separate`: 기존 로마자 특징을 보존하고 `1.8 * (0.6 * raw_jamo_sim + 0.4 * normalized_jamo_sim)`을 한 번 더한다. lattice에는 자모를 재반영하지 않는다. 외부 fold 성적으로 가중치를 조정하지 않는다.
- `predict_rows(..., jamo_score_mode="separate")` 또는 모델의 `jamo_score_mode` 키로 명시한다. 검증 전 기본 동작은 legacy다.
- `candidate_diagnostics`는 외부 리스트에 후보 개수·카탈로그 index 집합 해시를 기록하는 진단 옵션이다. 예측의 입력 특징이 아니며 정답을 후보에 넣지 않는다. 조회를 끈 통제 비교에서 행 전체를 검사한다.

PowerShell:

```powershell
.\.venv-server\Scripts\python.exe -X utf8 -m unittest tests.test_jamo_scoring -v
```

실험 실행기는 `experiments/evaluate_jamo_scoring.py`다. 사전 등록된 실험의 기존 결과는 덮어쓸 수 없다. 분할 이력은 `experiments/audit_generator_lineage.py`와 `AUDIT-GENERATOR-LINEAGE-01.json`에 기록한다.

비교 조건: fold0 2,372행, 조회 OFF, 동일 사전/후보, clean/default/severe, noise seed20261022. 원본 엔진과 legacy의 실제 보류64행 Top5까지 동일성 검사 후 전체 비교한다. 공식 평가기의 subset 비교 및 선박 단위 bootstrap2000회 seed0을 사용한다. 원본·노이즈 행은 공개 train 유래이고 비공개 평가 자료를 읽지 않는다.

원본 AS-06 코드/사전/기록 스냅샷은 `artifacts/JAMO-SCORE-01/reference/`에 보관한다. 행별 결과·오답 사례는 Git 제외 경로 `artifacts/JAMO-SCORE-01/evaluation/`에 보관하고, 집계는 `reports/matching_improvements/JAMO-SCORE-01/`에 둔다.

현재 Python은3.11.15로 서버 목표3.11.17과 다르다. fold0은 이미 개발에서 사용했고, 수동 사전 이력·합성 사전의 과거 생성 metadata에 제한이 있다. 한 fold의 결과를 독립 최종 평가나 서버 향상으로 해석하지 않는다.
