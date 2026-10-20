# 실행 성능 개선 — 2026-10-20

AS-02의 후속 코드 개선이며 새 OOF 성능 결과가 아니다. 전체 학습이나 ZIP 생성은 실행하지 않았다.

## 변경 내용

- 입력 행의 정규화·로마자·발음·이름 구조·톤수 변환을 한 번 계산하고 검색과 특징 생성에서 재사용한다. 카탈로그 이름 길이와 정렬된 톤수도 재사용한다.
- 후보 특징을 기본 64행 단위로 모아 LightGBM을 한 번 호출하고 행별 후보 구간에서 최고점을 선택한다. 직접 조회 성공 행은 검색·점수 계산을 생략한다.
- 특징 추출은 외부 스레드, 모델 학습은 별도 내부 스레드를 사용한다. 외부 작업자는 affinity와 Linux cgroup v2 quota 및 최대 6개 제한을 적용한다.
- 학습 결과 전체 목록을 없애고 묶음별 소비로 중간 후보 ID 등을 해제한다. 최종 학습 특징 행렬은 여전히 메모리에 유지하며 concatenate 순간의 복사는 남아 있다.
- TF-IDF 카탈로그는 CPU CSR, GPU sparse COO float32로 저장한다. 카탈로그 전체를 밀집 행렬로 만들지 않는다. 필요한 검색 채널만 인덱스를 구축한다.
- 검색 묶음은 최대 64개 쿼리 변환을 공유한다. 점수 연산은 단일 쿼리별 고정 형태를 유지한다. GPU 전체 쿼리 GEMM을 강제하지 않아 행 독립성 위험을 줄였지만 GPU 최대 처리량 최적화는 미완료다.
- 양의 점수만 선택하고 동점은 카탈로그 순서(선박 ID 정렬)로 처리한다. K가 카탈로그 크기를 초과해도 처리한다.
- 학습 결과에 인덱스·특징·모델 fit 시간, 스레드·배치 설정을 기록한다. 추론 구간 시간은 `matcher.last_timings`, 인덱스 저장량은 `index.memory_info`로 확인한다. 저장량은 전체 프로세스 최고 RAM/VRAM이 아니다.

## 검색 알고리즘과 호환성

RapidFuzz와 TF-IDF는 서로 다른 검색 알고리즘이다. CUDA 사용 가능 여부나 인덱스 생성 실패에 따라 자동 전환하지 않는다.

새 학습은 `VESSEL_SEARCH_ALGORITHM=rapidfuzz`가 기본이다. `tfidf`로 명시하여 변경할 수 있으며 assets.json에 저장된다. 이는 AS-02의 자동 Torch 경로와 다른 설정일 수 있으므로 이전 OOF 점수를 그대로 승계하지 않는다.

기존 assets.json에는 실제 알고리즘 정보가 없다. 그대로 실행하면 명확한 오류가 발생한다. 이전 학습 환경·코드를 확인하여 `VESSEL_LEGACY_SEARCH_ALGORITHM=rapidfuzz` 또는 `tfidf`를 명시해야 한다. 기존 자산은 자동 수정하지 않는다. 이 설정은 저장된 명시적 알고리즘을 덮어쓰지 않는다.

TF-IDF는 기본 CPU이며 `VESSEL_SEARCH_DEVICE=cuda`로 GPU를 명시한다. CUDA가 없거나 인덱스 생성에 실패하면 다른 검색 방식으로 바꾸지 않고 실패한다. 과거 dense float16 GPU 검색과 새 sparse float32 검색의 후보가 완전히 같다고 보장하지 않는다. GPU 사용 전 별도 장치·행 독립성 검증이 필요하다.

## Windows PowerShell 설정 예시

```powershell
$env:VESSEL_SEARCH_ALGORITHM = 'rapidfuzz' # 새 학습에 저장
$env:VESSEL_SEARCH_DEVICE = 'cpu'
$env:VESSEL_FEATURE_WORKERS = '4'
$env:VESSEL_PREDICT_WORKERS = '4'
$env:VESSEL_TRAIN_THREADS = '4'
$env:VESSEL_SCORE_THREADS = '1'
$env:VESSEL_TRAIN_BATCH = '64'
$env:VESSEL_PREDICT_BATCH = '64'
```

제출에 필요한 검색 방식은 반드시 자산에 저장한다. 로컬 환경변수가 평가 서버에도 전달된다고 가정하지 않는다. 실행 설정을 변경하면 실제 적용 작업자 수·행 수·구간 시간을 함께 기록한다.

## 검증 범위

관련 단위 테스트 22개 통과. 합성 데이터에서 실제 소형 LightGBM을 사용해 RapidFuzz/TF-IDF 각각 단독·배치·역순·작업자 수 변경의 예측 일치를 확인했다. 기존 누수 차단 검사는 내부 파생 필드만 허용하도록 갱신하고 타깃·fold 전달 금지 검사를 유지했다.

합성 400행의 1회 비용 비교는 `project_log/performance_revision.json`을 참조한다. 작은 카탈로그·소형 모델이므로 대회 속도 향상률이나 최적 작업자 수의 근거로 사용하지 않는다. 전체 OOF·전체 데이터 성능·Windows·GPU는 이번에 실행하지 않았다.
