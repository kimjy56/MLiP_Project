# AR-06 모델 저장소

본 디렉터리는 규칙 기반 및 후보 검색 엔진의 최종 완성판인 **AR-06 (`1022-AR06`)** 모델 코드 및 아티팩트를 보관합니다.

---

## 1. 모델 핵심 정보
- **모델 버전**: `1022-AR06`
- **한글 처리 (`KR_MODE`)**: `AR06` (중국 병음 2음절 축약 + 여객선 35종 + 훈독/의역 사전 + 연속 약어 직결 + 고유사도 톤수 감점 완화)
- **비한글 처리 (`EN_MODE`)**: `EN_V3` (L-WIN 톤수창 RapidFuzz C++ OSA Top-40 + L-ORD 어순 무관 정렬 유사도 + L-EN0 미등장 감점 면제)
- **제출 형식 준수**: `train_AR-06` 패키지 구성 (외부 의존성 없이 `script.py` + `model/model.json`으로 단독 실행 가능)

---

## 2. 파일 구성
- `run_ar06.py`: 5-fold OOF 교차검증 및 전체 훈련/패키징 파이프라인
- `script.py`: 단독 추론 스크립트 (DACON 제출 샌드박스 규격)
- `model/model.json`: 학습된 메타데이터 (콜사인 룩업 5,851건, 선박명 룩업 5,857건, 단어/음절 매핑 및 선종/국적 로그 리프트)
- `train_AR-06/`: 제출 ZIP 패키징용 빌드 폴더
  - `script.py`
  - `requirements.txt`
  - `model/model.json`

---

## 3. 실행 방법

### OOF 평가 및 재학습 파이프라인 실행:
```bash
python -X utf8 ./AR-06/run_ar06.py --data-dir <데이터_경로> --eval-dir ./eval_kit
```

### 단독 추론 실행 (test.csv & vessel_catalog.csv 필요):
```bash
python ./AR-06/script.py
```
