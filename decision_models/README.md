# 같은 후보·특징의 결정 모델 비교

CatBoost는 `catboost_model.py`, MLP는 `mlp_model.py`에 각각 독립 구현했다. 두 클래스의 API는 `fit(x, y, w, feature_names)`, `predict(x)`, `save(directory)`, `load(directory)`다. 다른 모델을 import하거나 학습하지 않는다. 공통 캐시를 한 번 생성하고 동일 파일을 두 모델에 전달한다. 증강 없음. 기존 제출 모델을 교체하지 않는다.

프로젝트 루트의 PowerShell에서 현재 승인된 `.venv`를 사용한다. CatBoost는 현재 환경에 없어 설치가 필요하다. 서버 패키지 버전 목록을 확인한 뒤 맞춰 설치한다. 아래 명령은 실행 예시이며 전체 비교 실험은 아직 실행하지 않았다.

```powershell
.\.venv\Scripts\python.exe -m decision_models.prepare_pairs --fold 0 --output artifacts/decision_pairs/fold_0
.\.venv\Scripts\python.exe -m decision_models.catboost_model train --pairs artifacts/decision_pairs/fold_0/train_pairs.npz --model artifacts/decision_models/catboost/fold_0
.\.venv\Scripts\python.exe -m decision_models.mlp_model train --pairs artifacts/decision_pairs/fold_0/train_pairs.npz --model artifacts/decision_models/mlp/fold_0
```

기준 분할은 배포된 G5-s42다. 각 fold의 학습 데이터에서만 조회 자산과 학습 쌍을 생성한다. 후보 설정은 RapidFuzz, S/P/T, 음성 검색 24, 학습 후보 한도 50이며 정답은 학습 쌍에만 보충한다. 추론 후보 한도는 30으로 두 모델에 동일하게 설정해야 한다. 후보·특징·가중치 파일 해시는 각 모델의 training.json에 기록된다. MLP의 평균·표준편차는 학습 쌍에서만 fit하고 저장하며 검증에서는 재계산하지 않는다. 모델별 최적화 설정은 다르므로 이번 비교는 이 고정 설정의 효과를 비교하는 것이며 모델 계열의 절대적 우열을 의미하지 않는다.

기존 검색과 추론 코드를 그대로 사용하는 방법:

```python
import os
from train import read_data
from pathlib import Path
from decision_models.mlp_model import MLPDecisionModel
from decision_models.matcher_adapter import DecisionMatcher
os.environ['VESSEL_PREDICT_CANDIDATES'] = '30'
train, catalog = read_data(Path('data'))
scorer = MLPDecisionModel.load('artifacts/decision_models/mlp/fold_0')
matcher = DecisionMatcher(catalog, 'artifacts/decision_pairs/fold_0/assets.json', scorer)
# fold 0의 실제 보류 행에서 QUERY_COLUMNS만 전달한다.
# pred, candidates, direct = matcher.predict_details(queries)
```

CatBoost를 사용하려면 scorer 클래스를 CatBoostDecisionModel로 바꾼다. 모델별 예측 후 후보 ID 목록과 순서가 동일한지 확인하고, 동일한 5-fold OOF를 공통 평가기로 비교한다. 단일 fold나 합성 테스트를 성능 비교로 사용하지 않는다. MLP는 배치 크기에 따른 점수 변화를 피하기 위해 각 후보에 고정된 연산 형태를 사용하므로 대량 추론 비용은 별도 측정해야 한다. ZIP 생성 기능은 없다.
