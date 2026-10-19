# 모델 코드와 학습 산출물

기존 학습·추론 경로를 유지하면서 같은 폴더 안에서 파일 역할로 구분한다.

| 파일 | 역할 | Git 동기화 |
|---|---|---|
| matcher.py, name_processing.py, ngram_retrieval.py | 모델 코드 | 포함 |
| README.md | 파일 배치 안내 | 포함 |
| assets.json | 학습으로 생성한 사전·설정 | 제외 |
| ranker.txt | 학습한 LightGBM 모델 | 제외 |

현재 PC에는 코드만 있다. 산출물은 로컬 학습으로 만들거나 승인된 공유 공간에서 받아 이 폴더에 둔다. 두 산출물의 실험 ID와 해시를 대조하고 서로 다른 실험의 파일을 섞지 않는다. 검증용 산출물은 별도 validation_model/에 저장되며 Git에서 제외된다.