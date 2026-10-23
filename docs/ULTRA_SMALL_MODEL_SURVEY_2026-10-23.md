# 초소형 언어모델 조사 — 선박 정보 매칭

조사·기록일: 2026-10-23 KST. 상태: 문헌 조사 완료, 구현·학습·성능 비교 미실행.

## 범위와 판단 기준

사용자 요청에 따라 주로 1B 이하 공개 모델을 조사했다. 생성형 모델, 이름 변환용 encoder-decoder, 검색용 encoder는 역할이 다르므로 하나의 벤치마크 순위로 합치지 않는다. 현재 집합 모델에 추가하거나 교체하기로 결정한 것은 아니다.

한글 가중치 0.7, 고유명사의 철자·숫자·음역 구별, 학습에 없던 선박, 추론 비용을 우선한다. 아래 용도는 모델 구조와 공개 자료에서 도출한 **적용 가설**이며 선박 매칭에서 검증한 결과가 아니다. 공개 일반 벤치마크를 우리 OOF 또는 Public 점수로 환산하지 않는다.

## 우선 검토 후보

| 모델 | 규모와 최신성 | 제안 용도 | 판단과 한계 |
|---|---|---|---|
| LFM2.5-350M / Base | 350M, 공식 발표 2026-03-31 | 한 행의 이름·메타정보 구조화, 관측 표기의 조건부 점수화 | 생성형 초소형 후보 중 우선 비교. 한국어 포함 9개 언어 명시. 일반 지식보다 구조화 출력·지시 수행에 적합하다는 개발사 설명. 독자 라이선스 |
| HyperCLOVAX-SEED-Text-Instruct-0.5B | 실제 총 570M, 2025년 공개 모델 | 한국어 표기 해석·음역 및 이름 쌍 판별 파인튜닝 | 한국어 벤치마크 근거가 명확한 후보. 영문 고유명사 매칭 우위는 미확인. 독자 라이선스 |
| Qwen3.5-0.8B / Base | 언어모델 부분 0.8B, 2026년 계열 | 범용 다국어 소형 모델 비교군, 조건부 점수화·쌍 판별 | Apache-2.0. 비전 모듈을 포함한 전체 자산 크기와 언어모델 크기를 구분. 최신 런타임·오프라인 로딩 확인 필요 |
| ByT5-small | 약 300M, 2021년 계열 | 카탈로그 이름 → 관측 이름의 바이트 단위 조건부 우도, 표기 변환 | 철자·오타 중심의 독립 설계에서는 우선 비교할 후보. UTF-8 바이트라 한글 입력 길이가 늘고 생성 비용이 생김. 사전학습 모델이므로 과제 파인튜닝 필요 |
| EmbeddingGemma 2 | 텍스트 270M, 전체 740M, 2026-10-06 공개 | 기록과 카탈로그를 각각 벡터화하는 독립 검색 모델 | 최신 검색형 후보. 이미지·음성 모듈 선택 로딩 가능. 의미 유사성과 선박 동일성은 다르므로 유사 이름·번호가 다른 선박을 음성 쌍으로 구별하는 학습 가설이 필요 |
| T5Gemma 2 270M-270M | 텍스트 약 370M, 비전 제외; 2025-12-18 공개 | ByT5와 비교할 encoder-decoder 이름 변환·조건부 우도 모델 | 모델명 두 숫자를 단순 합산한 540M이 아님. 공개판은 사전학습 모델. Gemma 라이선스와 새 모델 구현의 호환성 확인 필요 |
| paraphrase-multilingual-MiniLM-L12-v2 | 약 0.1B급, 기존 세대 | 저비용 다국어 벡터 검색 기준선 | 한국어 포함, 384차원 표현, Apache-2.0. 가장 작은 실용 비교군이나 선박 이름·번호 구별 성능의 근거는 없음 |

우선순위는 목적별로 잡는다. **생성형의 크기 대비 활용성은 LFM2.5-350M**, **한국어 중심은 HyperCLOVA X 0.5B**, **철자·음역 변형 학습은 ByT5-small**, **최신 검색형은 EmbeddingGemma 2**를 먼저 비교할 근거가 있다. 전체 과제에서 이들 중 누가 이기는지는 미측정이다.

출처: [LFM 모델 카드](https://huggingface.co/LiquidAI/LFM2.5-350M), [LFM 발표](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind), [HyperCLOVA 모델 카드](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B), [Qwen 모델 카드](https://huggingface.co/Qwen/Qwen3.5-0.8B), [ByT5 모델 카드](https://huggingface.co/google/byt5-small), [ByT5 원저자 저장소](https://github.com/google-research/byt5), [EmbeddingGemma 2 모델 카드](https://huggingface.co/google/embeddinggemma-2), [Google 출시 목록](https://ai.google.dev/gemma/docs/releases), [T5Gemma 2 발표](https://blog.google/innovation-and-ai/technology/developers-tools/t5gemma-2/), [MiniLM 모델 카드](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).

## 성능 근거와 비교 한계

### LFM2.5-350M과 Qwen3.5-0.8B

Liquid AI가 같은 공개 비교표에서 보고한 값이다. 우리의 재현 결과가 아니며 다른 기관의 평가 수치와 섞지 않는다.

| 모델 | IFEval | IFBench | MMLU-Pro |
|---|---:|---:|---:|
| LFM2.5-350M | 76.96 | 40.69 | 20.01 |
| Qwen3.5-0.8B (Instruct) | 59.94 | 22.87 | 37.42 |

이 비교에서는 LFM이 지시 수행, Qwen이 지식·추론 평가에서 높다. 따라서 LFM을 모든 용도에서 더 강하다고 해석하지 않는다. Qwen 자체 카드의 비사고 모드 MMLU-Pro는 29.7로 Liquid 보고와 다르므로 평가 설정의 영향을 확인해야 한다. [Liquid 비교표](https://huggingface.co/LiquidAI/LFM2.5-350M#-performance), [Qwen 자체 평가](https://huggingface.co/Qwen/Qwen3.5-0.8B).

### HyperCLOVA X 0.5B

NAVER 공개 5-shot 평가에서 Instruct-0.5B는 KMMLU 38.15, KoBEST 62.99이며, 같은 표의 Qwen2.5-0.5B-Instruct는 각각 29.68, 50.25다. 한국어 후보로 고를 근거지만 최신 Qwen3.5보다 우수하다는 증거는 아니다. 카드에는 Base-0.5B 점수도 있으나 공개 다운로드 가능 여부를 별도로 확인하지 않아 Base판을 사용 가능한 후보로 확정하지 않았다. [NAVER 평가표](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B#benchmarks).

### EmbeddingGemma 2

Google 공개 MTEB multilingual v2 평균은 61.36, 전작은 61.15다. 다국어 텍스트 평균 차이는 0.21점이며 코드 검색에서의 큰 개선을 선박명 검색 개선으로 옮겨 해석하지 않는다. 공식 문서는 100개 이상 언어를 명시하지만 이번 조사에서 한국어 선박명 또는 한국어 단독 점수는 확인하지 않았다. [Google 평가표](https://huggingface.co/google/embeddinggemma-2#benchmark-results).

## 추가로 검토했으나 우선순위를 낮춘 후보

- **LFM2.5-Encoder-230M/350M**: 생성형 LFM2.5-350M과 별개다. 개발사 17개 과제별 파인튜닝·5시드 평균에서 350M판은 81.02, XLM-R base는 77.46이다. 그러나 명시된 15개 지원 언어에 한국어가 없어서 한글 0.7 과제의 우선 후보로 올리지 않았다. 한국어 처리가 불가능하다는 뜻은 아니다. `trust_remote_code` 경로와 코드 번들도 검토 대상이다. [공식 카드](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M).
- **Gemma 3 270M**: 더 작은 생성형 비교군이지만 현재 과제의 한국어·고유명사 판별 우위를 확인하지 못했다. T5Gemma 2와 역할을 구분한다. [공식 카드](https://huggingface.co/google/gemma-3-270m-it).
- **SmolLM2-135M/360M**: 매우 작지만 영어 중심이어서 한국어 우선 실험 순위는 낮다. [공식 360M 카드](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).
- LFM2.5-1.2B-Instruct 같은 1B 초과 모델은 후속 비교군이다. 이번 초소형 표에서는 제외했다. [공식 카드](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct).

## 대회 규칙·라이선스·실행 환경

공개 범용 가중치를 공식 train.csv·vessel_catalog.csv만으로 추가 학습하는 방식은 규칙 문구상 사용 후보에 해당한다. 모델별 운영진 승인이나 제출 실행 검증 완료를 의미하지 않는다. 외부 선박 정답·매핑 추가, 테스트 행 간 정보 공유는 금지된다. 일반 벤치마크 보고서를 읽은 것을 외부 벤치마크 데이터로 우리 모델을 추가 학습·검증한 것으로 혼동하지 않는다. [공식 규칙](https://dacon.io/competitions/official/236771/overview/rules), [기존 규정 검토](../project_log/reviews/2026-10-23_language_model_rules.md).

- **Apache-2.0**: Qwen3.5, ByT5, EmbeddingGemma 2, 위 MiniLM. 배포 시 라이선스·필요 고지 보존.
- **LFM Open License v1.0**: 수정·배포 권한은 조건부로 부여되며, 상업 사용에는 연 매출 1,000만 달러 기준 제한이 있다. 제출 번들에는 라이선스·변경 표시·기존 고지를 보존해야 한다. 개인/팀/기관 중 실제 사용 주체 조건을 대조해야 한다. [LFM 라이선스](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE).
- **HyperCLOVA X SEED**: 배포 시 약관·NOTICE·변경 표시, 문서의 Powered by HyperCLOVA X 표시 및 파생 모델 명칭 조건 등이 있다. 1,000만 MAU와 NAVER 경쟁 서비스 관련 추가 조건도 있으므로 Apache와 동일하게 취급하지 않는다. [NAVER 라이선스](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B/blob/main/LICENSE).
- **Gemma 3 / T5Gemma 2**: Gemma 약관 적용. EmbeddingGemma 2의 Apache-2.0과 혼동하지 않는다. [T5Gemma 2 카드·약관 링크](https://huggingface.co/google/t5gemma-2-270m-270m).

서버 기본 transformers 4.57.6에서 새 모델이 곧바로 로드된다고 보장하지 않는다. Qwen3.5·EmbeddingGemma 2·T5Gemma 2는 특히 아키텍처 지원·설치 시간을 확인해야 한다. 본 조사에서는 어떤 모델도 다운로드·설치·실행하지 않았다. 350M FP16 가중치만의 산술 크기는 약 0.70GB지만 실제 RAM/VRAM에는 활성값·작업 공간·KV 캐시 등이 추가되므로 이를 실측 메모리나 ZIP 크기로 표현하지 않는다.

추론은 완전 로컬 번들로 실행하며 기본 가중치·토크나이저·설정을 포함한다. 최종 ID가 입력 행의 배치·순서·부분집합 여부에 따라 바뀌지 않아야 한다. 한 행의 여러 카탈로그 후보를 비교하는 것은 다른 테스트 행의 정보를 사용하는 것과 다르다. 가중치가 작다는 사실만으로 L4에서 전체 1분 미만 실행을 보장할 수 없다. [공식 실행 조건](https://dacon.io/competitions/official/236771/overview/evaluation).

## 후속 비교 설계 — 미실행 제안

목적을 먼저 하나 정하고 서로 다른 계열을 비교한다. 이름 변환·조건부 우도라면 ByT5-small과 소형 생성형, 다국어 벡터 검색이라면 MiniLM과 EmbeddingGemma 2를 비교한다. 공식 학습 fold 안에서만 학습·튜닝하고 동일 G5-s42 분할에서 한글/비한글·가중 점수와 새 선박 성능을 기록한다. 검색 모델 교체라면 고정 K 후보 포함률과 최종 ID 정확도를 함께 보며, 점수화 모델 비교라면 후보를 고정한다. 모델 크기·추론 구간별 시간·행 독립 검증을 함께 기록한다. G5 결과를 Public 개선 보장으로 해석하지 않는다.

이번 작업의 성능·속도·메모리 측정값은 모두 **미측정**이다. 학습 실험이 아니므로 실험 JSON·OOF·제출 ZIP은 생성하지 않았다.
