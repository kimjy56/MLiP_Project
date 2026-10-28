# SPEC-LEX-01 Rev.2 구현·검증 기록

이 문서는 사용자의 EXAONE 단독 신규 사전 명세를 적용한 실제 구현과 미완료 조건을 구분한다. 기존 제출 ZIP/모델/완료 기록을 덮어쓰지 않았다. 명세의 목표 정확도는 목표이며 실측이 아니다.

## 현재 결과

- 기존 IDEA_B_WORD_VARS, catalog_full_ko_map.json, 학습 단어/음절 조회 자산을 로드하지 않는 새 파이프라인을 구현했다.
- 신규 기능14개 + 기존 decoder/catalog 관련18개, 총 **32개 검사 통과**.
- 실제 생성 사전이 필요한 BOW CECIL/CNC DREAM/KEOYOUNG STAR 3 진단 **3개는 skip**. 정답 문자열을 사전 생성·학습·후보 선택에 주입하지 않았다.
- 배포 fold0 학습9850행/보류2372행을 검증하고 새 학습 쌍1573개를 준비했다. 실제 모델 학습/사전 생성/정확도 평가는 완료하지 못했다.
- 로컬에 요청 EXAONE 가중치가 없고 공식 Hugging Face 모델 API 다운로드는 DNS 이름 해석 실패로 막혔다. 대체 모델이나 기존 사전을 사용하지 않았다.
- 제출 ZIP 생성/갱신, 서버 제출, 커밋/push 없음.

## 명세에서 바로잡은 전제

1. 기존 사전 배제는 계보를 명확하게 만드는 조치다. 이것만으로 모든 누수가 사라지거나 로컬 점수가 리더보드와 같아지지는 않는다. 분포 차이·모델 선택·전처리·사전 생성의 fit 범위를 계속 검사한다.
2. HD→현대, SS→특정 회사명은 일반적인 일대일 법칙이 아니다. 결정론적 코드는 HD→에이치디 같은 일반 철자 독음과 구식 로마자·숫자 대안만 생성한다. 회사명 의미 대응은 EXAONE의 학습/생성 영역이다.
3. 원시 프롬프트 형식은 유지하되 끝 토큰은 하드코딩한 endoftext 문자열 대신 실제 tokenizer.eos_token_id를 사용한다. EXAONE의 tokenizer와 다른 모델의 특수 토큰을 혼용하지 않는다.
4. batch-size64는 유효 배치이며 기본 물리 microbatch4로 누적한다. 생성 batch128도 CUDA OOM 발생 시 물리 microbatch만 줄이고 실제 값을 캐시에 기록한다.
5. 250개 자모 후보와 35개 Dense 후보의 합집합 포함률은 Recall@35가 아니다. 합집합 포함률과 명시적으로 순위를 매긴 상위35 포함률을 따로 출력한다.
6. 기존 DPR은 기존 학습 계보를 가진 별도 모델이다. 기본 평가/정적 추론은 EXAONE 사전만 사용한다. 선택적 hybrid 평가에는 명시적인 기존 DPR 사용 동의 플래그와 checkpoint/embedding/fold/fit hash 검증 manifest를 요구하고 EXAONE-only=False로 기록한다.
7. 정적 JSON 사용은 온라인 LLM 호출이0회라는 뜻이다. JSON 로딩·색인·검색·점수 계산 시간이나 파일 용량이0 또는5MB로 보장되는 것은 아니다.
8. 4개 후보로 모든 어절의 모든 발음 조합을 보장할 수 없다. 문맥 Top2와 중복되지 않는 어절 조합을 제한적으로 선택한다.
9. 실제 측정은 카탈로그78595행, 정규화 고유 이름78592개, 토큰35418개다. 이름이 같은 여러 선박 ID는 합치지 않고 카탈로그에서 별도 후보로 유지한다.
10. 지정 세 사례는 이미 제시된 공개 개발 사례다. 사후 생성 진단으로만 사용하며 독립 최종 검증이나 조정용 정답으로 사용하지 않는다.

## 주요 구현 파일

| 파일 | 역할 |
|---|---|
| name_decoder/deterministic_ko_rules.py | 일반 알파벳 독음, 숫자/서수/로마 숫자, 보수적 구식 로마자 대안, 새 자모 정규화 |
| name_decoder/exaone_data.py | 공식 CSV/fold 검사, fit 한글 이름 쌍·동일 어절 수의 단어 쌍·일반 규칙 쌍 생성 |
| name_decoder/exaone_translit.py | EXAONE4 확인, 7개 projection의 rank24/alpha48/dropout0.05 LoRA, tokenizer EOS, HangulOnlyLogitsProcessor |
| name_decoder/exaone_lexicon.py | 단어 후보 최대3개, 제한된 Cartesian beam, 전체 이름 Top2와 어절 후보 다양성 병합 |
| experiments/exaone_common.py | 새 ID 예약·실패/차단 기록·원자 JSON·코드/데이터 해시와 실행 소스 snapshot |
| experiments/train_exaone_translit.py | fold별/전체 학습, 유효 batch64, 마지막 epoch 선택, adapter/기반 파일 해시 저장 |
| experiments/build_exaone_ko_lexicon.py | Track1/Track2 생성, hash-bound batch cache/resume, 최대4 후보 신규 map/manifest |
| experimental_matchers/exaone_lexicon.py | 기존 엔진을 import하지 않는 자모 검색·정적 JSON 추론·CSV 출력 CLI |
| experiments/evaluate_pure_exaone_oof.py | fold별 map provenance 거부/검사, 별도 Recall@35/합집합 포함률, 공통 평가기 호출 |
| tools/fetch_exaone.py | 공식 모델 revision을 고정한 다운로드 시도; 대회 데이터 접근 없음 |
| tests/test_exaone_lexicon.py | 기능·분리·로딩·정적 CLI·행 독립 검사14개 |
| tests/test_exaone_generated_acceptance.py | 실제 map 생성 후 실행할 지정 세 사례 진단 |

name_decoder/__init__.py만 기존 파일에서 수정했다. 공개 API 이름을 유지하면서 필요할 때 해당 모듈을 불러오도록 변경해 정적 사전 추론이 torch/transformers를 import하지 않게 했다. 기존 decoder/catalog 관련18개 검사가 통과했다. 원본은 artifacts/IMPL-EXAONE-LEX-01/before/에 보존했다.

기존 AS-06/07/08 스냅샷을 수정하거나 새 엔진으로 덮어쓰지 않았다. 새 matcher의 점수 규칙은 독립 실험 구현이며 기존 AS-06과 수치 동등하다고 주장하지 않는다. 현재 이름 점수는 최대 자모 유사도×10에 숫자·톤수·동일 문자열 메타 점수를 더한다. 기존 train lookup/학습 전이표를 불러오지 않는다. 후속 정확도 비교에서 이 새 scorer의 효과도 분리해야 한다.

## 데이터와 누수 검사

공식 파일의 현재 로컬 경로는 저장소의 data/이다. 배포 eval_kit/folds_group5_seed42.csv를 읽고 sample_id·target ID·vessel-group 일치를 검사한다. fold를 재생성하지 않는다.

| 항목 | 실측 |
|---|---:|
| 외부 fold0 fit 행 | 9850 |
| 보류 행 | 2372 |
| 신규 고유 학습 쌍 | 1573 |
| 일반 규칙 쌍 | 976 |
| 어절 정렬 쌍 | 328 |
| 전체 이름 쌍 | 269 |
| 카탈로그 행 / 고유 정규화 이름 | 78595 / 78592 |
| 카탈로그 고유 토큰 | 35418 |

쌍 종류는 동일 source/target 중복을 제거한 뒤의 분류다. 어절 수가 같다는 사실만으로 의미 정렬이 항상 정확한 것은 아니다. 출력 형식에 맞지 않는 혼합 문자 신고명은 현재 보수적으로 제외한다. 로컬 점수로 선택하지 않은 데이터 정제 정책이며 후속 검토 대상이다.

검증 선박 ID와 fit 선박 ID 교집합은 없다. 검증 행의 이름을 바꿔도 파생 학습 쌍이 변하지 않는 합성 검사도 통과했다. 검증 정답은 예측·후보·점수가 확정된 뒤에만 점수/순위 진단에 사용한다.

학습은 마지막 epoch를 사전 고정해 선택하며 외부 fold를 early stopping이나 best checkpoint 선택에 쓰지 않는다. 모든 카탈로그 이름을 생성 입력으로 쓰는 것은 공식 무라벨 카탈로그 사용이다. 검증 신고명을 생성 입력/후보 교정에 사용하지 않는다.

map manifest는 train/catalog/fold·fit/heldout ID 집합·학습 쌍·base·adapter·생성 코드·설정·배치 소스 순서의 해시를 연결한다. 평가 시 다른 fold map, full-train map, 바뀐 map/catalog, provenance가 없는 기존 map을 거부한다. 5-fold는 서로 다른5개 fold map이 필요하다.

## 생성과 실행 의미

- 제약 디코딩은 개별 토큰을 실제로 decode해 한글 완성형/ASCII 숫자/공백만 포함하는 토큰을 허용한다. EOS는 tokenizer에서 얻고 첫 토큰 EOS/공백만 출력은 막는다. 영문·줄바꿈·replacement 문자 토큰은 제외한다.
- 마지막 decode 결과도 다시 검사하며 EOS 없이 잘린 출력은 버린다. 빈 출력이나 낮은 커버리지를 정답으로 채우지 않는다.
- 빔 출력 순위는 기록하되 이를 확률/보정된 신뢰도로 표현하지 않는다. 거대한 step별 전체 vocabulary score 보관을 피하기 위해 출력 순위를 사용한다.
- batch cache는 입력 순서·설정·모델/소스 해시가 일치할 때만 resume한다. 완료 사전/manifest를 덮어쓰지 않는다.
- 생성 사전의 이름 키는 고유78592개가 될 수 있다. matcher는 원래78595개 vessel ID에 후보를 확장하고 동점은 ID로 처리한다.
- 정적 추론은 신규 JSON만 읽는다. 기존 사전·EXAONE/DPR 가중치·온라인 네트워크 호출은 기본 경로에 없다.

## 실제 실행 상태

| ID | 상태 | 의미 |
|---|---|---|
| IMPL-EXAONE-LEX-01 | 구현·검증 완료 / 실제 모델 실험 차단 | 이 구현 작업 |
| EXAONE-DATA-AUDIT-01 | completed_data_audit | 최초 실제 데이터 분리·집계 |
| EXAONE-DATA-AUDIT-02 | completed_data_audit | 해시와 소스 snapshot을 포함한 후속 데이터 확인 |
| LEX-EXAONE-PURE-01 | blocked_missing_base | 데이터 준비 후 EXAONE base 부재로 학습 시작 전 중단 |
| EXAONE-BASE-FETCH-01 | blocked_network_or_missing_base | huggingface.co DNS 실패. 모델 revision/가중치 미수령 |
| LEX-EXAONE-GENERATE-01 | planned, 의존성 차단 | 실제 adapter가 없어 미실행 |
| EVAL-EXAONE-PURE-01 | planned, 의존성 차단 | 실제 생성 map이 없어 미실행 |

최초 데이터/학습 preflight는 실행 소스 해시를 남겼으나 개별 source snapshot은 없었다. 이후 공통 실행기에 source snapshot 저장을 추가했고 DATA-AUDIT-02부터 적용했다. 초기 기록을 다른 설정으로 재표기하지 않았다.

32개 검사 통과에는 작은 무작위 EXAONE4 구조의 LoRA gradient/adapter 복원 검사와 합성 map의 정적 CLI/행 독립 검사가 포함된다. 이것은 실제 EXAONE-4.0-1.2B의 학습·생성 품질·성능을 확인한 결과가 아니다.

검사 확대 첫 실행에서 기존 test_name_decoder의 기본 temp 디렉터리 권한 오류1개가 있었다. 기대값/허용오차는 그대로 두고 검증된 scratch 하위 TEMP/TMP 경로에서 재실행해 통과했다. 지정 세 실제 생성 진단은 map이 없어 skip이며 성공으로 집계하지 않는다.

현재 지정 .venv-server는 Python3.11.15로 목표3.11.17과 다르고, 지정 패키지5개의 불일치/누락은 앞선 환경 검사에 기록돼 있다. 설치 환경을 변경하지 않았다. TEAM_RULES 실제v4, eval_kit README 실제1.0/사용자 적용 기준1.1 차이를 유지한다. 이 결과를 서버 환경 완전 호환성으로 표현하지 않는다.

## 이후 실행 명령

아래는 PowerShell, 프로젝트 루트 기준이다. 실패한 ID를 재사용하지 않기 위해 학습은 새 ID02를 사용한다. 다운로드가 가능한 환경 또는 이미 준비한 요청 모델의 로컬 경로가 필요하다. 원본 대회 데이터의 채팅 업로드는 필요하지 않다.

다운로드 재시도:

    .\.venv-server\Scripts\python.exe -B -X utf8 tools/fetch_exaone.py --experiment-id EXAONE-BASE-FETCH-02

또는 공식 모델 snapshot의 config/tokenizer/safetensors를 준비한 경로를 학습의 --local-model에 지정한다. 자동 대체 모델은 없다.

실제 학습:

    .\.venv-server\Scripts\python.exe -B -X utf8 experiments/train_exaone_translit.py --experiment-id LEX-EXAONE-PURE-02 --model-id LGAI-EXAONE/EXAONE-4.0-1.2B --fold 0 --epochs 4 --batch-size 64 --microbatch-size 4 --offline

사전 전수 생성:

    .\.venv-server\Scripts\python.exe -B -X utf8 experiments/build_exaone_ko_lexicon.py --experiment-id LEX-EXAONE-GENERATE-01 --checkpoint artifacts/LEX-EXAONE-PURE-02/fold_0_lora --output-word-lex artifacts/LEX-EXAONE-GENERATE-01/word_exaone_lexicon.json --output-catalog-map artifacts/LEX-EXAONE-GENERATE-01/catalog_exaone_ko_map.json

중단 생성은 동일 코드/모델/설정에서만 위 명령에 --resume를 추가한다. 파일/해시가 다르면 새 ID를 사용한다.

fold0 평가:

    .\.venv-server\Scripts\python.exe -B -X utf8 experiments/evaluate_pure_exaone_oof.py --experiment-id EVAL-EXAONE-PURE-01 --ko-map artifacts/LEX-EXAONE-GENERATE-01/catalog_exaone_ko_map.json --fold 0

5-fold는 각 fold별로 따로 학습/생성한 map을 --all-folds --map-pattern으로 지정한다. 하나의 fold0 map을 전체 OOF에 재사용할 수 없다. 기존 DPR 병합은 --dense-mode hybrid --allow-existing-dpr와 checkpoint/embedding 및 --dense-provenance로 검증된 fold/fit/hash manifest 명시가 필요하며 EXAONE-only 결과와 구분한다.

생성 사례 진단:

    $env:EXAONE_ACCEPTANCE_MAP='artifacts/LEX-EXAONE-GENERATE-01/catalog_exaone_ko_map.json'
    .\.venv-server\Scripts\python.exe -B -X utf8 -m unittest tests.test_exaone_generated_acceptance -v

정적 JSON 추론:

    .\.venv-server\Scripts\python.exe -B -X utf8 experimental_matchers/exaone_lexicon.py --ko-map <실제_생성_map_경로> --test-csv <실제_입력_CSV_경로> --catalog-csv data/vessel_catalog.csv --output-csv output/submission.csv

테스트는35개 중32개 통과/3개 skip이며 실제 카탈로그 사전 생성 후에만 지정 사례와 목표 Recall@35≥99%, Top1≥97.5%를 측정할 수 있다. 제출 checker는 ZIP을 새로 만들지 않았으므로 이번에는 실행하지 않았다.

## 근거와 다음 작업

- [구현 실험 기록](../project_log/experiments/IMPL-EXAONE-LEX-01.json)
- [최종 기능 검사 로그](../reports/exaone_lexicon/IMPL-EXAONE-LEX-01/tests_complete.txt)
- [공식 데이터 집계](../artifacts/EXAONE-DATA-AUDIT-02/data_manifest.json)
- [학습 차단 기록](../project_log/experiments/LEX-EXAONE-PURE-01.json)
- [다운로드 차단 기록](../project_log/experiments/EXAONE-BASE-FETCH-01.json)
- [공식 모델 카드](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B)
- [공식 tokenizer 설정](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/raw/main/tokenizer_config.json)

가중치 확보 후의 첫 작업은 실제 base/tokenizer 로딩과 작은 생성 기능 확인이다. 이후 고정4epoch 학습→전수 생성→fold 평가 순서로 진행한다. 목표를 맞추기 위해 알려진 보류 사례를 규칙/사전에 추가하지 않는다.

권장 커밋은 feat(exaone): add fold-isolated legacy-free lexicon pipeline / test(exaone): verify constraints provenance and static inference / docs(exaone): record blocked base acquisition and execution handoff 이다. 기존 미커밋 변경을 함께 stage하지 않도록 이번 변경만 검토한다.


## 동시 변경 인계

최종 해시 대조 중 이번 작업이 수정하지 않은 phonetic_encoder의6개 소스와 Bi-Encoder 학습/평가 스크립트의 동시 변경을 확인했다. 이 파일은 되돌리거나 이번 구현으로 집계하지 않았다. 변경 전/관측 후 해시는 reports/exaone_lexicon/IMPL-EXAONE-LEX-01/concurrent_changes.json에 분리했다.

이 변경은 기본 EXAONE-only 경로의 의존성이 아니다. 선택적 기존 DPR 경로는 해당 최신 코드와의 동작 일치/성능을 이번에 검증한 것으로 취급하지 않는다. 제가 수정한 기존 실행 소스는 name_decoder/__init__.py이며, 그 외의 EXAONE 파일은 새 파일이다.


동시 변경의 후속 검사에서 submit_AS-08.zip·submit.zip·기존 AS-08.json·train_AS-08 일부 소스도 변경됐다. 따라서 현재 작업공간 전체의 과거 ZIP/완료 기록 해시 보존 검사는 통과하지 않았다. 이 EXAONE 작업은 해당 파일을 생성/수정/되돌리지 않았으며, 이전/현재 해시는 위 concurrent_changes.json에 보존했다. AS-08 공식 점수는 이전 a2e6e6d380910a7cdfa99587135d04e59af4e7844b6666c6f47aa2b63e5a6d9e ZIP 기록과 연결된 과거 결과다. 현재 같은 파일명의 새 해시를 그 제출 원본이라고 취급하지 않는다.
