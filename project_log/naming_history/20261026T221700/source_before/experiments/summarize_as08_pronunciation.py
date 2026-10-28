"""Summarize completed AS08 generation and fixed-engine comparison records."""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT=Path(__file__).resolve().parents[1]


def load(rel):
    return json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))


def main():
    generated=load('project_log/experiments/AS08-KO-DIVERSE-01.json')
    filtered=load('project_log/experiments/AS08-KO-OWNERSHIP-01.json')
    expanded=load('project_log/experiments/EVAL-AS08-KO-DIVERSE-01.json')
    guarded=load('project_log/experiments/EVAL-AS08-KO-OWNERSHIP-01.json')
    if expanded['status']!='completed_not_promoted' or guarded['status']!='completed_not_promoted':
        raise ValueError('Comparisons not complete')
    stats=generated['diagnostics'];guardstats=filtered['diagnostics']
    lines=['# AS-08 합성 발음 보강·역검색 필터 실험 결과','',
        '기록 시각: '+datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),'',
        '**결정: 두 후보 모두 기본 적용하지 않고 AS-06을 유지한다.**', '',
        '실서버에서 새로 평가한 결과가 아니다. 비교 기준은 사용자 제공 AS-06 제출115506의0.9122153056이며, 이번 측정은 이미 개발에 사용한 공개 train의 fold0 보류2,372행(한글239 / 비한글2,133)에 대한 원본·합성 노이즈 실험이다.', '',
        '## 생성·후처리', '',
        f"- 기존 발음이 하나 이하인 이름{stats['scheduled_names']:,}개를 정답 정보 없이 선택했다. 범위는 정확도 평가 전에 기존 사전 커버리지와 실행 비용을 기준으로 정했다. 전체78,592개 이름을 새로 생성한 실험이 아니다.",
        '- 같은 LFM2.5-350M 가중치·V2 prompt를 고정하고 전체 생성 과정의 sampling을 사용했다. 첫 토큰만 분기하던 기존 방식의 후반 발음 탐색 한계를 보완하려는 가설이다.',
        '- seed20261026, batch64, 이름당4샘플, temperature0.7, top_p0.9, top_k0, 최대32새token. 기존 발음은 유지하고 최대4개까지 저장했다.',
        '- EOS 종료, 한글 출력, 숫자 signature 보존, 원자모 중복, 기존 발음과 유사도0.55 이상, 샘플 내 상대 logprob0.75 이내 규칙을 적용했다. 이 수치는 외부 fold 결과로 튜닝하지 않았다. sampling 후 logprob은 정답 확률이나 teacher-forced NLL이 아니다.',
        f"- 보강: {stats['names_with_added_variants']:,}개 이름에{stats['added_variants']:,}개 발음 추가. 단일 발음 항목4,348→{stats['variant_count_histogram'].get('1',0):,}. 기존 발음은 모두 보존했다.",
        f"- 후속 역검색 필터: 추가{guardstats['checked_additions']:,}개 중{guardstats['removed_additions']:,}개 제외, {guardstats['accepted_additions']:,}개 유지. 정규화 자모·같은 숫자 signature 그룹에서 원래 소유 이름이 유일한 최근접인 경우만 허용했다. 검증 정답이나 test 통계는 사용하지 않았다.",
        '- 역검색 필터는 첫 보강 실험의 집계 결과를 본 뒤 제안한 후속 개발 실험이다. 독립 확증 실험으로 해석하지 않는다.', '',
        '## 동일 점수식 비교', '',
        'AS-06의 legacy 점수와 조회 OFF를 고정했다. 바뀐 것은 사전이다. 후보 집합은 사전 변경에 따라 달라질 수 있으므로 고정 후보 실험이라고 부르지 않는다.', '',
        '| 조건 | AS-06 한글 정답 | 발음 보강 | 역검색 필터 | 보강 고침/망침 | 필터 고침/망침 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for phase in ['clean','default','severe','severe_20261026','severe_20261027']:
        a=expanded['phase_results'][phase];b=guarded['phase_results'][phase]
        x=a['paired']['hangul'];y=b['paired']['hangul']
        lines.append(f"| {phase} | {a['old']['hangul_correct']}/239 | {a['new']['hangul_correct']}/239 | {b['new']['hangul_correct']}/239 | {x['fixes']}/{x['breaks']} | {y['fixes']}/{y['breaks']} |")
    lines += ['',
        '모든 조건에서 비한글 예측은 동일했다. 원본에서는 두 후보 모두 예측이 같았지만 기본 노이즈에서는 각각 한글1건을 새로 틀렸다. 보강 후보의 severe20261026 한글+1은 다른 시드에서 일관되게 재현되지 않았다. 역검색 필터는 해당 이득도 보존하지 못했다.', '',
        '| 조건 | 보강 가중 점수 차이 (%p) · 95% CI | 필터 가중 점수 차이 (%p) · 95% CI |',
        '|---|---:|---:|',
    ]
    for phase in ['clean','default','severe','severe_20261026','severe_20261027']:
        cells=[]
        for run in [expanded,guarded]:
            ci=run['phase_results'][phase]['official']['compare']['boot']['w']
            cells.append(f"{ci['est']*100:+.3f} [{ci['lo']*100:+.3f}, {ci['hi']*100:+.3f}]")
        lines.append('| '+phase+' | '+' | '.join(cells)+' |')
    lines += ['',
        '공식 평가기 subset 짝지은 비교, 선박 단위 bootstrap2,000회 seed0. 모든 CI가0을 포함한다. 따라서 통계적으로 악화가 확정됐다고 쓰지 않으며, 안정적인 개선 근거가 없고 일부 점 추정치가 낮아 채택하지 않는다. 같은 선박에 대한 여러 노이즈 변형을 독립 표본으로 합산하지 않는다.', '',
        '## 관측 사실과 해석', '',
        '- 이번 다섯 조건에서는 한글 정답의 전체 가변 후보군 포함률이 모두100%였다. 새 사전의 한글 득실은 검색 복구가 아니라 후보 선택 변화였다. 비공개 test에도 검색 누락이 없다는 뜻은 아니다.',
        '- 사전 보강으로 새 발음이 생기면 오답 후보의 max 유사도도 올라갈 수 있다. 발음 개수 증가를 정확도 향상으로 간주할 수 없다.',
        '- 역검색 필터와 보강 후보를 행별 비교하면 severe20261022와 severe20261026에서 각각 보강 후보가 맞히던1개 변형 행을 필터 후보가 틀렸다. 해당 정답 카탈로그 이름의 추가 발음이 실제 제거돼 있었다. 이 동시 관측만으로 전체 점수 역전의 유일한 원인을 입증한 것은 아니다.',
        '- 남은 사전 누락은6개다. 이 중3개는 숫자로만 된 카탈로그 이름이며, 나머지3개는 영문을 포함한다(공개 카탈로그의 최소 예:383-B,390-B). 사전 누락6개를 평가 오답6건으로 해석하지 않는다.', '',
        '## 검증·비용·재현', '',
        '- 품질/다양성 테스트7개와 역검색 ownership 테스트5개 통과. 숫자 그룹, 중복, 기존 발음 보존, 경쟁 이름·동률, 기존 소유자의 여러 hit가 경쟁자를 숨기지 않는지 검사했다.',
        '- 원본 사전 기준 예측 캐시는 입력·fit 모델·파일 해시, 과거 Git 소스의 universal-newline 동일성, 조건별64행 예측·정답 순위 동일성 검사 후 재사용했다. 각 후보의 전체2,372행은 새로 예측했고 원본 캐시와 공식 평가기로 비교했다.',
        '- 두 후보 모두 공개 보류8행의 순서 역전·단독 예측 차이0. 배포용 ZIP 검사를 실행한 것은 아니다.',
        f"- CUDA 생성 전체{generated['cost']['total_seconds']:.2f}초, 최대 allocated VRAM{generated['cost']['peak_cuda_allocated_bytes']/2**30:.2f}GiB. 로딩/해시/생성/품질 검사/저장을 포함하며 신경망 학습은 없었다.",
        f"- 역검색 필터는 CPU {filtered['cost']['filtering_seconds']:.2f}초(로딩·색인·검사·저장 포함). 모델 가중치·검색 사전·ZIP 용량은 구분하며, ZIP은 생성하지 않았다. 로컬 시간과 서버 표시6분6초는 직접 비교하지 않는다.",
        '- Python3.11.15, torch2.7.1+cu128, RTX5080, CPU threads2. 목표 서버 Python3.11.17과 차이가 있다.',
        '- 생성 manifest에 모델/tokenizer/prompt/코드/입력/설정 해시와 배치별 seed·필터 결과, 최종 사전 해시를 기록했다. 과거 실행 소스와 최신 CLI 개선판이 다를 때는 해시가 일치하는 실행 당시 사본을 artifacts/source_snapshot 경로에 보존했다.',
        '- 기존 AS-06 사전·엔진은 수정하지 않았다. 행별 예측·대표 사례·생성 자산은 Git 제외 artifacts 경로에 두고 JSON 기록에 경로·해시를 남겼다. 비공개 평가 입력·개별 예측은 사용하지 않았다.', '',
        '## 한계와 다음 방향', '',
        '생성기는 fold0을 제외한 기록으로 학습됐지만 수동 사전·과거 baseline map의 생성 이력 한계는 남아 있다. 이 공유 사전으로 누수 없는 전체5-fold OOF를 주장하지 않는다. fold0 자체도 이미 개발에 사용했다.', '',
        '균일 자모 보정(AS-07), 단순 발음 확대, 원래 발음에 대한 최근접 검사는 채택 근거를 얻지 못했다. 다음에는 학습 내부 분할에서 생성기·사전까지 격리한 뒤, 이름·GT/IGT·숫자·선종/국적 충돌을 함께 고려하는 후보 선택을 비교해야 한다. 이번 외부 fold에서 실패 사례에 맞춰 필터 임계값을 재조정하지 않는다.', '',
        '실험 근거: AS08-KO-DIVERSE-PILOT01, AS08-KO-DIVERSE-01, AS08-KO-OWNERSHIP-01, EVAL-AS08-KO-DIVERSE-01, EVAL-AS08-KO-OWNERSHIP-01. 사용법: AS08_PRONUNCIATION_USAGE.md.',
    ]
    (ROOT/'docs/AS08_PRONUNCIATION_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(ROOT/'docs/AS08_PRONUNCIATION_RESULTS.md')


if __name__=='__main__':main()