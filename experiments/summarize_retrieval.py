"""Aggregate all four-channel subsets without changing experimental results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifact-root',type=Path,default=Path('/workspace/vessel_matching'))
    args=parser.parse_args()
    records={eid:json.loads((ROOT/f'project_log/experiments/{eid}.json').read_text()) for eid in ['E007','E008']}
    assert all(r['status']=='completed' and len(r['variants'])==16 for r in records.values())
    rows=[];summary={}
    for combo,v in records['E007']['variants'].items():
        summary[combo]={
            'channels':v['channels'],
            'mean_vessel_weighted_accuracy':np.mean([r['variants'][combo]['vessel_holdout']['pure']['weighted_accuracy'] for r in records.values()]).item(),
            'mean_mixed_production_weighted_accuracy':np.mean([r['variants'][combo]['row_holdout']['production']['weighted_accuracy'] for r in records.values()]).item(),
            'mean_vessel_candidates':np.mean([r['variants'][combo]['vessel_holdout']['pure']['candidate_count_nonlookup']['mean'] for r in records.values()]).item()}
        for eid,r in records.items():
            result=r['variants'][combo]
            for holdout in ['vessel_holdout','row_holdout']:
                for mode in ['pure','production']:
                    m=result[holdout][mode]
                    rows.append({'experiment':eid,'combination':combo,'channels':'+'.join(v['channels']),
                        'holdout':holdout,'evaluation':mode,'weighted_accuracy':m['weighted_accuracy'],
                        'hangul_rows':m['hangul']['rows'],'non_hangul_rows':m['non_hangul']['rows'],
                        'hangul_accuracy':m['hangul']['accuracy'],'non_hangul_accuracy':m['non_hangul']['accuracy'],
                        'hangul_candidate_recall':m['hangul']['candidate_recall'],'non_hangul_candidate_recall':m['non_hangul']['candidate_recall'],
                        'mean_candidates':m['candidate_count_nonlookup']['mean'],
                        'empty_search_rows':m['empty_search_rows_nonlookup'],'errors':m['errors'],
                        'training_pairs':result['fit']['training_pairs'],
                        'cached_evaluation_seconds':m['cached_evaluation_seconds']})
    def key(c):
        v=summary[c]
        return (v['mean_vessel_weighted_accuracy'],v['mean_mixed_production_weighted_accuracy'],-v['mean_vessel_candidates'],c)
    unrestricted=max((c for c in summary if c!='NONE'),key=key)
    selected=max((c for c in summary if 'N' in summary[c]['channels']),key=key)
    selection={'selected_variant':selected,'selected_channels':summary[selected]['channels'],
        'unrestricted_best':unrestricted,'selection_rule':'Best N-enabled combination to implement requested n-gram. Mean pure vessel score, mean production mixed score, then fewer candidates; report unrestricted best separately.',
        'summary':summary,'limitations':['Two overlapping development splits, not OOF or an untouched final test.',
            'All 37 reranking features are fixed; only retrieval channels and training negatives change.',
            'Variable union size; channel additions are not equal-budget comparisons.',
            'Cached evaluation time excludes retrieval/features and is not end-to-end inference latency.']}
    (ROOT/'reports/retrieval_combinations.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(ROOT/'reports/retrieval_combinations.csv',index=False)
    order=sorted(summary,key=lambda c:(len(summary[c]['channels']),c))
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    for j,(eid,r) in enumerate(records.items()):
        scores=[100*r['variants'][c]['vessel_holdout']['pure']['weighted_accuracy'] for c in order]
        ax.bar(np.arange(len(order))+(j-.5)*.38,scores,width=.38,label=eid)
    ax.set_xticks(np.arange(len(order)),order,rotation=45,ha='right')
    ax.set_ylim(0,105);ax.set_ylabel('Weighted vessel-holdout accuracy (%)')
    ax.set_title('All 16 retrieval combinations; 37 reranking features fixed')
    ax.legend();fig.savefig(ROOT/'reports/retrieval_combinations.png',dpi=180);plt.close(fig)
    lines=['# 네 검색 방법의 모든 조합 실험','',
        '요청한 문자 n-gram 검색을 구현하고, 4개 검색 경로의 모든 부분집합 16개를 기존 두 개발 분할에서 평가했다. 총 32개 조합·분할 결과이며, 실제 학습은 비어 있지 않은 15개 조합 × 2분할 = 30개 모델이다. NONE 2건은 학습하지 않는 고정 기준선이다.','',
        '## 무엇을 켜고 껐는가','',
        '| 기호 | 검색 방법 | 고정한 구현 |','|---|---|---|',
        '| S | 기존 문자열 | 비한글 정규화 이름 RapidFuzz 상위 24개. 기존 구현을 보존해 한글은 이 경로를 사용하지 않음 |',
        '| N | 문자 n-gram | 로마자화한 카탈로그 이름의 문자 2~4gram TF-IDF·코사인, 한글 60개/비한글 24개; 양수 점수만 |',
        '| P | 로마자·발음 | 한글 로마자 60개 + 기존 자음 키 60개; 모든 언어에 추가 발음 키 24개 |',
        '| T | 톤수·제원 | 두 톤수별 근접 후보와 한글의 추정 선종별 근접 후보; 기존 설정 유지 |','',
        '**검색 경로의 조합 실험**이다. 최종 후보를 평가하는 37개 특징은 모든 조합에서 유지했다. 예를 들어 T를 꺼도 다른 경로로 찾은 후보와 입력의 톤수 차이는 분류기 특징으로 남는다. N에서도 한영 문자열을 비교할 수 있도록 로마자 변환을 공통 전처리로 사용하므로, P를 끈다는 것은 로마자·발음 유사도 검색 경로를 끈다는 뜻이다.','',
        'N의 어휘와 IDF는 공식 카탈로그 78,595행으로만 만들었다. 학습/검증/테스트 이름을 모아 fit하지 않는다. 각 입력 행은 고정된 희소 인덱스에 독립적으로 질의하며, 동점은 카탈로그 ID 순서로 결정한다.','',
        '예를 들어 OCEAN의 2글자 조각은 OC·CE·EA·AN이다. TF-IDF는 여러 선박에서 흔한 조각의 비중을 낮추고 드문 조각의 비중을 높이며, 2~4글자 조각을 함께 비교한다. 일부 철자가 달라도 나머지 조각으로 후보를 찾을 수 있지만, 이 방법 자체가 발음을 이해하거나 한글 음역의 원래 영문 철자를 복원하는 것은 아니다. 이번 카탈로그의 어휘는 89,536개 조각, 희소 행렬의 0이 아닌 항목은 1,869,600개였다.','',
        '## 공정한 비교 조건과 시간 해석','',
        '- 기존 분할·학습 행 순서·37개 특징·LightGBM 350개 트리와 모든 하이퍼파라미터를 유지했다.',
        '- 각 조합은 자신의 검색 결과에서 음성 후보를 골라 별도로 학습했다. 정답 후보 주입은 학습에만 수행했다.',
        '- 공통 기록–후보 쌍의 특징 계산만 재사용했다. 다른 조합의 후보를 해당 조합의 검색 결과로 포함하지 않았다.',
        '- pure는 호출부호 조회와 학습 별칭을 꺼 검색 경로 자체를 비교한다. production은 고정된 학습 호출부호·별칭을 사용하는 실제 추론 조건이다.',
        '- NONE과 검색 결과가 없는 행은 카탈로그 ID 정렬의 첫 ID로 처리한다. 빈 검색 건수도 기록한다. NONE의 production 점수에는 호출부호·별칭 효과가 들어갈 수 있다.',
        '- 각 경로의 후보 예산은 고정했지만 합집합 크기는 달라진다. 이를 같은 총 후보 수의 비교나 Recall@K라고 부르지 않는다.',
        '- 두 분할 작업을 동시에 실행했다. 학습 시간은 자원 경합 영향을 받으며, cached_evaluation_seconds는 검색·특징 계산을 제외한 재사용 평가 시간이다. 어느 것도 실제 제출 추론 시간으로 제시하지 않는다.',
        '- 두 분할은 겹치는 개발 검증이며 2-fold OOF 또는 독립 최종 테스트가 아니다.','',
        '## 16개 조합 전체 결과','',
        '아래는 새 선박 pure 가중 정확도이며, 산식은 0.7×한글 정확도 + 0.3×비한글 정확도이다.','',
        '![모든 검색 조합 비교](../reports/retrieval_combinations.png)','',
        '| 조합 | S | N | P | T | 분할 1 | 분할 2 | 단순 평균 | 평균 후보 수 |',
        '|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|']
    for c in order:
        scores=[records[e]['variants'][c]['vessel_holdout']['pure']['weighted_accuracy'] for e in records]
        flags=' | '.join('✓' if ch in summary[c]['channels'] else '—' for ch in 'SNPT')
        lines.append(f"| {c} | {flags} | {100*scores[0]:.3f}% | {100*scores[1]:.3f}% | {100*summary[c]['mean_vessel_weighted_accuracy']:.3f}% | {summary[c]['mean_vessel_candidates']:.1f} |")
    lines += ['', '한글·비한글 정확도/표본 수/후보 포함률, 혼합 보류 pure/production, 학습 쌍 수, 오류 수를 포함한 전체 128행 결과는 `reports/retrieval_combinations.csv`에 저장했다. 후보 수에는 검색이 비었을 때의 고정 fallback 1개가 포함되며 빈 검색 건수는 별도 컬럼이다.','',
        '## n-gram을 추가했을 때의 짝지은 비교','',
        '다른 검색 경로를 같게 놓고 N만 추가한 8쌍이다. 차이는 퍼센트포인트(%p)이며 각 조합의 모델은 다시 학습했다.','',
        '| N 없는 조합 | N 추가 조합 | 분할 1 차이 | 분할 2 차이 | 평균 차이 |','|---|---|---:|---:|---:|']
    for before in order:
        if 'N' in summary[before]['channels']:continue
        after=''.join(c for c in 'SNPT' if c=='N' or c in summary[before]['channels'])
        delta=[100*(r['variants'][after]['vessel_holdout']['pure']['weighted_accuracy']-r['variants'][before]['vessel_holdout']['pure']['weighted_accuracy']) for r in records.values()]
        lines.append(f'| {before} | {after} | {delta[0]:+.3f} | {delta[1]:+.3f} | {np.mean(delta):+.3f} |')
    lines += ['', '## 선택 결과와 한계','',
        f'- 전체 최고 조합: **{unrestricted}**. 새 선박 점수 평균 {100*summary[unrestricted]["mean_vessel_weighted_accuracy"]:.3f}%.',
        f'- N 포함 최고 조합: **{selected}**. 새 선박 점수 평균 {100*summary[selected]["mean_vessel_weighted_accuracy"]:.3f}%. n-gram 적용 요청을 반영한 제출물은 이 구성으로 만든다.',
        '- 선택 기준은 실험 전에 기록한 새 선박 점수 평균 → 혼합 production 점수 평균 → 적은 후보 수 순이다.','',
        '| 실험 | SPT의 기존 결과 재현 | N 포함 최고: 혼합 production 점수 |','|---|---|---:|']
    for eid,r in records.items():
        assert r['variants']['SPT']['previous_search_reproduction']['passed']
        lines.append(f"| {eid} | 학습 쌍·후보 해시·전체 예측 일치 | {100*r['variants'][selected]['row_holdout']['production']['weighted_accuracy']:.3f}% |")
    lines += ['', '## 기존 SPT 대비 선택 조합의 오류 변화','',
        '| 실험 | 기존 오류 수정 | 새 오류 | 기존/선택 조합 오류 수 |','|---|---:|---:|---|']
    case_lines=[]
    public_train=pd.read_csv(args.artifact_root/'data/train.csv').set_index('sample_id')
    catalog=pd.read_csv(args.artifact_root/'data/vessel_catalog.csv').set_index('vessel_id')
    for eid in records:
        folder=args.artifact_root/f'retrieval_{eid.lower()}'
        before=pd.read_csv(folder/'SPT/vessel_holdout_pure_predictions.csv').set_index('sample_id')
        after=pd.read_csv(folder/selected/'vessel_holdout_pure_predictions.csv').set_index('sample_id').reindex(before.index)
        b=before.truth.eq(before.prediction);a=after.truth.eq(after.prediction)
        lines.append(f'| {eid} | {int((~b&a).sum())} | {int((b&~a).sum())} | {int((~b).sum())}/{int((~a).sum())} |')
        for label,mask in [('수정',~b&a),('새 오류',b&~a)]:
            groups={}
            for s in before.index[mask]:
                key=(public_train.loc[s,'vessel_name'],catalog.loc[before.loc[s,'truth'],'vessel_name'],
                     catalog.loc[before.loc[s,'prediction'],'vessel_name'],catalog.loc[after.loc[s,'prediction'],'vessel_name'])
                groups[key]=groups.get(key,0)+1
            for values,count in list(groups.items())[:3]:
                case_lines.append(f'| {eid} | {label} | '+' | '.join(str(v).replace('|','/') for v in values)+f' | {count} |')
    lines += ['', '공식 공개 train 보류에서만 고른 대표 사례다. 반복 기록이 같은 선박에 속하므로 행 수를 독립적인 선박 사례 수로 해석하지 않는다.','',
        '| 실험 | 변화 | 입력 이름 | 정답 | 기존 SPT 예측 | 선택 조합 예측 | 행 수 |','|---|---|---|---|---|---|---:|']+case_lines
    lines += ['', '새 검색 경로를 추가해도 정확도가 반드시 오르지는 않는다. 정답 후보를 더 찾는 효과와 비슷한 오답 후보가 늘어나는 효과가 함께 생긴다. 후보 수와 학습 음성 구성이 달라지므로 점수 변화를 단일 문자열 특징의 인과적 기여도로 해석하지 않는다.','',
        '보고서의 점수는 공식 리더보드 점수가 아니다. 이미 관찰한 개발 분할 16개 조합에서 모델을 선택했으므로 선택 편향이 있고, 같은 선박의 반복 기록도 존재한다. 일반화 확인은 별도의 그룹 검증이나 공식 평가가 필요하다.','',
        '## 재현 근거','',
        '- 구현: `model/ngram_retrieval.py`, `model/matcher.py`.',
        '- 모든 조합 학습·검증: `experiments/retrieval_factorial.py`.',
        '- 실행 명령·데이터/분할/코드 해시·환경·실측: `project_log/experiments/E007.json`, `E008.json`.',
        '- 요약 재생성: `python experiments/summarize_retrieval.py`.',
        '- 전체 학습 및 제출 ZIP: E009의 완료 상태와 검사 결과를 확인한다.','']
    final=json.loads((ROOT/'project_log/experiments/E009.json').read_text())
    if final['status']=='completed':
        checks=final['submission_checks'];bench=final.get('benchmark')
        lines += ['## 최종 학습·제출 파일(E009)','',
            f"- 채택 조합 {final['selected_variant']}, 전체 학습 {final['fit']['training_rows']:,}행/{final['fit']['training_pairs']:,}쌍.",
            f"- ZIP {checks['zip_bytes']:,}바이트, SHA-256 `{checks['sha256']}`.",
            '- 깨끗한 ZIP 해제, 오프라인 실행, 입력 보존, 출력 형식, 전체/순서변경/부분/단독/빈 입력 검사를 통과했다.',
            '- 파일: `/workspace/vessel_matching/ngram_submission/submit.zip`. 이전 두 제출 ZIP은 보존했다.','']
        if bench:
            lines += [f"공개 train {bench['rows']:,}행 전체의 호출부호를 미등록 값으로 바꾸어 일반 추론 경로를 실행했다. {bench['seconds']:.2f}초, 최대 RSS {bench['peak_rss_bytes']/1024**2:.1f} MiB였다. 시작·import·CSV 로딩·카탈로그 및 n-gram 인덱스·검색·특징·예측·CSV 저장을 포함한다. ZIP 해제·입력 준비는 제외하며 학습 별칭은 유지했다. 정확도 평가가 아닌 로컬 실행 비용 검사이고, 공식 서버·비공개 테스트 규모의 측정은 아니다.",'']
    (ROOT/'docs/RETRIEVAL_COMBINATIONS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'unrestricted_best':unrestricted,'selected_variant':selected,'selected_summary':summary[selected]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
