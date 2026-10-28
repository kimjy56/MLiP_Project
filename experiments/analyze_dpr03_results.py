"""Summarize fixed DPR03 outputs and paired component comparisons; no new inference."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.dpr03_common import start_run,finish_run,write_json,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id',default='ANALYZE-DPR03-01')
    args=parser.parse_args()
    out,path,record=start_run(args.experiment_id)
    try:
        import pandas as pd
        report_path=ROOT/'reports/matching_improvements/DPR-HYBRID-03/dpr_hybrid_robustness_benchmark.json'
        report=json.loads(report_path.read_text(encoding='utf-8'))
        summary={'benchmark_sha256':sha256(report_path),'conditions':{},'component_comparisons':{}}
        for name,result in report['results'].items():
            old=result['variants']['reference02']
            new=result['variants']['complete03']
            summary['conditions'][name]={
                'reference02':old,'complete03':new,
                'delta_weighted':new['weighted_accuracy']-old['weighted_accuracy'],
                'direct_fixes':result['direct03_vs_retrieval03']['num_fixes'],
                'direct_breaks':result['direct03_vs_retrieval03']['num_breaks'],
                'tail_fixes':result['complete03_vs_direct03']['num_fixes'],
                'tail_breaks':result['complete03_vs_direct03']['num_breaks'],
            }
        root=ROOT/'artifacts/DPR-HYBRID-03'
        for name,current,baseline in [('direct','direct03','retrieval03'),('tail','complete03','direct03')]:
            command=[sys.executable,'-B','-X','utf8',str(ROOT/'eval_kit/evaluate.py'),
                     '--oof',str(root/f'oof_DPR-HYBRID-03_true_unseen_severe_{current}.csv'),
                     '--compare',str(root/f'oof_DPR-HYBRID-03_true_unseen_severe_{baseline}.csv'),
                     '--train',str(ROOT/'data/train.csv'),'--folds',str(ROOT/'eval_kit/folds_group5_seed42.csv'),'--subset']
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True,timeout=240)
            (out/f'official_component_{name}.txt').write_text(result.stdout,encoding='utf-8')
            summary['component_comparisons'][name]=[line for line in result.stdout.splitlines() if '판정:' in line]
        train=pd.read_csv(ROOT/'data/train.csv',dtype=str,keep_default_na=False)
        catalog=pd.read_csv(ROOT/'data/vessel_catalog.csv',dtype=str,keep_default_na=False)
        names=dict(zip(catalog.vessel_id,catalog.vessel_name))
        errors=[]
        for condition in ['true_unseen_clean','true_unseen_severe']:
            oof=pd.read_csv(root/f'oof_DPR-HYBRID-03_{condition}_complete03.csv',dtype={'sample_id':str,'pred_vessel_id':str})
            joined=train.merge(oof,on='sample_id',validate='one_to_one')
            for r in joined.loc[joined.target_vessel_id.ne(joined.pred_vessel_id)].itertuples(index=False):
                errors.append({'condition':condition,'sample_id':r.sample_id,
                               'original_public_train_name':r.vessel_name,
                               'evaluated_noisy_name':None if condition.endswith('severe') else r.vessel_name,
                               'gold_id':r.target_vessel_id,'pred_id':r.pred_vessel_id,
                               'gold_catalog_name':names[r.target_vessel_id],'pred_catalog_name':names[r.pred_vessel_id],
                               'gold_final_rank':int(r.gold_rank),
                               'confirmed_error_category':'candidate_selection' if r.gold_rank>0 else 'retrieval_omission',
                               'cause_hypothesis':None})
        write_json(out/'public_heldout_error_cases.json',errors)
        summary['public_error_cases']=len(errors)
        summary['error_counts']={key:sum(r['confirmed_error_category']==key for r in errors)
                                 for key in ['candidate_selection','retrieval_omission']}
        summary['decision']='Do not promote DPR03 checkpoint or its direct-similarity rule; preserve AS09 submission reference. Packing and score reuse have separate latency evidence.'
        summary['limitations']=['No untouched test or new server result.','Known cases and outer fold used during development.',
                                'Full-system legacy map provenance limits retained.','All error cases use public train holdout only.']
        write_json(out/'summary.json',summary)
        record.update(analysis=summary,artifacts=[str(out/'summary.json'),str(out/'public_heldout_error_cases.json')])
        record['metrics']['metric_source']='derived analysis of saved public-fold predictions and official paired evaluator; no new model inference'
        record['decision']=summary['decision']
        finish_run(path,record,'analyzed_saved_development_outputs')
        print(summary['component_comparisons'],summary['error_counts'],flush=True)
    except BaseException as exc:
        finish_run(path,record,'failed',exc)
        raise


if __name__=='__main__':
    main()
