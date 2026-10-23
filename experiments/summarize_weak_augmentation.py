"""Official evaluation and aggregate-only report for AUG-02 / AUG-03."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import torch
import pandas as pd
import json
import subprocess
import zipfile
import hashlib
import argparse
from train_oof import now_kst,sha256
from train import read_data
from experiments.weak_augmentation import augment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-id',default='AUG-02')
    parser.add_argument('--augmentation-id',default='AUG-03')
    parser.add_argument('--prefix',default='augmentation')
    args=parser.parse_args()
    names=[args.baseline_id,args.augmentation_id]
    records=[json.loads((ROOT/f'project_log/experiments/{n}.json').read_text(encoding='utf-8')) for n in names]
    assert all(r['status']=='completed' and len(r['fold_results'])==5 for r in records)
    leaderboard=ROOT/f'reports/{args.prefix}_leaderboard.csv'
    if leaderboard.exists():
        raise ValueError('Preserve existing evaluation report; do not append a duplicate run')
    for i,name in enumerate(names):
        cmd=[sys.executable,'-X','utf8','eval_kit/evaluate.py','--train','data/train.csv',
             '--oof',records[i]['oof']['path'],'--id',name,'--leaderboard',str(leaderboard)]
        if i:
            cmd += ['--compare',records[0]['oof']['path']]
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        path=ROOT/f'reports/{name}_official_evaluation.txt'
        path.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Official evaluation failed: {path}')
        print(result.stdout,flush=True)
        records[i]['team_evaluation']={'command':cmd,'output':str(path),'sha256':sha256(path),'exit_code':0}
    table=pd.read_csv(leaderboard,keep_default_na=False)
    train,catalog=read_data(ROOT/'data')
    folds=pd.read_csv(ROOT/'eval_kit/folds_group5_seed42.csv')
    aligned=train.merge(folds[['sample_id','fold']],on='sample_id',validate='one_to_one')
    predictions=[]
    for r in records:
        oof=pd.read_csv(r['oof']['path'],dtype=str)
        predictions.append(oof.set_index('sample_id').loc[train.sample_id].pred_vessel_id.to_numpy())
    gold=train.target_vessel_id.to_numpy()
    ko=train.vessel_name.str.contains('[가-힣]').to_numpy()
    changes={}
    for label,mask in [('hangul',ko),('non_hangul',~ko)]:
        changes[label]={'rows':int(mask.sum()),
          'fixed':int(((predictions[0]!=gold)&(predictions[1]==gold)&mask).sum()),
          'broken':int(((predictions[0]==gold)&(predictions[1]!=gold)&mask).sum()),
          'prediction_changed':int(((predictions[0]!=predictions[1])&mask).sum())}
    columns=['vessel_name','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','target_vessel_id']
    for i,r in enumerate(records):
        for f in r['fold_results']:
            if i:
                source=aligned[aligned.fold!=f['fold']].drop(columns='fold')
                extra,stats=augment(source,catalog,seed=42+f['fold'])
                assert stats['synthetic_sha256']==f['fit']['augmentation']['synthetic_sha256']
                existing=set(map(tuple,source[columns].astype(str).to_numpy()))
                f['fit']['augmentation'].update(hangul_rows=int(extra.vessel_name.str.contains('[가-힣]').sum()),
                     duplicate_of_real_rows=sum(tuple(row) in existing for row in extra[columns].astype(str).to_numpy()))
                # This copied diagnostic describes real base pairs only, not all augmented rows.
                f['fit']['base_real_training_candidate_recall']=f['fit'].pop('training_candidate_recall')
                f['fit']['base_real_unique_feature_rows']=f['fit'].pop('unique_feature_rows')
        d=r['local_diagnostics']
        r['metrics'].update(metric_source='Official eval_kit/evaluate.py G5-s42 OOF; diagnostics from same predictions',
             hangul_rows=d['hangul_rows'],non_hangul_rows=d['non_hangul_rows'],
             hangul_accuracy=d['hangul_accuracy'],non_hangul_accuracy=d['non_hangul_accuracy'],
             weighted_accuracy=d['weighted_accuracy'],candidate_recall_hangul=d['hangul_candidate_recall'],
             candidate_recall_non_hangul=d['non_hangul_candidate_recall'],seen_vessel_fraction=0.0)
        r['cost'].update(training_seconds=d['training_seconds'],inference_seconds=d['prediction_and_diagnostics_seconds'],
            timing_scope='Sum over five folds: fit includes index/features/model save (augmented cost attributes reused base preparation); inference includes predict_details only, excludes Matcher construction/CSV and independence probes',
            model_bytes=sum(f['artifacts']['ranker.txt']['bytes'] for f in r['fold_results']),
            assets_bytes=sum(f['artifacts']['assets.json']['bytes'] for f in r['fold_results']))
        r['features']=json.loads(Path(r['fold_results'][0]['artifacts']['assets.json']['path']).read_text(encoding='utf-8'))['features']
        r['data'].update(validation_rows=12222,split_file_sha256=sha256(ROOT/'eval_kit/folds_group5_seed42.csv'))
        r['artifacts'] += [r['oof'],r['team_evaluation'],{'path':str(leaderboard),'sha256':sha256(leaderboard)}]
        r['decision']='Review aggregate score difference and confidence interval before adoption'
        r['limitations'] += ['Name deletion and masking are combined; individual contributions not identified.',
            'Hard-negative selection configuration is recorded in the experiment source hash and limitations; comparisons across revisions must identify this change.']
        (ROOT/f"project_log/experiments/{r['experiment_id']}.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    difference=records[1]['metrics']['weighted_accuracy']-records[0]['metrics']['weighted_accuracy']
    cmp=table.iloc[1]
    summary={'recorded_at_kst':now_kst(),'baseline':names[0],'augmentation':names[1],
             'weighted_difference':difference,'bootstrap_95_ci':[float(cmp.d_lo),float(cmp.d_hi)],
             'changes':changes,'metrics':{r['experiment_id']:r['metrics'] for r in records},
             'costs':{r['experiment_id']:r['cost'] for r in records}}
    with zipfile.ZipFile(ROOT/'submit_JY-01.zip') as z:
        summary['deployed_model_unchanged']={p:hashlib.sha256(z.read(p)).hexdigest()==sha256(ROOT/p)
                                          for p in ['model/assets.json','model/ranker.txt']}
    (ROOT/f'reports/{args.prefix}_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
