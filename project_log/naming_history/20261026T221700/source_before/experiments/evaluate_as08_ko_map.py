"""Fixed-engine comparison of the original and deficient-entry-repaired KO maps."""
import argparse
import re
import json
from pathlib import Path
import subprocess
import sys
import time
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.noisy_channel_common import configure_cpu, reserve_experiment, write_json, record_failure, sha256, now_kst
from experiments.evaluate_as07_jamo import load_engine, predict_chunks, metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id',default='EVAL-AS08-KO-DIVERSE-01')
    parser.add_argument('--map-experiment-id',default='AS08-KO-DIVERSE-01')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.map_experiment_id):
        parser.error('Invalid map experiment ID')
    configure_cpu(2)
    eid = args.experiment_id
    out,path,record = reserve_experiment(eid,ROOT / 'reports/as06_improvements',
        'Compare fixed AS-06 legacy scoring with old and repaired pronunciation maps', ' '.join(sys.argv))
    start = time.perf_counter()
    try:
        import pandas as pd
        import numpy as np
        from name_decoder.data import read_frames
        from eval_kit.robustness import PRESETS, RobustnessPerturber
        from eval_kit.evaluate import run as official_evaluate
        generated_dir = ROOT / 'artifacts' / args.map_experiment_id
        generation = json.loads((ROOT / 'project_log/experiments' / (args.map_experiment_id+'.json')).read_text(encoding='utf-8'))
        if generation['status'] != 'generated_unvalidated' or not generation['diagnostics']['processed_scope_complete']:
            raise ValueError('Scoped generation did not finish')
        if sha256(generated_dir / 'catalog_full_ko_map.json') != generation['artifact_sha256']['catalog_full_ko_map.json']:
            raise ValueError('New map hash differs from generation record')
        primary = json.loads((ROOT / 'project_log/experiments/AS07-JAMO-SCORE-01.json').read_text(encoding='utf-8'))
        secondary = json.loads((ROOT / 'project_log/experiments/AS07-JAMO-SCORE-02.json').read_text(encoding='utf-8'))
        if sha256(ROOT / 'AS-07/script.py') != primary['code']['file_sha256']['AS-07/script.py']:
            raise ValueError('Diagnostic legacy engine changed')
        old_map = json.loads((ROOT / 'AR-06/catalog_full_ko_map.json').read_text(encoding='utf-8-sig'))
        new_map = json.loads((generated_dir / 'catalog_full_ko_map.json').read_text(encoding='utf-8'))
        if any(new_map.get(name,[])[:len(variants)] != variants for name,variants in old_map.items()):
            raise AssertionError('Original variants not preserved')
        _,val,fit,_ = read_frames(ROOT / 'data',ROOT / 'eval_kit/folds_group5_seed42.csv',0)
        catalog = pd.read_csv(ROOT / 'data/vessel_catalog.csv',keep_default_na=False)
        engine = load_engine(ROOT / 'AS-07/script.py','_as08_fixed_legacy_engine')
        model = engine.fit_matching_model(fit,catalog,kr_mode='AR06',en_mode='EN_V3',use_train_lookup=False)
        saved_model = json.loads((ROOT / 'artifacts/AS07-JAMO-SCORE-01/evaluation/fit_model.json').read_text(encoding='utf-8'))
        model_equal = json.loads(json.dumps(model)) == saved_model
        # A cache is used only with immutable input/model/source checks and a
        # balanced64-row prediction+rank parity check independently per condition.
        hashes = {k:sha256(ROOT / p) for k,p in {'train.csv':'data/train.csv','vessel_catalog.csv':'data/vessel_catalog.csv','folds':'eval_kit/folds_group5_seed42.csv'}.items()}
        if hashes != primary['data']['source_files_sha256']:
            raise ValueError('Source inputs changed since cached baseline')
        historical_revision = primary['code']['git_revision_at_run']
        original_robustness = subprocess.check_output(['git','show',historical_revision + ':eval_kit/robustness.py'],cwd=ROOT)
        robustness_equal = (ROOT / 'eval_kit/robustness.py').read_bytes().replace(b'\r\n',b'\n') == original_robustness.replace(b'\r\n',b'\n')
        artifact_dir = ROOT / 'artifacts' / eid
        artifact_dir.mkdir(parents=True,exist_ok=False)
        write_json(artifact_dir / 'fit_model.json',model)
        record['data'] = dict(primary['data'])
        record['data']['noise_seeds'] = [20261022,20261026,20261027]
        record['data']['seed'] = None
        record['code']['file_sha256'].update({p:sha256(ROOT / p) for p in ['experiments/evaluate_as08_ko_map.py','AS-07/script.py','eval_kit/robustness.py']})
        record['environment'].update(device='cpu',threads=2)
        record['generation_experiment_id']=args.map_experiment_id
        record['map_hashes'] = {'old':sha256(ROOT / 'AR-06/catalog_full_ko_map.json'),'new':sha256(generated_dir / 'catalog_full_ko_map.json')}
        index_start=time.perf_counter()
        old_index = engine.CatalogIndex(catalog,full_ko_map=old_map)
        new_index = engine.CatalogIndex(catalog,full_ko_map=new_map)
        record['cost']['index_seconds_both_maps'] = time.perf_counter()-index_start
        gold=val.target_vessel_id.tolist();ko=val.hangul.to_numpy(dtype=bool)
        positions=list(np.flatnonzero(ko)[:32])+list(np.flatnonzero(~ko)[:32])
        drop=['target_vessel_id','target_vessel_id_fold','source','target','hangul','fold']
        phases=[('clean',None,None,primary),('default','default',20261022,primary),('severe','severe',20261022,primary),
                ('severe_20261026','severe',20261026,secondary),('severe_20261027','severe',20261027,secondary)]
        record['phase_results']={};errors=[]
        record['baseline_cache_checks']={'same_fit_model':model_equal,'same_inputs':True,'robustness_matches_historical_git_blob':robustness_equal,'per_condition_probe_rows':64}
        write_json(path,record)
        for phase,preset,seed,previous in phases:
            print('PHASE '+phase,flush=True)
            if preset is None:
                data=val.drop(columns=drop,errors='ignore').reset_index(drop=True);noise=None
            else:
                noisy,noise=RobustnessPerturber(deepcopy(PRESETS[preset]),seed=seed).perturb_dataset(val,catalog)
                data=noisy.drop(columns=drop,errors='ignore').reset_index(drop=True)
            source_dir=ROOT / ('artifacts/AS07-JAMO-SCORE-01/evaluation' if previous is primary else 'artifacts/AS07-JAMO-SCORE-02')
            baseline_path=source_dir/f'oof_{phase}_legacy.csv'
            expected=previous['phase_results'][phase]['legacy']['prediction_file_sha256']
            cache=pd.read_csv(baseline_path,keep_default_na=False,dtype={'sample_id':str,'pred_vessel_id':str})
            cache=cache.set_index('sample_id').loc[val.sample_id].reset_index()
            a,b,_=engine.predict_rows(data.iloc[positions],model,old_index,use_train_lookup=False,
                gold_ids=[gold[i] for i in positions],top_candidates=2,jamo_score_mode='legacy')
            cache_valid=(model_equal and robustness_equal and sha256(baseline_path)==expected
                and cache.iloc[positions].pred_vessel_id.tolist()==a and cache.iloc[positions].gold_rank.astype(int).tolist()==b)
            if cache_valid:
                old_p=cache.pred_vessel_id.tolist();old_r=cache.gold_rank.astype(int).tolist()
                old_seconds=None
                old_stats=dict(previous['phase_results'][phase]['legacy'])
                old_stats.pop('prediction_seconds',None)
                print('  Original map cache verified (hashes and64-row parity)',flush=True)
            else:
                old_p,old_r,_,_,old_seconds=predict_chunks(engine,data,model,old_index,'legacy',gold)
                old_stats=metrics(old_p,gold,ko,old_r)
                print('  Cache not used; original map recomputed',flush=True)
            pd.DataFrame({'sample_id':val.sample_id.tolist(),'fold':0,'pred_vessel_id':old_p,'gold_rank':old_r}).to_csv(artifact_dir/f'oof_{phase}_old.csv',index=False,encoding='utf-8')
            p,r,top,diagnostics,seconds=predict_chunks(engine,data,model,new_index,'legacy',gold)
            pd.DataFrame({'sample_id':val.sample_id.tolist(),'fold':0,'pred_vessel_id':p,'gold_rank':r}).to_csv(artifact_dir/f'oof_{phase}_new.csv',index=False,encoding='utf-8')
            write_json(artifact_dir/f'{phase}_new_candidate_sets.json',diagnostics)
            result={'noise_summary':noise,'old':old_stats,'new':metrics(p,gold,ko,r),
                'baseline_cache_verified':cache_valid,'old_recompute_seconds':old_seconds,'new_prediction_seconds':seconds,
                'new_candidate_count_mean':float(np.mean([x['count'] for x in diagnostics])),
                'new_candidate_count_p95':float(np.percentile([x['count'] for x in diagnostics],95)),
                'prediction_sha256':{m:sha256(artifact_dir/f'oof_{phase}_{m}.csv') for m in ['old','new']}}
            oldc=np.asarray(old_p)==np.asarray(gold);newc=np.asarray(p)==np.asarray(gold)
            result['paired']={label:{'fixes':int((~oldc&newc&mask).sum()),'breaks':int((oldc&~newc&mask).sum()),
                'prediction_changes':int(((np.asarray(old_p)!=np.asarray(p))&mask).sum()),
                'newly_retrieved':int(((np.asarray(old_r)==0)&(np.asarray(r)>0)&mask).sum()),
                'lost_from_candidates':int(((np.asarray(old_r)>0)&(np.asarray(r)==0)&mask).sum())}
                for label,mask in [('hangul',ko),('non_hangul',~ko)]}
            if not np.array_equal(np.asarray(old_p)[~ko],np.asarray(p)[~ko]):
                raise AssertionError('Non-Hangul predictions changed despite fixed engine')
            lines=[]
            official=official_evaluate(['--oof',str(artifact_dir/f'oof_{phase}_new.csv'),'--compare',str(artifact_dir/f'oof_{phase}_old.csv'),
                '--train',str(ROOT/'data/train.csv'),'--folds',str(ROOT/'eval_kit/folds_group5_seed42.csv'),'--subset','--boot','2000','--seed','0'],out=lines.append)
            result['official']={k:official[k] for k in ['score','boot','compare','warnings','folds']}
            (out/f'{phase}_official.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            for kind,mask in [('fix',~oldc&newc),('break',oldc&~newc)]:
                for i in np.flatnonzero(mask)[:8]:
                    errors.append({'source':'public train heldout/derived stress','phase':phase,'kind':kind,
                        'sample_id':str(val.iloc[i].sample_id),'observed_name':str(data.iloc[i].vessel_name),
                        'gold_vessel_id':gold[i],'old_prediction':old_p[i],'new_prediction':p[i],
                        'old_gold_rank':old_r[i],'new_gold_rank':r[i],
                        'failure_class':'retrieval omission' if (old_r[i] if kind=='fix' else r[i])==0 else 'candidate selection'})
            record['phase_results'][phase]=result
            write_json(out/'summary.json',record['phase_results']);write_json(path,record)
            print(json.dumps({'phase':phase,'old_h':result['old']['hangul_correct'],'new_h':result['new']['hangul_correct'],
                'paired':result['paired'],'delta_ci':official['compare']['boot']['w']}),flush=True)
        write_json(artifact_dir/'public_heldout_errors.json',errors)
        # Small public-row order/single check with the new static asset.
        sample=val.drop(columns=drop,errors='ignore').iloc[positions[:8]]
        full,_=engine.predict_rows(sample,model,new_index,use_train_lookup=False,jamo_score_mode='legacy')
        reverse,_=engine.predict_rows(sample.iloc[::-1],model,new_index,use_train_lookup=False,jamo_score_mode='legacy')
        singles=[engine.predict_rows(sample.iloc[[i]],model,new_index,use_train_lookup=False,jamo_score_mode='legacy')[0][0] for i in range(len(sample))]
        if full!=reverse[::-1] or full!=singles:raise AssertionError('Static-map row independence failed')
        regressions=[phase for phase,v in record['phase_results'].items() if v['new']['weighted_accuracy']<v['old']['weighted_accuracy']]
        record.update(status='completed_not_promoted',finished_at_kst=now_kst(),artifacts=[str(out),str(artifact_dir)],
            error_cases_file=str(artifact_dir/'public_heldout_errors.json'),
            decision=('Keep AS-06; repaired-map candidate regressed on: '+','.join(regressions) if regressions else 'Retain repaired map experimentally; no automatic promotion from one explored fold.'),
            row_independence={'rows':8,'reverse_differences':0,'single_differences':0},
            limitations=['One previously explored outerfold0, not untouched final test/full5fold/server score.',
                'Map-generation/filter rules use public catalog and original map only; earlier aggregate development results informed follow-up selection.',
                'Fixed rules jointly filter and diversify samples; not separate ablations of every filter.',
                'Existing baseline map historical generation provenance/manual lexicon limits remain.',
                'Cached old-map results have source/model/file hashes and64-row per-phase parity checks; timing is not an old/new speed comparison.'])
        record['cost']['total_seconds']=time.perf_counter()-start
        record['cost']['timing_scope']='CPU matching/map indexing, cache verification, official evaluation and saving; not server runtime.'
        clean=record['phase_results']['clean']['new']
        record['metrics'].update(metric_source='Fold0 clean development; not official server',**{k:clean[k] for k in ['hangul_rows','non_hangul_rows','hangul_accuracy','non_hangul_accuracy','weighted_accuracy']})
        write_json(path,record)
        print('COMPLETE: '+record['decision'],flush=True)
    except BaseException as exc:
        record_failure(path,record,exc);raise


if __name__=='__main__':
    main()