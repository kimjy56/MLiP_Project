# -*- coding: utf-8 -*-
"""experiments/evaluate_dpr_robustness.py — Evaluate DPR-Hybrid vs Baseline AS-06 on Noise Stress (SPEC-DPR-01).

Benchmarks Heldout Fold 0 (2,372 rows) under clean, default, and severe noise presets,
comparing Baseline AS-06 against DPR-Hybrid across both ko_map ON and OFF.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_kit.robustness import PRESETS, RobustnessPerturber
from experimental_matchers.dpr_hybrid.script import (
    CatalogIndex,
    fit_matching_model,
    is_korean_name,
    predict_rows,
)
from phonetic_encoder.hybrid_matcher import HybridDenseLexicalMatcher
from experiments.dpr03_common import (start_run,finish_run,write_json,sha256,object_hash,load_fold_data,now_kst)


def calc_metrics(preds: List[str], gold: List[str], is_ko: np.ndarray) -> Dict[str, Any]:
    correct = np.asarray(preds, dtype=object) == np.asarray(gold, dtype=object)
    h_acc = float(correct[is_ko].mean()) if is_ko.any() else None
    n_acc = float(correct[~is_ko].mean()) if (~is_ko).any() else None
    w_acc = 0.7 * h_acc + 0.3 * n_acc if (h_acc is not None and n_acc is not None) else None
    return {
        "hangul_accuracy": h_acc,
        "hangul_correct": int(correct[is_ko].sum()),
        "hangul_rows": int(is_ko.sum()),
        "non_hangul_accuracy": n_acc,
        "non_hangul_correct": int(correct[~is_ko].sum()),
        "non_hangul_rows": int((~is_ko).sum()),
        "weighted_accuracy": w_acc,
    }


def find_diffs(
    df: pd.DataFrame,
    preds_base: List[str],
    preds_hybrid: List[str],
    gold: List[str],
    is_ko: np.ndarray,
) -> Dict[str, Any]:
    fixes = []
    breaks = []
    for i in range(len(df)):
        g = gold[i]
        pb = preds_base[i]
        ph = preds_hybrid[i]
        sid = str(df.iloc[i].get("sample_id", f"row_{i}"))
        vname = str(df.iloc[i].get("vessel_name", ""))
        ko_flag = bool(is_ko[i])

        if pb != g and ph == g:
            fixes.append({
                "sample_id": sid,
                "vessel_name": vname,
                "is_korean": ko_flag,
                "gold_id": g,
                "baseline_pred": pb,
                "hybrid_pred": ph,
            })
        elif pb == g and ph != g:
            breaks.append({
                "sample_id": sid,
                "vessel_name": vname,
                "is_korean": ko_flag,
                "gold_id": g,
                "baseline_pred": pb,
                "hybrid_pred": ph,
            })

    return {
        "num_fixes": len(fixes),
        "num_breaks": len(breaks),
        "net_gain": len(fixes) - len(breaks),
        "fixes": fixes,
        "breaks": breaks,
    }


def verify_prediction_independence(frame, model, index, matcher, options, predict_function=predict_rows):
    language=frame.vessel_name.str.contains('[가-힣]',regex=True)
    probes=pd.concat([frame.loc[language].head(6),frame.loc[~language].head(6)]).drop(
        columns=['target_vessel_id','target_vessel_id_fold','fold'],errors='ignore')
    def predict(data):
        return predict_function(data,model,index,use_train_lookup=False,dense_matcher=matcher,
                                dense_mode='hybrid',**options)[0]
    expected=predict(probes)
    return {'rows':len(probes), 'reverse':predict(probes.iloc[::-1])[::-1]==expected,
            'subset':predict(probes.iloc[::2])==expected[::2],
            'single':all(predict(probes.iloc[[i]])[0]==expected[i] for i in range(len(probes))),
            'repeat':predict(probes)==expected, 'empty':predict(probes.iloc[:0])==[]}


def main():
    p=argparse.ArgumentParser(description='SPEC-DPR-03 development benchmark; never creates ZIPs')
    p.add_argument('--experiment-id',default='DPR-HYBRID-03')
    p.add_argument('--catalog-csv',default=str(ROOT/'data/vessel_catalog.csv'))
    p.add_argument('--train-csv',default=str(ROOT/'data/train.csv'))
    p.add_argument('--folds-csv',default=str(ROOT/'eval_kit/folds_group5_seed42.csv'))
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--checkpoint',default=str(ROOT/'artifacts/DPR-SYMPHON-03/fold_0_encoder.pt'))
    p.add_argument('--catalog-embeds',default=str(ROOT/'artifacts/DPR-SYMPHON-03/catalog_embeds_fp16.npy'))
    p.add_argument('--reference-checkpoint',default=str(ROOT/'artifacts/DPR-SYMPHON-02/fold_0_encoder.pt'))
    p.add_argument('--reference-embeds',default=str(ROOT/'artifacts/DPR-SYMPHON-02/catalog_embeds_fp16.npy'))
    p.add_argument('--output-json',default=str(ROOT/'reports/matching_improvements/DPR-HYBRID-03/dpr_hybrid_robustness_benchmark.json'))
    p.add_argument('--seed',type=int,default=20261022)
    p.add_argument('--dense-weight',type=float,default=2.5)
    p.add_argument('--presets',default='default,severe')
    p.add_argument('--skip-clean',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=6)
    p.add_argument('--ko-map',default=str(ROOT/'AR-06/catalog_full_ko_map.json'))
    args=p.parse_args()
    if Path(args.output_json).exists():raise FileExistsError('Do not overwrite completed benchmark report')
    out,path,record=start_run(args.experiment_id)
    started=time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits
        import subprocess
        import importlib.util
        threadpool_limits(limits=args.threads)
        torch.set_num_threads(args.threads)
        catalog,fit,val,data=load_fold_data(args.train_csv,args.catalog_csv,args.folds_csv,args.fold)
        gold=val.target_vessel_id.tolist()
        language=val.vessel_name.apply(is_korean_name).to_numpy()
        # Fixed manual policy for every benchmark; fit-only learned maps remain enabled.
        from experimental_matchers.dpr_hybrid import script as current_script
        if current_script.IDEA_B_WORD_VARS:
            raise AssertionError('Primary benchmark requires IDEA_B_WORD_VARS == {}')
        tick=time.perf_counter()
        model=fit_matching_model(fit,catalog,kr_mode='AR06',en_mode='EN_V3',use_train_lookup=False)
        fit_seconds=time.perf_counter()-tick
        ko_map=json.loads(Path(args.ko_map).read_text(encoding='utf-8'))
        indices={}
        index_times={}
        for name,mapping in [('true_unseen',{}),('full_system',ko_map)]:
            tick=time.perf_counter()
            indices[name]=CatalogIndex(catalog,full_ko_map=mapping)
            index_times[name]=time.perf_counter()-tick
        load_tick=time.perf_counter()
        matcher=HybridDenseLexicalMatcher(args.checkpoint,args.catalog_embeds,catalog_df=catalog,device=args.device)
        reference=HybridDenseLexicalMatcher(args.reference_checkpoint,args.reference_embeds,catalog_df=catalog,device=args.device)
        load_seconds=time.perf_counter()-load_tick
        # Original scoring source is immutable and kept separate from the new implementation.
        source=ROOT/'artifacts/IMPL-DPR-SYMPHON-03/before/experimental_matchers/dpr_hybrid/script.py'
        spec=importlib.util.spec_from_file_location('_dpr03_reference_script',source)
        legacy=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        legacy.IDEA_B_WORD_VARS={}
        report={'experiment_id':args.experiment_id,'heldout_fold':args.fold,'rows':len(val),
                'evaluated_at_kst':now_kst(),'seed':args.seed,'device':args.device,'threads':args.threads,
                'initialization_seconds':{'lexical_fit':fit_seconds,'indices':index_times,'neural_load':load_seconds},
                'policy':{'manual_dictionary':False,'fit_only_learned_maps':True,
                          'true_unseen_legacy_catalog_map':False,'full_system_legacy_catalog_map':True,
                          'full_system_lookup':True,'true_unseen_lookup':False,
                          'language_partition':'original public train names, including under noise'},
                'results':{},'limitations':record.get('limitations',[])+[
                    'Full-system uses the legacy generated map; its historical lineage limits remain.',
                    'Reference DPR02 is remeasured with accelerated row encoding; this is not historical server timing.',
                    'Known Glovis/WREN cases motivated rules, so case success is development evidence, not untouched validation.',
                    'External fold0 is used for neural epoch selection; no leaderboard-equivalence claim.']}
        conditions=[]
        if not args.skip_clean:conditions.append(('clean',val,0.0))
        for preset in [v.strip() for v in args.presets.split(',') if v.strip()]:
            perturber=RobustnessPerturber(deepcopy(PRESETS[preset]),seed=args.seed)
            noisy,summary,_=perturber.perturb_dataset(val,catalog,return_row_audits=True)
            conditions.append((preset,noisy,float(summary.get('any_noise_applied_rate',0.))))
        for condition,frame,noise_rate in conditions:
            query=frame.drop(columns=['target_vessel_id','target_vessel_id_fold','fold'],errors='ignore')
            for system,index in indices.items():
                key=f'{system}_{condition}'
                variants=[('reference02',legacy.predict_rows,reference,{}),
                          ('retrieval03',predict_rows,matcher,{'dense_top_k':60,'dense_name_direct':False,'tail_rules':False}),
                          ('direct03',predict_rows,matcher,{'dense_top_k':60,'dense_name_direct':True,'tail_rules':False}),
                          ('complete03',predict_rows,matcher,{'dense_top_k':60,'dense_name_direct':True,'tail_rules':True})]
                result={'noise_applied_rate':noise_rate,'variants':{}}
                predicted={}
                paths={}
                for variant,function,neural,options in variants:
                    tick=time.perf_counter()
                    predictions,ranks=function(query,model,index,use_train_lookup=(system=='full_system'),
                                               gold_ids=gold,dense_matcher=neural,dense_mode='hybrid',
                                               dense_weight=args.dense_weight,**options)
                    seconds=time.perf_counter()-tick
                    metrics=calc_metrics(predictions,gold,language)
                    included=np.asarray(ranks)>0
                    metrics.update(inference_time_sec=seconds,
                                   candidate_inclusion_hangul=float(included[language].mean()),
                                   candidate_inclusion_non_hangul=float(included[~language].mean()),
                                   timing_scope='row normalization, candidate search, features and scoring; excludes load/index/save')
                    oof=out/f'oof_{args.experiment_id}_{key}_{variant}.csv'
                    pd.DataFrame({'sample_id':val.sample_id,'fold':args.fold,
                                  'pred_vessel_id':predictions,'gold_rank':ranks}).to_csv(oof,index=False)
                    result['variants'][variant]=metrics
                    predicted[variant]=predictions
                    paths[variant]=oof
                    print(key,variant,metrics,flush=True)
                for variant in ['retrieval03','direct03','complete03']:
                    base='reference02' if variant=='retrieval03' else 'retrieval03' if variant=='direct03' else 'direct03'
                    result[variant+'_vs_'+base]=find_diffs(frame,predicted[base],predicted[variant],gold,language)
                # Official evaluator and paired vessel-bootstrap CI for each complete system.
                command=[sys.executable,'-B','-X','utf8',str(ROOT/'eval_kit/evaluate.py'),
                         '--oof',str(paths['complete03']),'--compare',str(paths['reference02']),
                         '--train',args.train_csv,'--folds',args.folds_csv,'--subset']
                evaluation=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True,timeout=240)
                (out/f'official_{key}.txt').write_text(evaluation.stdout,encoding='utf-8')
                report['results'][key]=result
                write_json(args.output_json,report)
        options={'dense_top_k':60,'dense_name_direct':True,'tail_rules':True}
        independent=verify_prediction_independence(val,model,indices['true_unseen'],matcher,options)
        if not all(v for k,v in independent.items() if k!='rows'):
            raise AssertionError('Hybrid prediction row independence failed')
        report['row_independence']=independent
        report['input_sha256']={'checkpoint':sha256(args.checkpoint),'embeddings':sha256(args.catalog_embeds),
                                'reference_checkpoint':sha256(args.reference_checkpoint),
                                'reference_embeddings':sha256(args.reference_embeds),'legacy_map':sha256(args.ko_map),
                                'reference_scoring_source':sha256(source)}
        report['total_seconds']=time.perf_counter()-started
        write_json(args.output_json,report)
        primary=report['results'].get('true_unseen_clean',{}).get('variants',{}).get('complete03',{})
        record.update(data=data,benchmark=report,artifacts=[args.output_json,str(out)],
                      retrieval={'dense_top_k':60,'candidate_inclusion_is_not_fixed_k_recall':True})
        record['metrics'].update({k:v for k,v in primary.items() if k in record['metrics']})
        record['metrics']['metric_source']='outer-fold development-selected hybrid benchmark; official local evaluator, not server'
        record['cost'].update(inference_seconds=report['total_seconds'],timing_scope='entire benchmark including loading, all variants/noise, evaluator and saves')
        finish_run(path,record,'evaluated_development_selected')
        print('Saved benchmark:',args.output_json,flush=True)
    except BaseException as exc:
        finish_run(path,record,'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',exc)
        raise


if __name__=='__main__':
    main()
