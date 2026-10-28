"""Clean full fold-0 fixed-candidate GPU evaluation. No ZIP or weight tuning."""
import sys
import argparse
from pathlib import Path
import time
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.noisy_channel_common import configure_cpu, reserve_experiment, record_failure, write_json, sha256, source_hashes, now_kst

def run(out,path,record,args):
    configure_cpu(2)
    import numpy as np
    import pandas as pd
    import torch
    from name_decoder.data import read_frames
    from name_decoder.noisy_channel import NoisyChannelScorer, MANIFEST_NAME, encode_channel_example, collate_channel_examples
    from name_decoder.channel_training import sequence_nll
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(ROOT/'AR-06'))
    from script import CatalogIndex, fit_matching_model, predict_rows
    begin=time.perf_counter()
    hashes=source_hashes(ROOT/'data',ROOT/'eval_kit/folds_group5_seed42.csv')
    _,val,fit,catalog=read_frames(ROOT/'data',ROOT/'eval_kit/folds_group5_seed42.csv',0)
    val=val.sort_values('sample_id').reset_index(drop=True)
    modeldir=args.model_path
    manifest=json.loads((modeldir/MANIFEST_NAME).read_text(encoding='utf-8'))
    assert manifest['heldout_fold']==0 and manifest['data_sha256']==hashes
    if json.loads((modeldir/'config.json').read_text(encoding='utf-8')).get('model_type') == 'lfm2':
        torch.backends.cudnn.enabled=False  # Avoid batch-size-dependent convolution algorithm differences.
    assert not set(fit.target_vessel_id)&set(val.target_vessel_id)
    record.update(hypothesis='Evaluate new reverse LM standalone and fixed AS05 fusion without tuning.',
        fixed_conditions=['clean full fold0','AR06/EN_V3 fit only fold != 0','lookup disabled','top35 fixed shortlist','fusion weight 0.20; gate margin <=1.0'],
        data={'source_files_sha256':hashes,'split_id':'G5-s42 fold0 only','seed':42,'validation_rows':len(val),'training_rows':len(fit),'scope':'single_fold_not_full_OOF','seen_vessels':0},
        model={'checkpoint':str(modeldir),'manifest_sha256':sha256(modeldir/MANIFEST_NAME)},
        environment={'python':sys.version,'torch':torch.__version__,'device':'cuda','gpu':torch.cuda.get_device_name(0)},
        limitations=['Not an untouched final test; public train already explored.','Previous LM weights unavailable: no direct V1/V2 improvement claim.','Fold0 only; not five-fold OOF.','Combined changes cannot isolate individual factor effects.'])
    record['code']['file_sha256']['experiments/evaluate_lm_rank_gpu.py']=sha256(__file__)
    write_json(path,record)
    catdf=pd.read_csv(ROOT/'data/vessel_catalog.csv',keep_default_na=False,dtype={'vessel_id':str})
    if args.candidates_csv:
        fixed=pd.read_csv(args.candidates_csv,keep_default_na=False,dtype={'sample_id':str,'candidate_vessel_id':str})
        assert set(fixed.sample_id)==set(val.sample_id)
        assert not fixed.duplicated(['sample_id','candidate_vessel_id']).any()
        assert set(fixed.candidate_vessel_id)<=set(catalog.name_by_id)
        assert np.isfinite(fixed.baseline_score.to_numpy()).all()
        groups={sid:g for sid,g in fixed.groupby('sample_id',sort=False)}
        lists=[list(zip(groups[sid].baseline_score,groups[sid].candidate_vessel_id)) for sid in val.sample_id]
        basepred=[items[0][1] for items in lists]
        candidate_rows=fixed.to_dict('records')
        record['cost'].update(baseline_index_fit_seconds=None,baseline_inference_seconds=None)
        record['fixed_candidate_source']={'path':str(args.candidates_csv),'sha256':sha256(args.candidates_csv),'retrieval_recomputed':False}
    else:
        t=time.perf_counter()
        index=CatalogIndex(catdf)
        base=fit_matching_model(fit,catdf,kr_mode='AR06',en_mode='EN_V3',use_train_lookup=False)
        record['cost']['baseline_index_fit_seconds']=time.perf_counter()-t
        t=time.perf_counter()
        basepred,_,lists=predict_rows(val.drop(columns=['target_vessel_id','target','fold'],errors='ignore'),base,index,use_train_lookup=False,top_candidates=35)
        record['cost']['baseline_inference_seconds']=time.perf_counter()-t
        candidate_rows=[{'sample_id':row.sample_id,'candidate_vessel_id':vid,'baseline_score':score}
                        for row,items in zip(val.itertuples(),lists) for score,vid in items]
    cpath=out/'candidates.csv'
    pd.DataFrame(candidate_rows).to_csv(cpath,index=False)
    record['retrieval']={'count_max':35,'candidate_pairs':len(candidate_rows),'candidate_sha256':sha256(cpath),'gold_injected':False,'fit_fold_verified':True,'baseline_code_sha256':sha256(ROOT/'AR-06/script.py')}
    print(f'baseline done rows={len(val)} pairs={len(candidate_rows)}',flush=True)
    t=time.perf_counter()
    scorer=NoisyChannelScorer.load(modeldir)
    scorer.model.to('cuda');scorer.device=torch.device('cuda')
    record['cost']['model_load_seconds']=time.perf_counter()-t
    results=[]
    max_check=0.0
    lfm = scorer.model.config.model_type in ("lfm2", "exaone4")
    tolerance = 1e-3 if lfm else 1e-4
    refined_rows = 0
    unsupported=0
    started=time.perf_counter()
    for k,(row,cands,bpred) in enumerate(zip(val.itertuples(),lists,basepred),1):
        names=sorted({catalog.name_by_id[vid] for _,vid in cands})
        name_scores={}
        try:
            examples=[encode_channel_example(scorer.tokenizer,n,row.vessel_name,scorer.max_length,prompt_version=manifest['prompt_version']) for n in names]
            with torch.inference_mode():
                for offset in range(0,len(examples),8):
                    chunk=examples[offset:offset+8]
                    batch={key:value.to('cuda') for key,value in collate_channel_examples(chunk,scorer.pad_token_id).items()}
                    outputs=scorer.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
                    means=-sequence_nll(outputs.logits,batch['labels'])
                    for name,score in zip(names[offset:offset+8],means.cpu().tolist()):
                        name_scores[name]=score
                    del outputs,batch,means
            if k==1:
                for name in names:
                    delta=abs(scorer.score_pair(name,row.vessel_name).mean_log_probability-name_scores[name])
                    max_check=max(max_check,delta)
                if max_check>tolerance:raise RuntimeError(f'Batch/single likelihood mismatch: {max_check}')
            if not all(np.isfinite(list(name_scores.values()))):raise FloatingPointError('Nonfinite scores')
            # Recheck near-tied top candidates with single-pair scoring.
            lm_order=sorted(name_scores,key=lambda n:-name_scores[n])
            fused_names={catalog.name_by_id[vid]:float(sc)+0.2*name_scores[catalog.name_by_id[vid]] for sc,vid in cands}
            best_lm=name_scores[lm_order[0]]
            best_fused=max(fused_names.values())
            near={name for name in names if best_lm-name_scores[name]<=2*tolerance or best_fused-fused_names[name]<=0.4*tolerance}
            if len(near)>2:
                for name in near:
                    name_scores[name]=scorer.score_pair(name,row.vessel_name).mean_log_probability
                refined_rows+=1
            ranked=sorted([(sc,vid,name_scores[catalog.name_by_id[vid]]) for sc,vid in cands],key=lambda v:(-v[0],v[1]))
            lm=sorted(ranked,key=lambda v:(-v[2],v[1]))
            fused=sorted(ranked,key=lambda v:(-(v[0]+0.2*v[2]),v[1]))
            eligible=len(ranked)>1 and ranked[0][0]-ranked[1][0]<=1.0 and len(names)>1
            lmid=lm[0][1]; fid=fused[0][1]; gated=fid if eligible else bpred
            lrank=next((i for i,v in enumerate(lm,1) if v[1]==row.target_vessel_id),0)
        except (ValueError,TypeError):
            unsupported+=1
            lmid=fid=gated=bpred;lrank=0;eligible=False
        goldrank=next((i for i,(_,vid) in enumerate(cands,1) if vid==row.target_vessel_id),0)
        results.append({'sample_id':row.sample_id,'fold':0,'target_vessel_id':row.target_vessel_id,'hangul':bool(row.hangul),'seen':False,'baseline':bpred,'lm':lmid,'fusion':fid,'gated':gated,'gold_rank':goldrank,'lm_gold_rank':lrank,'eligible':eligible})
        if k==1 or k%100==0 or k==len(val):
            pd.DataFrame(results).to_csv(out/'progress_predictions.csv',index=False)
            print(f'scored={k}/{len(val)} seconds={time.perf_counter()-started:.1f}',flush=True)
    torch.cuda.synchronize()
    record['cost'].update(inference_seconds=time.perf_counter()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),total_seconds=time.perf_counter()-begin,timing_scope='baseline index/fit, baseline retrieval and LM load separate; inference includes encoding, batched scoring, ranks and progress writes',submission_zip_bytes=None)
    df=pd.DataFrame(results)
    metrics={}
    for mode in ['baseline','lm','fusion','gated']:
        corr=df[mode].eq(df.target_vessel_id); bc=df.baseline.eq(df.target_vessel_id)
        h=df.hangul; nh=~h
        metrics[mode]={'hangul_correct':int(corr[h].sum()),'non_hangul_correct':int(corr[nh].sum()),'hangul_rows':int(h.sum()),'non_hangul_rows':int(nh.sum()),'hangul_accuracy':float(corr[h].mean()),'non_hangul_accuracy':float(corr[nh].mean()),'weighted_accuracy':float(0.7*corr[h].mean()+0.3*corr[nh].mean()),'fixes':int((corr&~bc).sum()),'breaks':int((~corr&bc).sum())}
        ranks=df.lm_gold_rank if mode=='lm' else df.gold_rank
        pd.DataFrame({'sample_id':df.sample_id,'fold':0,'pred_vessel_id':df[mode],'gold_rank':ranks}).to_csv(out/f'predictions_{mode}.csv',index=False)
    record['metrics']=metrics
    record['retrieval'].update(hangul_candidate_inclusion=float(df.loc[df.hangul,'gold_rank'].gt(0).mean()),non_hangul_candidate_inclusion=float(df.loc[~df.hangul,'gold_rank'].gt(0).mean()),count_min=min(map(len,lists)))
    record.update(status='evaluated_single_fold',finished_at_kst=now_kst(),batch_single_max_abs_difference=max_check, batch_single_tolerance=tolerance, single_refined_rows=refined_rows,unsupported_rows=unsupported,eligible_rows=int(df.eligible.sum()),failure=None,artifacts=[str(out)],decision='No weight tuning or deployment; interpret measured fixes/breaks before adoption.')
    df.to_csv(out/'paired_predictions.csv',index=False)
    record['artifact_sha256']={p.name:sha256(p) for p in out.glob('*.csv')}
    write_json(path,record);write_json(out/'run_record.json',record)
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':
    configure_cpu(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id',default='EVAL-LM-RANK-01')
    parser.add_argument('--model-path',type=Path,default=ROOT/'artifacts/LM-RANK-GPU-02/model')
    parser.add_argument('--candidates-csv',type=Path)
    args=parser.parse_args()
    out,path,record=reserve_experiment(args.experiment_id,ROOT/'reports/noisy_channel','Clean heldout fold0 LM ranking evaluation',' '.join(sys.argv))
    try:run(out,path,record,args)
    except BaseException as exc:
        record_failure(path,record,exc)
        raise
