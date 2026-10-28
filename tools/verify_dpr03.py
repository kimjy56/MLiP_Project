"""Verify selected weights, protected artifacts and native row independence without ZIPs."""
from pathlib import Path
import argparse
import json
import sys
import time
import torch
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.dpr03_common import (start_run,finish_run,write_json,sha256,load_fold_data,composite_score)
from experimental_matchers.dpr_hybrid.script import CatalogIndex,fit_matching_model,predict_rows,IDEA_B_WORD_VARS
from phonetic_encoder.hybrid_matcher import HybridDenseLexicalMatcher


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-id',default='VERIFY-DPR03-01')
    p.add_argument('--rows',type=int,default=300)
    args=p.parse_args()
    out,path,record=start_run(args.experiment_id)
    started=time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=6)
        torch.set_num_threads(6)
        checkpoint=ROOT/'artifacts/DPR-SYMPHON-03/fold_0_encoder.pt'
        history=json.loads((checkpoint.parent/'training_history.json').read_text(encoding='utf-8'))
        best=max(history['epochs'],key=lambda e:composite_score(e['retrieval']))
        assert best['epoch']==history['best']['epoch']
        assert sha256(checkpoint)==history['best']['checkpoint_sha256']
        selected=torch.load(checkpoint,map_location='cpu',weights_only=True)
        archived=torch.load(checkpoint.parent/'epochs'/f"epoch_{best['epoch']}.pt",map_location='cpu',weights_only=True)
        for group in ['encoder_state_dict','fusion_state_dict']:
            assert selected[group].keys()==archived[group].keys()
            assert all(torch.equal(selected[group][key],archived[group][key]) for key in selected[group])
        protected=json.loads((ROOT/'artifacts/IMPL-DPR-SYMPHON-03/reference_manifest.json').read_text(encoding='utf-8'))
        changed=[file for file,digest in protected['protected_files_sha256'].items() if sha256(ROOT/file)!=digest]
        if changed:raise AssertionError('Protected assets changed: '+str(changed))
        catalog,fit,val,data=load_fold_data(ROOT/'data/train.csv',ROOT/'data/vessel_catalog.csv',ROOT/'eval_kit/folds_group5_seed42.csv',0)
        mask=val.vessel_name.str.contains('[가-힣]',regex=True)
        probes=pd.concat([val.loc[mask].head(args.rows//2),val.loc[~mask].head(args.rows-args.rows//2)]).drop(
            columns=['target_vessel_id','target_vessel_id_fold','fold'],errors='ignore').reset_index(drop=True)
        if IDEA_B_WORD_VARS:raise AssertionError('Manual dictionary must be empty')
        model=fit_matching_model(fit,catalog,use_train_lookup=False)
        index=CatalogIndex(catalog,full_ko_map={})
        dense=HybridDenseLexicalMatcher(checkpoint,checkpoint.parent/'catalog_embeds_fp16.npy',catalog_df=catalog,device='cpu')
        def predict(frame):
            return predict_rows(frame,model,index,use_train_lookup=False,dense_matcher=dense,dense_mode='hybrid',
                                dense_top_k=60,dense_name_direct=True,tail_rules=True)[0]
        expected=predict(probes)
        print('Computed native full prediction for',len(probes),'public rows',flush=True)
        reverse=predict(probes.iloc[::-1])[::-1]
        subset=predict(probes.iloc[::2])
        sample_indices=list(dict.fromkeys([0,len(probes)//2-1,len(probes)-1]))
        single=[predict(probes.iloc[[i]])[0] for i in sample_indices]
        repeat=predict(probes)
        result={'rows':len(probes),'hangul_rows':int(probes.vessel_name.str.contains('[가-힣]',regex=True).sum()),
                'reverse_mismatches':sum(a!=b for a,b in zip(expected,reverse)),
                'subset_rows':len(subset),'subset_mismatches':sum(a!=b for a,b in zip(expected[::2],subset)),
                'single_rows':len(single),'single_mismatches':sum(expected[i]!=v for i,v in zip(sample_indices,single)),
                'repeat_mismatches':sum(a!=b for a,b in zip(expected,repeat)),
                'empty_pass':predict(probes.iloc[:0])==[], 'selected_epoch':best['epoch'],
                'selected_weights_match_archived_epoch':True,'protected_files_verified':len(protected['protected_files_sha256']),
                'embedding_checkpoint_provenance_verified':dense.provenance_verified,
                'scope':'native source and selected model on public fold0; official ZIP checker not executed'}
        assert all(v==0 for k,v in result.items() if k.endswith('mismatches'))
        assert result['empty_pass'] and dense.provenance_verified
        write_json(out/'verification.json',result)
        record.update(data=data,verification=result,artifacts=[str(out/'verification.json')])
        record['metrics']['metric_source']='native row-independence and asset verification, no new accuracy experiment'
        record['cost'].update(inference_seconds=time.perf_counter()-started,timing_scope='verification including data/model load, all prediction arrangements and file hashes')
        finish_run(path,record,'verified_native_no_zip')
        print(result,flush=True)
    except BaseException as exc:
        finish_run(path,record,'failed',exc)
        raise


if __name__=='__main__':
    main()
