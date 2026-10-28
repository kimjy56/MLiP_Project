"""Cross-fit AR06 confusion mining inside external fit folds; no external fold0."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'AR-06'))
from experiments.noisy_channel_common import configure_cpu,reserve_experiment,write_json,record_failure,sha256,now_kst
configure_cpu(2)
out,path,record=reserve_experiment('INNER-CONFUSIONS-01',ROOT/'artifacts','Mine observed model confusions inside external training folds',' '.join(sys.argv))
record.update(hypothesis='Actual high-scoring wrong candidates may be more useful than lexical-only negatives.',changed_factors=['fit-inner group cross-fitting'],fixed_conditions=['external fold0 excluded','no lookup','same public catalog'],next_actions=['Use cross-fitted candidate names for reverse LM ranking; compare at300 steps.'])
write_json(path,record)
try:
 import pandas as pd
 from script import CatalogIndex,fit_matching_model,predict_rows
 train=pd.read_csv(ROOT/'data/train.csv',keep_default_na=False)
 folds=pd.read_csv(ROOT/'eval_kit/folds_group5_seed42.csv',keep_default_na=False)
 cat=pd.read_csv(ROOT/'data/vessel_catalog.csv',keep_default_na=False)
 rows=train.merge(folds[['sample_id','fold']],on='sample_id',validate='one_to_one')
 fit=rows[rows.fold!=0].copy();held=rows[rows.fold==0]
 assert not set(fit.target_vessel_id)&set(held.target_vessel_id)
 index=CatalogIndex(cat);names=cat.set_index('vessel_id').vessel_name.to_dict()
 records=[];start=time.perf_counter()
 for inner in [1,2,3,4]:
  learning=fit[fit.fold!=inner];probe=fit[fit.fold==inner]
  assert not set(learning.target_vessel_id)&set(probe.target_vessel_id)
  model=fit_matching_model(learning,cat,kr_mode='AR06',en_mode='EN_V3',use_train_lookup=False)
  preds,_,cands=predict_rows(probe.drop(columns=['target_vessel_id','fold']),model,index,use_train_lookup=False,top_candidates=8)
  for row,pred,items in zip(probe.itertuples(),preds,cands):
   for rank,(score,vid) in enumerate(items,1):
    if vid!=row.target_vessel_id and names[vid].upper()!=names[row.target_vessel_id].upper():
     records.append(dict(sample_id=row.sample_id,inner_fold=inner,observed_name=row.vessel_name,catalog_name=names[row.target_vessel_id],negative_name=names[vid],negative_vessel_id=vid,baseline_rank=rank,baseline_score=score,baseline_wrong=pred!=row.target_vessel_id))
  pd.DataFrame(records).to_csv(out/'confusions.csv',index=False)
  print(f'inner={inner} completed probe_rows={len(probe)} candidates={len(records)}',flush=True)
 record.update(status='mined_untrained',finished_at_kst=now_kst(),data={'external_heldout_fold':0,'training_rows':len(fit),'inner_folds':[1,2,3,4],'fold_sha256':sha256(ROOT/'eval_kit/folds_group5_seed42.csv')},retrieval={'top_candidates':8,'lookup':False,'gold_injected':False},artifacts=[str(out)],artifact_sha256={'confusions.csv':sha256(out/'confusions.csv')},observations=['AR06 confusions, not reverse-LM confusions; lexical fallback and ambiguous-positive exclusions required before training.'],cost={'mining_seconds':time.perf_counter()-start},decision='Mining only; do not claim accuracy improvement or promote.')
 write_json(path,record)
except BaseException as exc:
 record_failure(path,record,exc);raise
