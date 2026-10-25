"""Reproduce AS-05 server-result audit without model inference or ZIP modification."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
import json,hashlib,subprocess,zipfile,unicodedata
from pathlib import Path
from decimal import Decimal
from datetime import datetime,timezone,timedelta
import pandas as pd
R=Path(__file__).resolve().parents[1]
def readj(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
def sha(b):return hashlib.sha256(b).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',str(s)).upper().split())
def decode(s):
 out=[]
 for k in range(533):
  e=int(((s-Decimal('.7')*k/532)*3568/Decimal('.3')).to_integral_value())
  pred=Decimal('.7')*k/532+Decimal('.3')*e/3568
  if 0<=e<=3568 and abs(pred-s)<=Decimal('.00000000005'):
   out.append({'hangul_correct':k,'non_hangul_correct':e,'score':str(pred)})
 return out

def main():
 tr=pd.read_csv(R/'data/train.csv',encoding='utf-8-sig')
 cat=pd.read_csv(R/'data/vessel_catalog.csv',encoding='utf-8-sig')
 seen=set(tr.target_vessel_id)
 ci=cat.set_index('vessel_id',verify_integrity=True)
 kr=tr.vessel_name.str.contains('[가-힣]',na=False)
 obs=pd.to_numeric(tr.international_gross_tonnage,errors='coerce')
 obs=obs.where(obs.gt(0),pd.to_numeric(tr.gross_tonnage,errors='coerce'))
 gold=pd.to_numeric(tr.target_vessel_id.map(ci.gross_tonnage),errors='coerce')
 valid=obs.gt(0)&gold.gt(0)
 rel=((obs-gold).abs()/gold).where(valid)
 names=tr.vessel_name.map(norm)
 cat_names=tr.target_vessel_id.map(ci.vessel_name).map(norm)
 different=names.ne(cat_names)
 languages={}
 for name,mask in [('hangul',kr),('non_hangul',~kr)]:
  sub=tr[mask]
  d={'rows':len(sub),'unique_vessels':int(sub.target_vessel_id.nunique()),'tonnage_missing_rows':int((mask&obs.isna()).sum()),'flag_missing_rows':int(sub.flag_name.isna().sum()),'type_missing_rows':int(sub.vessel_type.isna().sum()),'observed_name_digit_fraction':float(sub.vessel_name.str.contains(r'\d').mean()),'tonnage_comparable_rows':int((mask&valid).sum()),'tonnage_relative_error_quantiles':{str(q):float(rel[mask].quantile(q)) for q in (.5,.9,.99,1)}}
  if name=='non_hangul':
   no_ton=mask&obs.isna()
   n=names.str.replace(r'[^A-Z0-9]','',regex=True)
   c=cat_names.str.replace(r'[^A-Z0-9]','',regex=True)
   d.update(name_mismatch_rows=int((mask&different).sum()),name_mismatch_fraction=float(different[mask].mean()),first_latin_alnum_preserved_fraction=float(n[mask].str[:1].eq(c[mask].str[:1]).mean()),name_mismatch_given_tonnage_missing=float(different[no_ton].mean()))
  languages[name]=d
 catalog_groups={}
 seen_types=set(cat.loc[cat.vessel_id.isin(seen),'vessel_type'])
 for name,mask in [('train_targets',cat.vessel_id.isin(seen)),('other_catalog',~cat.vessel_id.isin(seen))]:
  sub=cat[mask]
  catalog_groups[name]={'vessels':len(sub),'gross_tonnage_median':float(pd.to_numeric(sub.gross_tonnage,errors='coerce').median()),'name_contains_digit_fraction':float(sub.vessel_name.str.contains(r'\d').mean()),'type_absent_from_train_targets_fraction':float((~sub.vessel_type.isin(seen_types)).mean()),'top_types':{str(k):int(v) for k,v in sub.vessel_type.value_counts().head(10).items()}}
 with zipfile.ZipFile(R/'submit_AS-04.zip') as z:
  s4=z.read('script.py');b4=z.read('model/model.json');m4=json.loads(b4)
 with zipfile.ZipFile(R/'submit_AS-05.zip') as z:
  s5=z.read('script.py');b5=z.read('model/model.json')
  config=json.loads(z.read('model/as05_config.json'))
  manifest=json.loads(z.read('model/noisy_channel_weights/noisy_channel_manifest.json'))
 ar=readj('artifacts/ar06_handoff/train_AR-06/model/model.json')
 package={'as04_as05_base_asset_bytes_identical':b4==b5,'base_asset_sha256':sha(b4),'as04_script_sha256':sha(s4),'as05_script_sha256':sha(s5),'ar06_handoff_common_keys_equal':{k:ar[k]==m4[k] for k in sorted(set(ar)&set(m4))},'as04_added_keys':sorted(set(m4)-set(ar)),'as05_config':config,'as05_checkpoint':manifest['experiment_id'],'uploaded_ar06_zip_hash_verified':False,'source_diff_review':'AR06 handoff -> AS04 changes Korean metadata scoring and adds transition probability tables. AS04 -> AS05 adds both-language reranker calls and runtime/entrypoint setup.'}
 cached=pd.read_csv(R/'reports/diagnostics/baseline_candidates_fold0_top35.csv').sort_values(['sample_id','baseline_score','candidate_vessel_id'],ascending=[True,False,True])
 targets=tr.set_index('sample_id').target_vessel_id
 kr_map=pd.Series(kr.to_numpy(),index=tr.sample_id)
 gate={}
 for label,use_kr in [('hangul',True),('non_hangul',False)]:
  groups=[(sid,g) for sid,g in cached.groupby('sample_id',sort=False) if bool(kr_map[sid])==use_kr]
  eligible={sid for sid,g in groups if len(g)>1 and g.baseline_score.iloc[0]-g.baseline_score.iloc[1]<=1}
  errors={sid for sid,g in groups if g.candidate_vessel_id.iloc[0]!=targets[sid]}
  gate[label]={'rows':len(groups),'gate_le1_rows':len(eligible),'baseline_errors':len(errors),'errors_inside_gate':len(errors&eligible),'gold_in_top35_rows':sum(targets[sid] in set(g.candidate_vessel_id) for sid,g in groups)}
 a=Decimal('.9056363583');b=Decimal('.9054976988')
 score={'difference':str(b-a),'percentage_point_difference':str(100*(b-a)),'displayed_seconds':{'ar06':234,'as05':769,'extra':535},'time_ratio':769/234,'official_language_accuracies':None,'assumed_group_counts':{'hangul':532,'non_hangul':3568,'officially_confirmed':False},'all_ar06_solutions':decode(a),'all_as05_solutions':decode(b),'same_branch_net_delta':{'hangul':-1,'non_hangul':14},'crossbranch_deltas':[[56,-878],[-58,906]],'one_kr_value_in_en_rows':(.7/532)/(.3/3568),'conditional_hybrid_score':str(Decimal('.7')*473/532+Decimal('.3')*3383/3568),'hybrid_score_is_measured':False}
 sources=['data/train.csv','data/vessel_catalog.csv','eval_kit/folds_group5_seed42.csv','artifacts/ar06_handoff/train_AR-06/script.py','artifacts/ar06_handoff/train_AR-06/model/model.json','reports/diagnostics/baseline_candidates_fold0_top35.csv','experiments/analyze_as05_server_result.py']
 result={'analysis_id':'ANALYSIS-AS05-SERVER-01','recorded_at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),'scope':'Existing scores, source/assets, public data aggregates and cached candidates; no model inference/training','score_analysis':score,'package_audit':package,'public_data':{'train_rows':len(tr),'catalog_rows':len(cat),'train_target_vessels':len(seen),'catalog_coverage':len(seen)/len(cat),'languages':languages,'catalog_groups':catalog_groups,'definitions':'Names: NFKC uppercase collapsed whitespace; differences include renaming, punctuation and typos. Tonnage: positive international else gross; relative denominator catalog gross tonnage.'},'cached_fold0_gate_diagnostic':gate,'gate_scope':'Fold0-excluded cached candidates with rounded scores; not full-fit submission inference','test_file_present':(R/'data/test.csv').exists(),'source_sha256':{p:sha((R/p).read_bytes()) for p in sources},'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'limitations':['Hidden test distribution and individual predictions unavailable','Official scores and display durations supplied by user, not independently fetched','Group counts unconfirmed and decoding nonunique','Original uploaded AR06 ZIP hash unavailable','Server timing phases and LM call count unknown']}
 out=R/'reports/as05_server_analysis.json'
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({'score_analysis':score,'public_data':result['public_data'],'cached_fold0_gate_diagnostic':gate,'all_common_asset_keys_equal':all(package['ar06_handoff_common_keys_equal'].values())},ensure_ascii=False))
if __name__=='__main__':main()
