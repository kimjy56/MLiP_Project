"""Aggregate-only effect and independence checks; never exports copied train rows."""
import itertools
import json
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact
from independent_noise import noise_copy, PROBABILITIES

ROOT=Path(__file__).resolve().parents[2]
train=pd.read_csv(ROOT/'data/train.csv',dtype={'sample_id':str,'target_vessel_id':str})
catalog=pd.read_csv(ROOT/'data/vessel_catalog.csv',dtype={'vessel_id':str}).set_index('vessel_id',drop=False).to_dict('index')
flags=[]
for row in train.to_dict('records'):
    _, flag=noise_copy(row,catalog[row['target_vessel_id']])
    if flag['eligible']:
        flags.append(flag)
frame=pd.DataFrame(flags)
result={'rows':len(frame),'pairwise':{},'actual_name_change_conditional':{},
    'interpretation':'Separate deterministic PRNG streams by event; finite sample rates need not match. No seed selection based on these checks.',
    'multiple_testing':'Six pairwise tests; Bonferroni threshold 0.05/6. Test of draws, not resulting original-plus-noise field independence.'}
for a,b in itertools.combinations(PROBABILITIES,2):
    table=pd.crosstab(frame[a],frame[b]).reindex(index=[False,True],columns=[False,True],fill_value=0)
    odds,p=fisher_exact(table.to_numpy())
    result['pairwise'][a+' vs '+b]={'counts':table.to_numpy().tolist(),'odds_ratio':float(odds),'p_value':float(p),
        'reject_bonferroni_005':bool(p<.05/6)}
for kind in ['ton_missing','ton_shift','flag_missing']:
    result['actual_name_change_conditional'][kind]={str(v):float(frame.loc[frame[kind]==v,'name_applied'].mean()) for v in [False,True]}
path=ROOT/'artifacts/as02_handoff/results/noise_independence.json'
path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
