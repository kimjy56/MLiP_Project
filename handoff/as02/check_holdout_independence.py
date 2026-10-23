"""Exercise actual scoring on public held-out rows, not full-train direct lookups."""
import json
import pandas as pd
from run_checks import ROOT, load

module=load(ROOT/'handoff/as02/script.py','as02_holdout')
model=json.loads((ROOT/'artifacts/as02_handoff/results/model_fold_0.json').read_text(encoding='utf-8'))
train=pd.read_csv(ROOT/'data/train.csv',dtype={'sample_id':str})
folds=pd.read_csv(ROOT/'eval_kit/folds_group5_seed42.csv',dtype={'sample_id':str})
held=train.merge(folds[['sample_id','fold']],on='sample_id')
held=held[held.fold==0]
ko=held.vessel_name.map(module.is_korean_name)
query=pd.concat([held[ko].head(12),held[~ko].head(12)]).drop(columns=['target_vessel_id','fold']).reset_index(drop=True)
index=module.CatalogIndex(pd.read_csv(ROOT/'data/vessel_catalog.csv'))
def predict(frame):
    return module.predict_rows(frame,model,index,use_train_lookup=False,disable_korean_metadata=True)[0]
base=predict(query)
assert predict(query.iloc[::-1])==base[::-1]
assert predict(query.iloc[:9])==base[:9]
assert [predict(query.iloc[[i]])[0] for i in range(len(query))]==base
assert predict(query.iloc[:0])==[]
result={'passed':True,'rows':len(query),'lookup':False,'checks':['reverse','subset','single','empty'],
        'source':'official public train fold 0, fold-trained model'}
(ROOT/'artifacts/as02_handoff/results/holdout_independence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)
