"""Leakage-aware sklearn Pipeline for query-catalog PAIRS.

The binary label is whether a candidate is the correct vessel. At inference,
argmax over the retrieved candidates gives the original target_vessel_id.
Run a small execution check: python analysis/pair_pipeline.py --smoke
This demonstration does not overwrite the existing submission or its model.
"""
from pathlib import Path
import argparse
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

NUMERIC = ['name_ratio','roman_ratio','digits_equal','name_length_ratio',
           'gross_gap','international_gap','best_gap','tonnage_disagreement',
           'log_query_tonnage','log_catalog_tonnage','query_gt_missing',
           'query_igt_missing','catalog_gt_missing','query_name_missing','is_hangul']
CATEGORICAL = ['query_type','catalog_type','query_flag','catalog_flag','catalog_unit',
               'type_pair','flag_pair']


def normalize(x):
    return ''.join(c for c in unicodedata.normalize('NFKC',x).upper() if c.isalnum()) if isinstance(x,str) else ''


def roman(x):
    return normalize(unidecode(x)) if isinstance(x,str) else ''


def positive(series):
    x=pd.to_numeric(series,errors='coerce').astype(float)
    return x.where(np.isfinite(x)&x.gt(0))


class PairFeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless per-pair transformations. No query-group statistics."""
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X=X.reset_index(drop=True)
        q=X.query_name.map(normalize); c=X.catalog_name.map(normalize)
        qr=X.query_name.map(roman); cr=X.catalog_name.map(roman)
        gt=positive(X.query_gross); it=positive(X.query_international); ct=positive(X.catalog_gross)
        def gap(a,b):
            return abs(a-b)/np.maximum(a,b)
        dg=gap(gt,ct); di=gap(it,ct)
        result=pd.DataFrame({
            'name_ratio':[fuzz.ratio(a,b)/100 if a and b else 0 for a,b in zip(q,c)],
            'roman_ratio':[fuzz.ratio(a,b)/100 if a and b else 0 for a,b in zip(qr,cr)],
            'digits_equal':[int(re.sub(r'\D','',a)==re.sub(r'\D','',b)) for a,b in zip(q,c)],
            'name_length_ratio':[min(len(a),len(b))/max(len(a),len(b),1) for a,b in zip(q,c)],
            'gross_gap':dg,'international_gap':di,
            'best_gap':pd.concat([dg,di],axis=1).min(axis=1),
            'tonnage_disagreement':gap(gt,it),
            'log_query_tonnage':np.log1p(it.fillna(gt)),
            'log_catalog_tonnage':np.log1p(ct),
            'query_gt_missing':gt.isna().astype(int),
            'query_igt_missing':it.isna().astype(int),
            'catalog_gt_missing':ct.isna().astype(int),
            'query_name_missing':q.eq('').astype(int),
            'is_hangul':X.query_name.fillna('').str.contains('[가-힣]').astype(int),
        })
        for col in CATEGORICAL[:5]:
            result[col]=X[col].where(X[col].notna(),np.nan)
        # Both strings are inputs to the pair, not a target-encoded crosswalk.
        result['type_pair']=result.query_type.fillna('__MISSING__')+' | '+result.catalog_type.fillna('__MISSING__')
        result['flag_pair']=result.query_flag.fillna('__MISSING__')+' | '+result.catalog_flag.fillna('__MISSING__')
        return result


def make_pipeline():
    numeric=Pipeline([
        # Applied to comparison features, not to the original matching tonnage.
        ('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
        ('scale',RobustScaler()),
    ])
    categorical=Pipeline([
        ('impute',SimpleImputer(strategy='constant',fill_value='__MISSING__')),
        ('encode',OneHotEncoder(handle_unknown='infrequent_if_exist',
                                min_frequency=5,sparse_output=True)),
    ])
    prep=ColumnTransformer([('numeric',numeric,NUMERIC),('categorical',categorical,CATEGORICAL)],
                           remainder='drop',sparse_threshold=1.0)
    return Pipeline([('pair_features',PairFeatureBuilder()),('preprocess',prep),
                     ('classifier',LogisticRegression(C=1.0,class_weight='balanced',
                                                       solver='liblinear',max_iter=1000,random_state=42))])


class PairRetriever:
    """Catalog-only hard negatives and candidates, independent of all test rows."""
    def __init__(self,catalog):
        self.catalog=catalog.sort_values('vessel_id',kind='stable').reset_index(drop=True)
        self.ids=self.catalog.vessel_id.tolist()
        self.lookup={v:i for i,v in enumerate(self.ids)}
        self.names=self.catalog.vessel_name.map(roman).tolist()
        self.gt=positive(self.catalog.gross_tonnage).to_numpy()
        valid=np.flatnonzero(np.isfinite(self.gt))
        self.order=valid[np.argsort(self.gt[valid],kind='stable')]

    def retrieve(self,row):
        q=roman(row['vessel_name'])
        selected=set(i for _,_,i in process.extract(q,self.names,scorer=fuzz.ratio,limit=30)) if q else set()
        for col in ['gross_tonnage','international_gross_tonnage']:
            v=row[col]
            if pd.notna(v) and np.isfinite(v) and v>0:
                pos=np.searchsorted(self.gt[self.order],v)
                near=self.order[max(0,pos-30):pos+30]
                order=np.lexsort((near,np.abs(self.gt[near]-v)))[:30]
                selected.update(near[order].tolist())
        return sorted(selected) if selected else [0]

    def pairs(self,queries,training=False):
        records=[];labels=[];meta=[]
        catalog_rows=self.catalog.to_dict('records')
        for row in queries.to_dict('records'):
            candidates=self.retrieve(row)
            truth=row.get('target_vessel_id')
            if training:
                # Training needs a positive pair. Validation MUST NOT add it.
                candidates=sorted(set(candidates)|{self.lookup[truth]})
            for i in candidates:
                c=catalog_rows[i]
                records.append({'query_name':row['vessel_name'],'catalog_name':c['vessel_name'],
                    'query_gross':row['gross_tonnage'],'query_international':row['international_gross_tonnage'],
                    'catalog_gross':c['gross_tonnage'],'query_type':row['vessel_type'],
                    'catalog_type':c['vessel_type'],'query_flag':row['flag_name'],
                    'catalog_flag':c['flag_name'],'catalog_unit':c['unit']})
                labels.append(int(c['vessel_id']==truth))
                meta.append({'sample_id':row['sample_id'],'candidate_id':c['vessel_id']})
        return pd.DataFrame(records),np.asarray(labels),pd.DataFrame(meta)


def choose_ids(meta,scores):
    # This groups candidate pairs belonging to ONE input row, never test rows
    # with the same vessel/name/call sign. Stable ID tie-break within the row.
    ranked=meta.assign(score=scores).sort_values(['sample_id','score','candidate_id'],
                                                ascending=[True,False,True],kind='stable')
    return ranked.drop_duplicates('sample_id').set_index('sample_id').candidate_id


def smoke(data_dir,output_dir):
    t=pd.read_csv(data_dir/'train.csv');c=pd.read_csv(data_dir/'vessel_catalog.csv')
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(t,groups=t.target_vessel_id))
    fitted=t.iloc[a].sample(n=300,random_state=42)
    valid=t.iloc[b].sample(n=80,random_state=43)
    assert not set(fitted.target_vessel_id)&set(valid.target_vessel_id)
    retriever=PairRetriever(c)
    X,y,meta=retriever.pairs(fitted,training=True)
    V,vy,vm=retriever.pairs(valid,training=False)
    pipeline=make_pipeline()
    # Equal total weight per query and target language weighting within train.
    ko=fitted.vessel_name.str.contains('[가-힣]',na=False)
    lang_weight=pd.Series(np.where(ko,.7/max(ko.mean(),1e-9),.3/max((~ko).mean(),1e-9)),index=fitted.sample_id)
    sizes=meta.sample_id.value_counts()
    w=meta.sample_id.map(lang_weight).to_numpy()/meta.sample_id.map(sizes).to_numpy()
    w=w/w.mean()
    pipeline.fit(X,y,classifier__sample_weight=w)
    scores=pipeline.predict_proba(V)[:,1]
    predicted=choose_ids(vm,scores)
    # Feature preprocessing must not learn from validation, and predict must
    # remain stable for one row's candidate pairs in isolation.
    for sid in valid.sample_id.head(3):
        mask=vm.sample_id.eq(sid).to_numpy()
        separate=pipeline.predict_proba(V.loc[mask])[:,1]
        np.testing.assert_allclose(separate,scores[mask],rtol=1e-10,atol=1e-10)
        assert choose_ids(vm.loc[mask],separate).loc[sid]==predicted.loc[sid]
    order=np.random.default_rng(42).permutation(len(V))
    shuffled=choose_ids(vm.iloc[order],pipeline.predict_proba(V.iloc[order])[:,1])
    assert shuffled.sort_index().equals(predicted.sort_index())
    # Missing and unknown values must transform without refitting.
    edge=V.head(1).copy()
    edge[['query_gross','query_international','catalog_gross']]=np.nan
    edge['query_flag']='UNSEEN_FLAG';edge['query_type']='UNSEEN_TYPE';edge['query_name']=np.nan
    assert np.isfinite(pipeline.predict_proba(edge)).all()
    output_dir.mkdir(parents=True,exist_ok=True)
    result={'passed':True,'purpose':'execution check only, not a model performance estimate',
        'training_queries':len(fitted),'validation_queries':len(valid),
        'training_pairs':len(X),'validation_pairs':len(V),
        'training_positive_pair_fraction':float(y.mean()),
        'transformed_columns':len(pipeline.named_steps['preprocess'].get_feature_names_out()),
        'checks':['vessel-disjoint split','no forced validation positive','fit only on train',
                  'single-query prediction equality','shuffled-pair prediction equality','missing and unseen categories']}
    (output_dir/'pipeline_smoke.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output')
    args=parser.parse_args()
    if args.smoke:
        smoke(args.data_dir,args.output_dir)
    else:
        parser.print_help()
