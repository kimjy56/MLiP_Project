"""All 16 S/N/P/T subsets, retrained separately on fixed public-train splits.

Only deterministic, per-row candidate features are cached. Each subset builds
its own negatives and model. Test distributions are never read or fitted.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train import MODEL_PARAMETERS
from matcher import CatalogIndex, Matcher, make_assets, norm, hangul

CHANNELS='SNPT'


def name_for(mask):
    return ''.join(c for j,c in enumerate(CHANNELS) if mask & (1<<j)) or 'NONE'


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def build_cache(frame,index,training,folder,label):
    entries=[]
    times={c:0. for c in CHANNELS}
    feature_seconds=0.
    started=time.monotonic()
    for k,row in enumerate(frame.to_dict('records')):
        membership={0:0}  # catalog-ID-first fallback, not a search channel
        for j,c in enumerate(CHANNELS):
            s=time.monotonic()
            ids=index.channel_candidates(row,c)
            times[c]+=time.monotonic()-s
            for i in ids:membership[i]=membership.get(i,0)|(1<<j)
        for vessel in index.assets['aliases'].get(norm(row.get('vessel_name')),[]):
            if vessel in index.lookup:
                i=index.lookup[vessel];membership[i]=membership.get(i,0)|16
        truth=index.lookup[row['target_vessel_id']]
        if training:membership.setdefault(truth,0)  # training positive only
        ids=sorted(membership)
        s=time.monotonic()
        x=index.features(row,ids)
        feature_seconds+=time.monotonic()-s
        entries.append({'row':row,'ids':np.asarray(ids,dtype=np.int32),
                        'bits':np.asarray([membership[i] for i in ids],dtype=np.uint8),
                        'x':x,'truth':truth,'direct':index.direct(row)})
        if (k+1)%500==0:
            print(f'{label}: cache {k+1}/{len(frame)} rows; {time.monotonic()-started:.1f}s',flush=True)
    # Public-train features stay in memory; persist only reproducibility evidence.
    summary={'rows':len(entries),'pairs_in_union':sum(len(e['ids']) for e in entries),
             'channel_search_seconds':times,'all_union_feature_seconds':feature_seconds,
             'wall_seconds':time.monotonic()-started,
             'timing_scope':'one pass: all four channels, union features, bookkeeping; no model fitting/scoring',
             'positive_injection':training}
    save(folder/(label+'_cache.json'),summary)
    return entries,summary


def select_positions(entry,mask,aliases=False):
    bits=entry['bits'];keep=(bits & mask)!=0
    if aliases:keep|=(bits & 16)!=0
    positions=np.flatnonzero(keep)
    empty=not len(positions)
    if empty:positions=np.asarray([int(np.searchsorted(entry['ids'],0))])
    return positions,empty


def fit_subset(entries,index,mask,destination):
    destination.mkdir(parents=True,exist_ok=True)
    assets={**index.assets,'retrieval_channels':[c for j,c in enumerate(CHANNELS) if mask&(1<<j)]}
    save(destination/'assets.json',assets)
    if not mask:
        return None,{'training_pairs':0,'model_trained':False,
                     'reason':'No search: fixed first-catalog-ID baseline, with optional production lookup/alias handling.'}
    start=time.monotonic()
    kh=np.mean([hangul(e['row']['vessel_name']) for e in entries])
    matrices=[];labels=[];weights=[];recalled=[]
    fingerprint=hashlib.sha256()
    for e in entries:
        p,_=select_positions(e,mask)
        truth_pos=int(np.searchsorted(e['ids'],e['truth']))
        recalled.append(truth_pos in p)
        if truth_pos not in p:p=np.sort(np.append(p,truth_pos))
        x=e['x'][p]
        if len(p)>100:
            hardness=np.maximum(x[:,1],x[:,4])-.5*x[:,11]
            selected=set(np.argsort(-hardness,kind='stable')[:80].tolist())
            selected.update(np.argsort(x[:,11],kind='stable')[:20].tolist())
            selected.add(int(np.flatnonzero(p==truth_pos)[0]))
            selected=sorted(selected);p=p[selected];x=x[selected]
        ids=e['ids'][p]
        fingerprint.update(json.dumps([e['row']['sample_id'],[index.ids[i] for i in ids]],ensure_ascii=False).encode())
        matrices.append(x);labels.extend((ids==e['truth']).astype(int).tolist())
        weight=.7/max(kh,1e-9) if hangul(e['row']['vessel_name']) else .3/max(1-kh,1e-9)
        weights.extend([weight]*len(p))
    x=np.concatenate(matrices)
    model=lgb.LGBMClassifier(**MODEL_PARAMETERS)
    model.fit(x,np.asarray(labels),sample_weight=np.asarray(weights),feature_name=index.feature_names)
    model.booster_.save_model(str(destination/'ranker.txt'))
    result={'training_pairs':len(labels),'unique_feature_rows':len(entries),'model_trained':True,
            'training_pair_sha256':fingerprint.hexdigest(),'training_candidate_recall':float(np.mean(recalled)),
            'pair_selection_and_fit_seconds':time.monotonic()-start,
            'timing_scope':'subset pair selection, concatenate, LightGBM fit and model write; shared feature cache excluded',
            'model_bytes':(destination/'ranker.txt').stat().st_size,'model_sha256':sha(destination/'ranker.txt')}
    return model.booster_,result


def evaluate(entries,index,model,mask,training_ids,production,destination,label):
    start=time.monotonic()
    chosen=[];matrices=[];direct=[];empty=[]
    for e in entries:
        d=e['direct'] if production else None
        direct.append(d)
        p,is_empty=select_positions(e,mask,aliases=production)
        chosen.append(p);empty.append(is_empty)
        if not d:matrices.append(e['x'][p])
    all_x=np.concatenate(matrices) if matrices else np.empty((0,len(index.feature_names)),dtype=np.float32)
    scores=model.predict(all_x,num_threads=1) if model is not None and len(all_x) else np.zeros(len(all_x))
    offset=0;pred=[];recalled=[];counts=[];errors=[]
    fingerprint=hashlib.sha256()
    for e,p,d,is_empty in zip(entries,chosen,direct,empty):
        if d:
            prediction=d;candidate_ids=[d];recall=d==e['row']['target_vessel_id'];count=1
        else:
            candidate_ids=[index.ids[i] for i in e['ids'][p]]
            prediction=candidate_ids[int(np.argmax(scores[offset:offset+len(p)]))]
            offset+=len(p);recall=e['truth'] in e['ids'][p];count=len(p)
        fingerprint.update(json.dumps([e['row']['sample_id'],candidate_ids]).encode())
        pred.append(prediction);recalled.append(recall);counts.append(count)
        if prediction!=e['row']['target_vessel_id']:
            stage='lookup' if d else 'empty_fallback' if is_empty else 'selection' if recall else 'retrieval'
            errors.append({'sample_id':e['row']['sample_id'],'truth':e['row']['target_vessel_id'],
                           'prediction':prediction,'stage':stage})
    elapsed=time.monotonic()-start
    frame=pd.DataFrame([e['row'] for e in entries])
    ko=frame.vessel_name.map(hangul).to_numpy();seen=frame.target_vessel_id.isin(training_ids).to_numpy()
    correct=frame.target_vessel_id.to_numpy()==np.asarray(pred)
    has_direct=np.array([bool(d) for d in direct])
    nonlookup_counts=np.asarray(counts)[~has_direct]
    result={'rows':len(frame),'seen_vessel_fraction':float(seen.mean()),
            'call_lookup_rows':int(has_direct.sum()),'empty_search_rows_nonlookup':int((np.asarray(empty)&~has_direct).sum()),
            'candidate_sha256':fingerprint.hexdigest(),'errors':len(errors),
            'error_stages':{s:sum(e['stage']==s for e in errors) for s in ['lookup','empty_fallback','retrieval','selection']},
            'cached_evaluation_seconds':elapsed,
            'timing_scope':'cached feature selection, batched model scoring and diagnostics; excludes retrieval, feature generation and CSV writes',
            'candidate_count_nonlookup':{'mean':float(nonlookup_counts.mean()),'p95':float(np.quantile(nonlookup_counts,.95)),
                                         'max':int(nonlookup_counts.max())} if len(nonlookup_counts) else None}
    for name,m in [('hangul',ko),('non_hangul',~ko),('seen',seen),('unseen',~seen)]:
        result[name]={'rows':int(m.sum()),'unique_vessels':int(frame.loc[m,'target_vessel_id'].nunique()),
                      'accuracy':float(correct[m].mean()) if m.any() else None,
                      'candidate_recall':float(np.asarray(recalled)[m].mean()) if m.any() else None}
    result['weighted_accuracy']=.7*result['hangul']['accuracy']+.3*result['non_hangul']['accuracy']
    output=pd.DataFrame({'sample_id':frame.sample_id,'truth':frame.target_vessel_id,'prediction':pred})
    output.to_csv(destination/(label+'_predictions.csv'),index=False)
    save(destination/(label+'_errors.json'),errors)
    return result,output


def probe(matcher,frame,expected):
    # Run actual per-row inference, not cached evaluator, including N-gram search.
    selected=pd.concat([frame.head(3),frame[frame.vessel_name.map(hangul)].head(3)]).drop_duplicates('sample_id')
    want=[expected.loc[s,'prediction'] for s in selected.sample_id]
    assert matcher.predict(selected)==want
    shuffled=selected.sample(frac=1,random_state=7)
    assert matcher.predict(shuffled)==[expected.loc[s,'prediction'] for s in shuffled.sample_id]
    for i in range(len(selected)):
        assert matcher.predict(selected.iloc[[i]])==[want[i]]
    assert matcher.predict(selected.iloc[::2])==want[::2]
    return {'passed':True,'rows':len(selected),'checks':['cached-vs-runtime','full','shuffled','single','subset']}


def run(args):
    record=json.loads(args.record.read_text())
    assert record['status']=='planned','Use a new planned record, not a completed run'
    assert not args.output_dir.exists(),'Use a new artifact folder'
    args.output_dir.mkdir(parents=True)
    record.update(status='running',started_at_kst=now(),commands=[' '.join(sys.argv)],variants={})
    sources=['model/matcher.py','model/name_processing.py','model/ngram_retrieval.py','train.py','experiments/retrieval_factorial.py']
    record['code']={'git_revision_at_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
                    'file_sha256':{p:sha(ROOT/p) for p in sources}}
    for p in sources:
        dest=args.output_dir/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    record['environment']={'python':platform.python_version(),'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','rapidfuzz','Unidecode']}}
    record['model']={'type':'LGBMClassifier','loss':'binary_logloss','parameters':MODEL_PARAMETERS}
    record['data'].update(source_files_sha256={p:sha(args.data_dir/p) for p in ['train.csv','vessel_catalog.csv']})
    save(args.record,record)
    try:
        frame=pd.read_csv(args.data_dir/'train.csv');catalog=pd.read_csv(args.data_dir/'vessel_catalog.csv')
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed).split(frame,groups=frame.target_vessel_id))
        fitted,mixed=train_test_split(frame.iloc[a],test_size=.2,random_state=args.seed,stratify=frame.iloc[a].vessel_name.map(hangul))
        unseen=frame.iloc[b];assert not set(fitted.target_vessel_id)&set(unseen.target_vessel_id)
        splits=pd.DataFrame({'sample_id':frame.sample_id,'split':np.where(frame.index.isin(fitted.index),'fit',np.where(frame.index.isin(unseen.index),'vessel_holdout','row_holdout'))})
        splits.to_csv(args.output_dir/'split.csv',index=False)
        previous=json.loads(args.previous_record.read_text())
        assert sha(args.output_dir/'split.csv')==previous['data']['split_file_sha256']
        record['data'].update(split_file_sha256=sha(args.output_dir/'split.csv'),training_rows=len(fitted),
                              validation_rows={'vessel_holdout':len(unseen),'row_holdout':len(mixed)})
        start=time.monotonic()
        assets=make_assets(fitted,catalog,'all',24,list(CHANNELS));index=CatalogIndex(catalog,assets)
        record['index_build_seconds']=time.monotonic()-start
        record['features']=index.feature_names
        record['ngram_index']={'fit_source':'official catalog only','n_documents':len(index.ids),
                               'vocabulary_size':len(index.ngram_index.vectorizer.vocabulary_),
                               'nonzero_entries':int(index.ngram_index.transpose.nnz),
                               'ngram_range':[2,4],'analyzer':'char','dtype':'float32','sublinear_tf':True,
                               'query_state':'no fit, no query aggregation'}
        unique=fitted.drop_duplicates(['vessel_name','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','target_vessel_id'])
        cache={};record['cache']={}
        for label,subset in [('fit',unique),('vessel_holdout',unseen),('row_holdout',mixed)]:
            record['active_stage']='cache '+label;save(args.record,record)
            cache[label],record['cache'][label]=build_cache(subset,index,label=='fit',args.output_dir,label)
            save(args.record,record)
        for mask in range(16):
            name=name_for(mask);record['active_stage']='combination '+name;save(args.record,record)
            print(f'Training/evaluating combination {mask+1}/16: {name}',flush=True)
            folder=args.output_dir/name;folder.mkdir()
            model,fit_result=fit_subset(cache['fit'],index,mask,folder/'model')
            result={'mask':mask,'channels':[c for j,c in enumerate(CHANNELS) if mask&(1<<j)],'fit':fit_result}
            for label in ['vessel_holdout','row_holdout']:
                result[label]={}
                for production in [False,True]:
                    mode='production' if production else 'pure'
                    metrics,pred=evaluate(cache[label],index,model,mask,set(fitted.target_vessel_id),production,folder,label+'_'+mode)
                    result[label][mode]=metrics
                    if production and model is not None:
                        matcher=Matcher.__new__(Matcher);matcher.index=index;matcher.model=model
                        index.retrieval_channels=result['channels']
                        result[label]['row_independence']=probe(matcher,unseen if label=='vessel_holdout' else mixed,pred.set_index('sample_id'))
                print(name,label,'pure=',result[label]['pure']['weighted_accuracy'],
                      'production=',result[label]['production']['weighted_accuracy'],flush=True)
            if name=='SPT':
                assert fit_result['training_pair_sha256']==previous['variants']['search']['fit']['training_pair_sha256']
                for label in ['vessel_holdout','row_holdout']:
                    assert result[label]['production']['candidate_sha256']==previous['variants']['search'][label]['candidate_sha256']
                    old=pd.read_csv(args.previous_artifacts/'search'/f'{label}_predictions.csv').set_index('sample_id')
                    new=pd.read_csv(folder/f'{label}_production_predictions.csv').set_index('sample_id')
                    assert old.sort_index().equals(new.sort_index())
                result['previous_search_reproduction']={'passed':True,'checks':['training pair hash','validation candidate hash','all production predictions']}
            save(folder/'result.json',result);record['variants'][name]=result;save(args.record,record)
        assert len(record['variants'])==16
        record.update(status='completed',finished_at_kst=now(),peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        record.pop('active_stage',None)
    except Exception:
        record.update(status='failed',finished_at_kst=now(),failure=traceback.format_exc())
        save(args.record,record)
        raise
    save(args.record,record)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--previous-record',type=Path,required=True)
    p.add_argument('--previous-artifacts',type=Path,required=True)
    run(p.parse_args())
