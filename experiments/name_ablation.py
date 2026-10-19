"""Reproducible public-train ablation. Never use hidden test rows here."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import fit
from matcher import Matcher, hangul, feature_names


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def evaluate(matcher, frame, training_ids):
    predictions, recalled, direct, counts, errors = [], [], [], [], []
    fingerprint = hashlib.sha256()
    start = time.monotonic()
    for row in frame.to_dict('records'):
        lookup = matcher.index.direct(row)
        if lookup:
            pred, recall, count = lookup, lookup == row['target_vessel_id'], 1
            candidates_ids = [lookup]
        else:
            candidates = matcher.index.candidates(row)
            candidates_ids = [matcher.index.ids[i] for i in candidates]
            scores = matcher.model.predict(matcher.index.features(row, candidates), num_threads=1)
            pred = candidates_ids[int(np.argmax(scores))]
            recall, count = row['target_vessel_id'] in candidates_ids, len(candidates)
        fingerprint.update(json.dumps([row['sample_id'], candidates_ids]).encode())
        predictions.append(pred); recalled.append(recall); direct.append(bool(lookup)); counts.append(count)
        if pred != row['target_vessel_id']:
            errors.append({'sample_id':row['sample_id'], 'vessel_name':row['vessel_name'],
                           'truth':row['target_vessel_id'], 'prediction':pred,
                           'stage':'lookup' if lookup else 'selection' if recall else 'retrieval'})
    seconds = time.monotonic()-start
    correct = np.asarray(predictions) == frame.target_vessel_id.to_numpy()
    ko = frame.vessel_name.map(hangul).to_numpy()
    seen = frame.target_vessel_id.isin(training_ids).to_numpy()
    output = {'rows':len(frame), 'predict_seconds':seconds,
              'seen_vessel_fraction':float(seen.mean()), 'candidate_sha256':fingerprint.hexdigest(),
              'candidate_count_nonlookup': {}, 'errors':len(errors),
              'error_stages':{s:sum(e['stage']==s for e in errors) for s in ['lookup','retrieval','selection']}}
    for label, mask in [('hangul',ko),('non_hangul',~ko),('seen',seen),('unseen',~seen)]:
        output[label] = {'rows':int(mask.sum()), 'unique_vessels':int(frame.loc[mask,'target_vessel_id'].nunique()),
                         'accuracy':float(correct[mask].mean()) if mask.any() else None,
                         'candidate_recall':float(np.asarray(recalled)[mask].mean()) if mask.any() else None}
    output['weighted_accuracy'] = .7*output['hangul']['accuracy']+.3*output['non_hangul']['accuracy']
    counts = np.asarray(counts)[~np.asarray(direct)]
    if len(counts):
        output['candidate_count_nonlookup'] = {'mean':float(counts.mean()), 'p95':float(np.quantile(counts,.95)), 'max':int(counts.max())}
    frame_pred = pd.DataFrame({'sample_id':frame.sample_id, 'truth':frame.target_vessel_id,
                               'prediction':predictions})
    return output, frame_pred, errors


def check_independence(matcher, unseen, mixed):
    probe = pd.concat([unseen.head(8), unseen[unseen.vessel_name.map(hangul)].head(8), mixed.head(8)]).drop_duplicates('sample_id')
    pred = matcher.predict(probe)
    expected = dict(zip(probe.sample_id, pred))
    assert dict(zip(probe.sample(frac=1, random_state=7).sample_id,
                    matcher.predict(probe.sample(frac=1, random_state=7)))) == expected
    for i in range(len(probe)):
        assert matcher.predict(probe.iloc[[i]]) == [expected[probe.iloc[i].sample_id]]
    subset = probe.iloc[::3]
    assert matcher.predict(subset) == [expected[s] for s in subset.sample_id]
    return {'passed':True, 'rows':len(probe), 'checks':['full','permutation','single','subset']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--baseline-model', type=Path)
    parser.add_argument('--variants', nargs='+', choices=['baseline','structure','all','search'],
                        default=['baseline','structure','all','search'])
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    if record['status'] not in ['planned']:
        raise ValueError('Create a new planned record; do not overwrite a historical run')
    record.update(status='running', started_at_kst=now(), commands=[' '.join(sys.argv)])
    record['code'] = {'git_revision_at_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
                      'file_sha256':{p:sha(ROOT/p) for p in ['model/matcher.py','model/name_processing.py','train.py','experiments/name_ablation.py']}}
    record['environment'] = {'python':platform.python_version(), 'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','lightgbm','rapidfuzz','Unidecode','scikit-learn']}}
    record['data'] = {'source_files_sha256':{p:sha(args.data_dir/p) for p in ['train.csv','vessel_catalog.csv']}, 'seed':args.seed}
    write(args.record,record)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(args.data_dir/'train.csv')
        catalog = pd.read_csv(args.data_dir/'vessel_catalog.csv')
        if args.split_file:
            splits = pd.read_csv(args.split_file)
            assert splits.sample_id.tolist() == frame.sample_id.tolist()
        else:
            a,b = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed).split(frame,groups=frame.target_vessel_id))
            pool = frame.iloc[a]
            fitted,mixed = train_test_split(pool,test_size=.2,random_state=args.seed,stratify=pool.vessel_name.map(hangul))
            splits = pd.DataFrame({'sample_id':frame.sample_id,'split':np.where(frame.index.isin(fitted.index),'fit',np.where(frame.index.isin(b),'vessel_holdout','row_holdout'))})
        splits.to_csv(args.output_dir/'split.csv',index=False)
        # Preserve original train_test_split row order for training: tree learning
        # and negative construction must reproduce the original baseline order.
        a,b = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed).split(frame,groups=frame.target_vessel_id))
        fitted,mixed = train_test_split(frame.iloc[a],test_size=.2,random_state=args.seed,stratify=frame.iloc[a].vessel_name.map(hangul))
        unseen=frame.iloc[b]
        for label, subset in [('fit',fitted),('vessel_holdout',unseen),('row_holdout',mixed)]:
            assert set(subset.sample_id)==set(splits.loc[splits.split==label,'sample_id'])
        assert not set(fitted.target_vessel_id) & set(unseen.target_vessel_id)
        record['data'].update(split_file_sha256=sha(args.output_dir/'split.csv'),training_rows=len(fitted),
                              validation_rows={'vessel_holdout':len(unseen),'row_holdout':len(mixed)})
        record['variants'] = {}
        write(args.record,record)
        for variant in args.variants:
            mode, search = ('all',24) if variant=='search' else (variant,0)
            dest=args.output_dir/variant
            dest.mkdir(exist_ok=True)
            result={'name_mode':mode,'phonetic_search':search,'features':feature_names(mode)}
            record['active_variant']=variant
            write(args.record,record)
            start=time.monotonic()
            if variant=='baseline' and args.baseline_model:
                modeldir=args.baseline_model
                result['fit']=None
                result['baseline_artifact_sha256']={p:sha(modeldir/p) for p in ['ranker.txt','assets.json']}
            else:
                modeldir=dest/'model'
                result['fit']=fit(fitted,catalog,modeldir,name_mode=mode,phonetic_search=search)
            result['fit_wall_seconds']=time.monotonic()-start
            start=time.monotonic()
            matcher=Matcher(catalog,modeldir)
            result['index_load_seconds']=time.monotonic()-start
            for label, subset in [('vessel_holdout',unseen),('row_holdout',mixed)]:
                metrics,pred,errors=evaluate(matcher,subset,set(fitted.target_vessel_id))
                result[label]=metrics
                pred.to_csv(dest/(label+'_predictions.csv'),index=False)
                write(dest/(label+'_errors.json'),errors)
                print(variant,label,json.dumps(metrics),flush=True)
            result['row_independence']=check_independence(matcher,unseen,mixed)
            result['artifact_bytes']={p:(modeldir/p).stat().st_size for p in ['ranker.txt','assets.json']}
            result['artifact_sha256']={p:sha(modeldir/p) for p in ['ranker.txt','assets.json']}
            write(dest/'result.json',result)
            record['variants'][variant]=result
            write(args.record,record)
        # Exact candidate lists, not just their counts, must match feature-only variants.
        baseline=record['variants'].get('baseline')
        if baseline:
            for v in ['structure','all']:
                if v in record['variants']:
                    for label in ['vessel_holdout','row_holdout']:
                        assert record['variants'][v][label]['candidate_sha256']==baseline[label]['candidate_sha256']
        trained=[record['variants'][v]['fit'] for v in ['baseline','structure','all'] if v in record['variants'] and record['variants'][v]['fit']]
        assert len({r['training_pair_sha256'] for r in trained})<=1
        record.update(status='completed',finished_at_kst=now(),peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        record['timing_scope']={'fit_wall':'assets/index + features + fit + model writes, excludes CSV load',
            'predict_seconds':'candidate lookup, features, prediction, diagnostics; excludes index and CSV I/O',
            'peak_process_rss':'whole ablation process, not standalone submission inference'}
        record.pop('active_variant',None)
    except Exception:
        record.update(status='failed',finished_at_kst=now(),failure=traceback.format_exc())
        write(args.record,record)
        raise
    write(args.record,record)


if __name__=='__main__':
    main()
