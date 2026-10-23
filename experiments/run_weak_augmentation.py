"""Paired five-fold augmentation study. Never writes a submission ZIP."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
# Explicit user-approved local environment. Import torch first on Windows.
import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
from threadpoolctl import threadpool_limits
import concurrent.futures
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from train import fit, read_data, training_pair_for_row, MODEL_PARAMETERS, prepare_row, Matcher, QUERY_COLUMNS, evaluate
from train_oof import validate_folds, now_kst, sha256, check_independence, summarize_folds
from experiments.weak_augmentation import augment

IDS = ['AUG-04', 'AUG-05']
OUT = ROOT / 'artifacts/oof'
FOLD_HASH = '9d09ede0f9254949aacf8cf80638a6c24de0ca5516ca3e9db329f55b9476ca51'


def save(record):
    (ROOT / f"project_log/experiments/{record['experiment_id']}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    for name in IDS:
        if (OUT / name).exists() or (ROOT / f'project_log/experiments/{name}.json').exists():
            raise ValueError('Run ID already exists; preserve previous results')
    # Keep identical JY-01 default behavior; explicit budget <=6 CPUs.
    settings = {'VESSEL_SEARCH_ALGORITHM': 'rapidfuzz', 'VESSEL_PHONE_SIMILARITY': 'rapidfuzz',
                'VESSEL_FEATURE_WORKERS': '4', 'VESSEL_TRAIN_THREADS': '4',
                'VESSEL_PREDICT_WORKERS': '4', 'VESSEL_SCORE_THREADS': '1',
                'VESSEL_TRAIN_CANDIDATES': '50', 'VESSEL_MAX_CANDIDATES': '0',
                'VESSEL_PREDICT_CANDIDATES': '30', 'VESSEL_TRAIN_BATCH': '64',
                'VESSEL_PREDICT_BATCH': '64', 'VESSEL_SEARCH_DEVICE': 'cpu'}
    os.environ.update(settings)
    torch.set_num_threads(1)
    code = {p: sha256(ROOT / p) for p in ['train.py', 'train_oof.py', 'model/matcher.py',
            'model/name_processing.py', 'model/ngram_retrieval.py',
            'experiments/weak_augmentation.py', 'experiments/run_weak_augmentation.py',
            'eval_kit/evaluate.py']}
    data = {p: sha256(ROOT / p) for p in ['data/train.csv', 'data/vessel_catalog.csv',
                                         'eval_kit/folds_group5_seed42.csv']}
    assert data['eval_kit/folds_group5_seed42.csv'] == FOLD_HASH
    records = []
    for name in IDS:
        r = json.loads((ROOT / 'project_log/templates/experiment.json').read_text(encoding='utf-8-sig'))
        r.update(experiment_id=name, status='planned', recorded_at_kst=now_kst(),
                 purpose='Paired actual-query group OOF: JY-01 configuration with/without weak train-only augmentation',
                 hypothesis='Weak name deletion and metadata masking may improve unseen-vessel generalization',
                 baseline_experiment_id=IDS[0] if name == IDS[1] else None,
                 changed_factors=['0.5 row-selection probability; synthetic pair weights 0.5'] if name == IDS[1] else [],
                 fixed_conditions=['same folds, real-row lookup assets, features, search, candidate selection, model hyperparameters'],
                 code={'git_revision_at_run': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                       'dirty': True, 'file_sha256': code},
                 data={'source_files_sha256':data, 'split_id':'G5-s42', 'seed':42},
                 model={'type':'LightGBM binary candidate classifier', 'parameters':MODEL_PARAMETERS},
                 environment={'python':platform.python_version(),'platform':platform.platform(),
                     'executable':sys.executable,'packages':{p:importlib.metadata.version(p) for p in
                       ['torch','numpy','pandas','scikit-learn','lightgbm','rapidfuzz','Unidecode']}},
                 commands=[sys.executable, *sys.argv], runtime=settings, fold_results=[],
                 limitations=['Previously used public train; development OOF, not independent final test.',
                   'Local .venv explicitly approved; Python differs from server.',
                   'Fixed weak corruption hypotheses, not learned test-distribution alignment.',
                   'Hard-negative feature references corrected to ratio/phonetic_ratio/best_gap by name in both arms; compare AUG-04 with AUG-02 separately.',
                   'Official evaluator README locally v1.0; stricter project v1.1 requirements govern validation.',
                   'No catalog-only synthetic new IDs in this initial experiment.'])
        save(r); records.append(r); (OUT / name).mkdir(parents=True)
    started = time.perf_counter()
    try:
        train, catalog = read_data(ROOT / 'data')
        folds = pd.read_csv(ROOT / 'eval_kit/folds_group5_seed42.csv',keep_default_na=False,
                           dtype={'sample_id':str,'target_vessel_id':str})
        joined = validate_folds(train,catalog,folds)
        parts = {name:[] for name in IDS}
        for r in records:
            r.update(status='running', started_at_kst=now_kst()); save(r)
        for k in range(5):
            print(f'Fold {k}: baseline training',flush=True)
            real = joined[joined.fold != k].drop(columns='fold').copy()
            val = joined[joined.fold == k].drop(columns='fold').copy()
            captured = {}
            def capture(x,y,w,index,unique,kh):
                captured.update(x=x,y=y,w=w,index=index,kh=kh)
            base_dir = OUT / IDS[0] / f'fold_{k}/model'
            result0 = fit(real,catalog,base_dir,'all',24,['S','P','T'],True,pair_observer=capture)
            extra, aug_stats = augment(real,catalog,seed=42+k)
            assert not set(extra.target_vessel_id) & set(val.target_vessel_id)
            print(f'Fold {k}: {len(extra)} synthetic rows',flush=True)
            extra_start = time.perf_counter()
            def build(row):
                return training_pair_for_row(prepare_row(row), captured['index'],captured['kh'])
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                pairs = list(pool.map(build,extra.to_dict('records')))
            x = np.concatenate([captured['x']] + [p[0] for p in pairs])
            y = np.concatenate([captured['y']] + [np.asarray(p[1]) for p in pairs])
            w = np.concatenate([captured['w']] + [np.asarray(p[2])*0.5 for p in pairs])
            extra_feature_seconds=time.perf_counter()-extra_start
            aug_dir = OUT / IDS[1] / f'fold_{k}/model'; aug_dir.mkdir(parents=True)
            parameters = dict(MODEL_PARAMETERS,n_jobs=4)
            tick=time.perf_counter()
            model=lgb.LGBMClassifier(**parameters)
            with threadpool_limits(limits=4):
                model.fit(x,y,sample_weight=w,feature_name=captured['index'].feature_names)
            model_seconds=time.perf_counter()-tick
            (aug_dir/'ranker.txt').write_text(model.booster_.model_to_string(),encoding='utf-8')
            (aug_dir/'assets.json').write_bytes((base_dir/'assets.json').read_bytes())
            assert sha256(aug_dir/'assets.json') == sha256(base_dir/'assets.json')
            pair_digest=hashlib.sha256()
            for p in pairs:
                pair_digest.update(json.dumps([p[4],p[5]],ensure_ascii=False).encode())
            result1 = dict(result0)
            result1.update(augmentation=aug_stats,training_pairs=len(y),
                synthetic_pair_count=len(y)-len(captured['y']), synthetic_pair_sha256=pair_digest.hexdigest(),
                base_training_pair_sha256=result0['training_pair_sha256'],
                base_training_pair_hash_reused=True,
                fit_seconds=result0['fit_seconds']-result0['model_fit_seconds']+extra_feature_seconds+model_seconds,
                extra_feature_seconds=extra_feature_seconds, model_fit_seconds=model_seconds,
                timing_scope='Base feature/index preparation reused; attributed fit cost excludes baseline classifier fit')
            result1.pop('training_pair_sha256',None)
            result1.pop('feature_importance_gain',None)
            del x,y,w,pairs,model,captured
            for record,destination,result in zip(records,[base_dir,aug_dir],[result0,result1]):
                matcher=Matcher(catalog,destination)
                metrics,pred=evaluate(matcher,val,set(real.target_vessel_id))
                independence=check_independence(matcher,val.loc[:,QUERY_COLUMNS],pred)
                summary={'fold':k,'fit':result,'validation':metrics,'row_independence':independence,
                         'artifacts':{n:{'path':str(destination/n),'sha256':sha256(destination/n),
                                            'bytes':(destination/n).stat().st_size} for n in ['assets.json','ranker.txt']}}
                record['fold_results'].append(summary); save(record)
                parts[record['experiment_id']].append(pd.DataFrame({'sample_id':val.sample_id,
                                                'pred_vessel_id':pred,'fold':k}))
                print(f"{record['experiment_id']} fold {k}: weighted={metrics['weighted_accuracy']:.6f}",flush=True)
        for r in records:
            name=r['experiment_id']; oof=pd.concat(parts[name],ignore_index=True)
            oof=oof.set_index('sample_id').loc[train.sample_id].reset_index()
            assert len(oof)==12222 and oof.sample_id.is_unique
            path=OUT/name/f'oof_{name}.csv'; oof.to_csv(path,index=False)
            r.update(status='completed',finished_at_kst=now_kst(),
                     local_diagnostics=summarize_folds(r['fold_results']),
                     oof={'path':str(path),'sha256':sha256(path),'rows':len(oof)},
                     paired_study_wall_seconds=time.perf_counter()-started)
            save(r)
    except BaseException as exc:
        for r in records:
            r.update(status='failed',finished_at_kst=now_kst(),
                     failure={'type':type(exc).__name__,'message':str(exc)})
            save(r)
        raise


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
