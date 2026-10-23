"""Comprehensive 5-fold OOF comparison: LightGBM vs CatBoost vs MLP on identical folds/features/candidates."""
import os
import sys
import time
import json
import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

from train import read_data, training_pair_for_row, prepare_row, Matcher, QUERY_COLUMNS, evaluate
from train_oof import validate_folds, now_kst, sha256, check_independence, summarize_folds
from model.matcher import make_assets, CatalogIndex, hangul
from decision_models.catboost_model import CatBoostDecisionModel
from decision_models.mlp_model import MLPDecisionModel
from decision_models.matcher_adapter import DecisionMatcher

OUT_BASE = ROOT / 'artifacts/decision_models'
REPORT_DIR = ROOT / 'reports'
FOLD_HASH = '9d09ede0f9254949aacf8cf80638a6c24de0ca5516ca3e9db329f55b9476ca51'


def run_comparison():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    settings = {
        'VESSEL_SEARCH_ALGORITHM': 'rapidfuzz',
        'VESSEL_PHONE_SIMILARITY': 'rapidfuzz',
        'VESSEL_FEATURE_WORKERS': '4',
        'VESSEL_TRAIN_THREADS': '4',
        'VESSEL_PREDICT_WORKERS': '4',
        'VESSEL_SCORE_THREADS': '1',
        'VESSEL_TRAIN_CANDIDATES': '50',
        'VESSEL_MAX_CANDIDATES': '0',
        'VESSEL_PREDICT_CANDIDATES': '30',
        'VESSEL_TRAIN_BATCH': '64',
        'VESSEL_PREDICT_BATCH': '64',
        'VESSEL_SEARCH_DEVICE': 'cpu'
    }
    os.environ.update(settings)
    torch.set_num_threads(1)

    train, catalog = read_data(ROOT / 'data')
    folds = pd.read_csv(ROOT / 'eval_kit/folds_group5_seed42.csv', dtype={'sample_id': str, 'target_vessel_id': str})
    joined = validate_folds(train, catalog, folds)

    MODELS = {
        'catboost': {
            'cls': CatBoostDecisionModel,
            'id': 'COMP-CATBOOST',
            'desc': 'CatBoostClassifier (iterations=350, depth=6, lr=0.05, threads=4)'
        },
        'mlp': {
            'cls': MLPDecisionModel,
            'id': 'COMP-MLP',
            'desc': 'MLPDecisionModel (64-32 ReLU, epochs=20, lr=0.001, train-only scaling)'
        }
    }

    records = {}
    parts = {m_key: [] for m_key in MODELS}

    for m_key, m_info in MODELS.items():
        out_dir = OUT_BASE / m_key
        out_dir.mkdir(parents=True, exist_ok=True)
        r = {
            'experiment_id': m_info['id'],
            'model_name': m_key,
            'description': m_info['desc'],
            'status': 'running',
            'started_at_kst': now_kst(),
            'split_id': 'G5-s42',
            'fold_results': [],
            'timings': {'fit_seconds': [], 'predict_seconds': []}
        }
        records[m_key] = r

    for k in range(5):
        print(f"\n=================== FOLD {k} ===================", flush=True)
        real = joined[joined.fold != k].drop(columns='fold').copy()
        val = joined[joined.fold == k].drop(columns='fold').copy()

        # Step 1: Build shared training assets and training pairs
        print(f"Fold {k}: Generating shared retrieval assets and training pairs...", flush=True)
        t_pair_start = time.perf_counter()
        assets = make_assets(real, catalog, 'all', 24, ['S', 'P', 'T'], True)
        index = CatalogIndex(catalog, assets)
        cols = ['vessel_name', 'gross_tonnage', 'international_gross_tonnage', 'flag_name', 'vessel_type', 'target_vessel_id']
        unique = real.drop_duplicates(cols)
        kh = unique.vessel_name.map(hangul).mean()

        import concurrent.futures
        def build_pair(row):
            return training_pair_for_row(prepare_row(row), index, kh)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            pairs = list(executor.map(build_pair, unique.to_dict('records')))

        x_train = np.concatenate([p[0] for p in pairs])
        y_train = np.concatenate([p[1] for p in pairs])
        w_train = np.concatenate([p[2] for p in pairs])
        pair_sec = time.perf_counter() - t_pair_start
        print(f"Fold {k}: {len(x_train)} training pairs generated in {pair_sec:.2f}s", flush=True)

        # Step 2: Train and evaluate each decision model
        for m_key, m_info in MODELS.items():
            print(f"Fold {k}: Training {m_key.upper()}...", flush=True)
            model_dir = OUT_BASE / m_key / f'fold_{k}'
            model_dir.mkdir(parents=True, exist_ok=True)
            assets_path = model_dir / 'assets.json'
            assets_path.write_text(json.dumps(assets, ensure_ascii=False), encoding='utf-8')

            # Fit model
            t_fit_start = time.perf_counter()
            scorer = m_info['cls']()
            scorer.fit(x_train, y_train, w_train, index.feature_names)
            scorer.save(model_dir)
            fit_sec = time.perf_counter() - t_fit_start

            # Evaluate with DecisionMatcher
            print(f"Fold {k}: Evaluating {m_key.upper()} on validation set ({len(val)} rows)...", flush=True)
            matcher = DecisionMatcher(catalog, assets_path, scorer)
            t_pred_start = time.perf_counter()
            metrics, pred = evaluate(matcher, val, set(real.target_vessel_id))
            pred_sec = time.perf_counter() - t_pred_start

            independence = check_independence(matcher, val.loc[:, QUERY_COLUMNS], pred)
            print(f"[{m_key.upper()}] Fold {k} Weighted Acc: {metrics['weighted_accuracy']:.5f} (Hangul: {metrics['hangul_accuracy']:.4f}, NonHangul: {metrics['non_hangul_accuracy']:.4f}) in {pred_sec:.2f}s", flush=True)

            fold_summary = {
                'fold': k,
                'fit_seconds': fit_sec,
                'predict_seconds': pred_sec,
                'validation': metrics,
                'row_independence': independence
            }
            records[m_key]['fold_results'].append(fold_summary)
            records[m_key]['timings']['fit_seconds'].append(fit_sec)
            records[m_key]['timings']['predict_seconds'].append(pred_sec)

            parts[m_key].append(pd.DataFrame({'sample_id': val.sample_id, 'pred_vessel_id': pred, 'fold': k}))

    # Step 3: Finalize OOF files and run official evaluation against LightGBM (AUG-04)
    aug04_oof = ROOT / 'artifacts/oof/AUG-04/oof_AUG-04.csv'
    results_summary = {}

    for m_key, m_info in MODELS.items():
        oof_df = pd.concat(parts[m_key], ignore_index=True)
        oof_df = oof_df.set_index('sample_id').loc[train.sample_id].reset_index()
        oof_path = OUT_BASE / m_key / f'oof_{m_info["id"]}.csv'
        oof_df.to_csv(oof_path, index=False)

        records[m_key]['status'] = 'completed'
        records[m_key]['finished_at_kst'] = now_kst()
        records[m_key]['local_diagnostics'] = summarize_folds(records[m_key]['fold_results'])
        records[m_key]['oof'] = {'path': str(oof_path), 'sha256': sha256(oof_path), 'rows': len(oof_df)}

        # Save experiment json
        (ROOT / f'project_log/experiments/{m_info["id"]}.json').write_text(
            json.dumps(records[m_key], ensure_ascii=False, indent=2), encoding='utf-8')

        # Run official evaluate.py
        lb_path = REPORT_DIR / f'eval_{m_info["id"]}_vs_LGBM.csv'
        if lb_path.exists():
            lb_path.unlink()
        cmd_eval = [
            str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8',
            'eval_kit/evaluate.py',
            '--train', 'data/train.csv',
            '--oof', str(oof_path),
            '--id', m_info['id'],
            '--compare', str(aug04_oof),
            '--leaderboard', str(lb_path)
        ]
        res_eval = subprocess.run(cmd_eval, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        print(f"\nOfficial Evaluation for {m_key.upper()}:\n{res_eval.stdout}")
        results_summary[m_key] = {
            'id': m_info['id'],
            'eval_output': res_eval.stdout,
            'diagnostics': records[m_key]['local_diagnostics'],
            'avg_fit_seconds': float(np.mean(records[m_key]['timings']['fit_seconds'])),
            'avg_pred_seconds': float(np.mean(records[m_key]['timings']['predict_seconds']))
        }

    (REPORT_DIR / 'model_comparison_summary.json').write_text(
        json.dumps(results_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print("\nModel comparison completed successfully!")


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run_comparison()
