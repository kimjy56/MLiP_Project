"""Train a separate candidate-set model, or run its offline CSV inference.

No ZIP generation and no writes to the existing production model directory.
A limited holdout run is an integration smoke test, not a model comparison.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

from train import read_data, training_pair_for_row, evaluate
from train_oof import validate_folds, check_independence
from model.matcher import CatalogIndex, Matcher, QUERY_COLUMNS, make_assets, prepare_row, hangul
from decision_models.set_model import SetDecisionModel
from handoff.as02.independent_noise import noise_copy

FOLD_HASH = '9d09ede0f9254949aacf8cf80638a6c24de0ca5516ca3e9db329f55b9476ca51'
SETTINGS = {
    'VESSEL_SEARCH_ALGORITHM': 'rapidfuzz', 'VESSEL_PHONE_SIMILARITY': 'rapidfuzz',
    'VESSEL_SEARCH_DEVICE': 'cpu', 'VESSEL_FEATURE_WORKERS': '4',
    'VESSEL_PREDICT_WORKERS': '4', 'VESSEL_SCORE_THREADS': '1',
    'VESSEL_TRAIN_CANDIDATES': '50', 'VESSEL_MAX_CANDIDATES': '0',
    'VESSEL_PREDICT_CANDIDATES': '30', 'VESSEL_PREDICT_BATCH': '64',
    'VESSEL_RELAX_TONNAGE_PENALTY': '0', 'VESSEL_HANGUL_IGNORE_TYPE_FLAG': '0',
    'VESSEL_HANGUL_SOFT_META': '1', 'VESSEL_KOREAN_EXPANSION': '1',
}


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def bounded_rows(frame, limit):
    """Deterministic, language-balanced execution sample; not a validation split."""
    if limit is None or len(frame) <= limit:
        return frame.copy()
    ko = frame.vessel_name.map(hangul)
    selected = list(frame[ko].index[:limit // 2]) + list(frame[~ko].index[:limit // 2])
    selected_set = set(selected)
    selected += [i for i in frame.index if i not in selected_set][:limit - len(selected)]
    return frame.loc[selected].copy()


def collect_pairs(training, index):
    cols = ['vessel_name', 'gross_tonnage', 'international_gross_tonnage',
            'flag_name', 'vessel_type', 'target_vessel_id']
    unique = training.drop_duplicates(cols)
    kh = unique.vessel_name.map(hangul).mean()
    matrices, labels, weights, sizes, retrieved = [], [], [], [], []
    pair_hash = hashlib.sha256()
    records = unique.to_dict('records')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for offset in range(0, len(records), 64):
            batch = [prepare_row(r, index.korean_expansion) for r in records[offset:offset + 64]]
            index.prepare_retrieval(batch)
            for x, y, w, found, sid, ids in executor.map(
                    lambda row: training_pair_for_row(row, index, kh), batch):
                matrices.append(x)
                labels.extend(y)
                weights.extend(w)
                sizes.append(len(x))  # Keep every original/augmented query separate.
                retrieved.append(found)
                pair_hash.update(json.dumps([sid, ids], ensure_ascii=False).encode('utf-8'))
    return (np.concatenate(matrices), np.asarray(labels, dtype=np.float32),
            np.asarray(weights, dtype=np.float32), np.asarray(sizes, dtype=np.int64),
            {'queries': len(sizes), 'pairs': len(labels), 'pair_order_sha256': pair_hash.hexdigest(),
             'retrieved_before_gold_insertion_fraction': float(np.mean(retrieved))})


def train_model(args):
    if args.output.resolve() == (ROOT / 'model').resolve():
        raise ValueError('Use a separate output directory for this experimental model')
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'run.json'
    report = {
        'status': 'running', 'started_at_kst': now(), 'scope': 'implementation smoke' if
        args.max_train_rows or args.max_valid_rows else 'single fold development' if args.holdout_fold is not None else 'full training',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'platform': platform.system(), 'device': args.device},
        'runtime_settings': SETTINGS, 'fold': args.holdout_fold,
        'seed': args.seed, 'epochs': args.epochs, 'independent_noise': args.independent_noise,
        'metrics': None, 'submission_zip': None, 'timings': {},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    started = time.perf_counter()
    try:
        os.environ.update(SETTINGS)
        torch.set_num_threads(6)
        tick = time.perf_counter()
        training, catalog = read_data(args.data_dir)
        report['data_sha256'] = {name: digest(args.data_dir / name) for name in ('train.csv', 'vessel_catalog.csv')}
        validation = None
        if args.holdout_fold is not None:
            if digest(args.folds) != args.folds_sha256:
                raise ValueError('Supplied fold file hash differs from the expected hash')
            folds = pd.read_csv(args.folds, dtype={'sample_id': str, 'target_vessel_id': str})
            joined = validate_folds(training, catalog, folds)
            report['fold_sha256'] = digest(args.folds)
            training = joined[joined.fold != args.holdout_fold].drop(columns='fold')
            validation = bounded_rows(joined[joined.fold == args.holdout_fold].drop(columns='fold'), args.max_valid_rows)
        training = bounded_rows(training, args.max_train_rows)
        training_ids = set(training.target_vessel_id)
        report['training_real_rows'] = len(training)
        report['training_vessels'] = len(training_ids)
        report['timings']['load_seconds'] = time.perf_counter() - tick
        tick = time.perf_counter()
        if args.independent_noise:
            catalog_rows = catalog.set_index('vessel_id').to_dict('index')
            copies = []
            for row in training[~training.vessel_name.map(hangul)].to_dict('records'):
                vessel = row['target_vessel_id']
                copy, _ = noise_copy(row, dict(catalog_rows[vessel], vessel_id=vessel),
                                     seed=42 + (args.holdout_fold or 0))
                copies.append(copy)
            training = pd.concat([training, pd.DataFrame(copies)], ignore_index=True)
        report['training_rows_with_augmentation'] = len(training)
        assets = make_assets(training, catalog, name_mode='all', phonetic_search=24,
                             retrieval_channels=['S', 'P', 'T'], use_train_lookup=False,
                             korean_expansion=True, hangul_soft_meta=True)
        assets['decision_model'] = 'set_mlp'
        model_dir = args.output / 'model'
        model_dir.mkdir()
        (model_dir / 'assets.json').write_text(json.dumps(assets, ensure_ascii=False, indent=2), encoding='utf-8')
        index = CatalogIndex(catalog, assets)
        report['timings']['augmentation_assets_index_seconds'] = time.perf_counter() - tick
        tick = time.perf_counter()
        x, y, w, sizes, pair_summary = collect_pairs(training, index)
        report['timings']['candidate_features_seconds'] = time.perf_counter() - tick
        report['training_pairs'] = pair_summary
        report['features'] = index.feature_names
        if args.save_pairs:
            np.savez_compressed(args.output / 'pairs.npz', x=x, y=y, w=w,
                                group_sizes=sizes, feature_names=np.asarray(index.feature_names))
        tick = time.perf_counter()
        scorer = SetDecisionModel(epochs=args.epochs, seed=args.seed, device=args.device)
        scorer.fit(x, y, w, index.feature_names, group_sizes=sizes)
        scorer.save(model_dir)
        report['parameter_count'] = sum(p.numel() for p in scorer.model.parameters())
        report['loss_history'] = scorer.loss_history
        report['timings']['fit_and_save_seconds'] = time.perf_counter() - tick
        if validation is not None:
            tick = time.perf_counter()
            matcher = Matcher(catalog, model_dir)
            report['timings']['inference_reload_seconds'] = time.perf_counter() - tick
            metrics, predictions = evaluate(matcher, validation, training_ids)
            report['metrics'] = metrics
            report['prediction_stage_seconds'] = matcher.last_timings
            report['independence'] = check_independence(matcher, validation.loc[:, QUERY_COLUMNS], predictions)
            print(json.dumps({'scope': report['scope'], 'validation_rows': len(validation),
                              'independence': report['independence']['passed']}), flush=True)
        report['artifacts'] = {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                               for p in model_dir.iterdir() if p.is_file()}
        if args.save_pairs:
            report['artifacts']['pairs.npz'] = {'sha256': digest(args.output / 'pairs.npz'),
                                              'bytes': (args.output / 'pairs.npz').stat().st_size}
        report['status'] = 'completed'
    except Exception as error:
        report['status'] = 'failed'
        report['failure_type'] = type(error).__name__
        raise
    finally:
        report['finished_at_kst'] = now()
        report['timings']['total_seconds'] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')


def predict(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    assets = json.loads((args.model / 'assets.json').read_text(encoding='utf-8'))
    if assets.get('decision_model') != 'set_mlp':
        raise ValueError('The model directory is not a trained candidate-set model')
    # Deterministic retrieval configuration used by this runner, never test-fitted.
    os.environ.update(SETTINGS)
    queries = pd.read_csv(args.test, encoding='utf-8-sig', keep_default_na=False,
                          dtype={'sample_id': str, 'call_sign': str})
    catalog = pd.read_csv(args.catalog, encoding='utf-8-sig', keep_default_na=False, dtype={'vessel_id': str})
    if not queries.sample_id.is_unique or queries.sample_id.eq('').any():
        raise ValueError('sample_id must be nonempty and unique')
    matcher = Matcher(catalog, args.model)
    predictions = matcher.predict(queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'sample_id': queries.sample_id, 'target_vessel_id': predictions}).to_csv(
        args.output, index=False, encoding='utf-8', mode='x')
    print(f'Saved {len(predictions)} predictions.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    fit = actions.add_parser('train')
    fit.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--holdout-fold', type=int, choices=range(5))
    fit.add_argument('--folds', type=Path, default=ROOT / 'eval_kit/folds_group5_seed42.csv')
    fit.add_argument('--folds-sha256', default=FOLD_HASH)
    fit.add_argument('--epochs', type=int, default=25)
    fit.add_argument('--seed', type=int, default=20261016)
    fit.add_argument('--device', default='cpu')
    fit.add_argument('--max-train-rows', type=int)
    fit.add_argument('--max-valid-rows', type=int)
    fit.add_argument('--independent-noise', action=argparse.BooleanOptionalAction, default=True)
    fit.add_argument('--save-pairs', action='store_true')
    infer = actions.add_parser('predict')
    infer.add_argument('--model', type=Path, required=True)
    infer.add_argument('--test', type=Path, required=True)
    infer.add_argument('--catalog', type=Path, required=True)
    infer.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'train':
        if any(value is not None and value < 2 for value in (args.max_train_rows, args.max_valid_rows)):
            parser.error('Execution sample limits must be at least 2')
        if args.max_valid_rows is not None and args.holdout_fold is None:
            parser.error('--max-valid-rows requires --holdout-fold')
        with threadpool_limits(limits=1):
            train_model(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
