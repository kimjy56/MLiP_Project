"""Generate public-train OOF from the team's supplied, fixed group folds.

Does not generate folds, tune on holdouts, run an unofficial team evaluator,
train a final model, or create a submission ZIP.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
try:
    import torch
except Exception:
    pass

import numpy as np
import pandas as pd

from train import ROOT, MODEL_PARAMETERS, QUERY_COLUMNS, Matcher, evaluate, fit, read_data


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


sha256 = _sha256


def now_kst():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def validate_folds(train, catalog, folds, expected_rows=12222):
    """Align by exact IDs, preserving train order; never generate a split."""
    if len(train) != expected_rows or len(folds) != len(train):
        raise ValueError('Unexpected train/fold row count')
    if 'fold' in train.columns:
        raise ValueError('Train must not already contain fold metadata')
    if not set(QUERY_COLUMNS + ['target_vessel_id']).issubset(train.columns):
        raise ValueError('Train schema is incomplete')
    if not {'sample_id', 'target_vessel_id', 'fold'}.issubset(folds.columns):
        raise ValueError('Fold schema is incomplete')
    if not {'vessel_id', 'vessel_name', 'gross_tonnage', 'vessel_type', 'flag_name'}.issubset(catalog.columns):
        raise ValueError('Catalog schema is incomplete')
    for frame, columns in ((train, ['sample_id', 'target_vessel_id']),
                           (folds, ['sample_id', 'target_vessel_id']),
                           (catalog, ['vessel_id'])):
        for column in columns:
            if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                raise ValueError('IDs must be nonempty strings, with no missing values')
    if not train.sample_id.is_unique or not folds.sample_id.is_unique:
        raise ValueError('Duplicate sample IDs')
    if catalog.empty or not catalog.vessel_id.is_unique:
        raise ValueError('Catalog must have unique, nonempty IDs')
    if set(train.sample_id) != set(folds.sample_id):
        raise ValueError('Train and fold sample IDs differ')
    if not set(train.target_vessel_id).issubset(set(catalog.vessel_id)):
        raise ValueError('Training target missing from catalog')
    if folds.fold.isna().any() or folds.fold.map(lambda v: isinstance(v, (bool, np.bool_))).any():
        raise ValueError('Fold must be an integer from 0 to 4')
    numeric = pd.to_numeric(folds.fold, errors='raise')
    if not numeric.isin(range(5)).all() or set(numeric) != set(range(5)):
        raise ValueError('Fold must contain exactly 0,1,2,3,4')
    aligned = folds.set_index('sample_id').loc[train.sample_id]
    if not np.array_equal(train.target_vessel_id.to_numpy(), aligned.target_vessel_id.to_numpy()):
        raise ValueError('Train/fold target IDs disagree')
    joined = train.reset_index(drop=True).copy()
    joined['fold'] = pd.to_numeric(aligned.fold).to_numpy(dtype=int)
    if not joined.groupby('target_vessel_id').fold.nunique().eq(1).all():
        raise ValueError('A vessel appears in multiple folds')
    return joined


def check_independence(matcher, queries, predictions):
    """Bounded public-train probes; labels never enter prediction."""
    probe = queries.iloc[:12].copy()
    expected = dict(zip(probe.sample_id, predictions[:len(probe)]))
    for selected in (probe, probe.iloc[::-1], probe.iloc[::2], probe.iloc[:1], probe.iloc[:0]):
        if matcher.predict(selected) != [expected[s] for s in selected.sample_id]:
            raise ValueError('Row independence check failed')
    return {'passed': True, 'rows': len(probe),
            'checks': ['repeat', 'reversed', 'subset', 'single', 'empty']}


def summarize_folds(summaries):
    """Aggregate row-weighted diagnostics, not an average of fold scores."""
    result = {'metric_source': 'local diagnostic; distributed team evaluation pending',
              'folds': len(summaries)}
    for group in ('hangul', 'non_hangul', 'seen', 'unseen'):
        count = sum(s['validation'][group + '_rows'] for s in summaries)
        result[group + '_rows'] = count
        for metric in ('accuracy', 'candidate_recall'):
            key = group + '_' + metric
            result[key] = (sum(s['validation'][key] * s['validation'][group + '_rows']
                               for s in summaries if s['validation'][group + '_rows']) / count
                           if count else None)
    result['weighted_accuracy'] = (
        .7 * result['hangul_accuracy'] + .3 * result['non_hangul_accuracy']
        if result['hangul_rows'] and result['non_hangul_rows'] else None)
    result['candidate_recall_definition'] = 'variable candidate-set coverage, including direct lookup; not Recall@K'
    result['training_seconds'] = sum(s['fit']['fit_seconds'] for s in summaries)
    result['prediction_and_diagnostics_seconds'] = sum(s['validation']['seconds'] for s in summaries)
    return result


def generate_oof(train, catalog, folds, output_dir, *, name_mode='baseline',
                 phonetic_search=0, retrieval_channels=None, use_train_lookup=True,
                 expected_rows=12222, on_fold=None):
    joined = validate_folds(train, catalog, folds, expected_rows)
    output_dir = Path(output_dir)
    # Do not overwrite any existing run, including partially completed runs.
    output_dir.mkdir(parents=True, exist_ok=False)
    parts, summaries = [], []
    for k in range(5):
        fitted = joined.loc[joined.fold != k].drop(columns='fold').copy()
        validation = joined.loc[joined.fold == k].drop(columns='fold').copy()
        destination = output_dir / f'fold_{k}' / 'model'
        fit_result = fit(fitted, catalog, destination, name_mode, phonetic_search,
                         retrieval_channels, use_train_lookup)
        matcher = Matcher(catalog, destination)
        # evaluate passes QUERY_COLUMNS only, and computes metrics afterwards.
        metrics, predictions = evaluate(matcher, validation, set(fitted.target_vessel_id))
        if len(predictions) != len(validation) or any(
                not isinstance(p, str) or p not in matcher.index.lookup for p in predictions):
            raise ValueError('Invalid OOF predictions')
        independence = check_independence(matcher, validation.loc[:, QUERY_COLUMNS], predictions)
        parts.append(pd.DataFrame({'sample_id': validation.sample_id.to_numpy(),
                                   'pred_vessel_id': predictions, 'fold': k}))
        artifacts = {name: {'path': str(destination / name),
                             'sha256': _sha256(destination / name),
                             'bytes': (destination / name).stat().st_size}
                     for name in ('assets.json', 'ranker.txt')}
        summary = {'fold': k, 'fit': fit_result, 'validation': metrics,
                   'row_independence': independence, 'artifacts': artifacts}
        summaries.append(summary)
        if on_fold is not None:
            on_fold(summary)
    oof = pd.concat(parts, ignore_index=True)
    if len(oof) != len(train) or not oof.sample_id.is_unique or set(oof.sample_id) != set(train.sample_id):
        raise ValueError('OOF row coverage is incomplete')
    oof = oof.set_index('sample_id').loc[train.sample_id].reset_index()
    if not np.array_equal(oof.fold.to_numpy(), joined.fold.to_numpy()):
        raise ValueError('OOF fold alignment failed')
    return oof, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id', required=True, help='Assigned team ID, e.g. O-01')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--folds', type=Path, default=ROOT / 'eval_kit/folds_group5_seed42.csv',
                        help='Default: <data-dir>/eval_kit/folds_group5_seed42.csv')
    parser.add_argument('--folds-sha256', required=True, help='Expected hash from the team distribution')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'artifacts/oof')
    parser.add_argument('--name-mode', choices=['baseline', 'structure', 'all'], default='all')
    parser.add_argument('--phonetic-search', type=int, default=24)
    parser.add_argument('--retrieval-channels', nargs='*', choices=['S', 'N', 'P', 'T'], default=None)
    parser.add_argument('--use-train-lookup', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.folds is None:
        args.folds = args.data_dir / 'eval_kit/folds_group5_seed42.csv'
    if not re.fullmatch(r'[A-Za-z]+-[0-9]+', args.experiment_id):
        parser.error('Use an assigned team ID such as O-01')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', args.folds_sha256):
        parser.error('--folds-sha256 must be a 64-digit SHA-256 hash')
    if args.phonetic_search < 0:
        parser.error('--phonetic-search must be nonnegative')
    record_path = ROOT / 'project_log/experiments' / f'{args.experiment_id}.json'
    output_dir = args.output_root / args.experiment_id
    if record_path.exists() or output_dir.exists():
        parser.error('Experiment ID already exists; use a new ID and preserve previous results')
    code_files = ['train_oof.py', 'train.py', 'model/matcher.py', 'model/name_processing.py',
                  'model/ngram_retrieval.py', 'model/torch_indexer.py', 'script.py']
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True)
    record = {
        'experiment_id': args.experiment_id, 'status': 'planned', 'recorded_at_kst': now_kst(),
        'purpose': 'Generate full OOF using supplied fixed group folds; no model selection or tuning',
        'hypothesis': 'Not a performance-improvement claim; evaluate the specified configuration',
        'code': {'git_revision': revision.stdout.strip() if revision.returncode == 0 else None,
                 'dirty': bool(status.stdout.strip()) if status.returncode == 0 else None,
                 'sha256': {name: _sha256(ROOT / name) for name in code_files}},
        'configuration': {'name_mode': args.name_mode, 'phonetic_search': args.phonetic_search,
                          'retrieval_channels': args.retrieval_channels,
                          'use_train_lookup': args.use_train_lookup, 'model': MODEL_PARAMETERS,
                          'runtime_overrides': {k: v for k, v in os.environ.items() if k.startswith('VESSEL_')},
                          'search_algorithm': os.environ.get('VESSEL_SEARCH_ALGORITHM', 'rapidfuzz')},
        'command': [sys.executable, *sys.argv],
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'executable': sys.executable},
        'data': {}, 'fold_results': [], 'team_evaluation': None,
        'limitations': ['Local diagnostics are not the distributed team evaluator result',
                        'E00/selftest, CI comparison and auxiliary folds require the team evaluation kit',
                        'No final fit, submission ZIP or hidden-test result'],
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open('x', encoding='utf-8') as target:
        json.dump(record, target, ensure_ascii=False, indent=2)

    def save_record():
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def completed_fold(summary):
        record['fold_results'].append(summary)
        save_record()
        print(f'Fold {summary["fold"]} completed', flush=True)

    start = time.monotonic()
    try:
        record.update(status='running', started_at_kst=now_kst())
        save_record()
        for name, path in [('train', args.data_dir / 'train.csv'),
                            ('catalog', args.data_dir / 'vessel_catalog.csv'), ('folds', args.folds)]:
            record['data'][name] = {'path': str(path), 'sha256': _sha256(path)}
        if record['data']['folds']['sha256'] != args.folds_sha256.lower():
            raise ValueError('Fold hash differs from the supplied team hash')
        record['data']['expected_folds_sha256'] = args.folds_sha256.lower()
        train, catalog = read_data(args.data_dir)
        folds = pd.read_csv(args.folds, encoding='utf-8-sig', keep_default_na=False,
                            dtype={'sample_id': str, 'target_vessel_id': str})
        oof, summaries = generate_oof(train, catalog, folds, output_dir,
            name_mode=args.name_mode, phonetic_search=args.phonetic_search,
            retrieval_channels=args.retrieval_channels, use_train_lookup=args.use_train_lookup,
            on_fold=completed_fold)
        oof_path = output_dir / f'oof_{args.experiment_id}.csv'
        oof.to_csv(oof_path, index=False, encoding='utf-8')
        record.update(status='completed', local_diagnostics=summarize_folds(summaries),
                      oof={'path': str(oof_path), 'rows': len(oof),
                                               'sha256': _sha256(oof_path)})
    except BaseException as exc:
        record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      failure={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        record.update(finished_at_kst=now_kst(), elapsed_seconds=time.monotonic()-start,
                      timing_scope='input hashes/loading, five sequential folds, probes and output save')
        save_record()
    print(f'OOF saved: {oof_path}; run the distributed eval_kit separately.')


if __name__ == '__main__':
    main()
