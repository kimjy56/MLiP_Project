"""Run provenance for SPEC-DPR-03. No submission packaging or external calls."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    'phonetic_encoder/model.py', 'phonetic_encoder/tokenizer.py',
    'phonetic_encoder/dataset.py', 'phonetic_encoder/loss.py',
    'phonetic_encoder/retrieval.py', 'phonetic_encoder/hybrid_matcher.py',
    'experimental_matchers/dpr_hybrid/script.py', 'experiments/dpr03_common.py',
    'experiments/train_bi_encoder.py', 'experiments/export_catalog_embeddings.py',
    'experiments/evaluate_bi_encoder_oof.py', 'experiments/evaluate_dpr_robustness.py',
    'experiments/benchmark_dpr03_latency.py', 'tests/test_dpr03.py', 'tests/run_dpr_tests.py',
    'tools/verify_dpr03.py', 'eval_kit/evaluate.py', 'eval_kit/robustness.py',
    'experiments/analyze_dpr03_results.py',
]


def now_kst():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def start_run(run_id, output_dir=None):
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', run_id):
        raise ValueError('Invalid run ID')
    record_path = ROOT / 'project_log/experiments' / (run_id + '.json')
    record = json.loads((record_path if record_path.exists() else ROOT / 'project_log/templates/experiment.json').read_text(encoding='utf-8-sig'))
    if record.get('status') != 'planned':
        raise FileExistsError('Completed/failed/running ID cannot be reused: ' + run_id)
    out = Path(output_dir or ROOT / 'artifacts' / run_id)
    if out.exists():
        raise FileExistsError('Refuse to overwrite existing output directory: ' + str(out))
    out.mkdir(parents=True)
    record.update(experiment_id=run_id, status='running', started_at_kst=now_kst(),
                  recorded_at_kst=record.get('recorded_at_kst') or now_kst(),
                  commands=[subprocess.list2cmdline([sys.executable, *sys.argv])],
                  environment={'python': platform.python_version(), 'executable': sys.executable,
                               'platform': platform.platform(), 'environment_report': 'reports/dpr03_environment.json'},
                  code={'git_revision_at_run': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                        'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
                        'file_sha256': {s: sha256(ROOT / s) for s in SOURCE_FILES}})
    for s in SOURCE_FILES:
        dest = out / 'source_snapshot' / s
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / s).read_bytes())
    write_json(record_path, record)
    return out, record_path, record


def finish_run(path, record, status, error=None):
    record.update(status=status, finished_at_kst=now_kst())
    if error is not None:
        record['failure'] = {'type': type(error).__name__, 'message': str(error), 'retry': 'Use a new run ID; preserve failed outputs'}
    write_json(path, record)
    import csv
    index = ROOT / 'project_log/experiments/INDEX.csv'
    with index.open(encoding='utf-8-sig', newline='') as stream:
        known = {r['experiment_id'] for r in csv.DictReader(stream)}
    if record['experiment_id'] not in known:
        with index.open('a', encoding='utf-8', newline='') as stream:
            csv.writer(stream, lineterminator='\n').writerow([
                record['experiment_id'], 'SPEC-DPR-03', status,
                record.get('metrics', {}).get('metric_source') or 'not evaluated', '',
                f"experiments/{record['experiment_id']}.json"])


def load_fold_data(train_path, catalog_path, folds_path, fold):
    import pandas as pd
    train = pd.read_csv(train_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    folds = pd.read_csv(folds_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    for frame, key in [(train, 'sample_id'), (folds, 'sample_id'), (catalog, 'vessel_id')]:
        if frame[key].eq('').any() or not frame[key].is_unique:
            raise ValueError('Nonunique/empty ' + key)
    if set(train.sample_id) != set(folds.sample_id):
        raise ValueError('Fold IDs do not cover train exactly')
    if not set(train.target_vessel_id).issubset(set(catalog.vessel_id)):
        raise ValueError('Training ID absent from catalog')
    folds['fold'] = pd.to_numeric(folds['fold'], errors='raise')
    if not folds.fold.isin(range(5)).all():
        raise ValueError('Invalid fold number')
    merged = train.merge(folds, on='sample_id', suffixes=('', '_fold'), validate='one_to_one')
    if not merged.target_vessel_id.eq(merged.target_vessel_id_fold).all():
        raise ValueError('Fold target mismatch')
    if merged.groupby('target_vessel_id').fold.nunique().max() != 1:
        raise ValueError('Vessels span folds')
    if fold == -1:
        fit, val = merged.copy(), merged.iloc[:0].copy()
    elif fold in range(5):
        fit, val = merged.loc[merged.fold.ne(fold)].copy(), merged.loc[merged.fold.eq(fold)].copy()
    else:
        raise ValueError('fold must be -1 or 0..4')
    fit_ids, val_ids = sorted(set(fit.target_vessel_id)), sorted(set(val.target_vessel_id))
    assert not set(fit_ids) & set(val_ids)
    manifest = {'source_files_sha256': {str(p): sha256(p) for p in [train_path, catalog_path, folds_path]},
                'split_id': f'G5-s42 fold{fold}' if fold >= 0 else 'full_train',
                'split_file_sha256': sha256(folds_path), 'training_rows': len(fit), 'validation_rows': len(val),
                'fit_vessel_ids_sha256': object_hash(fit_ids), 'heldout_vessel_ids_sha256': object_hash(val_ids),
                'fit_vessel_ids': fit_ids, 'heldout_vessel_ids': val_ids}
    return catalog, fit.reset_index(drop=True), val.reset_index(drop=True), manifest


def composite_score(metrics):
    import math
    keys = ['recall@35', 'recall@10', 'recall@1', 'mrr']
    if any(not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1 for k in keys):
        raise ValueError('Composite inputs must be finite fractions in [0,1]')
    return .45 * metrics['recall@35'] + .30 * metrics['recall@10'] + .25 * metrics['recall@1'] + 10.0 * metrics['mrr']
