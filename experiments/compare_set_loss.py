"""Loss function comparison experiments: AS-SET-06 (row_bce) and AS-SET-07 (row_ce).

Evaluates Experiment C (Row-wise mean BCE) directly vs Experiment A (AS-SET-04):
- Row-wise mean BCE: Average candidate loss within each row -> apply language weight (0.7/0.3) -> batch row mean
- Evaluates on exact same candidate sets and features from AS-SET-04 with baseline mean_max context.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

from decision_models.set_model import SetDecisionModel
from decision_models.matcher_adapter import DecisionMatcher
from experiments.train_set_model import digest, now, read_data, validate_folds
from model.matcher import CatalogIndex, QUERY_COLUMNS, hangul
from train_oof import check_independence
from eval_kit.evaluate import run as official_evaluate

CODE_FILES = ['experiments/compare_set_loss.py', 'decision_models/set_model.py',
              'decision_models/matcher_adapter.py', 'model/matcher.py',
              'model/name_processing.py', 'model/torch_indexer.py', 'train.py',
              'train_oof.py', 'handoff/as02/independent_noise.py', 'eval_kit/evaluate.py']


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def artifact_info(path):
    path = Path(path)
    return {'path': relative(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def score_cache(model, x, sizes):
    return model.predict_groups(x, sizes)


def select_and_rank(scores, candidate_ids, gold_ids):
    offsets = np.r_[0, np.cumsum([len(ids) for ids in candidate_ids])]
    predictions, ranks = [], []
    for i, (cands, gold) in enumerate(zip(candidate_ids, gold_ids)):
        cscores = scores[offsets[i]:offsets[i + 1]]
        order = np.argsort(-cscores, kind='stable')
        predictions.append(cands[order[0]])
        ranks.append(int(order.tolist().index(cands.index(gold)) + 1) if gold in cands else 0)
    return predictions, ranks


def metrics(validation, predictions, ranks):
    ko = validation.vessel_name.map(hangul).to_numpy()
    correct = (np.asarray(predictions) == validation.target_vessel_id.to_numpy())
    kh = correct[ko].mean() if ko.any() else 0.0
    knh = correct[~ko].mean() if (~ko).any() else 0.0
    ranks_arr = np.asarray(ranks)
    return {
        'weighted_accuracy': float(.7 * kh + .3 * knh),
        'hangul_accuracy': float(kh),
        'non_hangul_accuracy': float(knh),
        'hangul_correct': int(correct[ko].sum()),
        'hangul_rows': int(ko.sum()),
        'non_hangul_correct': int(correct[~ko].sum()),
        'non_hangul_rows': int((~ko).sum()),
        'candidate_recall_hangul': float((ranks_arr[ko] > 0).mean()),
        'candidate_recall_non_hangul': float((ranks_arr[~ko] > 0).mean())
    }


def run_experiment(args):
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base_artifacts = Path(args.base_artifacts)

    exp_id = args.exp_id
    record_path = ROOT / f'project_log/experiments/{exp_id}.json'
    record = {
        'experiment_id': exp_id,
        'plan_id': f'N3-{args.loss_mode}',
        'status': 'running',
        'recorded_at_kst': now(),
        'started_at_kst': now(),
        'finished_at_kst': None,
        'purpose': f'Loss function comparison: {args.loss_mode} (Experiment C/D vs Baseline A AS-SET-04)',
        'hypothesis': f'Equal row weighting in {args.loss_mode} eliminates the sample-size bias where queries with 30 candidates dominate queries with 5 candidates, while preserving the official 0.7:0.3 Korean language weighting.',
        'baseline_experiment_id': 'AS-SET-04',
        'secondary_baselines': ['AS-SET-02', 'AS-SET-03', 'AS-SET-05'],
        'changed_factors': [
            f'SetDecisionModel loss_mode changed from pair_bce to {args.loss_mode}',
            'Candidate loss is averaged per query before applying row language weight and batch averaging',
            'Padded candidate positions are strictly excluded and variable valid candidate counts Nb are used'
        ],
        'fixed_conditions': [
            'identical training pairs and validation candidate sets from AS-SET-04 (natural candidate alignment max_candidates=30)',
            'same G5-s42 5-fold split (12,222 rows)',
            f'same context_mode={args.context_mode}',
            'same Adam optimizer, lr=0.001, weight_decay=0.001, seed=20261016, 25 epochs',
            'same 64-query training batch grouping',
            'no lookup (use_train_lookup=False)'
        ],
        'code': {},
        'data': {},
        'retrieval': {
            'channels': ['S', 'P', 'T'],
            'phonetic_search': 24,
            'train_candidates': 30,
            'predict_candidates': 30,
            'algorithm_alignment': 'prune_candidates_by_tonnage (same as AS-SET-04)'
        },
        'fold_results': [],
        'metrics': {},
        'cost': {},
        'artifacts': []
    }

    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        git_hash = None
    code_hashes = {f: digest(ROOT / f) for f in CODE_FILES if (ROOT / f).exists()}
    record['code'] = {'git_revision_at_run': git_hash, 'file_sha256': code_hashes}
    save_json(record_path, record)

    progress = {'status': 'running', 'started_at_kst': now(), 'stage': 'loading data'}
    save_json(output / 'progress.json', progress)

    try:
        training, catalog = read_data(args.data_dir)
        folds = pd.read_csv(args.folds, dtype={'sample_id': str, 'target_vessel_id': str})
        joined = validate_folds(training, catalog, folds)
        record['data'] = {
            'source_files_sha256': {
                'train.csv': digest(args.data_dir / 'train.csv'),
                'vessel_catalog.csv': digest(args.data_dir / 'vessel_catalog.csv')
            },
            'split_id': 'G5-s42',
            'split_file_sha256': digest(args.folds),
            'seed': 20261016,
            'validation_rows': len(training)
        }

        parts = []

        for fold in range(5):
            progress.update(stage=f'fold {fold + 1}/5 training SetDecisionModel {args.loss_mode}')
            save_json(output / 'progress.json', progress)

            fold_dir = output / f'fold_{fold}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            base_fold_dir = base_artifacts / f'fold_{fold}'

            train_cache = base_fold_dir / 'train_pairs.npz'
            val_cache = base_fold_dir / 'validation_pairs.npz'

            if not train_cache.exists() or not val_cache.exists():
                raise FileNotFoundError(f'Missing base artifacts for fold {fold}: {train_cache} or {val_cache}')

            with np.load(train_cache, allow_pickle=False) as tdata:
                tx = tdata['x']
                ty = tdata['y']
                tw = tdata['w']
                tsizes = tdata['group_sizes']
                feature_names = tdata['feature_names'].tolist()

            with np.load(val_cache, allow_pickle=False) as vdata:
                vx = vdata['x']
                vsizes = vdata['group_sizes']
                candidate_ids_flat = vdata['candidate_ids'].tolist()

            offsets = np.r_[0, np.cumsum(vsizes)]
            candidate_ids = [candidate_ids_flat[offsets[i]:offsets[i + 1]] for i in range(len(vsizes))]

            validation = joined[joined.fold == fold].drop(columns='fold').reset_index(drop=True)

            print(f'[{exp_id}] Fold {fold + 1}/5: training SetDecisionModel ({args.loss_mode}, {args.context_mode}) '
                  f'({len(tsizes)} queries / {len(tx)} pairs)...', flush=True)

            tick = time.perf_counter()
            fitted = SetDecisionModel(embedding_dim=32, hidden_dim=32, epochs=25, seed=20261016,
                                      batch_size=64, lr=0.001, weight_decay=0.001,
                                      device=args.device, context_mode=args.context_mode,
                                      loss_mode=args.loss_mode)
            fitted.fit(tx, ty, tw, feature_names, group_sizes=tsizes)
            fit_seconds = time.perf_counter() - tick

            destination = fold_dir / 'set'
            destination.mkdir(parents=True, exist_ok=True)
            tick = time.perf_counter()
            fitted.save(destination)
            scorer = SetDecisionModel.load(destination, device='cpu')
            persistence_seconds = time.perf_counter() - tick

            # Validation scoring
            tick = time.perf_counter()
            scores = score_cache(scorer, vx, vsizes)
            score_seconds = time.perf_counter() - tick
            pred, ranks = select_and_rank(scores, candidate_ids, validation.target_vessel_id)
            result = metrics(validation, pred, ranks)

            # Live prediction check with DecisionMatcher
            assets_path = base_fold_dir / 'set/assets.json'
            if not assets_path.exists():
                assets_path = ROOT / f'artifacts/AS-SET-02/fold_{fold}/set/assets.json'
            matcher = DecisionMatcher(catalog, assets_path, scorer)
            tick = time.perf_counter()
            live_pred = matcher.predict(validation.loc[:, QUERY_COLUMNS])
            pipeline_seconds = time.perf_counter() - tick
            if live_pred != pred:
                raise ValueError('Live predictions differ from shared-cache predictions')
            independence = check_independence(matcher, validation.loc[:, QUERY_COLUMNS], pred)

            frame = pd.DataFrame({'sample_id': validation.sample_id.to_numpy(), 'pred_vessel_id': pred,
                                  'fold': fold, 'gold_rank': ranks})
            frame.to_csv(destination / 'oof_part.csv', index=False)
            parts.append(frame)

            model_files = [p for p in destination.iterdir() if p.is_file() and p.name != 'oof_part.csv']
            summary = {
                'fold': fold, 'metrics': result, 'training_seconds': fit_seconds,
                'save_reload_seconds': persistence_seconds, 'cached_score_seconds': score_seconds,
                'prediction_pipeline_seconds': pipeline_seconds,
                'row_independence': independence, 'cache_live_predictions_equal': True,
                'loss_history': fitted.loss_history,
                'model_bytes': sum(p.stat().st_size for p in model_files),
                'artifacts': [artifact_info(p) for p in model_files]
            }
            record['fold_results'].append(summary)
            record['features'] = feature_names
            save_json(record_path, record)
            save_json(destination / 'result.json', summary)

            print(f"[{exp_id}] Fold {fold + 1}/5: weighted={result['weighted_accuracy']:.10f}, "
                  f"Hangul={result['hangul_correct']}/{result['hangul_rows']}, "
                  f"non-Hangul={result['non_hangul_correct']}/{result['non_hangul_rows']}, fit={fit_seconds:.1f}s", flush=True)

            del fitted, scorer, matcher
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate 5 folds to full OOF
        oof_frame = pd.concat(parts, ignore_index=True).set_index('sample_id').loc[training.sample_id].reset_index()
        if not oof_frame.sample_id.is_unique or len(oof_frame) != len(training):
            raise ValueError('Invalid full OOF coverage')
        oof_path = output / f'oof_{exp_id}.csv'
        oof_frame.to_csv(oof_path, index=False)

        # Compare with baseline in base_artifacts and AS-SET-02
        ref_oof_matches = list(base_artifacts.glob('oof_*.csv'))
        ref_oof_path = ref_oof_matches[0] if ref_oof_matches else base_artifacts / 'oof_AS-SET-04.csv'
        base_name = ref_oof_path.stem.replace('oof_', '')
        as02_oof_path = ROOT / 'artifacts/AS-SET-02/oof_AS-SET-02.csv'
        gold = training.target_vessel_id.to_numpy()
        ko = training.vessel_name.map(hangul).to_numpy()
        after = oof_frame.pred_vessel_id.to_numpy()

        comparisons = {}
        if ref_oof_path.exists():
            ref_oof = pd.read_csv(ref_oof_path).set_index('sample_id').loc[training.sample_id].reset_index()
            before = ref_oof.pred_vessel_id.to_numpy()
            for label, mask in [('hangul', ko), ('non_hangul', ~ko)]:
                comparisons[f'vs_{base_name}_{label}'] = {
                    'changed_predictions': int(((before != after) & mask).sum()),
                    'fixed': int(((before != gold) & (after == gold) & mask).sum()),
                    'broken': int(((before == gold) & (after != gold) & mask).sum()),
                    'both_wrong': int(((before != gold) & (after != gold) & mask).sum())
                }
            pd.DataFrame({'sample_id': training.sample_id, 'fold': joined.fold, f'before_{base_name}': before,
                          f'after_{exp_id}': after, 'gold': gold, 'hangul': ko}).loc[before != after].to_csv(
                              output / f'changed_predictions_vs_{base_name}.csv', index=False)

        if as02_oof_path.exists() and as02_oof_path != ref_oof_path:
            as02_oof = pd.read_csv(as02_oof_path).set_index('sample_id').loc[training.sample_id].reset_index()
            before_02 = as02_oof.pred_vessel_id.to_numpy()
            for label, mask in [('hangul', ko), ('non_hangul', ~ko)]:
                comparisons[f'vs_AS_SET_02_{label}'] = {
                    'changed_predictions': int(((before_02 != after) & mask).sum()),
                    'fixed': int(((before_02 != gold) & (after == gold) & mask).sum()),
                    'broken': int(((before_02 == gold) & (after != gold) & mask).sum()),
                    'both_wrong': int(((before_02 != gold) & (after != gold) & mask).sum())
                }

        # Official evaluation with eval_kit/evaluate.py
        lines = []
        arguments = ['--train', relative(args.data_dir / 'train.csv'), '--oof', relative(oof_path),
                     '--folds', relative(args.folds), '--id', exp_id]
        if ref_oof_path.exists():
            arguments += ['--compare', relative(ref_oof_path)]
        evaluation = official_evaluate(arguments, out=lambda line: lines.append(str(line)))
        evaluation.pop('rows', None)
        evaluation.pop('correct', None)
        portable_text = '\n'.join(lines).replace(str(ROOT), '.').replace(ROOT.as_posix(), '.')
        (output / f'evaluation_{exp_id}.txt').write_text(portable_text, encoding='utf-8')
        save_json(output / f'evaluation_{exp_id}.json', evaluation)

        score = evaluation['score']
        record.update(
            status='completed', finished_at_kst=now(),
            official_evaluation=evaluation,
            decision=f'Completed; {args.loss_mode} evaluated directly vs AS-SET-04'
        )
        record['metrics'].update(
            metric_source='eval_kit/evaluate.py, local full G5',
            hangul_rows=score['n_h'], non_hangul_rows=score['n_nh'],
            hangul_accuracy=score['h'], non_hangul_accuracy=score['nh'], weighted_accuracy=score['w'],
            candidate_recall_hangul=float((oof_frame.gold_rank.to_numpy()[ko] > 0).mean()),
            candidate_recall_non_hangul=float((oof_frame.gold_rank.to_numpy()[~ko] > 0).mean()),
            seen_vessel_fraction=0.0
        )
        cost = record['cost']
        cost.update(
            training_seconds=sum(f['training_seconds'] for f in record['fold_results']),
            inference_seconds=sum(f['prediction_pipeline_seconds'] for f in record['fold_results']),
            model_bytes=[f['model_bytes'] for f in record['fold_results']],
            timing_scope='Training is optimizer only; inference is prediction pipeline over all 12,222 held-out rows across five folds.'
        )
        record['artifacts'] = [artifact_info(oof_path), artifact_info(output / f'evaluation_{exp_id}.json')]
        save_json(record_path, record)

        report = {
            'status': 'completed', 'scope': 'local fixed G5 development OOF; no Public estimate',
            'comparisons': comparisons, 'code_sha256': code_hashes, 'metrics': record['metrics'], 'cost': cost,
            'evaluation': evaluation, 'oof': artifact_info(oof_path),
            'total_seconds': time.perf_counter() - started
        }
        save_json(output / 'summary.json', report)
        progress.update(status='completed', stage='done', finished_at_kst=now())
        save_json(output / 'progress.json', progress)
        print(f"\n[{exp_id} Completed] Weighted Accuracy: {score['w']:.10f} | Hangul: {score['h']:.6f} | Non-Hangul: {score['nh']:.6f}")
        print(json.dumps({'status': 'completed', 'comparisons': comparisons}, ensure_ascii=False, indent=2))

    except BaseException as error:
        record.update(status='failed', finished_at_kst=now(), failure={'type': type(error).__name__, 'stage': progress.get('stage')})
        save_json(record_path, record)
        progress.update(status='failed', error_type=type(error).__name__, finished_at_kst=now())
        save_json(output / 'progress.json', progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--folds', type=Path, default=ROOT / 'eval_kit/folds_group5_seed42.csv')
    parser.add_argument('--base-artifacts', type=Path, default=ROOT / 'artifacts/AS-SET-04')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/AS-SET-06')
    parser.add_argument('--exp-id', default='AS-SET-06')
    parser.add_argument('--context-mode', default='mean_max')
    parser.add_argument('--loss-mode', default='row_bce', choices=['pair_bce', 'row_bce', 'row_ce'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run_experiment(args)


if __name__ == '__main__':
    main()
