"""Fixed G5 comparison with identical query batches, candidates and features.

The independent 64/32 MLP shares the set scorer's training loop to control
batch order, optimizer and pair-weighted BCE. This is not a historical JY OOF.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

from decision_models.mlp_model import MLPDecisionModel
from decision_models.set_model import SetDecisionModel
from decision_models.matcher_adapter import DecisionMatcher
from experiments.train_set_model import (FOLD_HASH, SETTINGS, collect_pairs, digest, now,
                                         noise_copy, read_data, validate_folds)
from model.matcher import CatalogIndex, QUERY_COLUMNS, make_assets, prepare_row, hangul
from train_oof import check_independence
from eval_kit.evaluate import run as official_evaluate

MODEL_IDS = {'mlp': 'AS-MLP-02', 'set': 'AS-SET-02'}
CODE_FILES = ['experiments/compare_set_model.py', 'experiments/train_set_model.py',
              'decision_models/set_model.py', 'decision_models/mlp_model.py',
              'decision_models/matcher_adapter.py', 'model/matcher.py',
              'model/name_processing.py', 'model/torch_indexer.py', 'train.py',
              'train_oof.py', 'handoff/as02/independent_noise.py', 'eval_kit/evaluate.py']


class IndependentNetwork(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = MLPDecisionModel(hidden_dims=(64, 32), device='cpu').build_network(width)

    def forward(self, x, mask):
        # Mask is consumed by the shared BCE loop, not by the independent network.
        return self.network(x).squeeze(-1)


class MatchedBatchMLP(SetDecisionModel):
    """Only architecture differs; inherit scaling, batches, labels and loss."""
    def _network(self, width):
        return IndependentNetwork(width)

    def export_scorer(self):
        result = MLPDecisionModel(hidden_dims=(64, 32), epochs=self.epochs,
                                  lr=self.lr, weight_decay=self.weight_decay, device='cpu')
        result.feature_names = self.feature_names.copy()
        result.mean, result.scale = self.mean.copy(), self.scale.copy()
        result.model = self.model.network.to('cpu').eval()
        return result


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


def collect_validation(queries, index):
    """No truth column enters retrieval or feature generation."""
    matrices, sizes, candidates = [], [], []
    records = queries.loc[:, QUERY_COLUMNS].to_dict('records')
    def build(row):
        ids = index.candidates(row, max_candidates=30)
        return index.features(row, ids), [index.ids[i] for i in ids]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(records), 64):
            batch = [prepare_row(row, index.korean_expansion) for row in records[start:start + 64]]
            index.prepare_retrieval(batch, use_direct=False)
            for x, ids in executor.map(build, batch):
                matrices.append(x)
                sizes.append(len(ids))
                candidates.append(ids)
    return np.concatenate(matrices), np.asarray(sizes, dtype=np.int64), candidates


def score_cache(scorer, x, sizes):
    """Same 64-query execution grouping as the inference pipeline."""
    offsets = np.r_[0, np.cumsum(sizes)]
    scores = []
    for start in range(0, len(sizes), 64):
        stop = min(start + 64, len(sizes))
        batch = x[offsets[start]:offsets[stop]]
        if hasattr(scorer, 'predict_groups'):
            score = scorer.predict_groups(batch, sizes[start:stop], num_threads=1)
        else:
            score = scorer.predict(batch, num_threads=1)
        scores.append(score)
    return np.concatenate(scores)


def select_and_rank(scores, candidate_ids, truth):
    predictions, ranks = [], []
    offset = 0
    for ids, gold in zip(candidate_ids, truth):
        order = np.argsort(-scores[offset:offset + len(ids)], kind='stable')
        predictions.append(ids[order[0]])
        ranks.append(0 if gold not in ids else int(np.where(order == ids.index(gold))[0][0]) + 1)
        offset += len(ids)
    return predictions, ranks


def metrics(frame, predictions, ranks):
    ko = frame.vessel_name.map(hangul).to_numpy()
    correct = frame.target_vessel_id.to_numpy() == np.asarray(predictions)
    covered = np.asarray(ranks) > 0
    output = {'rows': len(frame), 'seen_vessel_fraction': 0.,
              'candidate_definition': 'variable candidate-set coverage after compression; not Recall@K'}
    for key, mask in [('hangul', ko), ('non_hangul', ~ko)]:
        output.update({key + '_rows': int(mask.sum()), key + '_correct': int(correct[mask].sum()),
                       key + '_accuracy': float(correct[mask].mean()),
                       key + '_candidate_coverage': float(covered[mask].mean()),
                       key + '_candidate_misses': int((mask & ~covered).sum()),
                       key + '_selection_errors': int((mask & covered & ~correct).sum())})
    output['weighted_accuracy'] = .7 * output['hangul_accuracy'] + .3 * output['non_hangul_accuracy']
    return output


def artifact_info(path):
    return {'path': relative(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def check_code(hashes):
    changed = [name for name, value in hashes.items() if digest(ROOT / name) != value]
    if changed:
        raise RuntimeError('Source changed during comparison: ' + ', '.join(changed))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output
    os.environ.update(SETTINGS)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    code_hashes = {name: digest(ROOT / name) for name in CODE_FILES}
    original_models = {p.name: digest(p) for p in (ROOT / 'model').iterdir()
                       if p.suffix in ('.pt', '.cbm', '.npz', '.json')}
    records = {}
    template = json.loads((ROOT / 'project_log/templates/experiment.json').read_text(encoding='utf-8'))
    for key, identifier in MODEL_IDS.items():
        path = ROOT / 'project_log/experiments' / f'{identifier}.json'
        if path.exists():
            raise FileExistsError(path)
        record = json.loads(json.dumps(template))
        record.update(experiment_id=identifier, plan_id='N2-G5-comparison', status='planned',
                      recorded_at_kst=now(), purpose='Fair full G5 independent MLP vs candidate-set comparison',
                      hypothesis='Candidate context may improve selection on unseen vessels',
                      baseline_experiment_id=MODEL_IDS['mlp'] if key == 'set' else None,
                      changed_factors=['network architecture: independent 64/32 MLP vs mean/max set 32/32'],
                      fixed_conditions=['shared candidates/features/augmentation', 'shared 64-query batches/order',
                                        'shared scaler/Adam/weighted BCE/25 epochs/seed', 'no lookup', 'same fixed G5'],
                      limitations=['Development G5, not independent final test or Public score',
                                   'User-approved Python 3.11.15; Windows RTX 5080, not evaluation L4',
                                   'MLP is retrained with matched query batches; historical MLP training batches differ'],
                      fold_results=[])
        record['code'] = {'git_revision_at_run': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                          'dirty': True, 'file_sha256': code_hashes}
        record['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
                                 'numpy': np.__version__, 'pandas': pd.__version__, 'os': platform.system(),
                                 'training_device': args.device, 'inference_device': 'cpu',
                                 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                 'python_exception_user_authorized': True}
        record['model'] = {'type': key, 'parameters': {'epochs': 25, 'seed': 20261016,
                           'batch_size_queries': 64, 'lr': .001, 'weight_decay': .001,
                           'parameter_count': 4225 if key == 'mlp' else 4193},
                           'loss': 'weighted candidate-pair BCE'}
        record['commands'] = ['.venv/Scripts/python.exe -X utf8 experiments/compare_set_model.py --device ' + args.device]
        record['retrieval'] = dict(SETTINGS, channels=['S', 'P', 'T'], phonetic_search=24)
        record['preprocessing'] = {'name_mode': 'all', 'korean_expansion': True, 'hangul_soft_meta': True,
                                   'train_lookup': False, 'augmentation': 'existing non-Hangul independent_noise, seed 42+fold'}
        records[key] = record
        save_json(path, record)
    started = time.perf_counter()
    progress = {'status': 'running', 'started_at_kst': now(), 'fold': None, 'stage': 'load'}
    save_json(output / 'progress.json', progress)
    try:
        load_start = time.perf_counter()
        training, catalog = read_data(args.data_dir)
        if digest(args.folds) != FOLD_HASH:
            raise ValueError('Unexpected common fold hash')
        folds = pd.read_csv(args.folds, dtype={'sample_id': str, 'target_vessel_id': str})
        joined = validate_folds(training, catalog, folds)
        load_seconds = time.perf_counter() - load_start
        data_hashes = {name: digest(args.data_dir / name) for name in ('train.csv', 'vessel_catalog.csv')}
        for key, record in records.items():
            record.update(status='running', started_at_kst=now())
            record['data'].update(source_files_sha256=data_hashes, split_id='G5-s42',
                                  split_file_sha256=FOLD_HASH, seed=20261016, validation_rows=len(training))
            save_json(ROOT / f'project_log/experiments/{MODEL_IDS[key]}.json', record)
        cat_rows = catalog.set_index('vessel_id').to_dict('index')
        parts = {key: [] for key in MODEL_IDS}
        shared = []
        for fold in range(5):
            check_code(code_hashes)
            fold_dir = output / f'fold_{fold}'
            fold_dir.mkdir()
            progress.update(fold=fold, stage='shared candidates/features')
            save_json(output / 'progress.json', progress)
            print(f'Fold {fold + 1}/5: preparing shared training and validation features', flush=True)
            real = joined[joined.fold != fold].drop(columns='fold').copy()
            validation = joined[joined.fold == fold].drop(columns='fold').copy()
            if set(real.target_vessel_id) & set(validation.target_vessel_id):
                raise ValueError('Vessel leakage in fixed folds')
            tick = time.perf_counter()
            copies = []
            for row in real[~real.vessel_name.map(hangul)].to_dict('records'):
                vessel = row['target_vessel_id']
                copy, _ = noise_copy(row, dict(cat_rows[vessel], vessel_id=vessel), seed=42 + fold)
                copies.append(copy)
            augmented = pd.concat([real, pd.DataFrame(copies)], ignore_index=True)
            assets = make_assets(augmented, catalog, name_mode='all', phonetic_search=24,
                                 retrieval_channels=['S', 'P', 'T'], use_train_lookup=False,
                                 korean_expansion=True, hangul_soft_meta=True)
            index = CatalogIndex(catalog, assets)
            prep_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            x, y, w, sizes, pair_summary = collect_pairs(augmented, index)
            pair_seconds = time.perf_counter() - tick
            train_cache = fold_dir / 'train_pairs.npz'
            np.savez_compressed(train_cache, x=x, y=y, w=w, group_sizes=sizes,
                                feature_names=np.asarray(index.feature_names))
            tick = time.perf_counter()
            vx, vsizes, candidate_ids = collect_validation(validation.loc[:, QUERY_COLUMNS], index)
            val_features_seconds = time.perf_counter() - tick
            val_cache = fold_dir / 'validation_pairs.npz'
            np.savez_compressed(val_cache, x=vx, group_sizes=vsizes,
                                feature_names=np.asarray(index.feature_names),
                                sample_ids=validation.sample_id.to_numpy(dtype=str),
                                candidate_ids=np.asarray([c for ids in candidate_ids for c in ids]))
            fold_shared = {'fold': fold, 'real_training_rows': len(real), 'augmented_training_rows': len(augmented),
                           'validation_rows': len(validation), 'pairs': pair_summary,
                           'preparation_seconds': prep_seconds, 'training_candidate_feature_seconds': pair_seconds,
                           'validation_candidate_feature_seconds': val_features_seconds,
                           'train_cache': artifact_info(train_cache), 'validation_cache': artifact_info(val_cache)}
            shared.append(fold_shared)
            save_json(fold_dir / 'shared.json', fold_shared)
            scaler = None
            for key in MODEL_IDS:
                progress.update(stage=f'train {key}')
                save_json(output / 'progress.json', progress)
                print(f'Fold {fold + 1}/5: training {key} ({len(sizes)} queries / {len(x)} pairs)', flush=True)
                cls = MatchedBatchMLP if key == 'mlp' else SetDecisionModel
                tick = time.perf_counter()
                fitted = cls(epochs=25, seed=20261016, batch_size=64, device=args.device)
                fitted.fit(x, y, w, index.feature_names, group_sizes=sizes)
                fit_seconds = time.perf_counter() - tick
                current_scaler = (fitted.mean, fitted.scale)
                if scaler is not None:
                    for a, b in zip(scaler, current_scaler):
                        np.testing.assert_array_equal(a, b)
                scaler = tuple(a.copy() for a in current_scaler)
                destination = fold_dir / key
                destination.mkdir()
                model_assets = dict(assets, decision_model='set_mlp' if key == 'set' else 'mlp')
                assets_path = destination / 'assets.json'
                save_json(assets_path, model_assets)
                tick = time.perf_counter()
                if key == 'mlp':
                    fitted.export_scorer().save(destination)
                    scorer = MLPDecisionModel.load(destination, device='cpu')
                else:
                    fitted.save(destination)
                    scorer = SetDecisionModel.load(destination, device='cpu')
                persistence_seconds = time.perf_counter() - tick
                progress.update(stage=f'validate {key}')
                save_json(output / 'progress.json', progress)
                tick = time.perf_counter()
                scores = score_cache(scorer, vx, vsizes)
                score_seconds = time.perf_counter() - tick
                pred, ranks = select_and_rank(scores, candidate_ids, validation.target_vessel_id)
                result = metrics(validation, pred, ranks)
                # Re-run the actual retrieval-to-prediction API and compare with
                # the frozen validation cache, measuring each executable path.
                tick = time.perf_counter()
                matcher = DecisionMatcher(catalog, assets_path, scorer)
                index_load_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                live_pred = matcher.predict(validation.loc[:, QUERY_COLUMNS])
                pipeline_seconds = time.perf_counter() - tick
                if live_pred != pred:
                    raise ValueError('Live predictions differ from shared-cache predictions')
                independence = check_independence(matcher, validation.loc[:, QUERY_COLUMNS], pred)
                frame = pd.DataFrame({'sample_id': validation.sample_id.to_numpy(), 'pred_vessel_id': pred,
                                      'fold': fold, 'gold_rank': ranks})
                frame.to_csv(destination / 'oof_part.csv', index=False)
                parts[key].append(frame)
                model_files = [p for p in destination.iterdir() if p.is_file() and p.name not in ('assets.json', 'oof_part.csv')]
                summary = {'fold': fold, 'metrics': result, 'training_seconds': fit_seconds,
                           'save_reload_seconds': persistence_seconds, 'cached_score_seconds': score_seconds,
                           'index_load_seconds': index_load_seconds, 'prediction_pipeline_seconds': pipeline_seconds,
                           'row_independence': independence, 'cache_live_predictions_equal': True,
                           'loss_history': fitted.loss_history, 'model_bytes': sum(p.stat().st_size for p in model_files),
                           'assets_bytes': assets_path.stat().st_size,
                           'artifacts': [artifact_info(p) for p in model_files] + [artifact_info(assets_path)]}
                records[key]['fold_results'].append(summary)
                records[key]['features'] = index.feature_names
                save_json(ROOT / f'project_log/experiments/{MODEL_IDS[key]}.json', records[key])
                save_json(destination / 'result.json', summary)
                print(f"Fold {fold + 1}/5 {key}: weighted={result['weighted_accuracy']:.10f}, "
                      f"Hangul={result['hangul_correct']}/{result['hangul_rows']}, "
                      f"non-Hangul={result['non_hangul_correct']}/{result['non_hangul_rows']}, fit={fit_seconds:.1f}s", flush=True)
                del fitted, scorer, matcher
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del x, y, w, vx, index, augmented
            gc.collect()
        check_code(code_hashes)
        output_frames = {}
        for key, pieces in parts.items():
            frame = pd.concat(pieces, ignore_index=True).set_index('sample_id').loc[training.sample_id].reset_index()
            if not frame.sample_id.is_unique or len(frame) != len(training):
                raise ValueError('Invalid full OOF coverage')
            frame.to_csv(output / f'oof_{MODEL_IDS[key]}.csv', index=False)
            output_frames[key] = frame
        gold = training.target_vessel_id.to_numpy()
        ko = training.vessel_name.map(hangul).to_numpy()
        before = output_frames['mlp'].pred_vessel_id.to_numpy()
        after = output_frames['set'].pred_vessel_id.to_numpy()
        comparisons = {}
        for label, mask in [('hangul', ko), ('non_hangul', ~ko)]:
            comparisons[label] = {'changed_predictions': int(((before != after) & mask).sum()),
                                  'fixed': int(((before != gold) & (after == gold) & mask).sum()),
                                  'broken': int(((before == gold) & (after != gold) & mask).sum()),
                                  'both_wrong': int(((before != gold) & (after != gold) & mask).sum())}
        pd.DataFrame({'sample_id': training.sample_id, 'fold': joined.fold, 'before': before,
                      'after': after, 'gold': gold, 'hangul': ko}).loc[before != after].to_csv(
                          output / 'changed_predictions.csv', index=False)
        report = {'status': 'completed', 'scope': 'local fixed G5 development OOF; no Public estimate',
                  'comparisons': comparisons, 'shared_folds': shared, 'data_load_seconds': load_seconds,
                  'code_sha256': code_hashes, 'models': {}}
        for key, record in records.items():
            identifier = MODEL_IDS[key]
            oof_path = output / f'oof_{identifier}.csv'
            lines = []
            arguments = ['--train', relative(args.data_dir / 'train.csv'), '--oof', relative(oof_path),
                         '--folds', relative(args.folds), '--id', identifier]
            if key == 'set':
                arguments += ['--compare', relative(output / f"oof_{MODEL_IDS['mlp']}.csv")]
            evaluation = official_evaluate(arguments, out=lambda line: lines.append(str(line)))
            evaluation.pop('rows')
            evaluation.pop('correct')
            portable_text = '\n'.join(lines).replace(str(ROOT), '.').replace(ROOT.as_posix(), '.')
            (output / f'evaluation_{identifier}.txt').write_text(portable_text, encoding='utf-8')
            save_json(output / f'evaluation_{identifier}.json', evaluation)
            score = evaluation['score']
            record.update(status='completed', finished_at_kst=now(),
                          official_evaluation=evaluation, shared_folds=shared,
                          decision='Comparison completed; selection requires paired result interpretation')
            record['metrics'].update(metric_source='eval_kit/evaluate.py, local full G5',
                hangul_rows=score['n_h'], non_hangul_rows=score['n_nh'],
                hangul_accuracy=score['h'], non_hangul_accuracy=score['nh'], weighted_accuracy=score['w'],
                candidate_recall_hangul=float((output_frames[key].gold_rank.to_numpy()[ko] > 0).mean()),
                candidate_recall_non_hangul=float((output_frames[key].gold_rank.to_numpy()[~ko] > 0).mean()),
                seen_vessel_fraction=0.)
            cost = record['cost']
            cost.update(training_seconds=sum(f['training_seconds'] for f in record['fold_results']),
                        inference_seconds=sum(f['prediction_pipeline_seconds'] for f in record['fold_results']),
                        model_bytes=[f['model_bytes'] for f in record['fold_results']],
                        assets_bytes=[f['assets_bytes'] for f in record['fold_results']],
                        timing_scope='Training is optimizer only; inference is candidate search/features/scoring over all 12,222 held-out rows across five folds. CSV load, model/index load and file save separate; not official-server time.')
            record['artifacts'] = [artifact_info(oof_path), artifact_info(output / f'evaluation_{identifier}.json')]
            record['error_cases_file'] = relative(output / 'changed_predictions.csv') if key == 'set' else None
            save_json(ROOT / f'project_log/experiments/{identifier}.json', record)
            report['models'][key] = {'id': identifier, 'metrics': record['metrics'], 'cost': cost,
                                    'evaluation': evaluation, 'oof': artifact_info(oof_path)}
        if original_models != {p.name: digest(p) for p in (ROOT / 'model').iterdir()
                               if p.suffix in ('.pt', '.cbm', '.npz', '.json')}:
            raise RuntimeError('Existing production model assets changed during comparison')
        report['total_seconds'] = time.perf_counter() - started
        save_json(output / 'summary.json', report)
        progress.update(status='completed', stage='done', finished_at_kst=now())
        save_json(output / 'progress.json', progress)
        print(json.dumps({'status': 'completed', 'comparison': comparisons,
                          'weighted': {k: v['metrics']['weighted_accuracy'] for k, v in report['models'].items()}}, ensure_ascii=False), flush=True)
    except BaseException as error:
        for key, record in records.items():
            record.update(status='failed', finished_at_kst=now(), failure={'type': type(error).__name__, 'stage': progress['stage']})
            save_json(ROOT / f'project_log/experiments/{MODEL_IDS[key]}.json', record)
        progress.update(status='failed', error_type=type(error).__name__, finished_at_kst=now())
        save_json(output / 'progress.json', progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--folds', type=Path, default=ROOT / 'eval_kit/folds_group5_seed42.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/AS-SET-02')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == '__main__':
    main()
