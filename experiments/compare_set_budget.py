"""Candidate-budget alignment experiment: AS-SET-03.

Tests whether aligning training candidates (max 30) with inference candidates (max 30)
reduces Train-Test discrepancy in candidate-set pooling statistics, compared to AS-SET-02 (train 50 vs infer 30).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
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

from decision_models.set_model import SetDecisionModel
from decision_models.matcher_adapter import DecisionMatcher
from experiments.train_set_model import (FOLD_HASH, digest, now, noise_copy,
                                         read_data, validate_folds)
from model.matcher import CatalogIndex, QUERY_COLUMNS, make_assets, prepare_row, hangul
from train import training_pair_for_row
from train_oof import check_independence
from eval_kit.evaluate import run as official_evaluate

SETTINGS = {
    'VESSEL_SEARCH_ALGORITHM': 'rapidfuzz', 'VESSEL_PHONE_SIMILARITY': 'rapidfuzz',
    'VESSEL_SEARCH_DEVICE': 'cpu', 'VESSEL_FEATURE_WORKERS': '4',
    'VESSEL_PREDICT_WORKERS': '4', 'VESSEL_SCORE_THREADS': '1',
    'VESSEL_TRAIN_CANDIDATES': '30',  # ALIGNED TO 30
    'VESSEL_MAX_CANDIDATES': '0',
    'VESSEL_PREDICT_CANDIDATES': '30', 'VESSEL_PREDICT_BATCH': '64',
    'VESSEL_RELAX_TONNAGE_PENALTY': '0', 'VESSEL_HANGUL_IGNORE_TYPE_FLAG': '0',
    'VESSEL_HANGUL_SOFT_META': '1', 'VESSEL_KOREAN_EXPANSION': '1',
}

CODE_FILES = ['experiments/compare_set_budget.py', 'decision_models/set_model.py',
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


def collect_pairs_budget30(training, index):
    """Collect training pairs with candidate limit = 30."""
    cols = ['vessel_name', 'gross_tonnage', 'international_gross_tonnage',
            'flag_name', 'vessel_type', 'target_vessel_id']
    unique = training.drop_duplicates(cols)
    kh = unique.vessel_name.map(hangul).mean()
    matrices, labels, weights, sizes = [], [], [], []
    retrieved_natural = []
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
                sizes.append(len(x))
                retrieved_natural.append(bool(found))
    retrieved_fraction = float(np.mean(retrieved_natural))
    summary = {
        'queries': len(sizes),
        'pairs': len(labels),
        'retrieved_before_gold_insertion_fraction': retrieved_fraction,
        'natural_queries': int(sum(retrieved_natural)),
        'injected_queries': int(len(retrieved_natural) - sum(retrieved_natural)),
        'avg_candidates_per_query': float(np.mean(sizes)),
        'max_candidates_per_query': int(max(sizes)),
        'min_candidates_per_query': int(min(sizes))
    }
    return (np.concatenate(matrices), np.asarray(labels, dtype=np.float32),
            np.asarray(weights, dtype=np.float32), np.asarray(sizes, dtype=np.int64), summary)


def score_cache(scorer, x, sizes):
    offsets = np.r_[0, np.cumsum(sizes)]
    scores = []
    for start in range(0, len(sizes), 64):
        stop = min(start + 64, len(sizes))
        batch = x[offsets[start]:offsets[stop]]
        score = scorer.predict_groups(batch, sizes[start:stop], num_threads=1)
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
    output = {'rows': len(frame), 'seen_vessel_fraction': 0.}
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


def run_experiment(args):
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output
    os.environ.update(SETTINGS)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    code_hashes = {name: digest(ROOT / name) for name in CODE_FILES if (ROOT / name).exists()}
    original_models = {p.name: digest(p) for p in (ROOT / 'model').iterdir()
                       if p.suffix in ('.pt', '.cbm', '.npz', '.json')}

    record_path = ROOT / 'project_log/experiments/AS-SET-03.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    record.update(status='running', started_at_kst=now(), fold_results=[])
    record['code'] = {
        'git_revision_at_run': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'dirty': True, 'file_sha256': code_hashes
    }
    save_json(record_path, record)

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
        record['data'].update(source_files_sha256=data_hashes, split_id='G5-s42',
                              split_file_sha256=FOLD_HASH, seed=20261016, validation_rows=len(training))
        save_json(record_path, record)

        cat_rows = catalog.set_index('vessel_id').to_dict('index')
        parts = []
        shared = []

        for fold in range(5):
            fold_dir = output / f'fold_{fold}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            progress.update(fold=fold, stage='candidates/features budget 30')
            save_json(output / 'progress.json', progress)
            print(f'[AS-SET-03] Fold {fold + 1}/5: generating training features with cand_budget=30...', flush=True)

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
            x, y, w, sizes, pair_summary = collect_pairs_budget30(augmented, index)
            pair_seconds = time.perf_counter() - tick

            train_cache = fold_dir / 'train_pairs.npz'
            np.savez_compressed(train_cache, x=x, y=y, w=w, group_sizes=sizes,
                                feature_names=np.asarray(index.feature_names))

            # Reuse the frozen G5 validation pairs from AS-SET-02 to guarantee exact comparability
            ref_val_cache = ROOT / f'artifacts/AS-SET-02/fold_{fold}/validation_pairs.npz'
            val_cache = fold_dir / 'validation_pairs.npz'
            if ref_val_cache.exists():
                with np.load(ref_val_cache, allow_pickle=False) as vdata:
                    vx = vdata['x']
                    vsizes = vdata['group_sizes']
                    candidate_ids_flat = vdata['candidate_ids'].tolist()
                np.savez_compressed(val_cache, x=vx, group_sizes=vsizes,
                                    feature_names=np.asarray(index.feature_names),
                                    sample_ids=validation.sample_id.to_numpy(dtype=str),
                                    candidate_ids=np.asarray(candidate_ids_flat))
                # Reconstruct candidate_ids per query
                offsets = np.r_[0, np.cumsum(vsizes)]
                candidate_ids = [candidate_ids_flat[offsets[i]:offsets[i+1]] for i in range(len(vsizes))]
                val_features_seconds = 0.0
            else:
                raise FileNotFoundError(f'Missing reference validation cache: {ref_val_cache}')

            fold_shared = {
                'fold': fold, 'real_training_rows': len(real), 'augmented_training_rows': len(augmented),
                'validation_rows': len(validation), 'pairs': pair_summary,
                'preparation_seconds': prep_seconds, 'training_candidate_feature_seconds': pair_seconds,
                'validation_candidate_feature_seconds': val_features_seconds,
                'train_cache': artifact_info(train_cache), 'validation_cache': artifact_info(val_cache)
            }
            shared.append(fold_shared)
            save_json(fold_dir / 'shared.json', fold_shared)

            # Train SetDecisionModel
            progress.update(stage='train SetDecisionModel')
            save_json(output / 'progress.json', progress)
            print(f'[AS-SET-03] Fold {fold + 1}/5: training SetDecisionModel ({len(sizes)} queries / {len(x)} pairs)...', flush=True)

            tick = time.perf_counter()
            fitted = SetDecisionModel(embedding_dim=32, hidden_dim=32, epochs=25, seed=20261016,
                                      batch_size=64, lr=0.001, weight_decay=0.001, device=args.device)
            fitted.fit(x, y, w, index.feature_names, group_sizes=sizes)
            fit_seconds = time.perf_counter() - tick

            destination = fold_dir / 'set'
            destination.mkdir(parents=True, exist_ok=True)
            model_assets = dict(assets, decision_model='set_mlp')
            assets_path = destination / 'assets.json'
            save_json(assets_path, model_assets)

            tick = time.perf_counter()
            fitted.save(destination)
            scorer = SetDecisionModel.load(destination, device='cpu')
            persistence_seconds = time.perf_counter() - tick

            # Validation Scoring
            progress.update(stage='validate SetDecisionModel')
            save_json(output / 'progress.json', progress)
            tick = time.perf_counter()
            scores = score_cache(scorer, vx, vsizes)
            score_seconds = time.perf_counter() - tick
            pred, ranks = select_and_rank(scores, candidate_ids, validation.target_vessel_id)
            result = metrics(validation, pred, ranks)

            # Live prediction check
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
            parts.append(frame)

            model_files = [p for p in destination.iterdir() if p.is_file() and p.name not in ('assets.json', 'oof_part.csv')]
            summary = {
                'fold': fold, 'metrics': result, 'training_seconds': fit_seconds,
                'save_reload_seconds': persistence_seconds, 'cached_score_seconds': score_seconds,
                'index_load_seconds': index_load_seconds, 'prediction_pipeline_seconds': pipeline_seconds,
                'row_independence': independence, 'cache_live_predictions_equal': True,
                'loss_history': fitted.loss_history, 'model_bytes': sum(p.stat().st_size for p in model_files),
                'assets_bytes': assets_path.stat().st_size,
                'artifacts': [artifact_info(p) for p in model_files] + [artifact_info(assets_path)]
            }
            record['fold_results'].append(summary)
            record['features'] = index.feature_names
            save_json(record_path, record)
            save_json(destination / 'result.json', summary)

            print(f"[AS-SET-03] Fold {fold + 1}/5: weighted={result['weighted_accuracy']:.10f}, "
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
        oof_path = output / 'oof_AS-SET-03.csv'
        oof_frame.to_csv(oof_path, index=False)

        # Compare with baseline AS-SET-02
        ref_oof_path = ROOT / 'artifacts/AS-SET-02/oof_AS-SET-02.csv'
        gold = training.target_vessel_id.to_numpy()
        ko = training.vessel_name.map(hangul).to_numpy()
        after = oof_frame.pred_vessel_id.to_numpy()

        comparisons = {}
        if ref_oof_path.exists():
            ref_oof = pd.read_csv(ref_oof_path).set_index('sample_id').loc[training.sample_id].reset_index()
            before = ref_oof.pred_vessel_id.to_numpy()
            for label, mask in [('hangul', ko), ('non_hangul', ~ko)]:
                comparisons[label] = {
                    'changed_predictions': int(((before != after) & mask).sum()),
                    'fixed': int(((before != gold) & (after == gold) & mask).sum()),
                    'broken': int(((before == gold) & (after != gold) & mask).sum()),
                    'both_wrong': int(((before != gold) & (after != gold) & mask).sum())
                }
            pd.DataFrame({'sample_id': training.sample_id, 'fold': joined.fold, 'before_AS_SET_02': before,
                          'after_AS_SET_03': after, 'gold': gold, 'hangul': ko}).loc[before != after].to_csv(
                              output / 'changed_predictions.csv', index=False)

        # Official Evaluation with evaluate.py
        lines = []
        arguments = ['--train', relative(args.data_dir / 'train.csv'), '--oof', relative(oof_path),
                     '--folds', relative(args.folds), '--id', 'AS-SET-03']
        if ref_oof_path.exists():
            arguments += ['--compare', relative(ref_oof_path)]
        evaluation = official_evaluate(arguments, out=lambda line: lines.append(str(line)))
        evaluation.pop('rows', None)
        evaluation.pop('correct', None)
        portable_text = '\n'.join(lines).replace(str(ROOT), '.').replace(ROOT.as_posix(), '.')
        (output / 'evaluation_AS-SET-03.txt').write_text(portable_text, encoding='utf-8')
        save_json(output / 'evaluation_AS-SET-03.json', evaluation)

        score = evaluation['score']
        record.update(
            status='completed', finished_at_kst=now(),
            official_evaluation=evaluation, shared_folds=shared,
            decision='Completed; comparison vs AS-SET-02 available'
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
            assets_bytes=[f['assets_bytes'] for f in record['fold_results']],
            timing_scope='Training is optimizer only; inference is candidate search/features/scoring over all 12,222 held-out rows across five folds. CSV load, model/index load and file save separate; not official-server time.'
        )
        record['artifacts'] = [artifact_info(oof_path), artifact_info(output / 'evaluation_AS-SET-03.json')]
        record['error_cases_file'] = relative(output / 'changed_predictions.csv') if (output / 'changed_predictions.csv').exists() else None
        save_json(record_path, record)

        if original_models != {p.name: digest(p) for p in (ROOT / 'model').iterdir()
                               if p.suffix in ('.pt', '.cbm', '.npz', '.json')}:
            raise RuntimeError('Existing production model assets changed during comparison')

        report = {
            'status': 'completed', 'scope': 'local fixed G5 development OOF; no Public estimate',
            'comparisons': comparisons, 'shared_folds': shared, 'data_load_seconds': load_seconds,
            'code_sha256': code_hashes, 'metrics': record['metrics'], 'cost': cost,
            'evaluation': evaluation, 'oof': artifact_info(oof_path),
            'total_seconds': time.perf_counter() - started
        }
        save_json(output / 'summary.json', report)
        progress.update(status='completed', stage='done', finished_at_kst=now())
        save_json(output / 'progress.json', progress)
        print(f"\n[AS-SET-03 Completed] Weighted Accuracy: {score['w']:.10f} | Hangul: {score['h']:.6f} | Non-Hangul: {score['nh']:.6f}")
        print(json.dumps({'status': 'completed', 'comparisons': comparisons}, ensure_ascii=False, indent=2))

    except BaseException as error:
        record.update(status='failed', finished_at_kst=now(), failure={'type': type(error).__name__, 'stage': progress['stage']})
        save_json(record_path, record)
        progress.update(status='failed', error_type=type(error).__name__, finished_at_kst=now())
        save_json(output / 'progress.json', progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--folds', type=Path, default=ROOT / 'eval_kit/folds_group5_seed42.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/AS-SET-03')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run_experiment(args)


if __name__ == '__main__':
    main()
