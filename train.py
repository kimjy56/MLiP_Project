"""Fit and validate exclusively on the official train.csv and catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import concurrent.futures
import os
try:
    import torch
except Exception:
    pass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
if str(ROOT / 'model') not in sys.path:
    sys.path.insert(0, str(ROOT / 'model'))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from model.matcher import CatalogIndex, FEATURES, QUERY_COLUMNS, Matcher, hangul, make_assets
except ImportError:
    from matcher import CatalogIndex, FEATURES, QUERY_COLUMNS, Matcher, hangul, make_assets


MODEL_PARAMETERS = dict(n_estimators=350, learning_rate=.05, num_leaves=31,
                        max_depth=-1, min_child_samples=40, colsample_bytree=.9,
                        reg_lambda=2, random_state=20261016, n_jobs=4,
                        verbosity=-1, deterministic=True, force_col_wise=True)


@threadpool_limits.wrap(limits=4)
def fit(train, catalog, destination, name_mode='baseline', phonetic_search=0, retrieval_channels=None,
        use_train_lookup=True):
    start = time.monotonic()
    destination.mkdir(parents=True, exist_ok=True)
    assets = make_assets(train, catalog, name_mode, phonetic_search, retrieval_channels, use_train_lookup)
    index = CatalogIndex(catalog, assets)
    # Calls and aliases are used only at inference; the pair model must learn
    # catalog similarity without seeing its own label as an alias feature.
    cols = ['vessel_name', 'gross_tonnage', 'international_gross_tonnage',
            'flag_name', 'vessel_type', 'target_vessel_id']
    unique = train.drop_duplicates(cols)
    records = unique.to_dict('records')
    kh = unique.vessel_name.map(hangul).mean()

    def _process_train_row(row):
        truth = index.lookup[row['target_vessel_id']]
        candidates = index.candidates(row, use_aliases=False)
        was_retrieved = truth in candidates
        if not was_retrieved:
            candidates = sorted(candidates + [truth])
        features = index.features(row, candidates, include_name_features=False)
        if len(candidates) > 100:
            hardness = np.maximum(features[:, 1], features[:, 4]) - features[:, 11]*0.5
            selected = set(np.argsort(-hardness, kind='stable')[:80].tolist())
            selected.update(np.argsort(features[:, 11], kind='stable')[:20].tolist())
            selected.add(candidates.index(truth))
            selected = sorted(selected)
            features = features[selected]
            candidates = [candidates[i] for i in selected]
        if name_mode != 'baseline':
            features = np.column_stack([features, index.name_index.features(
                row.get('vessel_name'), candidates, name_mode)])
        row_labels = [int(i == truth) for i in candidates]
        weight = .7/max(kh, 1e-9) if hangul(row['vessel_name']) else .3/max(1-kh, 1e-9)
        row_weights = [weight] * len(candidates)
        cand_ids = [index.ids[i] for i in candidates]
        return features, row_labels, row_weights, was_retrieved, row['sample_id'], cand_ids

    workers = min(8, os.cpu_count() or 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        row_results = list(executor.map(_process_train_row, records))

    matrices, labels, weights = [], [], []
    retrieved = []
    pair_hash = hashlib.sha256()
    for features, row_labels, row_weights, was_retrieved, sample_id, cand_ids in row_results:
        matrices.append(features)
        labels.extend(row_labels)
        weights.extend(row_weights)
        retrieved.append(was_retrieved)
        pair_hash.update(json.dumps([sample_id, cand_ids], ensure_ascii=False).encode('utf-8'))
    x = np.concatenate(matrices)
    model = lgb.LGBMClassifier(**MODEL_PARAMETERS)
    model.fit(x, np.asarray(labels), sample_weight=np.asarray(weights), feature_name=index.feature_names)
    # Python handles Unicode Windows paths; native LightGBM fopen does not.
    (destination / 'ranker.txt').write_text(model.booster_.model_to_string(), encoding='utf-8')
    (destination / 'assets.json').write_text(json.dumps(assets, ensure_ascii=False), encoding='utf-8')
    return {'training_rows': len(train), 'unique_feature_rows': len(unique),
            'training_pair_sha256': pair_hash.hexdigest(),
            'training_pairs': len(labels), 'training_candidate_recall': float(np.mean(retrieved)),
            'fit_seconds': time.monotonic()-start,
            'timing_scope': 'assets, catalog index, candidate/features, training and model save; excludes CSV load',
            'feature_importance_gain': dict(zip(index.feature_names, model.booster_.feature_importance('gain').tolist()))}


def evaluate(matcher, validation, training_ids):
    start = time.monotonic()
    pred, candidates, direct = matcher.predict_details(validation.loc[:, QUERY_COLUMNS])
    recall = [gold in ids for gold, ids in zip(validation.target_vessel_id, candidates)]
    ko = validation.vessel_name.map(hangul).to_numpy()
    correct = validation.target_vessel_id.to_numpy() == np.asarray(pred)
    seen = validation.target_vessel_id.isin(training_ids).to_numpy()
    output = {'rows': len(validation), 'seconds': time.monotonic()-start,
              'known_call_sign_fraction': float(np.mean(direct)),
              'seen_vessel_fraction': float(seen.mean())}
    for name, mask in [('hangul', ko), ('non_hangul', ~ko), ('seen', seen), ('unseen', ~seen)]:
        output[name+'_rows'] = int(mask.sum())
        output[name+'_accuracy'] = float(correct[mask].mean()) if mask.any() else None
        output[name+'_candidate_recall'] = float(np.asarray(recall)[mask].mean()) if mask.any() else None
    output['weighted_accuracy'] = (.7*output['hangul_accuracy'] + .3*output['non_hangul_accuracy']
                                   if ko.any() and (~ko).any() else None)
    output['metric_source'] = 'local diagnostic; use distributed eval_kit for team ranking'
    return output, pred


def read_data(data_dir):
    train = pd.read_csv(data_dir / 'train.csv', encoding='utf-8-sig', keep_default_na=False,
                        dtype={'sample_id': str, 'target_vessel_id': str, 'call_sign': str})
    catalog = pd.read_csv(data_dir / 'vessel_catalog.csv', encoding='utf-8-sig',
                          keep_default_na=False, dtype={'vessel_id': str})
    return train, catalog


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--final-only', action='store_true')
    modes.add_argument('--validate-only', action='store_true')
    parser.add_argument('--use-train-lookup', action=argparse.BooleanOptionalAction, default=True,
                        help='Persist whether calls and training-name aliases are both enabled')
    parser.add_argument('--name-mode', choices=['baseline', 'structure', 'all'], default='baseline')
    parser.add_argument('--phonetic-search', type=int, default=0)
    parser.add_argument('--retrieval-channels', nargs='*', choices=['S','N','P','T'], default=None)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    train, catalog = read_data(args.data_dir)
    reports = args.output_root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    if not args.final_only:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20261016).split(
            train, groups=train.target_vessel_id))
        pool, unseen = train.iloc[a], train.iloc[b]
        fitted, mixed = train_test_split(pool, test_size=.2, random_state=20261016,
                                        stratify=pool.vessel_name.map(hangul))
        assert not set(fitted.target_vessel_id) & set(unseen.target_vessel_id)
        dest = args.output_root / 'validation_model'
        results = {'scope': 'legacy development holdouts, not team 5-fold OOF',
                   'seed': 20261016, 'fit': fit(fitted, catalog, dest, args.name_mode,
                       args.phonetic_search, args.retrieval_channels, args.use_train_lookup)}
        matcher = Matcher(catalog, dest)
        for label, data in [('vessel_holdout', unseen), ('row_holdout', mixed)]:
            result, pred = evaluate(matcher, data, set(fitted.target_vessel_id))
            results[label] = result
            # These are public TRAIN holdouts, never hidden evaluation records.
            pd.DataFrame({'sample_id': data.sample_id, 'truth': data.target_vessel_id,
                          'prediction': pred}).to_csv(reports / (label+'_predictions.csv'), index=False)
            print(label, json.dumps(result), flush=True)
        # Exercise both lookup hits and the general matcher on public held-out data.
        probe = pd.concat([unseen.head(12), mixed.head(12), unseen[unseen.vessel_name.map(hangul)].head(8)]).loc[:, QUERY_COLUMNS]
        expected = dict(zip(probe.sample_id, matcher.predict(probe)))
        reordered = probe.sample(frac=1, random_state=7)
        assert dict(zip(reordered.sample_id, matcher.predict(reordered))) == expected
        for i in range(len(probe)):
            assert matcher.predict(probe.iloc[[i]])[0] == expected[probe.iloc[i].sample_id]
        subset = probe.iloc[::3]
        assert matcher.predict(subset) == [expected[s] for s in subset.sample_id]
        results['row_independence'] = {'passed': True, 'rows': len(probe),
                                       'checks': ['full', 'shuffled', 'single', 'subset']}
        (reports / 'validation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        pd.DataFrame({'sample_id': train.sample_id,
                      'split': np.where(train.index.isin(fitted.index), 'fit',
                                        np.where(train.index.isin(unseen.index), 'vessel_holdout', 'row_holdout'))
                      }).to_csv(reports / 'split.csv', index=False)
    if not args.validate_only:
        result = fit(train, catalog, args.output_root / 'model', args.name_mode,
                     args.phonetic_search, args.retrieval_channels, args.use_train_lookup)
        (reports / 'final_training.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('Final training complete', flush=True)


if __name__ == '__main__':
    main()
