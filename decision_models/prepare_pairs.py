"""Build one immutable training-fold cache shared by both scorers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from train import read_data, training_pair_for_row
from model.matcher import CatalogIndex, make_assets, prepare_row, hangul
from train_oof import validate_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--folds', type=Path, default=Path('eval_kit/folds_group5_seed42.csv'))
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.update(VESSEL_SEARCH_ALGORITHM='rapidfuzz', VESSEL_TRAIN_CANDIDATES='50',
                      VESSEL_MAX_CANDIDATES='0', VESSEL_PREDICT_CANDIDATES='30')
    train, catalog = read_data(args.data_dir)
    folds = pd.read_csv(args.folds, dtype={'sample_id': str, 'target_vessel_id': str})
    joined = validate_folds(train, catalog, folds)
    real = joined[joined.fold != args.fold].drop(columns='fold')
    assets = make_assets(real, catalog, 'all', 24, ['S', 'P', 'T'], True)
    index = CatalogIndex(catalog, assets)
    cols = ['vessel_name', 'gross_tonnage', 'international_gross_tonnage', 'flag_name', 'vessel_type', 'target_vessel_id']
    unique = real.drop_duplicates(cols)
    kh = unique.vessel_name.map(hangul).mean()
    pairs = [training_pair_for_row(prepare_row(row), index, kh) for row in unique.to_dict('records')]
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'train_pairs.npz', x=np.concatenate([p[0] for p in pairs]),
        y=np.concatenate([p[1] for p in pairs]), w=np.concatenate([p[2] for p in pairs]),
        feature_names=np.asarray(index.feature_names))
    (args.output / 'assets.json').write_text(json.dumps(assets, ensure_ascii=False), encoding='utf-8')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {'fold': args.fold, 'seed': 42, 'augmentation': False,
        'data_sha256': {str(p): digest(p) for p in [args.data_dir / 'train.csv', args.data_dir / 'vessel_catalog.csv', args.folds]},
        'pair_sha256': digest(args.output / 'train_pairs.npz'),
        'candidate_order_sha256': hashlib.sha256(json.dumps([[p[4], p[5]] for p in pairs]).encode()).hexdigest(),
        'training_rows': len(real), 'unique_rows': len(unique), 'features': index.feature_names,
        'search': 'rapidfuzz', 'channels': ['S', 'P', 'T'], 'phonetic_search': 24,
        'training_candidate_limit': 50, 'prediction_candidate_limit': 30}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
