"""Shared cached-pair CLI. No retrieval, augmentation or validation fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def file_sha256(path):
    """Hash in bounded chunks without loading a potentially large pair cache."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fit_standardizer(features):
    """Fit on training pairs only; preserve float64 reduction/float32 storage.

    Constant or nearly constant columns use scale 1, avoiding division by zero
    without changing the existing models' normalization convention.
    """
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1
    return mean, scale


def run_cli(model_class):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'predict'])
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with np.load(args.pairs, allow_pickle=False) as data:
        names = data['feature_names'].tolist()
        x = data['x']
        if x.ndim != 2 or x.shape[1] != len(names) or not np.isfinite(x).all():
            raise ValueError('Invalid feature matrix')
        tick = time.perf_counter()
        if args.action == 'train':
            if args.model.exists():
                raise FileExistsError(args.model)
            model = model_class().fit(x, data['y'], data['w'], names)
            args.model.mkdir(parents=True)
            model.save(args.model)
            report = {'pair_file_sha256': file_sha256(args.pairs),
                      'training_pairs': len(x), 'features': names,
                      'fit_and_save_seconds': time.perf_counter() - tick,
                      'model': model_class.__name__}
            (args.model / 'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        else:
            if args.output is None:
                parser.error('--output is required for predict')
            if args.output.exists():
                raise FileExistsError(args.output)
            model = model_class.load(args.model)
            if names != model.feature_names:
                raise ValueError('Feature order mismatch')
            scores = model.predict(x)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('wb') as handle:
                np.save(handle, scores, allow_pickle=False)


def validate_training(x, y, w):
    x = np.asarray(x, dtype=np.float32)
    y, w = np.asarray(y, dtype=np.float32), np.asarray(w, dtype=np.float32)
    if x.ndim != 2 or y.shape != (len(x),) or w.shape != y.shape:
        raise ValueError('Invalid training shapes')
    if not all(np.isfinite(a).all() for a in (x, y, w)) or (w <= 0).any():
        raise ValueError('Training values must be finite; weights positive')
    # Checking membership directly avoids sorting every training label.
    is_positive = y == 1
    is_negative = y == 0
    if not (is_positive | is_negative).all() or not is_positive.any() or not is_negative.any():
        raise ValueError('Both binary classes are required')
    return x, y, w
