"""DACON entry point. No network access or test-distribution features."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'model'))
from matcher import Matcher
import pandas as pd


def input_path(filename, explicit=None):
    if explicit is not None:
        return Path(explicit)
    for directory in ('data', 'open'):
        path = ROOT / directory / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f'Missing {filename}: expected data/ or open/ under {ROOT}')


def lookup_override():
    value = os.environ.get('USE_TRAIN_LOOKUP')
    if value is None:
        return None  # Respect the saved training configuration.
    if value.strip().lower() not in {'0', '1', 'false', 'true'}:
        raise ValueError('USE_TRAIN_LOOKUP must be 0/1 or false/true')
    return value.strip().lower() in {'1', 'true'}


def run(test_path=None, catalog_path=None, output_path=None):
    test_path = input_path('test.csv', test_path)
    catalog_path = input_path('vessel_catalog.csv', catalog_path)
    output_path = Path(output_path) if output_path is not None else ROOT / 'output/submission.csv'
    test = pd.read_csv(test_path, encoding='utf-8-sig', keep_default_na=False,
                       dtype={'sample_id': str, 'call_sign': str})
    catalog = pd.read_csv(catalog_path, encoding='utf-8-sig', keep_default_na=False,
                          dtype={'vessel_id': str})
    if test.sample_id.isna().any() or test.sample_id.eq('').any() or not test.sample_id.is_unique:
        raise ValueError('sample_id must be unique and non-null')
    matcher = Matcher(catalog, ROOT / 'model', use_train_lookup=lookup_override())
    predictions = matcher.predict(test)
    if len(predictions) != len(test) or not set(predictions).issubset(matcher.index.lookup):
        raise ValueError('Invalid predictions')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'sample_id': test.sample_id, 'target_vessel_id': predictions}).to_csv(
        output_path, index=False, encoding='utf-8')
    # Never log evaluation records, features, or individual predictions.
    print(f'Saved {len(predictions)} predictions.')
    return output_path


if __name__ == '__main__':
    run()
