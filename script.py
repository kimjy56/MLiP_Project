"""DACON entry point. No network access or test-distribution features."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'model'))
from matcher import Matcher


def run(test_path=None, catalog_path=None, output_path=None):
    test_path = Path(test_path) if test_path is not None else ROOT / 'data/test.csv'
    catalog_path = Path(catalog_path) if catalog_path is not None else ROOT / 'data/vessel_catalog.csv'
    output_path = Path(output_path) if output_path is not None else ROOT / 'output/submission.csv'
    test = pd.read_csv(test_path, encoding='utf-8-sig', dtype={'sample_id': str, 'call_sign': str})
    catalog = pd.read_csv(catalog_path, encoding='utf-8-sig', dtype={'vessel_id': str})
    if test.sample_id.isna().any() or not test.sample_id.is_unique:
        raise ValueError('sample_id must be unique and non-null')
    matcher = Matcher(catalog, ROOT / 'model')
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
