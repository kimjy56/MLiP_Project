"""Recheck saved PUBLIC validation errors with the original validation assets.

Usage:
  python project_log/recheck_errors.py --artifacts-root /path/to/original/project
The source root must contain data/, reports/ and validation_model/. No fitting.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
from matcher import CatalogIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'project_log/baseline_errors.json')
    args = parser.parse_args()
    src = args.artifacts_root
    provenance = json.loads((ROOT / 'project_log/provenance.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for relative, expected in provenance['original_error_artifacts_sha256'].items():
        if sha(src / relative) != expected:
            raise ValueError(f'Original experiment evidence changed: {relative}')
    if sha(ROOT / 'model/matcher.py') != provenance['evidence_sha256']['model/matcher.py']:
        raise ValueError('Use the original matcher version to recheck E001 errors.')
    train = pd.read_csv(src / 'data/train.csv')
    catalog = pd.read_csv(src / 'data/vessel_catalog.csv')
    predictions = pd.read_csv(src / 'reports/vessel_holdout_predictions.csv')
    wrong = predictions[predictions.truth != predictions.prediction].merge(
        train, on='sample_id', validate='one_to_one')
    assert wrong.truth.eq(wrong.target_vessel_id).all()
    index = CatalogIndex(catalog, json.loads((src / 'validation_model/assets.json').read_text()))
    names = catalog.set_index('vessel_id').vessel_name.to_dict()
    cases = []
    for row in wrong.to_dict('records'):
        candidates = index.candidates(row)
        stage = 'candidate_selection' if index.lookup[row['truth']] in candidates else 'candidate_retrieval'
        cases.append({'sample_id': row['sample_id'], 'query_name': row['vessel_name'],
                      'true_id': row['truth'], 'true_name': names[row['truth']],
                      'predicted_id': row['prediction'], 'predicted_name': names[row['prediction']],
                      'error_stage': stage, 'candidate_count': len(candidates)})
    result = {'source': 'Public train vessel_holdout; saved predictions; candidate presence rechecked with original validation assets',
              'new_model_trained': False, 'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {len(cases)} public validation error cases.')


if __name__ == '__main__':
    main()
