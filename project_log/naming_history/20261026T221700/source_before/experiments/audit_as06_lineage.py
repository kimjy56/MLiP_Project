"""Audit the recorded AS-06 generator lineage without training or private data."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.noisy_channel_common import (
    configure_cpu, reserve_experiment, write_json, record_failure, sha256, now_kst,
)


def audit():
    import pandas as pd
    from name_decoder.data import read_frames
    from name_decoder.noisy_channel import prepare_channel_pairs
    from name_decoder.noise_augmenter import generate_augmented_pairs
    from name_decoder.channel_training import learned_hangul_variants, use_inner_confusions
    from name_decoder.catalog import has_hangul, canonical_name

    run_dir = ROOT / 'artifacts/LM-LFM25-FULL-R24-01'
    record = json.loads((ROOT / 'project_log/experiments/LM-LFM25-FULL-R24-01.json').read_text(encoding='utf-8-sig'))
    manifest = json.loads((run_dir / 'model/noisy_channel_manifest.json').read_text(encoding='utf-8-sig'))
    checks = {}
    for name, expected in record['data']['source_files_sha256'].items():
        path = ROOT / ('eval_kit/folds_group5_seed42.csv' if name == 'folds' else 'data/' + name)
        checks['source_' + name] = sha256(path) == expected == manifest['data_sha256'][name]
    _, val, fit, catalog = read_frames(ROOT / 'data', ROOT / 'eval_kit/folds_group5_seed42.csv', 0)
    train_ids = json.loads((run_dir / 'training_vessel_ids.json').read_text(encoding='utf-8-sig'))
    checks['training_vessel_ids_hash'] = sha256(run_dir / 'training_vessel_ids.json') == record['data']['training_vessel_ids_sha256']
    checks['training_vessel_membership'] = sorted(fit.target_vessel_id.unique()) == train_ids
    checks['external_vessel_overlap_zero'] = not (set(train_ids) & set(val.target_vessel_id))
    checks['external_sample_overlap_zero'] = not (set(fit.sample_id) & set(val.sample_id))
    checks['record_and_manifest_fold0'] = record['data']['heldout_fold'] == manifest['heldout_fold'] == 0
    checks['training_rows_match'] = len(fit) == record['data']['training_rows']
    raw = prepare_channel_pairs(fit, catalog)
    augmented = pd.DataFrame(generate_augmented_pairs(raw, seed=42, augment_factor=0.5), columns=['catalog_name', 'observed_name'])
    augmented['hangul'] = augmented.observed_name.map(has_hangul)
    pairs = pd.concat([raw, augmented], ignore_index=True).drop_duplicates(['catalog_name', 'observed_name']).reset_index(drop=True)
    learned, learned_count = learned_hangul_variants(raw)
    extra = pd.DataFrame(learned, columns=['catalog_name', 'observed_name'])
    extra['hangul'] = extra.observed_name.map(has_hangul)
    pairs = pd.concat([pairs, extra], ignore_index=True).drop_duplicates(['catalog_name', 'observed_name']).reset_index(drop=True)
    known = raw.groupby('observed_name').catalog_name.agg(set).to_dict()
    tail = pairs.iloc[len(raw):]
    keep = [not (known.get(row.observed_name, set()) - {row.catalog_name}) for row in tail.itertuples(index=False)]
    pairs = pd.concat([pairs.iloc[:len(raw)], tail.loc[keep]], ignore_index=True)
    pair_hash = hashlib.sha256(json.dumps(pairs[['catalog_name','observed_name']].values.tolist(), ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    checks['reconstructed_pairs_hash'] = pair_hash == record['training_pairs']['ordered_pairs_sha256']
    checks['pair_count'] = len(pairs) == record['training_pairs']['count']
    checks['learned_token_count'] = learned_count == record['training_pairs']['learned_hangul_token_count']
    mined_path = ROOT / 'artifacts/INNER-CONFUSIONS-01/confusions.csv'
    checks['mined_hash'] = sha256(mined_path) == record['training_pairs']['confusions_sha256']
    mined = pd.read_csv(mined_path, keep_default_na=False, dtype={'sample_id': str})
    checks['mined_fit_samples_only'] = set(mined.sample_id) <= set(fit.sample_id)
    checks['mined_no_outer_fold'] = set(mined.inner_fold.astype(int)) == {1,2,3,4}
    negatives = json.loads((run_dir / 'negative_names.json').read_text(encoding='utf-8-sig'))
    checks['saved_negatives_hash'] = sha256(run_dir / 'negative_names.json') == record['training_pairs']['negative_names_sha256']
    recovered, mining_stats = use_inner_confusions(pairs, catalog, fit, mined, negatives)
    checks['mined_assignments_match'] = recovered == negatives
    positives = pairs.groupby('observed_name').catalog_name.agg(set).to_dict()
    checks['negatives_exclude_known_positives'] = all(
        len(items) == 2 and len(set(items)) == 2 and not (set(items) & positives[row.observed_name])
        for row, items in zip(pairs.itertuples(index=False), negatives))
    checks['negative_rows_match'] = len(negatives) == len(pairs)
    for filename, expected in manifest['checkpoint_sha256'].items():
        checks['checkpoint_' + filename] = sha256(run_dir / 'model' / filename) == expected
    for filename, expected in record['artifact_sha256'].items():
        checks['artifact_' + filename] = sha256(run_dir / filename) == expected
    # No historical generation-side manifest binds the map to these weights.
    map_path = ROOT / 'AR-06/catalog_full_ko_map.json'
    map_data = json.loads(map_path.read_text(encoding='utf-8-sig'))
    expected_map = json.loads((ROOT / 'project_log/experiments/AS-06.json').read_text(encoding='utf-8-sig'))['code']['file_sha256']['AR-06/catalog_full_ko_map.json']
    checks['map_matches_recorded_as06_hash'] = sha256(map_path).lower() == expected_map.lower()
    return {'checks': checks, 'all_checks_passed': all(checks.values()),
        'training_rows': len(fit), 'validation_rows': len(val), 'training_vessels': len(train_ids),
        'validation_vessels': int(val.target_vessel_id.nunique()), 'raw_pairs': len(raw), 'pairs': len(pairs),
        'reconstructed_pairs_sha256': pair_hash, 'mining': mining_stats,
        'generation_model_link': 'Historical AS-06 record declaration and matching map hash only; no generation-side model/seed manifest exists.',
        'fold_scope': 'Recorded and reproducible generator fit excludes fold0. It includes folds1-4; this map is unsuitable for leak-free full5fold OOF.',
        'historical_correction': 'The unqualified claim that this generator trained on all12222 rows is contradicted by the reproducible9850-row lineage.',
        'residual_limits': ['Historical miner source snapshot was not saved; CSV/record hashes and current source membership checks do not re-prove every historical miner fit.',
            'Public train and outer fold0 were already explored in previous development; this is not an untouched final test.',
            'Manual lexicon history is not fold-isolated. Shared lexicon stays fixed in this paired comparison.']}


def main():
    configure_cpu(2)
    out, path, record = reserve_experiment('AUDIT-AS06-LINEAGE-01', ROOT / 'artifacts', 'Reconstruct AS-06 generator fit lineage before paired scoring experiment', ' '.join(sys.argv))
    try:
        result = audit()
        write_json(out / 'audit.json', result)
        record.update(status='completed' if result['all_checks_passed'] else 'audit_failed', finished_at_kst=now_kst(),
            diagnostics=result, artifacts=[str(out / 'audit.json')],
            observations=[result['historical_correction'], result['fold_scope']],
            limitations=result['residual_limits'] + [result['generation_model_link']],
            decision='Permit conditional paired fold0 development comparison only; never present shared-map full5fold scores as leak-free.')
        record['code']['file_sha256']['experiments/audit_as06_lineage.py'] = sha256(__file__)
        write_json(path, record)
        print(json.dumps(result, indent=2), flush=True)
        if not result['all_checks_passed']:
            raise RuntimeError('Lineage audit mismatch; inspect preserved report')
    except BaseException as exc:
        record_failure(path, record, exc)
        raise


if __name__ == '__main__':
    main()