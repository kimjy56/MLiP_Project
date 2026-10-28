"""Pre-registered paired AS-06/experimental_matchers/jamo_scoring Jamo score comparison; never builds ZIPs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.noisy_channel_common import (
    configure_cpu, reserve_experiment, write_json, record_failure, sha256, now_kst,
)


def load_engine(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metrics(predictions, gold, ko, ranks):
    import numpy as np
    correct = np.asarray(predictions) == np.asarray(gold)
    ko = np.asarray(ko, dtype=bool)
    found = np.asarray(ranks) > 0
    return {
        'rows': len(gold), 'hangul_rows': int(ko.sum()), 'non_hangul_rows': int((~ko).sum()),
        'hangul_correct': int(correct[ko].sum()), 'non_hangul_correct': int(correct[~ko].sum()),
        'hangul_accuracy': float(correct[ko].mean()), 'non_hangul_accuracy': float(correct[~ko].mean()),
        'weighted_accuracy': float(0.7 * correct[ko].mean() + 0.3 * correct[~ko].mean()),
        'candidate_inclusion_hangul': float(found[ko].mean()),
        'candidate_inclusion_non_hangul': float(found[~ko].mean()),
        'retrieval_failures': int((~found).sum()),
        'selection_failures': int((found & ~correct).sum()),
    }


def predict_chunks(engine, frame, model, index, mode, gold):
    preds, ranks, top, candidates = [], [], [], []
    started = time.perf_counter()
    for start in range(0, len(frame), 256):
        stop = min(len(frame), start + 256)
        audit = []
        p, r, t = engine.predict_rows(frame.iloc[start:stop], model, index,
            use_train_lookup=False, gold_ids=gold[start:stop], top_candidates=2,
            jamo_score_mode=mode, candidate_diagnostics=audit)
        for row in audit:
            row['row_index'] += start
        preds.extend(p); ranks.extend(r); top.extend(t); candidates.extend(audit)
        print(f'  {mode}: {stop}/{len(frame)} rows ({time.perf_counter()-started:.1f}s)', flush=True)
    return preds, ranks, top, candidates, time.perf_counter() - started


def main():
    configure_cpu(2)
    eid = 'JAMO-SCORE-01'
    out, path, record = reserve_experiment(eid, ROOT / 'reports/matching_improvements',
        'Compare separate1.8*Jamo channel against frozen AS-06 with identical candidate sets', ' '.join(sys.argv))
    started = time.perf_counter()
    try:
        import numpy as np
        import pandas as pd
        from eval_kit.robustness import PRESETS, RobustnessPerturber
        from eval_kit.evaluate import run as official_evaluate
        from name_decoder.data import read_frames
        audit = json.loads((ROOT / 'artifacts/AUDIT-GENERATOR-LINEAGE-01/audit.json').read_text(encoding='utf-8'))
        if not audit['all_checks_passed']:
            raise ValueError('Lineage audit did not pass')
        reference = ROOT / 'artifacts' / eid / 'reference'
        snapshot = json.loads((reference / 'snapshot_manifest.json').read_text(encoding='utf-8'))
        if any(sha256(ROOT / rel) != digest or sha256(reference / rel) != digest for rel, digest in snapshot.items()):
            raise ValueError('AS-06 reference snapshot has changed')
        artifact_dir = ROOT / 'artifacts' / eid / 'evaluation'
        artifact_dir.mkdir(parents=True, exist_ok=False)
        base = load_engine(reference / 'artifacts/AS-06/script.py', '_as06_frozen')
        engine = load_engine(ROOT / 'experimental_matchers/jamo_scoring/script.py', '_jamo_scoring_score_experiment')
        _, val, fit, _ = read_frames(ROOT / 'data', ROOT / 'eval_kit/folds_group5_seed42.csv', 0)
        catalog = pd.read_csv(ROOT / 'data/vessel_catalog.csv', keep_default_na=False)
        ko_map = json.loads((reference / 'AR-06/catalog_full_ko_map.json').read_text(encoding='utf-8-sig'))
        record['data'].update(training_rows=len(fit), validation_rows=len(val),
            split_id='folds_group5_seed42.csv/fold0', seed=20261022,
            source_files_sha256={k: sha256(ROOT / p) for k, p in {
                'train.csv': 'data/train.csv', 'vessel_catalog.csv': 'data/vessel_catalog.csv',
                'folds': 'eval_kit/folds_group5_seed42.csv'}.items()})
        record['code']['file_sha256'].update({p: sha256(ROOT / p) for p in [
            'experimental_matchers/jamo_scoring/script.py', 'experiments/evaluate_jamo_scoring.py', 'experiments/audit_generator_lineage.py',
            'tests/test_jamo_scoring.py', 'artifacts/AS-06/script.py']})
        record['environment']['threads'] = 2
        record['environment']['device'] = 'cpu'
        t0 = time.perf_counter()
        model = engine.fit_matching_model(fit, catalog, kr_mode='AR06', en_mode='EN_V3', use_train_lookup=False)
        fit_seconds = time.perf_counter() - t0
        write_json(artifact_dir / 'fit_model.json', model)
        t0 = time.perf_counter()
        index = engine.CatalogIndex(catalog, full_ko_map=ko_map)
        index_seconds = time.perf_counter() - t0
        original_index = base.CatalogIndex(catalog, full_ko_map=ko_map)
        drop_columns = ['target_vessel_id', 'target_vessel_id_fold', 'target', 'source', 'hangul', 'fold']
        inputs = val.drop(columns=drop_columns, errors='ignore').reset_index(drop=True)
        ko = val.hangul.to_numpy(dtype=bool)
        gold = val.target_vessel_id.tolist()
        # Fixed balanced parity probe; labels are used only for diagnostics.
        probe_positions = list(np.flatnonzero(ko)[:32]) + list(np.flatnonzero(~ko)[:32])
        probe = inputs.iloc[probe_positions]
        a = base.predict_rows(probe, model, original_index, use_train_lookup=False, top_candidates=5)
        b = engine.predict_rows(probe, model, index, use_train_lookup=False, top_candidates=5, jamo_score_mode='legacy')
        if a != b:
            raise AssertionError('Legacy engine parity failed')
        del original_index
        record['reference_parity'] = {'rows': len(probe_positions), 'predictions_ranks_and_top5_equal': True}
        record['lineage_audit'] = {'record': 'AUDIT-GENERATOR-LINEAGE-01', 'all_checks_passed': True,
            'map_generator_link': audit['generation_model_link']}
        record['phase_results'] = {}
        record['cost'] = {'fit_seconds': fit_seconds, 'index_seconds': index_seconds,
            'timing_scope': 'CPU fit/index and chunked retrieval+features+scoring+candidate-hash capture; official evaluation/save reported within total. Not server timing.'}
        write_json(path, record)
        errors = []
        for phase in ['clean', 'default', 'severe']:
            print(f'PHASE {phase}', flush=True)
            if phase == 'clean':
                data, noise_summary = inputs, None
            else:
                noisy, noise_summary = RobustnessPerturber(deepcopy(PRESETS[phase]), seed=20261022).perturb_dataset(val, catalog)
                data = noisy.drop(columns=drop_columns, errors='ignore').reset_index(drop=True)
                assert data.sample_id.tolist() == inputs.sample_id.tolist()
            predictions = {}
            phase_stats = {'noise_summary': noise_summary}
            candidate_sets = None
            for mode in ['legacy', 'separate']:
                p, r, top, diag, seconds = predict_chunks(engine, data, model, index, mode, gold)
                if len(diag) != len(val):
                    raise AssertionError('Candidate diagnostics did not cover every row')
                if candidate_sets is None:
                    candidate_sets = diag
                elif diag != candidate_sets:
                    raise AssertionError('Candidate set mismatch between scoring modes')
                phase_stats[mode] = metrics(p, gold, ko, r)
                phase_stats[mode]['prediction_seconds'] = seconds
                phase_stats[mode]['candidate_count_mean'] = float(np.mean([x['count'] for x in diag]))
                phase_stats[mode]['candidate_count_p95'] = float(np.percentile([x['count'] for x in diag], 95))
                filename = artifact_dir / f'oof_{phase}_{mode}.csv'
                pd.DataFrame({'sample_id': val.sample_id.tolist(), 'fold': 0, 'pred_vessel_id': p, 'gold_rank': r}).to_csv(filename, index=False, encoding='utf-8')
                phase_stats[mode]['prediction_file_sha256'] = sha256(filename)
                predictions[mode] = (p, r, top)
            write_json(artifact_dir / f'{phase}_candidate_sets.json', candidate_sets)
            phase_stats['candidate_sets_equal'] = True
            phase_stats['candidate_sets_sha256'] = sha256(artifact_dir / f'{phase}_candidate_sets.json')
            old = np.asarray(predictions['legacy'][0]) == np.asarray(gold)
            new = np.asarray(predictions['separate'][0]) == np.asarray(gold)
            phase_stats['paired'] = {}
            for label, mask in [('all', np.ones(len(val), dtype=bool)), ('hangul', ko), ('non_hangul', ~ko)]:
                phase_stats['paired'][label] = {
                    'fixes': int((~old & new & mask).sum()), 'breaks': int((old & ~new & mask).sum()),
                    'changed_predictions': int(((np.asarray(predictions['legacy'][0]) != np.asarray(predictions['separate'][0])) & mask).sum())}
            official_text = []
            result = official_evaluate(['--oof', str(artifact_dir / f'oof_{phase}_separate.csv'),
                '--compare', str(artifact_dir / f'oof_{phase}_legacy.csv'), '--train', str(ROOT / 'data/train.csv'),
                '--folds', str(ROOT / 'eval_kit/folds_group5_seed42.csv'), '--subset', '--boot', '2000', '--seed', '0'],
                out=official_text.append)
            phase_stats['official'] = {k: result[k] for k in ['score', 'boot', 'compare', 'warnings', 'folds']}
            (out / f'{phase}_official.txt').write_text('\n'.join(official_text) + '\n', encoding='utf-8')
            if abs(result['score']['w'] - phase_stats['separate']['weighted_accuracy']) > 1e-12:
                raise AssertionError('Official score mismatch')
            for kind, mask in [('fix', ~old & new), ('break', old & ~new)]:
                for i in np.flatnonzero(mask)[:10]:
                    errors.append({'phase': phase, 'kind': kind, 'source': 'official public train heldout/derived stress only',
                        'sample_id': str(val.iloc[i].sample_id), 'observed_name': str(data.iloc[i].vessel_name),
                        'gold_vessel_id': gold[i], 'legacy_prediction': predictions['legacy'][0][i],
                        'separate_prediction': predictions['separate'][0][i],
                        'legacy_gold_rank': predictions['legacy'][1][i], 'separate_gold_rank': predictions['separate'][1][i],
                        'failure_class': 'retrieval omission' if predictions['separate'][1][i] == 0 else 'candidate selection',
                        'interpretation': 'Observed rank/output change; underlying linguistic cause unconfirmed.'})
            record['phase_results'][phase] = phase_stats
            write_json(out / 'summary.json', record['phase_results'])
            write_json(path, record)
            print(json.dumps({'phase': phase, 'legacy': phase_stats['legacy'], 'separate': phase_stats['separate'],
                'paired': phase_stats['paired'], 'official_delta_ci': result['compare']['boot']['w']}, ensure_ascii=False), flush=True)
        write_json(artifact_dir / 'public_heldout_errors.json', errors)
        # Audit representative real rows for all/half/shuffled/single consistency.
        selected = list(np.flatnonzero(ko)[:6]) + list(np.flatnonzero(~ko)[:6])
        probe = inputs.iloc[selected]
        independence = {}
        for mode in ['legacy', 'separate']:
            p, _ = engine.predict_rows(probe, model, index, use_train_lookup=False, jamo_score_mode=mode)
            q, _ = engine.predict_rows(probe.iloc[::-1], model, index, use_train_lookup=False, jamo_score_mode=mode)
            singles = [engine.predict_rows(probe.iloc[[i]], model, index, use_train_lookup=False, jamo_score_mode=mode)[0][0] for i in range(len(probe))]
            if p != q[::-1] or p != singles:
                raise AssertionError('Row independence failed')
            independence[mode] = {'rows': len(probe), 'reverse_differences': 0, 'single_differences': 0}
        if any(sha256(ROOT / rel) != digest for rel, digest in snapshot.items()):
            raise AssertionError('Baseline assets changed during evaluation')
        clean = record['phase_results']['clean']
        regressed = any(x['separate']['weighted_accuracy'] < x['legacy']['weighted_accuracy'] for x in record['phase_results'].values())
        record.update(status='completed_not_promoted', finished_at_kst=now_kst(),
            row_independence=independence, error_cases_file=str(artifact_dir / 'public_heldout_errors.json'),
            artifacts=[str(out), str(artifact_dir), str(reference)],
            decision=('Reject separate-channel candidate after observed regression; retain original AS-06. No outer-fold retuning.' if regressed else 'Keep separate-channel candidate experimental; one pre-explored fold cannot support production promotion.'),
            limitations=audit['residual_limits'] + [audit['generation_model_link'], 'One fold and one noise seed; not a full5fold or official server improvement.'],
            next_actions=['Use public heldout errors to document the result; do not tune weight on this outer fold.', 'Follow separately registered pronunciation/retrieval experiments if continuing.'])
        record['metrics'].update(metric_source='Clean outerfold0 development comparison; not server', **{k: clean['separate'][k] for k in ['hangul_rows','non_hangul_rows','hangul_accuracy','non_hangul_accuracy','weighted_accuracy']})
        record['cost']['total_seconds'] = time.perf_counter() - started
        write_json(path, record)
        print('COMPLETE: ' + record['decision'], flush=True)
    except BaseException as exc:
        record_failure(path, record, exc)
        raise


if __name__ == '__main__':
    main()