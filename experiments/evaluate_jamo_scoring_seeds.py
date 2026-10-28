"""Secondary severe-noise seed check; keeps Jamo scoring experiment01 settings fixed."""
from pathlib import Path
import sys
import json
import time
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.noisy_channel_common import (
    configure_cpu, reserve_experiment, write_json, record_failure, sha256, now_kst,
)
from experiments.evaluate_jamo_scoring import load_engine, predict_chunks, metrics


def main():
    configure_cpu(2)
    eid = 'JAMO-SCORE-02'
    out, path, record = reserve_experiment(eid, ROOT / 'reports/matching_improvements',
        'Secondary severe-noise check on two seeds; no parameter changes', ' '.join(sys.argv))
    start = time.perf_counter()
    try:
        import pandas as pd
        import numpy as np
        from name_decoder.data import read_frames
        from eval_kit.robustness import PRESETS, RobustnessPerturber
        from eval_kit.evaluate import run as official_evaluate
        primary = json.loads((ROOT / 'project_log/experiments/JAMO-SCORE-01.json').read_text(encoding='utf-8'))
        if primary['status'] != 'completed_not_promoted':
            raise ValueError('Primary comparison is incomplete')
        if sha256(ROOT / 'experimental_matchers/jamo_scoring/script.py') != primary['code']['file_sha256']['experimental_matchers/jamo_scoring/script.py']:
            raise ValueError('Scoring source changed after primary experiment')
        artifact_dir = ROOT / 'artifacts' / eid
        artifact_dir.mkdir(parents=True, exist_ok=False)
        model = json.loads((ROOT / 'artifacts/JAMO-SCORE-01/evaluation/fit_model.json').read_text(encoding='utf-8'))
        _, val, fit, _ = read_frames(ROOT / 'data', ROOT / 'eval_kit/folds_group5_seed42.csv', 0)
        catalog = pd.read_csv(ROOT / 'data/vessel_catalog.csv', keep_default_na=False)
        mapping = json.loads((ROOT / 'AR-06/catalog_full_ko_map.json').read_text(encoding='utf-8-sig'))
        reference = ROOT / 'artifacts/JAMO-SCORE-01/reference'
        snapshot = json.loads((reference / 'snapshot_manifest.json').read_text(encoding='utf-8'))
        if any(sha256(ROOT / rel) != digest for rel, digest in snapshot.items()):
            raise ValueError('Reference asset changed')
        engine = load_engine(ROOT / 'experimental_matchers/jamo_scoring/script.py', '_jamo_scoring_fixed_seed_check')
        index = engine.CatalogIndex(catalog, full_ko_map=mapping)
        gold = val.target_vessel_id.tolist()
        ko = val.hangul.to_numpy(dtype=bool)
        record['data'] = dict(primary['data'])
        record['data']['seed'] = None
        record['data']['noise_seeds'] = [20261026, 20261027]
        record['code']['file_sha256'].update({p: sha256(ROOT / p) for p in [
            'experimental_matchers/jamo_scoring/script.py', 'experiments/evaluate_jamo_scoring.py', 'experiments/evaluate_jamo_scoring_seeds.py']})
        record['phase_results'] = {}
        record['environment'].update(threads=2, device='cpu')
        write_json(path, record)
        for seed in [20261026, 20261027]:
            phase = f'severe_{seed}'
            print('PHASE ' + phase, flush=True)
            noisy, noise_summary = RobustnessPerturber(deepcopy(PRESETS['severe']), seed=seed).perturb_dataset(val, catalog)
            data = noisy.drop(columns=['target_vessel_id','target_vessel_id_fold','source','target','hangul','fold'], errors='ignore').reset_index(drop=True)
            assert data.sample_id.tolist() == val.sample_id.tolist()
            result = {'seed': seed, 'noise_summary': noise_summary}
            predictions = {}
            sets = None
            for mode in ['legacy', 'separate']:
                p, r, top, candidates, seconds = predict_chunks(engine, data, model, index, mode, gold)
                if len(candidates) != len(val) or (sets is not None and sets != candidates):
                    raise AssertionError('Candidate set mismatch')
                sets = candidates
                filename = artifact_dir / f'oof_{phase}_{mode}.csv'
                pd.DataFrame({'sample_id': val.sample_id.tolist(), 'fold': 0, 'pred_vessel_id': p, 'gold_rank': r}).to_csv(filename, index=False, encoding='utf-8')
                result[mode] = metrics(p, gold, ko, r)
                result[mode].update(prediction_seconds=seconds, prediction_file_sha256=sha256(filename))
                predictions[mode] = np.asarray(p)
            write_json(artifact_dir / f'{phase}_candidate_sets.json', sets)
            result['candidate_sets_equal'] = True
            result['candidate_sets_sha256'] = sha256(artifact_dir / f'{phase}_candidate_sets.json')
            old, new = predictions['legacy'] == np.asarray(gold), predictions['separate'] == np.asarray(gold)
            result['paired'] = {label: {'fixes': int((~old & new & mask).sum()), 'breaks': int((old & ~new & mask).sum())}
                for label, mask in [('hangul',ko), ('non_hangul',~ko)]}
            lines = []
            official = official_evaluate(['--oof', str(artifact_dir / f'oof_{phase}_separate.csv'),
                '--compare', str(artifact_dir / f'oof_{phase}_legacy.csv'), '--train', str(ROOT / 'data/train.csv'),
                '--folds', str(ROOT / 'eval_kit/folds_group5_seed42.csv'), '--subset', '--boot', '2000', '--seed', '0'], out=lines.append)
            result['official'] = {k: official[k] for k in ['score','boot','compare','warnings','folds']}
            if abs(official['score']['w'] - result['separate']['weighted_accuracy']) > 1e-12:
                raise AssertionError('Official score mismatch')
            (out / f'{phase}_official.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            record['phase_results'][phase] = result
            write_json(out / 'summary.json', record['phase_results'])
            write_json(path, record)
            print(json.dumps({'phase':phase, 'legacy_h':result['legacy']['hangul_correct'],
                'separate_h':result['separate']['hangul_correct'], 'paired':result['paired'],
                'delta_ci':official['compare']['boot']['w']}), flush=True)
        if any(sha256(ROOT / rel) != digest for rel, digest in snapshot.items()):
            raise AssertionError('Baseline changed')
        record.update(status='completed_not_promoted', finished_at_kst=now_kst(),
            artifacts=[str(out),str(artifact_dir)], cost={'total_seconds':time.perf_counter()-start,
                'timing_scope':'Load/index, predictions including candidate hashes, official bootstrap and saving; CPU, not server timing.'},
            decision='Retain as an experimental candidate only; no automatic production promotion or ZIP.',
            limitations=primary['limitations'] + ['Secondary seeds were selected before execution but after observing primary results.',
                'Repeated transformations of the same vessels are correlated; no independent sample-size increase claimed.'])
        write_json(path, record)
        print('SECONDARY CHECK COMPLETE', flush=True)
    except BaseException as exc:
        record_failure(path, record, exc)
        raise


if __name__ == '__main__':
    main()