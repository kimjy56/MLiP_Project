"""Read an existing bundle and verify it; never create or overwrite a ZIP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

FILES = ['script.py', 'model/matcher.py', 'model/name_processing.py',
         'model/ngram_retrieval.py', 'model/assets.json', 'model/ranker.txt']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--reference-python', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    train = pd.read_csv(args.data_dir / 'train.csv')
    hangul = train.vessel_name.str.contains('[가-힣]', na=False)
    probe = pd.concat([train[hangul].head(30), train[~hangul].head(30)]).drop(columns='target_vessel_id')
    probe.iloc[::2, probe.columns.get_loc('call_sign')] = 'LOCAL_UNKNOWN_CALL_ENV_CHECK'
    source_hashes = {n: sha(args.bundle / n) for n in FILES}
    expected, runs = None, []
    with tempfile.TemporaryDirectory(prefix='vessel-environment-') as tmp:
        tmp = Path(tmp)
        for name in FILES:
            target = tmp / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.bundle / name, target)
        (tmp / 'data').mkdir()
        catalog_path = tmp / 'data/vessel_catalog.csv'
        shutil.copy2(args.data_dir / 'vessel_catalog.csv', catalog_path)
        catalog_path.chmod(0o444)
        catalog_hash = sha(catalog_path)
        catalog_ids = set(pd.read_csv(catalog_path).vessel_id)
        runner = (
            "import socket,runpy; "
            "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled')); "
            "socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled')); "
            f"runpy.run_path({str(tmp / 'script.py')!r},run_name='__main__')"
        )
        cases = [('reference', probe, str(args.reference_python)),
                 ('full', probe, sys.executable), ('repeat', probe, sys.executable),
                 ('shuffled', probe.sample(frac=1, random_state=9), sys.executable),
                 ('half', probe.iloc[::2], sys.executable),
                 ('single', probe.iloc[[0]], sys.executable),
                 ('empty', probe.iloc[:0], sys.executable)]
        for label, rows, interpreter in cases:
            test_path = tmp / 'data/test.csv'
            if test_path.exists():
                test_path.chmod(0o644)
            rows.to_csv(test_path, index=False)
            test_path.chmod(0o444)
            before = sha(test_path)
            start = time.monotonic()
            run = subprocess.run([interpreter, '-c', runner], cwd='/tmp',
                                 capture_output=True, text=True, check=True, timeout=180,
                                 env={**os.environ, 'MPLCONFIGDIR': str(tmp / 'mpl')})
            elapsed = time.monotonic() - start
            pred = pd.read_csv(tmp / 'output/submission.csv')
            assert list(pred.columns) == ['sample_id', 'target_vessel_id']
            assert pred.sample_id.tolist() == rows.sample_id.tolist()
            assert set(pred.target_vessel_id).issubset(catalog_ids)
            observed = dict(zip(pred.sample_id, pred.target_vessel_id))
            if expected is None:
                expected = observed
            else:
                assert observed == {key: expected[key] for key in rows.sample_id}, label
            assert sha(test_path) == before and sha(catalog_path) == catalog_hash
            assert run.stdout.strip() == f'Saved {len(rows)} predictions.'
            runs.append({'mode': label, 'rows': len(rows), 'seconds': elapsed})
            print(f'{label}: {len(rows)} rows, passed, {elapsed:.2f}s', flush=True)
    assert source_hashes == {n: sha(args.bundle / n) for n in FILES}
    result = {'passed': True, 'runs': runs, 'source_sha256': source_hashes,
              'checks': ['same predictions as old Python environment on 60 public train probes',
                         'offline entry point', 'read-only inputs unchanged', 'schema/order/catalog membership',
                         'repeat/shuffled/half/single/empty consistency', 'source bundle unchanged'],
              'limitations': ['Public training probes only, not accuracy evaluation or exhaustive equivalence',
                              'No training, no ZIP creation, no full hidden-test timing, no GPU execution']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
