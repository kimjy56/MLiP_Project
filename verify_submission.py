"""Exercise a clean ZIP on public TRAIN probes, with socket networking disabled."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parent
FILES = ['script.py', 'requirements.txt', 'model/matcher.py',
         'model/name_processing.py', 'model/ngram_retrieval.py', 'model/ranker.txt', 'model/assets.json',
         'model/catboost.cbm']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='Standalone bundle root containing code/model and data/')
    args = parser.parse_args()
    verify(args.root.resolve())


def verify(ROOT):
    (ROOT/'reports').mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'submit.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for file in FILES:
            z.write(ROOT / file, file)
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == sorted(FILES)
        assert z.testzip() is None
        unpacked = sum(x.file_size for x in z.infolist())
    assert archive.stat().st_size < 10_000_000_000
    assert unpacked < 32_000_000_000
    train = pd.read_csv(ROOT / 'data/train.csv')
    ko = train.vessel_name.str.contains('[가-힣]', na=False)
    probe = pd.concat([train[ko].head(30), train[~ko].head(30)]).drop(columns='target_vessel_id')
    # Unknown calls exercise retrieval and ranker, even after final full-data fit.
    probe.loc[probe.index[::2], 'call_sign'] = 'LOCAL_UNKNOWN_CALL'
    original = probe.copy()
    durations = []
    with tempfile.TemporaryDirectory(prefix='vessel-verify-') as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        (tmp / 'data').mkdir()
        catalog_bytes = (ROOT / 'data/vessel_catalog.csv').read_bytes()
        (tmp / 'data/vessel_catalog.csv').write_bytes(catalog_bytes)
        (tmp / 'data/vessel_catalog.csv').chmod(0o444)
        runner = (
            "import socket, runpy; "
            "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled')); "
            "socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled')); "
            f"runpy.run_path({str(tmp / 'script.py')!r},run_name='__main__')"
        )
        expected = None
        for label, rows in [('full', original), ('shuffled', original.sample(frac=1, random_state=9)),
                            ('subset', original.iloc[::4]), ('single', original.iloc[[0]]),
                            ('empty', original.iloc[:0])]:
            test_path = tmp / 'data/test.csv'
            if test_path.exists():
                test_path.chmod(0o644)
            rows.to_csv(test_path, index=False)
            test_path.chmod(0o444)
            before = hashlib.sha256(test_path.read_bytes()).hexdigest()
            start = time.monotonic()
            result = subprocess.run([sys.executable, '-c', runner], cwd=tempfile.gettempdir(),
                                    capture_output=True, text=True, check=True,
                                    env={**os.environ, 'MPLCONFIGDIR': str(tmp/'mpl')}, timeout=120)
            durations.append({'mode': label, 'rows': len(rows), 'seconds': time.monotonic()-start})
            output = pd.read_csv(tmp / 'output/submission.csv')
            assert list(output.columns) == ['sample_id', 'target_vessel_id']
            assert output.sample_id.tolist() == rows.sample_id.tolist()
            assert output.target_vessel_id.isin(pd.read_csv(ROOT/'data/vessel_catalog.csv').vessel_id).all()
            observed = dict(zip(output.sample_id, output.target_vessel_id))
            if expected is None:
                expected = observed
            else:
                assert observed == {key: expected[key] for key in rows.sample_id}
            assert hashlib.sha256(test_path.read_bytes()).hexdigest() == before
            assert (tmp/'data/vessel_catalog.csv').read_bytes() == catalog_bytes
            assert result.stdout.strip() == f'Saved {len(rows)} predictions.'
    report = {'passed': True, 'files': FILES, 'zip_bytes': archive.stat().st_size,
              'uncompressed_bytes': unpacked, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'checks': ['clean extraction', 'offline entry point', 'read-only inputs',
                         'output schema', 'catalog membership', 'input order preserved',
                         'full/shuffled/subset/single/empty consistency'],
              'runs': durations,
              'limitation': 'Local Python 3.12 environment; not the official evaluation server.'}
    (ROOT/'reports/submission_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
