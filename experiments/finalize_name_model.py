"""Fit the chosen variant, package it, and check public-data execution offline."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train import fit
from verify_submission import verify, FILES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True)
    p.add_argument('--selection-file',type=Path,default=ROOT/'reports/name_improvement.json')
    p.add_argument('--benchmark',action='store_true',help='Run all public train rows with unknown calls, no accuracy evaluation')
    args=p.parse_args()
    record=json.loads(args.record.read_text())
    assert record['status']=='planned', 'Do not overwrite completed experiments'
    assert not args.output_dir.exists(), 'Use a new bundle directory'
    selection=json.loads(args.selection_file.read_text())
    chosen=selection['selected_variant']
    channels=selection.get('selected_channels')
    mode,search=('all',24) if chosen=='search' or channels is not None else (chosen,0)
    record.update(status='running',started_at_kst=now(),commands=[' '.join(sys.argv)],
                  selected_variant=chosen,name_mode=mode,phonetic_search=search,
                  retrieval_channels=channels,selection_file_sha256=sha(args.selection_file),
                  code_sha256={n:sha(ROOT/n) for n in ['train.py','model/matcher.py','model/name_processing.py','model/ngram_retrieval.py','script.py','verify_submission.py','experiments/finalize_name_model.py']},
                  data_sha256={n:sha(args.data_dir/n) for n in ['train.csv','vessel_catalog.csv']},
                  environment={'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','lightgbm','rapidfuzz','Unidecode']}})
    def save():
        args.record.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    save()
    try:
        args.output_dir.mkdir(parents=True)
        train=pd.read_csv(args.data_dir/'train.csv')
        catalog=pd.read_csv(args.data_dir/'vessel_catalog.csv')
        start=time.monotonic()
        record['fit']=fit(train,catalog,args.output_dir/'model',mode,search,channels)
        record['fit_wall_seconds']=time.monotonic()-start
        for n in ['script.py','requirements.txt','model/matcher.py','model/name_processing.py','model/ngram_retrieval.py']:
            shutil.copy2(ROOT/n,args.output_dir/n)
        (args.output_dir/'data').symlink_to(args.data_dir.resolve(),target_is_directory=True)
        save()
        verify(args.output_dir)
        record['submission_checks']=json.loads((args.output_dir/'reports/submission_checks.json').read_text())
        record['artifacts']={n:{'path':str(args.output_dir/n),'bytes':(args.output_dir/n).stat().st_size,
                                'sha256':sha(args.output_dir/n)} for n in FILES+['submit.zip']}
        save()
        if args.benchmark:
            # Evaluate execution cost only. No claim of holdout accuracy: these
            # are all public training rows, with forced general matching path.
            with tempfile.TemporaryDirectory(prefix='vessel-benchmark-') as tmp:
                tmp=Path(tmp)
                with zipfile.ZipFile(args.output_dir/'submit.zip') as z:z.extractall(tmp)
                (tmp/'data').mkdir()
                shutil.copy2(args.data_dir/'vessel_catalog.csv',tmp/'data/vessel_catalog.csv')
                queries=train.drop(columns='target_vessel_id').copy()
                queries['call_sign']='LOCAL_UNKNOWN_CALL_FOR_COST'
                queries.to_csv(tmp/'data/test.csv',index=False)
                inputs={n:sha(tmp/'data'/n) for n in ['test.csv','vessel_catalog.csv']}
                for n in inputs:(tmp/'data'/n).chmod(0o444)
                runner=("import socket,runpy,resource,json; "
                    "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled')); "
                    "socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled')); "
                    f"runpy.run_path({str(tmp/'script.py')!r},run_name='__main__'); "
                    "print(json.dumps({'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))")
                start=time.monotonic()
                result=subprocess.run([sys.executable,'-c',runner],cwd='/tmp',capture_output=True,text=True,
                                      timeout=1200,check=True,env={**os.environ,'MPLCONFIGDIR':'/tmp/mlip-mpl'})
                elapsed=time.monotonic()-start
                pred=pd.read_csv(tmp/'output/submission.csv')
                assert len(pred)==len(queries) and pred.sample_id.tolist()==queries.sample_id.tolist()
                assert pred.target_vessel_id.isin(catalog.vessel_id).all()
                assert inputs=={n:sha(tmp/'data'/n) for n in inputs}
                record['benchmark']={'rows':len(queries),'hangul_rows':int(queries.vessel_name.str.contains('[가-힣]',na=False).sum()),
                    'call_sign':'all deliberately unknown','aliases':'unchanged full-train assets',
                    'seconds':elapsed,**json.loads(result.stdout.strip().splitlines()[-1]),
                    'scope':'fresh process startup, imports, CSV reads, catalog index, features, prediction, CSV write; excludes ZIP extraction and preparing inputs',
                    'checks':['offline','read-only inputs unchanged','row count/order','catalog membership'],
                    'limitation':'Public training-input execution benchmark, not predictive evaluation or official server measurement; hidden test size is unknown.'}
                print(json.dumps(record['benchmark']),flush=True)
        record.update(status='completed',finished_at_kst=now())
    except Exception:
        record.update(status='failed',finished_at_kst=now(),failure=traceback.format_exc())
        save()
        raise
    save()


if __name__=='__main__':
    main()
