"""Download only the requested public EXAONE base; never read competition data."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.exaone_common import MODEL_ID,begin_run,finish_run,write_json
from name_decoder.exaone_translit import resolve_base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id",default="EXAONE-BASE-FETCH-01")
    args=p.parse_args()
    out,path,record=begin_run(args.experiment_id,"Resolve/download pinned official EXAONE4 1.2B base; no dataset access")
    try:
        base,provenance=resolve_base(MODEL_ID)
        write_json(out/"base_resolution.json",{"path":str(base),"provenance":provenance})
        record["artifacts"]=[str(base),str(out/"base_resolution.json")]
        finish_run(path,record,"base_available_unvalidated")
        print({"base_path":str(base),"revision":provenance.get("revision")})
    except BaseException as exc:
        finish_run(path,record,"blocked_network_or_missing_base",exc)
        raise


if __name__=="__main__":
    main()
