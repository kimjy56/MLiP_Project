"""Provenance/file helpers for the EXAONE pipeline; no legacy model assets."""
from __future__ import annotations
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MODEL_ID="LGAI-EXAONE/EXAONE-4.0-1.2B"
CODE_FILES=["name_decoder/deterministic_ko_rules.py","name_decoder/exaone_data.py",
            "name_decoder/exaone_translit.py","name_decoder/exaone_lexicon.py",
            "name_decoder/exaone_reverse.py","experiments/train_exaone_reverse.py",
            "experiments/exaone_common.py","experiments/train_exaone_translit.py",
            "experiments/build_exaone_ko_lexicon.py","experiments/evaluate_pure_exaone_oof.py",
            "experimental_matchers/exaone_lexicon.py","tools/fetch_exaone.py","name_decoder/__init__.py"]


def now_kst():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def code_state():
    return {"git_revision_at_run":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT).decode().strip(),
            "dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT)),
            "file_sha256":{p:sha256(ROOT/p) for p in CODE_FILES if (ROOT/p).is_file()}}


def begin_run(experiment_id,purpose,output_parent=None):
    if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]*",experiment_id):
        raise ValueError("Invalid experiment ID")
    path=ROOT/"project_log/experiments"/f"{experiment_id}.json"
    out=Path(output_parent or ROOT/"artifacts")/experiment_id
    if out.exists():raise FileExistsError(f"Output already exists; choose a new experiment ID: {out}")
    if path.exists():
        record=json.loads(path.read_text(encoding="utf-8-sig"))
        if record.get("status")!="planned" or record.get("experiment_id")!=experiment_id:
            raise FileExistsError("Do not overwrite a completed/failed record")
    else:
        record=json.loads((ROOT/"project_log/templates/experiment.json").read_text(encoding="utf-8-sig"))
    out.mkdir(parents=True,exist_ok=False)
    record.update(experiment_id=experiment_id,status="running",purpose=purpose,
        recorded_at_kst=now_kst(),started_at_kst=now_kst(),code=code_state(),
        commands=[subprocess.list2cmdline([sys.executable,*sys.argv])],
        environment={"python":platform.python_version(),"executable":sys.executable,"platform":platform.platform(),
                     "target_python":"3.11.17"})
    for relative in CODE_FILES:
        source=ROOT/relative
        if source.is_file():
            target=out/"source_snapshot"/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(source.read_bytes())
    write_json(path,record)
    return out,path,record


def finish_run(path,record,status="completed",error=None):
    record.update(status=status,finished_at_kst=now_kst())
    if error is not None:
        record["failure"]={"type":type(error).__name__,"message":str(error),
                           "retry":"Preserve partial outputs; use a new experiment ID"}
    write_json(path,record)
    # Append an aggregate-only index row; never change historical rows.
    index=ROOT/"project_log/experiments/INDEX.csv"
    import csv
    with index.open(encoding="utf-8-sig",newline="") as stream:
        rows=list(csv.DictReader(stream))
    if not any(row["experiment_id"]==record["experiment_id"] for row in rows):
        with index.open("a",encoding="utf-8",newline="") as stream:
            csv.writer(stream,lineterminator="\n").writerow([
                record["experiment_id"],"EXAONE legacy-free lexicon",status,
                record.get("metrics",{}).get("metric_source") or "not evaluated","",
                f"experiments/{record['experiment_id']}.json"])


def configure(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    os.environ["TOKENIZERS_PARALLELISM"]="false"
