"""Local-only provenance and file helpers for the noisy-channel experiments."""
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
CODE_FILES = ["name_decoder/lora.py", "name_decoder/noisy_channel.py",
              "name_decoder/catalog.py", "name_decoder/data.py",
              "experiments/noisy_channel_common.py",
              "experiments/train_noisy_channel.py",
              "experiments/score_noisy_channel_candidates.py"]


def now_kst():
    return datetime.now(KST).isoformat(timespec="seconds")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def configure_cpu(threads):
    if not 1 <= threads <= 4:
        raise ValueError("CPU threads must be between 1 and 4")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def environment():
    packages = {}
    for name in ("torch", "transformers", "safetensors", "pandas", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "device": "cpu"}


def code_state():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT,
                                           stderr=subprocess.DEVNULL).decode("utf-8").strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {"git_revision_at_run": git("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status),
            "file_sha256": {name: sha256(ROOT / name) for name in CODE_FILES}}


def source_hashes(data_dir, folds_file):
    return {"train.csv": sha256(data_dir / "train.csv"),
            "vessel_catalog.csv": sha256(data_dir / "vessel_catalog.csv"),
            "folds": sha256(folds_file)}


def reserve_experiment(experiment_id, output_parent, purpose, command):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", experiment_id):
        raise ValueError("Experiment ID may contain only letters, digits, '-' and '_'")
    record_path = ROOT / "project_log" / "experiments" / f"{experiment_id}.json"
    output_dir = Path(output_parent) / experiment_id
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; use a new experiment ID: {output_dir}")
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8-sig"))
        if record.get("experiment_id") != experiment_id or record.get("status") != "planned":
            raise FileExistsError("Existing non-planned experiment must not be overwritten")
    else:
        record = json.loads((ROOT / "project_log/templates/experiment.json").read_text(
            encoding="utf-8-sig"))
    record.update(experiment_id=experiment_id, status="running", purpose=purpose,
                  recorded_at_kst=now_kst(), started_at_kst=now_kst(),
                  commands=[command], code=code_state(), environment=environment())
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(record_path, record)
    return output_dir, record_path, record


def record_failure(record_path, record, exc):
    record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                  finished_at_kst=now_kst(),
                  failure={"type": type(exc).__name__, "message": str(exc),
                           "retry": "Use a new experiment ID; partial outputs are preserved."})
    write_json(record_path, record)
