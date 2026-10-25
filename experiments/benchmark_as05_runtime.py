"""Time the exact AS-05 package on a small public sample (row count explicit).

No model fitting and no private evaluation inputs. Repeated public rows are
explicitly a runtime workload, not accuracy/OOF evidence.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--experiment-id", default="AS-05-RUNTIME-02")
    args = parser.parse_args()
    if args.rows <= 0:
        parser.error("rows must be positive")
    stage = ROOT / "artifacts/AS-05/package"
    record_path = ROOT / "project_log/experiments" / (args.experiment_id + ".json")
    run_root = ROOT / "artifacts" / args.experiment_id
    if record_path.exists() or run_root.exists():
        raise FileExistsError("Use a fresh runtime experiment ID")
    now = datetime.now(timezone(timedelta(hours=9))).isoformat()
    record = {"experiment_id": args.experiment_id, "status": "planned", "recorded_at_kst": now,
              "purpose": "Measure AS-05 exact package end-to-end against 1800-second limit",
              "scope": "public train-derived runtime stress; no accuracy claim",
              "seed": 42, "requested_rows": args.rows, "budget_seconds": 1800,
              "source_package_sha256": sha(ROOT / "submit_AS-05.zip"),
              "code_sha256": sha(__file__), "results": None,
              "limitations": ["Local CPU runtime, not official L4 server measurement.",
                              "Repeated public observations; not a bound for every unseen input distribution."]}
    save(record_path, record)
    run_root.mkdir(parents=True)
    package = run_root / "package"
    package.mkdir()
    # Hard links preserve exact package bytes without a second multi-GB copy.
    # Only manifest-listed submission files are linked; no data or caches.
    import zipfile
    with zipfile.ZipFile(ROOT / "submit_AS-05.zip") as archive:
        members = archive.namelist()
    for name in members:
        src = (stage / name).resolve()
        dst = (package / name).resolve()
        if not src.is_relative_to(stage.resolve()) or not dst.is_relative_to(package.resolve()):
            raise ValueError("Unexpected package member")
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    import pandas as pd
    train = pd.read_csv(ROOT / "data/train.csv", keep_default_na=False)
    full_repeats, remainder = divmod(args.rows, len(train))
    chunks = [train.copy() for _ in range(full_repeats)]
    if remainder:
        chunks.append(train.sample(n=remainder, random_state=42))
    public = pd.concat(chunks, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    public["sample_id"] = [f"RUNTIME_{i:06d}" for i in range(len(public))]
    public = public.drop(columns=["target_vessel_id"])
    data_dir = package / "data"
    data_dir.mkdir()
    public.to_csv(data_dir / "test.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(ROOT / "data/vessel_catalog.csv", data_dir / "vessel_catalog.csv")
    record["input"] = {"rows": len(public), "hangul_rows": int(public.vessel_name.str.contains(r"[\uac00-\ud7a3]").sum()),
                       "full_public_repetitions": full_repeats, "sampled_remainder": remainder,
                       "train_sha256": sha(ROOT / "data/train.csv"),
                       "input_sha256": sha(data_dir / "test.csv"),
                       "catalog_sha256": sha(data_dir / "vessel_catalog.csv")}
    record["status"] = "running"
    record["started_at_kst"] = datetime.now(timezone(timedelta(hours=9))).isoformat()
    save(record_path, record)
    environment = os.environ.copy()
    environment.update({"PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                        "OPENBLAS_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                        "CUDA_VISIBLE_DEVICES": ""})
    print(f"Starting exact script: {len(public)} public-derived rows; CPU 4 threads; 1800s deadline", flush=True)
    entry = package / "script.py"
    if args.profile:
        entry = run_root / "profile_entry.py"
        shutil.copy2(ROOT / "experiments/profile_as05_entry.py", entry)
    started = time.perf_counter()
    with (run_root / "stdout.log").open("wb") as out, (run_root / "stderr.log").open("wb") as err:
        process = subprocess.Popen([sys.executable, "-X", "utf8", str(entry)],
                                   cwd=run_root, env=environment, stdout=out, stderr=err)
        next_report = 20
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            if elapsed >= 1800:
                process.kill()
                process.wait()
                record["status"] = "timeout"
                break
            if elapsed >= next_report:
                print(f"running: {elapsed:.0f}s / 1800s", flush=True)
                next_report += 20
            time.sleep(1)
    elapsed = time.perf_counter() - started
    if args.profile and (run_root / "profile.json").exists():
        record["profile"] = json.loads((run_root / "profile.json").read_text(encoding="utf-8"))
    record["finished_at_kst"] = datetime.now(timezone(timedelta(hours=9))).isoformat()
    record["results"] = {"seconds": elapsed, "exit_code": process.returncode,
                         "timing_scope": "Python startup/imports, input/model reads and hashes, catalog indexes, prediction and output write; excludes archive extraction/workload preparation",
                         "rows": len(public), "under_1800_seconds": elapsed < 1800 and process.returncode == 0,
                         "stdout": (run_root / "stdout.log").read_text(encoding="utf-8", errors="replace").strip()}
    if process.returncode == 0:
        predictions = pd.read_csv(package / "output/submission.csv", keep_default_na=False)
        catalog_ids = set(pd.read_csv(data_dir / "vessel_catalog.csv", usecols=["vessel_id"]).vessel_id)
        valid = (list(predictions.columns) == ["sample_id", "target_vessel_id"] and
                 predictions.sample_id.tolist() == public.sample_id.tolist() and
                 set(predictions.target_vessel_id) <= catalog_ids)
        record["results"]["output_valid"] = valid
        record["status"] = "completed" if valid else "failed"
    elif record["status"] != "timeout":
        record["status"] = "failed"
    record["artifacts"] = [str(run_root.relative_to(ROOT))]
    save(record_path, record)
    print(json.dumps(record["results"], ensure_ascii=False), flush=True)
    if record["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
