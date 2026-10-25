"""Build AS-05 from the existing AS-04 ZIP and trained reverse checkpoint.

Packaging only: no fitting, model conversion, private test inference or upload.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build():
    baseline = ROOT / "submit_AS-04.zip"
    checkpoint = ROOT / "artifacts/AS-HCX-NC-01/model"
    stage = ROOT / "artifacts/AS-05/package"
    destination = ROOT / "submit_AS-05.zip"
    record_path = ROOT / "project_log/experiments/AS-05-PACKAGE-01.json"
    if stage.exists() or destination.exists() or record_path.exists():
        raise FileExistsError("AS-05 package or record exists; do not silently overwrite")
    manifest = json.loads((checkpoint / "noisy_channel_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("direction") != "catalog_name_to_observed_name" or not manifest.get("merged"):
        raise ValueError("Requires the completed merged reverse checkpoint")
    record = {
        "experiment_id": "AS-05-PACKAGE-01", "submission_id": "AS-05", "status": "planned",
        "recorded_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "purpose": "User-requested AS-05 ZIP using the completed reverse-channel checkpoint",
        "hypothesis": "Gated reverse-name likelihood can refine ambiguous baseline candidates",
        "baseline": {"path": str(baseline.relative_to(ROOT)), "sha256": digest(baseline)},
        "checkpoint": {"path": str(checkpoint.relative_to(ROOT)),
                       "manifest_sha256": digest(checkpoint / "noisy_channel_manifest.json"),
                       "heldout_fold": manifest["heldout_fold"],
                       "checkpoint_sha256": manifest["checkpoint_sha256"]},
        "configuration": {"channel_weight": 0.20, "normalization": "mean_including_EOS",
                          "max_baseline_margin": 1.0, "candidate_limit": 35,
                          "bound_pruning": "candidate baseline score upper-bounds fused score",
                          "cpu_threads": 4, "device": "CUDA float32 when available; CPU float32 otherwise",
                          "row_independence": "no cross-row cache, aggregation or time-adaptive gate"},
        "code": {"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 "dirty": True,
                 "source_sha256": {name: digest(ROOT / name) for name in (
                     "experiments/build_as05_submission.py", "experiments/as05_reranker.py",
                     "name_decoder/noisy_channel.py", "name_decoder/catalog.py")}},
        "metrics": {"official_server_score": None, "full_oof_score": None},
        "checks": {}, "limitations": [
            "Uses the completed fold-0-excluded reverse checkpoint without retraining.",
            "Fold 1-4 examples may have been used to train this reverse checkpoint; they are not OOF evidence.",
            "128-row cached pilot is not full AS-05 ID validation; margin gating is a new integration choice.",
            "GPU execution and official server end-to-end runtime have not been verified locally."],
        "upload_performed": False, "retraining_performed": False,
    }
    write_json(record_path, record)
    try:
        stage.mkdir(parents=True, exist_ok=False)
        (stage / "model/channel").mkdir(parents=True)
        with zipfile.ZipFile(baseline) as archive:
            if set(archive.namelist()) != {"script.py", "requirements.txt", "model/model.json"}:
                raise ValueError("Unexpected AS-04 archive structure")
            script = archive.read("script.py").decode("utf-8-sig")
            (stage / "model/model.json").write_bytes(archive.read("model/model.json"))
        signature = "    gold_ids: Optional[List[str]] = None,\n) -> Tuple[List[str], List[int]]:"
        if script.count(signature) != 1:
            raise ValueError("AS-04 predict_rows signature changed")
        script = script.replace(signature, "    gold_ids: Optional[List[str]] = None,\n    name_reranker=None,\n) -> Tuple[List[str], List[int]]:", 1)
        target = "best_vid = cat_idx.vessel_ids[scored_cands[0][1]]"
        if script.count(target) != 2:
            raise ValueError("AS-04 must have both Korean and non-Korean selection sites")
        lines = []
        for line in script.splitlines():
            if line.strip() == target:
                indent = line[:len(line) - len(line.lstrip())]
                lines.extend([indent + "best_index = scored_cands[0][1]",
                              indent + "if name_reranker is not None:",
                              indent + "    best_index = name_reranker.select(raw_name, scored_cands, cat_idx)",
                              indent + "best_vid = cat_idx.vessel_ids[best_index]"])
            else:
                lines.append(line)
        script = "\n".join(lines) + "\n"
        # Thread/offline settings are installed BEFORE numpy/torch imports.
        script = script.replace("import os\n", '''import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
''', 1)
        old_call = "    preds, _ = predict_rows(test_df, model, cat_idx, use_train_lookup=use_lookup, gold_ids=None)"
        if script.count(old_call) != 1:
            raise ValueError("AS-04 entry point changed")
        script = script.replace(old_call, '''    import sys
    sys.path.insert(0, str(base_dir / "model"))
    from as05_reranker import NoisyChannelReranker
    reranker = NoisyChannelReranker(base_dir / "model", catalog_df)
    preds, _ = predict_rows(test_df, model, cat_idx, use_train_lookup=use_lookup,
                           gold_ids=None, name_reranker=reranker)
    print(f"AS-05 rows={len(test_df)} reranked={reranker.eligible_rows} "
          f"pairs={reranker.scored_pairs} changed={reranker.changed_rows} "
          f"unsupported={reranker.unsupported_rows}")''', 1)
        compile(script, "script.py", "exec")
        (stage / "script.py").write_text(script, encoding="utf-8")
        (stage / "requirements.txt").write_text(
            "# All runtime packages are preinstalled in the official competition image.\n"
            "# torch==2.7.1 transformers==4.57.6 safetensors==0.6.2\n"
            "# numpy==1.26.4 pandas==2.0.3 rapidfuzz==3.14.3\n", encoding="utf-8")
        (stage / "model/channel/__init__.py").write_text("", encoding="utf-8")
        for name in ("catalog.py", "noisy_channel.py"):
            shutil.copy2(ROOT / "name_decoder" / name, stage / "model/channel" / name)
        shutil.copy2(ROOT / "experiments/as05_reranker.py", stage / "model/as05_reranker.py")
        config = {"submission_id": "AS-05", "checkpoint_id": "AS-HCX-NC-01",
                  **record["configuration"]}
        write_json(stage / "model/as05_config.json", config)
        weights = stage / "model/noisy_channel_weights"
        weights.mkdir()
        for relative, expected in manifest["checkpoint_sha256"].items():
            src = (checkpoint / relative).resolve()
            dst = (weights / relative).resolve()
            if not src.is_relative_to(checkpoint.resolve()) or not dst.is_relative_to(weights.resolve()):
                raise ValueError("Invalid checkpoint path")
            if digest(src) != expected:
                raise ValueError(f"Trained checkpoint changed: {relative}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        shutil.copy2(checkpoint / "noisy_channel_manifest.json", weights / "noisy_channel_manifest.json")
        # Keep any locally supplied license notices with the inference assets.
        pretrained = ROOT / "artifacts/pretrained/hyperclovax-seed-0.5b"
        for pattern in ("LICENSE*", "NOTICE*"):
            for src in pretrained.glob(pattern):
                if src.is_file():
                    notice_dir = stage / "model/licenses"
                    notice_dir.mkdir(exist_ok=True)
                    shutil.copy2(src, notice_dir / src.name)
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        for path in files:
            if path.suffix == ".py":
                compile(path.read_text(encoding="utf-8-sig"), path.name, "exec")
        # Stored weights avoid a lengthy compression pass; archive is <10 GB.
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in files:
                archive.write(path, path.relative_to(stage).as_posix())
        record.update(status="packaged_pending_runtime_checks",
                      finished_at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                      archive={"path": destination.name, "sha256": digest(destination),
                               "bytes": destination.stat().st_size, "member_count": len(files)},
                      staging_directory=str(stage.relative_to(ROOT)),
                      checks={"source_compilation": True, "checkpoint_hashes_match": True,
                              "official_structure": True})
        record["package_code_sha256"] = {path.relative_to(stage).as_posix(): digest(path)
                                         for path in files if path.suffix == ".py"}
        write_json(record_path, record)
        print(json.dumps(record["archive"], ensure_ascii=False), flush=True)
    except BaseException as exc:
        record.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        write_json(record_path, record)
        raise


if __name__ == "__main__":
    build()
