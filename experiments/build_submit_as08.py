# -*- coding: utf-8 -*-
"""experiments/build_submit_as08.py — Build submit_AS-08.zip for DACON submission.

Features in AS-08:
  - Multi-View V2 Bi-Encoder (DPR-MULTIVIEW-02) with 4-category exception views ([ACR], [INIT], [SEM], [NUM])
  - Manual lexicon (IDEA_B_WORD_VARS) completely removed (0 hand-curated entries, DPR-ABLATE-LEXICON-01)
  - 2D DP Dynamic Programming lattice loop (lattice_word_similarity) completely excised
  - Synthetic Hangul catalog map (catalog_full_ko_map.json) turned OFF (0 synthetic distractor collisions)
  - 4 Dense similarity features in Stage-2 scoring (dense_sim, dense_rank_recip, dense_zscore, dense_margin_top1)
  - 100% strict row independence, zero BatchNorm, LayerNorm-only, pure deterministic inference.
  - Complete packaging of ALL phonetic_encoder package files (no missing loss.py / dataset.py).
  - Explicit fatal assertion preventing silent fallback from hybrid to none.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experimental_matchers/dpr_hybrid"))

from experimental_matchers.dpr_hybrid.script import fit_matching_model


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def build_submission():
    print("=== Building AS-08 Submission Bundle ===")

    train_path = ROOT / "data/train.csv"
    catalog_path = ROOT / "data/vessel_catalog.csv"
    enc_path = ROOT / "artifacts/DPR-SYMPHON-01/full_encoder.pt"
    embeds_path = ROOT / "artifacts/DPR-SYMPHON-01/catalog_embeds_full_multiview_fp16.npy"

    print("Reading train and catalog data...")
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog_df = pd.read_csv(catalog_path, encoding="utf-8-sig")

    print("Fitting lexical matching model on full train.csv (12,222 rows)...")
    model = fit_matching_model(train_df, catalog_df, kr_mode="AR06", en_mode="EN_V3", use_train_lookup=True)
    # AS-08: Synthetic Hangul catalog map is completely OFF to avoid distractor collisions
    model["catalog_full_ko_map"] = {}
    model["use_train_lookup_default"] = True
    model["dense_mode"] = "hybrid"
    model["dense_weight"] = 2.5

    bundle_dir = ROOT / "train_AS-08"
    model_dir = bundle_dir / "model"
    phonetic_pkg_dir = model_dir / "phonetic_encoder"

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    phonetic_pkg_dir.mkdir(parents=True, exist_ok=True)

    print("Copying script.py and requirements.txt...")
    src_script = (ROOT / "experimental_matchers/dpr_hybrid/script.py").read_text(encoding="utf-8")
    (bundle_dir / "script.py").write_text(src_script, encoding="utf-8")
    (bundle_dir / "requirements.txt").write_text("", encoding="utf-8")

    print("Writing model.json and model/__init__.py...")
    (model_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    (model_dir / "__init__.py").write_text("", encoding="utf-8")

    print("Copying Bi-Encoder weights and catalog embeddings...")
    shutil.copyfile(enc_path, model_dir / "full_encoder.pt")
    shutil.copyfile(embeds_path, model_dir / "catalog_embeds_full_fp16.npy")

    print("Copying ENTIRE phonetic_encoder module into model/phonetic_encoder...")
    for py_file in (ROOT / "phonetic_encoder").glob("*.py"):
        shutil.copyfile(py_file, phonetic_pkg_dir / py_file.name)
        print(f"  -> Copied {py_file.name}")

    # Build ZIPs
    target_zips = [ROOT / "submit_AS-08.zip", ROOT / "submit.zip"]
    zip_stats = {}
    for zip_path in target_zips:
        print(f"Creating {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in bundle_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(bundle_dir).as_posix()
                    zf.write(file_path, arcname)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        sha = compute_sha256(zip_path)
        zip_stats[zip_path.name] = {"size_mb": round(size_mb, 2), "sha256": sha}
        print(f"  -> {zip_path.name}: {size_mb:.2f} MB (SHA256: {sha})")

    # Post-build strict verification: Unpack ZIP into isolated directory and verify HybridDenseLexicalMatcher
    print("\n--- Running Strict In-Zip Isolation Smoke Test ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(ROOT / "submit_AS-08.zip") as zf:
            zf.extractall(tmp_path)

        # Test 1: Direct Python import test inside isolated directory
        cmd_import = [
            sys.executable,
            "-c",
            "import sys; "
            "from pathlib import Path; "
            "base = Path('.').resolve(); "
            "sys.path.insert(0, str(base)); "
            "sys.path.insert(0, str(base / 'model')); "
            "from phonetic_encoder.hybrid_matcher import HybridDenseLexicalMatcher; "
            "assert HybridDenseLexicalMatcher is not None, 'HybridDenseLexicalMatcher is None!'; "
            "print('Isolated Import Check: SUCCESS!')"
        ]
        res_import = subprocess.run(cmd_import, cwd=str(tmp_path), capture_output=True, text=True)
        if res_import.returncode != 0:
            print("STDERR:", res_import.stderr)
            raise RuntimeError(f"Strict In-Zip Import Check FAILED: {res_import.stderr}")
        print(res_import.stdout.strip())

        # Test 2: Run script.py on a 5-row test dataset and verify submission output
        data_dir = tmp_path / "data"
        out_dir = tmp_path / "output"
        data_dir.mkdir()
        out_dir.mkdir()

        # Slice 5 rows from train.csv to serve as test.csv
        sample_test = train_df.head(5).drop(columns=["target_vessel_id"], errors="ignore").copy()
        sample_test.to_csv(data_dir / "test.csv", index=False, encoding="utf-8-sig")
        # Copy catalog
        catalog_df.to_csv(data_dir / "vessel_catalog.csv", index=False, encoding="utf-8-sig")

        cmd_run = [sys.executable, "script.py"]
        res_run = subprocess.run(cmd_run, cwd=str(tmp_path), capture_output=True, text=True)
        if res_run.returncode != 0:
            print("STDERR:", res_run.stderr)
            raise RuntimeError(f"Strict In-Zip Inference Run FAILED: {res_run.stderr}")

        sub_csv = out_dir / "submission.csv"
        assert sub_csv.exists(), "output/submission.csv was not generated!"
        sub_df = pd.read_csv(sub_csv)
        assert len(sub_df) == 5, f"Expected 5 predictions, got {len(sub_df)}"
        print(f"Isolated 5-row Inference Check: SUCCESS! Predictions generated for all 5 rows.")

    print("\n[VERIFIED] Submission bundles created and strictly validated without errors!")
    return zip_stats


if __name__ == "__main__":
    build_submission()
