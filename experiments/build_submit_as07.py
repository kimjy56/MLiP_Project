# -*- coding: utf-8 -*-
"""experiments/build_submit_as07.py — Build submit_AS-07.zip for DACON submission."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experimental_matchers/dpr_hybrid"))

from experimental_matchers.dpr_hybrid.script import fit_matching_model


def build_submission():
    print("=== Building AS-07 Submission Bundle ===")

    train_path = ROOT / "data/train.csv"
    catalog_path = ROOT / "data/vessel_catalog.csv"
    ko_map_path = ROOT / "AR-06/catalog_full_ko_map.json"
    enc_path = ROOT / "artifacts/DPR-SYMPHON-01/full_encoder.pt"
    embeds_path = ROOT / "artifacts/DPR-SYMPHON-01/catalog_embeds_full_fp16.npy"

    print("Reading train and catalog data...")
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog_df = pd.read_csv(catalog_path, encoding="utf-8-sig")
    ko_map = json.loads(ko_map_path.read_text(encoding="utf-8"))

    print("Fitting lexical matching model on full train.csv (12,222 rows)...")
    model = fit_matching_model(train_df, catalog_df, kr_mode="AR06", en_mode="EN_V3", use_train_lookup=True)
    model["catalog_full_ko_map"] = ko_map
    model["dense_mode"] = "hybrid"
    model["dense_weight"] = 2.5

    bundle_dir = ROOT / "train_AS-07"
    model_dir = bundle_dir / "model"
    phonetic_pkg_dir = model_dir / "phonetic_encoder"

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    phonetic_pkg_dir.mkdir(parents=True, exist_ok=True)

    print("Copying script.py and requirements.txt...")
    src_script = (ROOT / "experimental_matchers/dpr_hybrid/script.py").read_text(encoding="utf-8")
    (bundle_dir / "script.py").write_text(src_script, encoding="utf-8")
    (bundle_dir / "requirements.txt").write_text("", encoding="utf-8")

    print("Writing model.json...")
    (model_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")

    print("Copying Bi-Encoder weights and catalog embeddings...")
    shutil.copyfile(enc_path, model_dir / "full_encoder.pt")
    shutil.copyfile(embeds_path, model_dir / "catalog_embeds_full_fp16.npy")

    print("Copying phonetic_encoder module into model/phonetic_encoder...")
    for mod_file in ("__init__.py", "model.py", "tokenizer.py", "hybrid_matcher.py"):
        shutil.copyfile(ROOT / "phonetic_encoder" / mod_file, phonetic_pkg_dir / mod_file)

    # Build ZIPs
    target_zips = [ROOT / "submit_AS-07.zip", ROOT / "submit.zip"]
    for zip_path in target_zips:
        print(f"Creating {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in bundle_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(bundle_dir).as_posix()
                    zf.write(file_path, arcname)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"  -> {zip_path.name}: {size_mb:.2f} MB")

    print("\nSubmission bundles created successfully!")


if __name__ == "__main__":
    build_submission()
