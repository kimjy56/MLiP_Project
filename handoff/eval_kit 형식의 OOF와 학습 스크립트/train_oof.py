"""AS-02 5-Fold OOF Training and Evaluation Script (eval_kit standard).
Reproduces AS-02 (Korean metadata score disabled) on G5-s42 split.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import script as as02_module


def main():
    parser = argparse.ArgumentParser(description="Train AS-02 5-Fold OOF and evaluate with eval_kit")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="Data directory with train.csv and vessel_catalog.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "eval_kit/folds_group5_seed42.csv", help="Fold CSV path")
    parser.add_argument("--output-oof", type=Path, default=SCRIPT_DIR / "oof_AS-02.csv", help="Target OOF CSV path")
    parser.add_argument("--evaluate", action="store_true", default=True, help="Run eval_kit evaluate.py after completion")
    args = parser.parse_args()

    print("=== AS-02 5-Fold OOF Generation (Idea D: Korean Metadata Score OFF) ===")
    t_start = time.monotonic()

    # 1. Load Data
    train_path = args.data_dir / "train.csv"
    catalog_path = args.data_dir / "vessel_catalog.csv"
    print(f"Loading data from {args.data_dir}...")
    train = pd.read_csv(train_path, dtype={"sample_id": str, "target_vessel_id": str})
    catalog = pd.read_csv(catalog_path, dtype={"vessel_id": str})
    folds = pd.read_csv(args.folds, dtype=str)

    assert set(folds.sample_id) == set(train.sample_id), "Fold sample_ids do not match train"
    joined = train.merge(folds[["sample_id", "fold"]], on="sample_id")

    index = as02_module.CatalogIndex(catalog)
    oof_rows = []

    # 2. Sequential 5-Fold Loop
    for k in range(5):
        t0 = time.monotonic()
        print(f"\n--- [Fold {k}/5] ---")
        real_df = joined[joined.fold != str(k)].drop(columns="fold")
        val_df = joined[joined.fold == str(k)].copy()

        # Fit model on training fold only
        print(f"Fitting AS-02 model on {len(real_df)} train rows...")
        model = as02_module.fit_matching_model(real_df, catalog)

        # Predict with Korean metadata disabled (Idea D)
        query = val_df.drop(columns=["target_vessel_id", "fold"])
        gold_ids = val_df.target_vessel_id.tolist()
        print(f"Predicting on {len(val_df)} validation rows (disable_korean_metadata=True)...")
        preds, ranks = as02_module.predict_rows(
            query, model, index, gold_ids=gold_ids, disable_korean_metadata=True
        )

        fold_frame = pd.DataFrame({
            "sample_id": val_df.sample_id,
            "pred_vessel_id": preds,
            "fold": int(k)
        })
        oof_rows.append(fold_frame)
        sec = time.monotonic() - t0
        print(f"Fold {k} completed in {sec:.2f}s")

    # 3. Concatenate and save OOF
    oof = pd.concat(oof_rows, ignore_index=True)
    assert len(oof) == len(train), f"OOF rows ({len(oof)}) != train rows ({len(train)})"
    oof.to_csv(args.output_oof, index=False, encoding="utf-8")
    total_sec = time.monotonic() - t_start
    print(f"\nSaved OOF ({len(oof)} rows) to: {args.output_oof} (Elapsed: {total_sec:.2f}s)")

    # 4. Optional eval_kit Evaluation
    if args.evaluate and (ROOT / "eval_kit/evaluate.py").exists():
        print("\n=== Running eval_kit/evaluate.py ===")
        cmd = [
            sys.executable,
            str(ROOT / "eval_kit/evaluate.py"),
            "--oof", str(args.output_oof),
            "--train", str(train_path)
        ]
        ref_ar02 = SCRIPT_DIR / "oof_AR02-reproduced.csv"
        if ref_ar02.exists():
            cmd.extend(["--compare", str(ref_ar02)])
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
