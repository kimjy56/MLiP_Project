# -*- coding: utf-8 -*-
"""experiments/train_all_folds.py — Sequential fine-tuning for Folds 1 to 4 (SPEC-DPR-01).

Completes the full 5-fold cross-validation checkpoints for zero-leakage evaluation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    py_exec = sys.executable
    print(f"Starting sequential fold fine-tuning with {py_exec}...")

    for fold in [1, 2, 3, 4]:
        print(f"\n==========================================")
        print(f"=== Training Fold {fold} (Stage B Fine-tuning) ===")
        print(f"==========================================")
        cmd = [
            py_exec,
            str(ROOT / "experiments/train_bi_encoder.py"),
            "--epochs-pretrain", "0",
            "--epochs-finetune", "10",
            "--fold", str(fold),
        ]
        res = subprocess.run(cmd, cwd=str(ROOT))
        if res.returncode != 0:
            print(f"Error training fold {fold}, exit code: {res.returncode}")
            sys.exit(res.returncode)

    print("\n[DONE] All 5 folds (0, 1, 2, 3, 4) fine-tuning complete!")

if __name__ == "__main__":
    main()
