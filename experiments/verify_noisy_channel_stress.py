"""Verify Noisy Channel robustness and candidate reranking under noise stress.

Tests:
1. Standard 128-row pilot (64 Korean, 64 Non-Korean) under 'default' noise preset (seed=20261022).
   - Injected perturbations: typos (36.5%), spacing (15%), tonnage shifts (37%), missing flags (17%).
2. Evaluates:
   - Baseline Clean Accuracy vs Baseline Noisy Accuracy.
   - Fused (Baseline + Noisy Channel lambda=0.20) on Noisy Data.
   - Specific noise rescue cases: where baseline failed due to typos/perturbations but Noisy Channel recovered the gold ship.
   - Retention Rate (% of clean score retained under noise).
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "AR-06") not in sys.path:
    sys.path.insert(0, str(ROOT / "AR-06"))

import numpy as np
import pandas as pd
import re

from script import CatalogIndex, fit_matching_model, predict_rows, is_korean_name
from eval_kit.robustness import RobustnessPerturber, PRESETS
from name_decoder.noisy_channel import NoisyChannelScorer


def is_kr(val: str) -> bool:
    return bool(re.search("[가-힣]", str(val)))


def run_noise_validation():
    print("=" * 80)
    print("   [Noisy Channel vs Baseline: Noise Stress Test Verification]")
    print("   Noise Preset: 'default' | Seed: 20261022 | Heldout Fold: 0")
    print("=" * 80)

    data_dir = ROOT / "data"
    train_path = data_dir / "train.csv"
    catalog_path = data_dir / "vessel_catalog.csv"
    folds_path = ROOT / "eval_kit/folds_group5_seed42.csv"

    # 1. Load data
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog_df = pd.read_csv(catalog_path, encoding="utf-8-sig")
    folds_df = pd.read_csv(folds_path, dtype={"sample_id": str, "fold": int}, encoding="utf-8-sig")
    name_by_id = dict(zip(catalog_df["vessel_id"], catalog_df["vessel_name"]))

    merged = train_df.merge(folds_df[["sample_id", "fold"]], on="sample_id", how="inner")
    fit_df = merged[merged["fold"] != 0].reset_index(drop=True)
    val_df = merged[merged["fold"] == 0].reset_index(drop=True)

    # 2. Select 128-row pilot (64 Korean, 64 Non-Korean)
    val_kr = val_df[val_df["vessel_name"].apply(is_kr)].reset_index(drop=True)
    val_en = val_df[~val_df["vessel_name"].apply(is_kr)].reset_index(drop=True)
    pilot_df = pd.concat([val_kr.iloc[:64], val_en.iloc[:64]], ignore_index=True)
    n_total = len(pilot_df)

    print(f"\n[1] Pilot dataset selected: {n_total} rows (Hangul: 64, Non-Hangul: 64)")

    # 3. Inject standard noise into pilot dataset
    perturber = RobustnessPerturber(deepcopy(PRESETS["default"]), seed=20261022)
    noisy_pilot, audit_summary, audits = perturber.perturb_dataset(pilot_df, catalog_df, return_row_audits=True)

    print(f"[2] Noise injected successfully:")
    print(f"  - Total perturbed rows: {audit_summary['any_noise_applied_count']}/{audit_summary['total_rows']} ({audit_summary['any_noise_applied_rate']*100:.1f}%)")
    print(f"  - Name typos: {audit_summary.get('name_typo_count', 0)} rows")
    print(f"  - Hangul spacing: {audit_summary.get('hangul_spacing_count', 0)} rows")
    print(f"  - Tonnage shifts: {audit_summary.get('ton_shift_count', 0)} rows")
    print(f"  - Flag missing: {audit_summary.get('flag_missing_count', 0)} rows")

    # 4. Build catalog and fit Fold 0 baseline
    t0 = time.perf_counter()
    cat_idx = CatalogIndex(catalog_df)
    model = fit_matching_model(fit_df, catalog_df, kr_mode="AR06", en_mode="EN_V3", use_train_lookup=False)
    print(f"[3] Catalog indexed and Fold 0 baseline model fitted ({time.perf_counter()-t0:.2f}s)")

    # 5. Load Noisy Channel Scorer
    t0 = time.perf_counter()
    scorer = NoisyChannelScorer.load(ROOT / "artifacts/AS-HCX-NC-01/model")
    print(f"[4] NoisyChannelScorer loaded ({time.perf_counter()-t0:.2f}s)")

    gold_ids = pilot_df["target_vessel_id"].tolist()
    is_kr_mask = pilot_df["vessel_name"].apply(is_kr).to_numpy()

    # 6. Predict on CLEAN pilot (Baseline)
    clean_preds, _, _ = predict_rows(
        pilot_df, model, cat_idx, use_train_lookup=False,
        gold_ids=gold_ids, top_candidates=1
    )
    clean_corr = (np.array(clean_preds) == np.array(gold_ids))
    clean_kr_acc = clean_corr[is_kr_mask].mean()
    clean_en_acc = clean_corr[~is_kr_mask].mean()
    clean_weighted = 0.7 * clean_kr_acc + 0.3 * clean_en_acc

    # 7. Predict on NOISY pilot (Baseline + top 8 candidate extraction)
    print("\n[5] Running inference and extracting candidates on NOISY pilot...")
    noisy_preds, _, noisy_cands_list = predict_rows(
        noisy_pilot, model, cat_idx, use_train_lookup=False,
        gold_ids=gold_ids, top_candidates=8
    )
    base_noisy_corr = (np.array(noisy_preds) == np.array(gold_ids))
    base_noisy_kr_acc = base_noisy_corr[is_kr_mask].mean()
    base_noisy_en_acc = base_noisy_corr[~is_kr_mask].mean()
    base_noisy_weighted = 0.7 * base_noisy_kr_acc + 0.3 * base_noisy_en_acc

    # 8. Rerank NOISY candidates with Noisy Channel
    print("[6] Scoring NOISY candidates with Noisy Channel (lambda=0.20)...")
    lambda_val = 0.20
    fused_preds = []
    rescues = []
    breaks = []

    for idx in range(n_total):
        s_id = pilot_df.iloc[idx]["sample_id"]
        orig_name = pilot_df.iloc[idx]["vessel_name"]
        noisy_name = noisy_pilot.iloc[idx]["vessel_name"]
        gold_vid = gold_ids[idx]
        base_pred = noisy_preds[idx]
        cands = noisy_cands_list[idx]

        # Score top candidates with Noisy Channel
        fused_cands = []
        for base_sc, c_vid in cands:
            c_name = name_by_id[c_vid]
            ch_score = scorer.score_pair(c_name, noisy_name)
            # Fused score = baseline_score + lambda * mean_log_probability
            combined = base_sc + lambda_val * ch_score.mean_log_probability
            fused_cands.append((combined, c_vid, c_name, base_sc, ch_score.mean_log_probability))

        fused_cands.sort(key=lambda x: (-x[0], x[1]))
        fused_top1_vid = fused_cands[0][1]
        fused_preds.append(fused_top1_vid)

        # Track rescues and breaks
        was_correct = (base_pred == gold_vid)
        is_correct = (fused_top1_vid == gold_vid)

        if not was_correct and is_correct:
            rescues.append({
                "sample_id": s_id,
                "orig_name": orig_name,
                "noisy_name": noisy_name,
                "gold_vid": gold_vid,
                "gold_name": name_by_id[gold_vid],
                "base_pred_vid": base_pred,
                "base_pred_name": name_by_id[base_pred],
            })
        elif was_correct and not is_correct:
            breaks.append({
                "sample_id": s_id,
                "orig_name": orig_name,
                "noisy_name": noisy_name,
                "gold_vid": gold_vid,
                "gold_name": name_by_id[gold_vid],
                "fused_pred_vid": fused_top1_vid,
                "fused_pred_name": name_by_id[fused_top1_vid],
            })

    fused_corr = (np.array(fused_preds) == np.array(gold_ids))
    fused_kr_acc = fused_corr[is_kr_mask].mean()
    fused_en_acc = fused_corr[~is_kr_mask].mean()
    fused_weighted = 0.7 * fused_kr_acc + 0.3 * fused_en_acc

    # 9. Results Summary
    retention_base = (base_noisy_weighted / clean_weighted) * 100
    retention_fused = (fused_weighted / clean_weighted) * 100

    print("\n" + "=" * 85)
    print(f"{'Condition':<28} | {'Hangul Acc':<12} | {'Non-Hangul':<12} | {'Weighted Acc':<14} | {'Retention'}")
    print("-" * 85)
    print(f"{'1. Baseline (Clean Data)':<28} | {clean_kr_acc*100:6.2f}% ({clean_corr[is_kr_mask].sum():2d}/64) | {clean_en_acc*100:6.2f}% ({clean_corr[~is_kr_mask].sum():2d}/64) | {clean_weighted*100:6.3f}%        | 100.00%")
    print(f"{'2. Baseline (Noisy Data)':<28} | {base_noisy_kr_acc*100:6.2f}% ({base_noisy_corr[is_kr_mask].sum():2d}/64) | {base_noisy_en_acc*100:6.2f}% ({base_noisy_corr[~is_kr_mask].sum():2d}/64) | {base_noisy_weighted*100:6.3f}%        | {retention_base:6.2f}%")
    print(f"{'3. Fused NC (Noisy Data, λ=0.20)':<28} | {fused_kr_acc*100:6.2f}% ({fused_corr[is_kr_mask].sum():2d}/64) | {fused_en_acc*100:6.2f}% ({fused_corr[~is_kr_mask].sum():2d}/64) | {fused_weighted*100:6.3f}%        | {retention_fused:6.2f}%")
    print("=" * 85)

    print(f"\nRescues (+Fix by Noisy Channel): {len(rescues)} rows")
    for r in rescues:
        print(f"  * [RESCUE] {r['sample_id']}: '{r['orig_name']}' -> Noisy: '{r['noisy_name']}'")
        print(f"    Gold: {r['gold_name']} ({r['gold_vid']}) vs Baseline Miss: {r['base_pred_name']} ({r['base_pred_vid']})")

    print(f"Breaks (Clean correct -> Broken by NC): {len(breaks)} rows")
    for b in breaks:
        print(f"  * [BREAK] {b['sample_id']}: '{b['orig_name']}' -> Noisy: '{b['noisy_name']}'")
        print(f"    Gold: {b['gold_name']} vs Fused Error: {b['fused_pred_name']}")

    # Save report
    out_dir = ROOT / "reports/robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "robustness_noisy_channel_pilot128.json"
    report = {
        "benchmark_id": "ROBUSTNESS-NC-PILOT128-01",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "noise_preset": "default",
        "noise_seed": 20261022,
        "sample_size": n_total,
        "perturbed_rows_count": audit_summary["any_noise_applied_count"],
        "perturbed_rows_rate": audit_summary["any_noise_applied_rate"],
        "lambda": lambda_val,
        "metrics": {
            "clean_baseline_weighted": round(clean_weighted, 5),
            "clean_baseline_hangul": round(clean_kr_acc, 5),
            "clean_baseline_non_hangul": round(clean_en_acc, 5),
            "noisy_baseline_weighted": round(base_noisy_weighted, 5),
            "noisy_baseline_hangul": round(base_noisy_kr_acc, 5),
            "noisy_baseline_non_hangul": round(base_noisy_en_acc, 5),
            "noisy_fused_weighted": round(fused_weighted, 5),
            "noisy_fused_hangul": round(fused_kr_acc, 5),
            "noisy_fused_non_hangul": round(fused_en_acc, 5),
            "retention_baseline": round(retention_base, 2),
            "retention_fused": round(retention_fused, 2),
            "rescues_count": len(rescues),
            "breaks_count": len(breaks),
            "net_gain": len(rescues) - len(breaks),
        },
        "rescues": rescues,
        "breaks": breaks,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved robustness report to: {report_path}")


if __name__ == "__main__":
    run_noise_validation()
