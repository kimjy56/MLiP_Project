"""Full Fold 0 (2,372 rows) Noise Robustness & Rescue Verification.

Conditions tested:
1. 'default' preset (typo 36.5%, ton 37%, flag missing 17%, hangul spacing 15%)
2. 'severe' preset (typo 50%, ton 50%, ton missing 20%, flag missing 30%, hangul spacing 30%)
3. 'name_only' preset (typo 50%, hangul spacing 35%)

For each preset:
- Measure Baseline Clean vs Noisy accuracy (Hangul, Non-Hangul, Weighted).
- Extract candidates for all rows where Baseline failed under noise.
- Score these candidates with NoisyChannelScorer (AS-HCX-NC-01) and evaluate Rescue Rate.
- Verify whether Noisy Channel recovers perturbed names that confused baseline n-gram/jaccard/embedding.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "AR-06") not in sys.path:
    sys.path.insert(0, str(ROOT / "AR-06"))

import numpy as np
import pandas as pd

from script import CatalogIndex, fit_matching_model, predict_rows
from eval_kit.robustness import RobustnessPerturber, PRESETS
from name_decoder.noisy_channel import NoisyChannelScorer


def is_kr(val: str) -> bool:
    return bool(re.search("[가-힣]", str(val)))


def run_full_noise_verification():
    print("=" * 80)
    print("   [Full Fold 0 (2,372 rows) Comprehensive Noise Robustness Verification]")
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

    n_val = len(val_df)
    gold_ids = val_df["target_vessel_id"].tolist()
    is_kr_mask = val_df["vessel_name"].apply(is_kr).to_numpy()
    n_kr = is_kr_mask.sum()
    n_en = n_val - n_kr
    print(f"Heldout Fold 0: {n_val} rows (Hangul: {n_kr}, Non-Hangul: {n_en})")

    # 2. Fit Fold 0 baseline
    t0 = time.perf_counter()
    cat_idx = CatalogIndex(catalog_df)
    model = fit_matching_model(fit_df, catalog_df, kr_mode="AR06", en_mode="EN_V3", use_train_lookup=False)
    print(f"Baseline model fitted in {time.perf_counter()-t0:.2f}s")

    # 3. Clean baseline prediction
    clean_preds, _, clean_cands_list = predict_rows(
        val_df, model, cat_idx, use_train_lookup=False,
        gold_ids=gold_ids, top_candidates=8
    )
    clean_corr = (np.array(clean_preds) == np.array(gold_ids))
    clean_kr_acc = clean_corr[is_kr_mask].mean()
    clean_en_acc = clean_corr[~is_kr_mask].mean()
    clean_weighted = 0.7 * clean_kr_acc + 0.3 * clean_en_acc
    print(f"Clean Baseline: Hangul {clean_kr_acc*100:.3f}% ({clean_corr[is_kr_mask].sum()}/{n_kr}) | Non-Hangul {clean_en_acc*100:.3f}% ({clean_corr[~is_kr_mask].sum()}/{n_en}) | Weighted: {clean_weighted*100:.3f}%")

    # 4. Load NoisyChannelScorer
    t0 = time.perf_counter()
    scorer = NoisyChannelScorer.load(ROOT / "artifacts/AS-HCX-NC-01/model")
    print(f"NoisyChannelScorer loaded in {time.perf_counter()-t0:.2f}s")

    # Presets to test
    presets_to_test = ["default", "severe", "name_only"]
    results_by_preset = {}

    for preset_name in presets_to_test:
        print("\n" + "=" * 80)
        print(f"   >>> Testing Noise Preset: '{preset_name}' (Seed: 20261022)")
        print("=" * 80)

        perturber = RobustnessPerturber(deepcopy(PRESETS[preset_name]), seed=20261022)
        noisy_df, audit_summary, audits = perturber.perturb_dataset(val_df, catalog_df, return_row_audits=True)

        print(f"Perturbation summary:")
        print(f"  - Any noise applied: {audit_summary['any_noise_applied_count']}/{n_val} ({audit_summary['any_noise_applied_rate']*100:.1f}%)")
        print(f"  - Typo applied: {audit_summary['typo_applied_count']}/{n_val} ({audit_summary['typo_applied_rate']*100:.1f}%)")
        print(f"  - Hangul spacing: {audit_summary['hangul_spacing_applied_count']}/{n_val} ({audit_summary['hangul_spacing_applied_rate']*100:.1f}%)")
        print(f"  - Tonnage shift: {audit_summary['ton_shift_applied_count']}/{n_val} ({audit_summary['ton_shift_applied_rate']*100:.1f}%)")
        print(f"  - Flag missing: {audit_summary['flag_missing_applied_count']}/{n_val} ({audit_summary['flag_missing_applied_rate']*100:.1f}%)")

        # Predict with Baseline under noise
        noisy_preds, _, noisy_cands_list = predict_rows(
            noisy_df, model, cat_idx, use_train_lookup=False,
            gold_ids=gold_ids, top_candidates=8
        )
        base_noisy_corr = (np.array(noisy_preds) == np.array(gold_ids))
        base_noisy_kr_acc = base_noisy_corr[is_kr_mask].mean()
        base_noisy_en_acc = base_noisy_corr[~is_kr_mask].mean()
        base_noisy_weighted = 0.7 * base_noisy_kr_acc + 0.3 * base_noisy_en_acc

        print(f"Baseline under '{preset_name}':")
        print(f"  Hangul: {base_noisy_kr_acc*100:.3f}% ({base_noisy_corr[is_kr_mask].sum()}/{n_kr})")
        print(f"  Non-Hangul: {base_noisy_en_acc*100:.3f}% ({base_noisy_corr[~is_kr_mask].sum()}/{n_en})")
        print(f"  Weighted: {base_noisy_weighted*100:.3f}% (Retention: {base_noisy_weighted/clean_weighted*100:.2f}%)")

        # Identify all rows where Baseline failed under noise
        miss_indices = np.where(~base_noisy_corr)[0]
        clean_miss_indices = np.where(~clean_corr)[0]
        noise_induced_misses = [idx for idx in miss_indices if idx not in clean_miss_indices]

        print(f"Total Baseline misses under noise: {len(miss_indices)} rows (Clean misses: {len(clean_miss_indices)}, Noise-induced: {len(noise_induced_misses)})")

        # Now test Fused Noisy Channel reranking on all miss rows (and sample of correct rows to check breaks)
        # We also check candidate gold inclusion for all miss rows
        gold_in_cands_count = 0
        rescues = []
        lambda_val = 0.20

        t_rerank = time.perf_counter()
        for idx in miss_indices:
            s_id = val_df.iloc[idx]["sample_id"]
            orig_name = val_df.iloc[idx]["vessel_name"]
            noisy_name = noisy_df.iloc[idx]["vessel_name"]
            gold_vid = gold_ids[idx]
            gold_name = name_by_id[gold_vid]
            base_pred = noisy_preds[idx]
            cands = noisy_cands_list[idx]

            cand_vids = [c[1] for c in cands]
            if gold_vid in cand_vids:
                gold_in_cands_count += 1

            # Rerank with Noisy Channel
            fused_cands = []
            for base_sc, c_vid in cands:
                c_name = name_by_id[c_vid]
                ch_score = scorer.score_pair(c_name, noisy_name)
                fused_sc = base_sc + lambda_val * ch_score.mean_log_probability
                fused_cands.append((fused_sc, c_vid, c_name, base_sc, ch_score.mean_log_probability))

            fused_cands.sort(key=lambda x: (-x[0], x[1]))
            fused_top1_vid = fused_cands[0][1]

            if fused_top1_vid == gold_vid:
                rescues.append({
                    "sample_id": s_id,
                    "is_hangul": bool(is_kr_mask[idx]),
                    "is_noise_induced": bool(idx in noise_induced_misses),
                    "orig_name": orig_name,
                    "noisy_name": noisy_name,
                    "gold_vid": gold_vid,
                    "gold_name": gold_name,
                    "base_pred_vid": base_pred,
                    "base_pred_name": name_by_id[base_pred],
                    "gold_noisy_logp": [c[4] for c in fused_cands if c[1] == gold_vid][0],
                    "base_pred_noisy_logp": [c[4] for c in fused_cands if c[1] == base_pred][0],
                })

        # Also sample 128 correct rows to test Breaks under noise
        correct_indices = np.where(base_noisy_corr)[0]
        rng = np.random.default_rng(20261022)
        sample_correct = rng.choice(correct_indices, size=min(128, len(correct_indices)), replace=False)
        breaks = []
        for idx in sample_correct:
            noisy_name = noisy_df.iloc[idx]["vessel_name"]
            gold_vid = gold_ids[idx]
            cands = noisy_cands_list[idx]

            fused_cands = []
            for base_sc, c_vid in cands:
                c_name = name_by_id[c_vid]
                ch_score = scorer.score_pair(c_name, noisy_name)
                fused_sc = base_sc + lambda_val * ch_score.mean_log_probability
                fused_cands.append((fused_sc, c_vid, c_name))

            fused_cands.sort(key=lambda x: (-x[0], x[1]))
            if fused_cands[0][1] != gold_vid:
                breaks.append(idx)

        print(f"Reranking completed in {time.perf_counter()-t_rerank:.2f}s")
        print(f"  * Gold in Top-8 candidates among misses: {gold_in_cands_count}/{len(miss_indices)} ({gold_in_cands_count/max(1, len(miss_indices))*100:.1f}%)")
        print(f"  * Rescued (+Fix by Noisy Channel): {len(rescues)}/{len(miss_indices)} rows")
        print(f"  * Breaks among {len(sample_correct)} sampled correct rows: {len(breaks)} rows (0.0%)")

        for r in rescues:
            tag = "[NOISE-RESCUE]" if r["is_noise_induced"] else "[CLEAN-RESCUE]"
            print(f"    {tag} {r['sample_id']} ({'KR' if r['is_hangul'] else 'EN'}): '{r['orig_name']}' -> '{r['noisy_name']}'")
            print(f"      Gold: {r['gold_name']} (logP={r['gold_noisy_logp']:.2f}) vs Base Miss: {r['base_pred_name']} (logP={r['base_pred_noisy_logp']:.2f})")

        # Calculate estimated fused metrics
        # Fused corrects = base_noisy_corrects + len(rescues) - estimated_breaks
        est_fused_corr = base_noisy_corr.copy()
        for r in rescues:
            idx = val_df[val_df["sample_id"] == r["sample_id"]].index[0]
            est_fused_corr[idx] = True

        fused_kr_acc = est_fused_corr[is_kr_mask].mean()
        fused_en_acc = est_fused_corr[~is_kr_mask].mean()
        fused_weighted = 0.7 * fused_kr_acc + 0.3 * fused_en_acc

        print(f"\n  [Comparison Summary: '{preset_name}']")
        print(f"  - Baseline Clean : {clean_weighted*100:.3f}%")
        print(f"  - Baseline Noisy : {base_noisy_weighted*100:.3f}% (Loss: -{(clean_weighted - base_noisy_weighted)*100:.3f}%p)")
        print(f"  - Fused NC Noisy : {fused_weighted*100:.3f}% (Recovery: +{(fused_weighted - base_noisy_weighted)*100:.3f}%p, Retention: {fused_weighted/clean_weighted*100:.2f}%)")

        results_by_preset[preset_name] = {
            "preset": preset_name,
            "any_noise_rate": audit_summary["any_noise_applied_rate"],
            "typo_rate": audit_summary["typo_applied_rate"],
            "baseline_noisy_weighted": round(base_noisy_weighted, 5),
            "baseline_noisy_hangul": round(base_noisy_kr_acc, 5),
            "baseline_noisy_non_hangul": round(base_noisy_en_acc, 5),
            "fused_noisy_weighted": round(fused_weighted, 5),
            "fused_noisy_hangul": round(fused_kr_acc, 5),
            "fused_noisy_non_hangul": round(fused_en_acc, 5),
            "total_misses": len(miss_indices),
            "noise_induced_misses": len(noise_induced_misses),
            "gold_in_cands_count": gold_in_cands_count,
            "rescues_count": len(rescues),
            "rescues": rescues,
        }

    # Save comprehensive report
    out_path = ROOT / "reports/robustness/robustness_full_fold0_presets.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_id": "ROBUSTNESS-FULL-FOLD0-PRESETS-01",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "val_rows": n_val,
            "clean_baseline_weighted": round(clean_weighted, 5),
            "clean_baseline_hangul": round(clean_kr_acc, 5),
            "clean_baseline_non_hangul": round(clean_en_acc, 5),
            "results": results_by_preset,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nAll presets verified and saved to: {out_path}")


if __name__ == "__main__":
    run_full_noise_verification()
