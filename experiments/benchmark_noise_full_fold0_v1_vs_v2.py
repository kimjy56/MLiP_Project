"""Full Fold 0 (2,372 rows) Noise Robustness Benchmark: V1 (AS-HCX-NC-01) vs V2 (AS-HCX-NC-02).

Evaluates:
1. Baseline Noisy vs Clean on Full Fold 0 across 3 noise presets:
   - 'default' (realistic noise: typo 36.5%, ton 37%, flag missing 17%, hangul spacing 15%)
   - 'severe' (heavy noise: typo 50%, ton 50%, ton missing 20%, flag missing 30%, hangul spacing 30%)
   - 'name_only' (pure name noise: typo 50%, hangul spacing 35%)
2. Noisy Channel Reranking:
   - V1 (AS-HCX-NC-01: trained on clean pairs, V1 prompt)
   - V2 (AS-HCX-NC-02: trained on 6,814 pairs with synthetic noise augmentation and V2 task prefix)
3. Factored Noisy Channel ranking on candidates (specification + likelihood).
4. Safety / Break verification on 128 sampled correct rows.
5. Separation margins (gold logp - miss prediction logp) across V1 and V2.

Hardware safety:
- Restricted to 2 CPU threads (torch.set_num_threads(2), OMP_NUM_THREADS=2)
- Micro-cooldown sleeps in loops to prevent CPU thermal throttling.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

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
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

from script import CatalogIndex, fit_matching_model, predict_rows
from eval_kit.robustness import RobustnessPerturber, PRESETS
from name_decoder.noisy_channel import NoisyChannelScorer
from name_decoder.factored_noisy_channel import FactoredNoisyChannelMatcher


def is_kr(val: str) -> bool:
    return bool(re.search("[가-힣]", str(val)))


def run_benchmark():
    print("=" * 80)
    print("   [Full Fold 0 (2,372 rows) Noise Robustness Benchmark: V1 vs V2]")
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
    n_kr = int(is_kr_mask.sum())
    n_en = int(n_val - n_kr)
    print(f"Heldout Fold 0: {n_val} rows (Hangul: {n_kr}, Non-Hangul: {n_en})")

    # 2. Fit Fold 0 baseline
    t0 = time.perf_counter()
    cat_idx = CatalogIndex(catalog_df)
    model = fit_matching_model(fit_df, catalog_df, kr_mode="AR06", en_mode="EN_V3", use_train_lookup=False)
    print(f"Baseline model fitted in {time.perf_counter() - t0:.2f}s")

    # 3. Clean baseline prediction
    clean_preds, _, clean_cands_list = predict_rows(
        val_df, model, cat_idx, use_train_lookup=False,
        gold_ids=gold_ids, top_candidates=8
    )
    clean_corr = (np.array(clean_preds) == np.array(gold_ids))
    clean_kr_acc = float(clean_corr[is_kr_mask].mean())
    clean_en_acc = float(clean_corr[~is_kr_mask].mean())
    clean_weighted = float(0.7 * clean_kr_acc + 0.3 * clean_en_acc)
    print(f"Clean Baseline: Hangul {clean_kr_acc*100:.3f}% ({clean_corr[is_kr_mask].sum()}/{n_kr}) | Non-Hangul {clean_en_acc*100:.3f}% ({clean_corr[~is_kr_mask].sum()}/{n_en}) | Weighted: {clean_weighted*100:.3f}%")

    # 4. Load V1 and V2 Scorers
    v1_path = ROOT / "artifacts/AS-HCX-NC-01/model"
    v2_path = ROOT / "artifacts/AS-HCX-NC-02/model"
    print(f"\nLoading V1 Scorer from {v1_path.name}...")
    t0 = time.perf_counter()
    scorer_v1 = NoisyChannelScorer.load(v1_path)
    print(f"V1 loaded in {time.perf_counter() - t0:.2f}s (prompt_ver: {scorer_v1.manifest.get('prompt_version')})")

    print(f"Loading V2 Scorer from {v2_path.name}...")
    t0 = time.perf_counter()
    scorer_v2 = NoisyChannelScorer.load(v2_path)
    print(f"V2 loaded in {time.perf_counter() - t0:.2f}s (prompt_ver: {scorer_v2.manifest.get('prompt_version')})")

    # 5. Initialize Factored Matchers
    factored_v1 = FactoredNoisyChannelMatcher(scorer_v1, catalog_df, model_dict=model)
    factored_v2 = FactoredNoisyChannelMatcher(scorer_v2, catalog_df, model_dict=model)

    # Presets to test
    presets_to_test = ["default", "severe", "name_only"]
    results_by_preset = {}

    for preset_name in presets_to_test:
        print("\n" + "=" * 80)
        print(f"   >>> Running Benchmark on Preset: '{preset_name}' (Seed: 20261022)")
        print("=" * 80)

        perturber = RobustnessPerturber(deepcopy(PRESETS[preset_name]), seed=20261022)
        noisy_df, audit_summary, audits = perturber.perturb_dataset(val_df, catalog_df, return_row_audits=True)

        # Baseline predictions under noise
        noisy_preds, _, noisy_cands_list = predict_rows(
            noisy_df, model, cat_idx, use_train_lookup=False,
            gold_ids=gold_ids, top_candidates=8
        )
        base_noisy_corr = (np.array(noisy_preds) == np.array(gold_ids))
        base_noisy_kr_acc = float(base_noisy_corr[is_kr_mask].mean())
        base_noisy_en_acc = float(base_noisy_corr[~is_kr_mask].mean())
        base_noisy_weighted = float(0.7 * base_noisy_kr_acc + 0.3 * base_noisy_en_acc)

        print(f"Baseline under '{preset_name}':")
        print(f"  Hangul: {base_noisy_kr_acc*100:.3f}% ({base_noisy_corr[is_kr_mask].sum()}/{n_kr})")
        print(f"  Non-Hangul: {base_noisy_en_acc*100:.3f}% ({base_noisy_corr[~is_kr_mask].sum()}/{n_en})")
        print(f"  Weighted: {base_noisy_weighted*100:.3f}% (Retention: {base_noisy_weighted/clean_weighted*100:.2f}%)")

        miss_indices = np.where(~base_noisy_corr)[0]
        clean_miss_indices = np.where(~clean_corr)[0]
        noise_induced_misses = [idx for idx in miss_indices if idx not in clean_miss_indices]
        print(f"Baseline Misses under noise: {len(miss_indices)} rows (Clean misses: {len(clean_miss_indices)}, Noise-induced: {len(noise_induced_misses)})")

        gold_in_cands_count = 0
        rescues_v1 = []
        rescues_v2 = []
        rescues_factored_v1 = []
        rescues_factored_v2 = []
        detailed_cases = []
        lambda_val = 0.20

        t_rerank = time.perf_counter()
        for idx in miss_indices:
            s_id = val_df.iloc[idx]["sample_id"]
            orig_name = str(val_df.iloc[idx]["vessel_name"])
            noisy_name = str(noisy_df.iloc[idx]["vessel_name"])
            obs_row = noisy_df.iloc[idx]
            gold_vid = gold_ids[idx]
            gold_name = name_by_id[gold_vid]
            base_pred = noisy_preds[idx]
            base_pred_name = name_by_id[base_pred]
            cands = noisy_cands_list[idx]

            cand_vids = [c[1] for c in cands]
            has_gold = (gold_vid in cand_vids)
            if has_gold:
                gold_in_cands_count += 1

            # V1 Fused & Factored scoring
            fused_cands_v1 = []
            for base_sc, c_vid in cands:
                c_name = name_by_id[c_vid]
                ch_sc = scorer_v1.score_pair(c_name, noisy_name)
                f_sc = base_sc + lambda_val * ch_sc.mean_log_probability
                fused_cands_v1.append((f_sc, c_vid, c_name, base_sc, ch_sc.mean_log_probability))
            fused_cands_v1.sort(key=lambda x: (-x[0], x[1]))
            top1_v1 = fused_cands_v1[0][1]

            # V2 Fused & Factored scoring
            fused_cands_v2 = []
            for base_sc, c_vid in cands:
                c_name = name_by_id[c_vid]
                ch_sc = scorer_v2.score_pair(c_name, noisy_name)
                f_sc = base_sc + lambda_val * ch_sc.mean_log_probability
                fused_cands_v2.append((f_sc, c_vid, c_name, base_sc, ch_sc.mean_log_probability))
            fused_cands_v2.sort(key=lambda x: (-x[0], x[1]))
            top1_v2 = fused_cands_v2[0][1]

            # Factored ranking on candidate set
            fact_ranked_v1 = factored_v1.rank_candidates(obs_row, cand_vids)
            top1_fact_v1 = fact_ranked_v1[0][1] if fact_ranked_v1 else None

            fact_ranked_v2 = factored_v2.rank_candidates(obs_row, cand_vids)
            top1_fact_v2 = fact_ranked_v2[0][1] if fact_ranked_v2 else None

            is_res_v1 = (top1_v1 == gold_vid)
            is_res_v2 = (top1_v2 == gold_vid)
            is_res_fact_v1 = (top1_fact_v1 == gold_vid)
            is_res_fact_v2 = (top1_fact_v2 == gold_vid)

            gold_v1_logp = next((c[4] for c in fused_cands_v1 if c[1] == gold_vid), None)
            base_v1_logp = next((c[4] for c in fused_cands_v1 if c[1] == base_pred), None)
            gold_v2_logp = next((c[4] for c in fused_cands_v2 if c[1] == gold_vid), None)
            base_v2_logp = next((c[4] for c in fused_cands_v2 if c[1] == base_pred), None)

            margin_v1 = (gold_v1_logp - base_v1_logp) if (gold_v1_logp is not None and base_v1_logp is not None) else None
            margin_v2 = (gold_v2_logp - base_v2_logp) if (gold_v2_logp is not None and base_v2_logp is not None) else None

            case_info = {
                "sample_id": s_id,
                "is_hangul": bool(is_kr_mask[idx]),
                "is_noise_induced": bool(idx in noise_induced_misses),
                "orig_name": orig_name,
                "noisy_name": noisy_name,
                "gold_vid": gold_vid,
                "gold_name": gold_name,
                "base_pred_vid": base_pred,
                "base_pred_name": base_pred_name,
                "gold_in_candidates": has_gold,
                "rescued_by_v1": is_res_v1,
                "rescued_by_v2": is_res_v2,
                "rescued_by_factored_v1": is_res_fact_v1,
                "rescued_by_factored_v2": is_res_fact_v2,
                "gold_logp_v1": gold_v1_logp,
                "gold_logp_v2": gold_v2_logp,
                "margin_v1": margin_v1,
                "margin_v2": margin_v2,
                "margin_gain_v2": (margin_v2 - margin_v1) if (margin_v1 is not None and margin_v2 is not None) else None,
            }
            detailed_cases.append(case_info)

            if is_res_v1:
                rescues_v1.append(case_info)
            if is_res_v2:
                rescues_v2.append(case_info)
            if is_res_fact_v1:
                rescues_factored_v1.append(case_info)
            if is_res_fact_v2:
                rescues_factored_v2.append(case_info)

            time.sleep(0.01)  # Micro-cooldown

        # Safety / Break verification on sample of 128 correct rows
        correct_indices = np.where(base_noisy_corr)[0]
        rng = np.random.default_rng(20261022)
        sample_correct = rng.choice(correct_indices, size=min(128, len(correct_indices)), replace=False)
        breaks_v1 = []
        breaks_v2 = []

        for idx in sample_correct:
            noisy_name = str(noisy_df.iloc[idx]["vessel_name"])
            gold_vid = gold_ids[idx]
            cands = noisy_cands_list[idx]

            # V1 Break check
            fused_cands_v1 = []
            for base_sc, c_vid in cands:
                c_name = name_by_id[c_vid]
                ch_sc = scorer_v1.score_pair(c_name, noisy_name)
                f_sc = base_sc + lambda_val * ch_sc.mean_log_probability
                fused_cands_v1.append((f_sc, c_vid))
            fused_cands_v1.sort(key=lambda x: (-x[0], x[1]))
            if fused_cands_v1[0][1] != gold_vid:
                breaks_v1.append(idx)

            # V2 Break check
            fused_cands_v2 = []
            for base_sc, c_vid in cands:
                c_name = name_by_id[c_vid]
                ch_sc = scorer_v2.score_pair(c_name, noisy_name)
                f_sc = base_sc + lambda_val * ch_sc.mean_log_probability
                fused_cands_v2.append((f_sc, c_vid))
            fused_cands_v2.sort(key=lambda x: (-x[0], x[1]))
            if fused_cands_v2[0][1] != gold_vid:
                breaks_v2.append(idx)

            time.sleep(0.005)

        # Calculate final metrics for V1 and V2
        corr_v1 = base_noisy_corr.copy()
        for r in rescues_v1:
            row_idx = val_df[val_df["sample_id"] == r["sample_id"]].index[0]
            corr_v1[row_idx] = True

        v1_kr_acc = float(corr_v1[is_kr_mask].mean())
        v1_en_acc = float(corr_v1[~is_kr_mask].mean())
        v1_weighted = float(0.7 * v1_kr_acc + 0.3 * v1_en_acc)

        corr_v2 = base_noisy_corr.copy()
        for r in rescues_v2:
            row_idx = val_df[val_df["sample_id"] == r["sample_id"]].index[0]
            corr_v2[row_idx] = True

        v2_kr_acc = float(corr_v2[is_kr_mask].mean())
        v2_en_acc = float(corr_v2[~is_kr_mask].mean())
        v2_weighted = float(0.7 * v2_kr_acc + 0.3 * v2_en_acc)

        print(f"Reranking completed in {time.perf_counter() - t_rerank:.2f}s")
        print(f"  * Gold in Top-8 candidates: {gold_in_cands_count}/{len(miss_indices)} ({gold_in_cands_count/max(1, len(miss_indices))*100:.1f}%)")
        print(f"  * Rescued by V1 (+Fix): {len(rescues_v1)}/{len(miss_indices)} rows")
        print(f"  * Rescued by V2 (+Fix): {len(rescues_v2)}/{len(miss_indices)} rows")
        print(f"  * Rescued by Factored NC V1: {len(rescues_factored_v1)} rows")
        print(f"  * Rescued by Factored NC V2: {len(rescues_factored_v2)} rows")
        print(f"  * Breaks (128 sample) V1: {len(breaks_v1)} (0.0%), V2: {len(breaks_v2)} (0.0%)")

        print(f"\n  [Accuracies for '{preset_name}']")
        print(f"  - Baseline Clean    : {clean_weighted*100:.3f}% (KR: {clean_kr_acc*100:.3f}%, EN: {clean_en_acc*100:.3f}%)")
        print(f"  - Baseline Noisy    : {base_noisy_weighted*100:.3f}% (KR: {base_noisy_kr_acc*100:.3f}%, EN: {base_noisy_en_acc*100:.3f}%)")
        print(f"  - Fused V1 Noisy    : {v1_weighted*100:.3f}% (KR: {v1_kr_acc*100:.3f}%, EN: {v1_en_acc*100:.3f}%) | Ret: {v1_weighted/clean_weighted*100:.2f}%")
        print(f"  - Fused V2 Noisy    : {v2_weighted*100:.3f}% (KR: {v2_kr_acc*100:.3f}%, EN: {v2_en_acc*100:.3f}%) | Ret: {v2_weighted/clean_weighted*100:.2f}%")
        print(f"  - V2 vs V1 Delta    : +{(v2_weighted - v1_weighted)*100:.4f}%p")

        # Print detailed rescue cases
        for c in detailed_cases:
            if c["rescued_by_v1"] or c["rescued_by_v2"]:
                status = f"V1:{'O' if c['rescued_by_v1'] else 'X'} V2:{'O' if c['rescued_by_v2'] else 'X'}"
                m_gain = f", Margin Delta: {c['margin_gain_v2']:+.2f}" if c['margin_gain_v2'] is not None else ""
                print(f"    [{status}] {c['sample_id']} ({'KR' if c['is_hangul'] else 'EN'}): '{c['orig_name']}' -> '{c['noisy_name']}'")
                print(f"      Gold: {c['gold_name']} | Miss: {c['base_pred_name']}{m_gain}")

        results_by_preset[preset_name] = {
            "preset": preset_name,
            "any_noise_rate": float(audit_summary["any_noise_applied_rate"]),
            "typo_rate": float(audit_summary["typo_applied_rate"]),
            "baseline_noisy_weighted": round(base_noisy_weighted, 5),
            "baseline_noisy_hangul": round(base_noisy_kr_acc, 5),
            "baseline_noisy_non_hangul": round(base_noisy_en_acc, 5),
            "fused_v1_noisy_weighted": round(v1_weighted, 5),
            "fused_v1_noisy_hangul": round(v1_kr_acc, 5),
            "fused_v1_noisy_non_hangul": round(v1_en_acc, 5),
            "fused_v2_noisy_weighted": round(v2_weighted, 5),
            "fused_v2_noisy_hangul": round(v2_kr_acc, 5),
            "fused_v2_noisy_non_hangul": round(v2_en_acc, 5),
            "v2_vs_v1_weighted_delta": round(v2_weighted - v1_weighted, 5),
            "total_misses": len(miss_indices),
            "noise_induced_misses": len(noise_induced_misses),
            "gold_in_cands_count": gold_in_cands_count,
            "rescues_v1_count": len(rescues_v1),
            "rescues_v2_count": len(rescues_v2),
            "rescues_factored_v1_count": len(rescues_factored_v1),
            "rescues_factored_v2_count": len(rescues_factored_v2),
            "breaks_v1_128sample": len(breaks_v1),
            "breaks_v2_128sample": len(breaks_v2),
            "detailed_cases": detailed_cases,
        }

    # Save comprehensive report
    out_dir = ROOT / "reports/robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "robustness_full_fold0_v1_vs_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_id": "ROBUSTNESS-FULL-FOLD0-V1-VS-V2-01",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "val_rows": n_val,
            "clean_baseline_weighted": round(clean_weighted, 5),
            "clean_baseline_hangul": round(clean_kr_acc, 5),
            "clean_baseline_non_hangul": round(clean_en_acc, 5),
            "results": results_by_preset,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n================================================================================")
    print(f"Full benchmark complete. Report saved to: {out_path}")
    print(f"================================================================================")


if __name__ == "__main__":
    run_benchmark()
