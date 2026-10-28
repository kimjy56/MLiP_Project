# -*- coding: utf-8 -*-
"""Prepare nested splits and candidate caches for KO-FUSION-01.

Implements SPEC-KO-FUSION-01 Section 6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_as09_dir = _repo_root / "train_AS-09"
if str(_as09_dir) not in sys.path:
    sys.path.insert(0, str(_as09_dir))

from experimental_matchers.ko_fusion.as09_adapter import AS09RerankAdapter, fit_split_model
from experimental_matchers.ko_fusion.cache import CandidateCacheWriter, label_trace
import script as as09


def main():
    parser = argparse.ArgumentParser(description="Prepare KO-FUSION-01 splits and trace caches")
    parser.add_argument("--experiment-id", type=str, default="KO-FUSION-01")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--folds", type=str, default="eval_kit/folds_group5_seed42.csv")
    parser.add_argument("--outer-fold", type=int, default=0, help="Outer fold to evaluate (default 0)")
    parser.add_argument("--out-dir", type=str, default="artifacts/KO-FUSION-01")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    folds_path = Path(args.folds).resolve()
    out_dir = Path(args.out_dir).resolve() / f"outer_{args.outer_fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== [1] Loading data and folds for outer fold {args.outer_fold} ===")
    train_df = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
    catalog_df = pd.read_csv(data_dir / "vessel_catalog.csv", encoding="utf-8-sig")
    folds_df = pd.read_csv(folds_path, dtype={"sample_id": str, "fold": int}, encoding="utf-8-sig")
    merged = train_df.merge(folds_df[["sample_id", "fold"]], on="sample_id", how="inner")

    with open(_as09_dir / "model" / "model.json", "r", encoding="utf-8") as f:
        base_model = json.load(f)

    t_df = merged[merged["fold"] != args.outer_fold].reset_index(drop=True)
    v_df = merged[merged["fold"] == args.outer_fold].reset_index(drop=True)
    print(f"  - T (Train fit): {len(t_df)} rows, {t_df['target_vessel_id'].nunique()} vessels")
    print(f"  - V (Heldout fold {args.outer_fold}): {len(v_df)} rows, {v_df['target_vessel_id'].nunique()} vessels")

    # Inner split of T into A (training) and B (model selection)
    print("=== [2] Creating inner A/B split on T (StratifiedGroupKFold, 5 folds, random_state=42) ===")
    kr_mask_t = t_df["vessel_name"].apply(as09.is_korean_name).astype(int)
    sgkf_ab = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    inner_folds_ab = np.zeros(len(t_df), dtype=int)
    for fold_i, (_, test_idx) in enumerate(sgkf_ab.split(t_df, kr_mask_t, t_df["target_vessel_id"])):
        inner_folds_ab[test_idx] = fold_i
    t_df["inner_fold"] = inner_folds_ab

    # B: inner_fold == 0, A: inner_fold != 0
    b_df = t_df[t_df["inner_fold"] == 0].reset_index(drop=True)
    a_df = t_df[t_df["inner_fold"] != 0].reset_index(drop=True)
    print(f"  - Inner A (Ranker train): {len(a_df)} rows ({a_df['vessel_name'].apply(as09.is_korean_name).sum()} Korean)")
    print(f"  - Inner B (Ranker selection): {len(b_df)} rows ({b_df['vessel_name'].apply(as09.is_korean_name).sum()} Korean)")

    # Save split manifests
    splits_dir = Path(args.out_dir).resolve() / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    inner_split_csv = splits_dir / f"outer_{args.outer_fold}_inner_selection.csv"
    t_df[["sample_id", "target_vessel_id", "inner_fold"]].to_csv(inner_split_csv, index=False, encoding="utf-8")

    # 1. Cross-fit A into 3 folds to create upstream OOF traces for A
    print("\n=== [3] Tracing Inner A queries via 3-fold cross-fit (random_state=20261029) ===")
    kr_mask_a = a_df["vessel_name"].apply(as09.is_korean_name).astype(int)
    sgkf_a3 = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261029)
    a_inner_folds = np.zeros(len(a_df), dtype=int)
    for fold_j, (_, test_idx) in enumerate(sgkf_a3.split(a_df, kr_mask_a, a_df["target_vessel_id"])):
        a_inner_folds[test_idx] = fold_j
    a_df["crossfit_fold"] = a_inner_folds

    traces_a: List[Any] = []
    golds_a: List[str] = []
    t0_a = time.perf_counter()

    for j in range(3):
        a_train_j = a_df[a_df["crossfit_fold"] != j].reset_index(drop=True)
        a_val_j = a_df[a_df["crossfit_fold"] == j].reset_index(drop=True)
        print(f"  - A Fold {j}: fit={len(a_train_j)}, val={len(a_val_j)}")

        model_j = fit_split_model(a_train_j, catalog_df, base_model=base_model)
        adapter_j = AS09RerankAdapter(data_dir=data_dir, device=args.device, custom_model=model_j)

        tr_j = adapter_j.trace_df(a_val_j)
        traces_a.extend(tr_j)
        golds_a.extend(a_val_j["target_vessel_id"].tolist())

    sec_a = time.perf_counter() - t0_a
    lqs_a = [label_trace(t, g) for t, g in zip(traces_a, golds_a)]
    print(f"  -> Traced A in {sec_a:.2f}s ({len(lqs_a)} rows)")

    cache_a_dir = out_dir / "cache" / "train"
    manifest_a = CandidateCacheWriter.write_cache(
        cache_a_dir,
        lqs_a,
        asset_manifest_sha256=adapter_j.asset_manifest_sha256,
        feature_schema_sha256=adapter_j.feature_schema_sha256,
        filter_trainable_only=True,
    )
    print(f"  -> Saved trainable A cache: {manifest_a['retained_queries']} queries, {manifest_a['total_candidates']} candidate pairs")

    # 2. Trace B using model fit strictly on A
    print("\n=== [4] Tracing Inner B queries using fit_assets(A) ===")
    t0_b = time.perf_counter()
    model_a_full = fit_split_model(a_df, catalog_df, base_model=base_model)
    adapter_b = AS09RerankAdapter(data_dir=data_dir, device=args.device, custom_model=model_a_full)

    traces_b = adapter_b.trace_df(b_df)
    golds_b = b_df["target_vessel_id"].tolist()
    lqs_b = [label_trace(t, g) for t, g in zip(traces_b, golds_b)]
    sec_b = time.perf_counter() - t0_b
    print(f"  -> Traced B in {sec_b:.2f}s ({len(lqs_b)} rows)")

    cache_b_dir = out_dir / "cache" / "selection"
    manifest_b = CandidateCacheWriter.write_cache(
        cache_b_dir,
        lqs_b,
        asset_manifest_sha256=adapter_b.asset_manifest_sha256,
        feature_schema_sha256=adapter_b.feature_schema_sha256,
        filter_trainable_only=False,  # Keep all rows for full accuracy evaluation on B
    )
    print(f"  -> Saved selection B cache: {manifest_b['retained_queries']} queries ({manifest_b['total_candidates']} candidate pairs)")

    # 3. Cross-fit T into 3 folds to create full training cache
    print("\n=== [5] Tracing Full T queries via 3-fold cross-fit (random_state=20261029) ===")
    kr_mask_t3 = t_df["vessel_name"].apply(as09.is_korean_name).astype(int)
    sgkf_t3 = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261029)
    t_inner_folds = np.zeros(len(t_df), dtype=int)
    for fold_k, (_, test_idx) in enumerate(sgkf_t3.split(t_df, kr_mask_t3, t_df["target_vessel_id"])):
        t_inner_folds[test_idx] = fold_k
    t_df["crossfit_fold"] = t_inner_folds

    traces_t: List[Any] = []
    golds_t: List[str] = []
    t0_t = time.perf_counter()

    for k in range(3):
        t_train_k = t_df[t_df["crossfit_fold"] != k].reset_index(drop=True)
        t_val_k = t_df[t_df["crossfit_fold"] == k].reset_index(drop=True)
        print(f"  - T Fold {k}: fit={len(t_train_k)}, val={len(t_val_k)}")

        model_k = fit_split_model(t_train_k, catalog_df, base_model=base_model)
        adapter_k = AS09RerankAdapter(data_dir=data_dir, device=args.device, custom_model=model_k)

        tr_k = adapter_k.trace_df(t_val_k)
        traces_t.extend(tr_k)
        golds_t.extend(t_val_k["target_vessel_id"].tolist())

    sec_t = time.perf_counter() - t0_t
    lqs_t = [label_trace(t, g) for t, g in zip(traces_t, golds_t)]
    print(f"  -> Traced T in {sec_t:.2f}s ({len(lqs_t)} rows)")

    cache_t_dir = out_dir / "cache" / "train_full"
    manifest_t = CandidateCacheWriter.write_cache(
        cache_t_dir,
        lqs_t,
        asset_manifest_sha256=adapter_k.asset_manifest_sha256,
        feature_schema_sha256=adapter_k.feature_schema_sha256,
        filter_trainable_only=True,
    )
    print(f"  -> Saved full T cache: {manifest_t['retained_queries']} queries, {manifest_t['total_candidates']} candidate pairs")

    # 4. Trace V (Fold 0 heldout) using fit_assets(T)
    print(f"\n=== [6] Tracing Heldout V queries (Fold {args.outer_fold}) using fit_assets(T) ===")
    t0_v = time.perf_counter()
    model_t_full = fit_split_model(t_df, catalog_df, base_model=base_model)
    adapter_v = AS09RerankAdapter(data_dir=data_dir, device=args.device, custom_model=model_t_full)

    traces_v = adapter_v.trace_df(v_df)
    golds_v = v_df["target_vessel_id"].tolist()
    lqs_v = [label_trace(t, g) for t, g in zip(traces_v, golds_v)]
    sec_v = time.perf_counter() - t0_v
    print(f"  -> Traced V in {sec_v:.2f}s ({len(lqs_v)} rows)")

    cache_v_dir = out_dir / "cache" / "validation"
    manifest_v = CandidateCacheWriter.write_cache(
        cache_v_dir,
        lqs_v,
        asset_manifest_sha256=adapter_v.asset_manifest_sha256,
        feature_schema_sha256=adapter_v.feature_schema_sha256,
        filter_trainable_only=False,  # Keep all rows for outer evaluation
    )
    print(f"  -> Saved validation V cache: {manifest_v['retained_queries']} queries ({manifest_v['total_candidates']} candidate pairs)")

    print(f"\nAll preparation finished successfully for outer fold {args.outer_fold}!")


if __name__ == "__main__":
    main()
