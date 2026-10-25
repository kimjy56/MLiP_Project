"""Benchmark use_cache=True vs use_cache=False on AS-HCX-03 heldout pilot (128 rows).

Validates:
1. Exact generation equality (100% token/text match across all 128 heldout rows).
2. Latency and throughput speedup on CPU with 4 threads.
3. Accuracy consistency under CatalogNameSearch.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from name_decoder.catalog import canonical_name
from name_decoder.catalog_search import CatalogNameSearch
from name_decoder.data import read_frames, select_pilot


def now_kst():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def build_prompt(source):
    return (
        f"<|im_start|>system\n"
        f"한국어 선박명을 국제 카탈로그의 공식 영문 대문자 선박명으로 1:1 변환합니다. 오직 영문 공식 선박명만 대문자로 한 줄로 출력하세요.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"입력: {source}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"출력:\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference cache for HyperCLOVA X")
    parser.add_argument("--model-path", default="artifacts/AS-HCX-03/model")
    parser.add_argument("--baseline-csv", default="artifacts/AS-HCX-03/pilot_predictions.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--folds", default="eval_kit/folds_group5_seed42.csv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--eval-per-language", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--catalog-scorer", default="ratio", choices=["ratio", "WRatio"])
    parser.add_argument("--output-json", default="reports/benchmark_use_cache.json")
    args = parser.parse_args()

    started_at = now_kst()
    print(f"=== Starting Inference Cache Benchmark at {started_at} ===", flush=True)

    # Thread configuration
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(min(2, args.threads))
    except RuntimeError:
        pass
    print(f"Configured CPU execution with {args.threads} threads (interop capped)", flush=True)

    # Load data and pilot
    data_dir = ROOT / args.data_dir
    folds_path = ROOT / args.folds
    pairs, validation, fit_rows, catalog_index = read_frames(data_dir, folds_path, heldout_fold=args.fold)
    pilot_df = select_pilot(validation, per_language=args.eval_per_language, seed=42)
    catalog_names = sorted(catalog_index.ids_by_name.keys())
    catalog_search = CatalogNameSearch(catalog_names, scorer=args.catalog_scorer)

    print(f"Loaded {len(pilot_df)} heldout pilot rows (Hangul: {pilot_df.hangul.sum()}, Non-Hangul: {(~pilot_df.hangul).sum()})", flush=True)

    # Load baseline (use_cache=False) predictions
    baseline_path = ROOT / args.baseline_csv
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        print(f"Loaded baseline predictions from {baseline_path} (evaluated with use_cache=False)", flush=True)
    else:
        baseline_df = None
        print(f"Warning: Baseline CSV {baseline_path} not found.", flush=True)

    # Load model
    model_path = ROOT / args.model_path
    print(f"Loading model and tokenizer from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, device_map="cpu")
    model.eval()

    # Run inference with use_cache=True
    print(f"Starting pilot inference with use_cache=True on {len(pilot_df)} samples...", flush=True)
    results = []
    t_start = time.perf_counter()

    for idx, (_, r) in enumerate(pilot_df.iterrows()):
        prompt = build_prompt(r.source)
        inp = tokenizer(prompt, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        t1 = time.perf_counter()
        gen_raw = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        gen_clean = gen_raw.strip().split("\n")[0].strip().upper()

        target = r.target
        direct_exact = (gen_clean == target)
        top_matches = catalog_search.top_names(gen_clean)
        is_top1 = (top_matches[0] == target)
        is_top5 = (target in top_matches[:5])
        is_top10 = (target in top_matches[:10])

        results.append({
            "sample_id": r.sample_id,
            "hangul": bool(r.hangul),
            "source": r.source,
            "target": target,
            "generated_text": gen_clean,
            "direct_exact_match": bool(direct_exact),
            "predicted_catalog_name": top_matches[0],
            "correct_top1": bool(is_top1),
            "correct_top5": bool(is_top5),
            "correct_top10": bool(is_top10),
            "top10_candidates": json.dumps(top_matches, ensure_ascii=False),
            "seconds": round(t1 - t0, 3),
        })

        if (idx + 1) % 16 == 0 or (idx + 1) == len(pilot_df):
            elapsed = time.perf_counter() - t_start
            avg_per_sample = elapsed / (idx + 1)
            print(f"Evaluated {idx + 1}/{len(pilot_df)} rows ({elapsed:.1f}s, {avg_per_sample:.2f}s/row)...", flush=True)

    cache_true_total_seconds = time.perf_counter() - t_start
    df_cache_true = pd.DataFrame(results)

    # Compare row-by-row against baseline (use_cache=False)
    equality_summary = {}
    if baseline_df is not None:
        text_matches = (df_cache_true["generated_text"] == baseline_df["generated_text"]).sum()
        total_rows = len(df_cache_true)
        exact_text_match_ratio = text_matches / total_rows
        equality_summary = {
            "total_rows": total_rows,
            "exact_text_matches": int(text_matches),
            "exact_text_match_ratio": float(exact_text_match_ratio),
            "all_identical": bool(text_matches == total_rows),
        }
        print(f"\n--- Output Equality Check (use_cache=True vs use_cache=False) ---")
        print(f"Identical generations: {text_matches}/{total_rows} ({exact_text_match_ratio*100:.2f}%)")
        if text_matches != total_rows:
            diff_mask = df_cache_true["generated_text"] != baseline_df["generated_text"]
            for _, diff_row in df_cache_true[diff_mask].iterrows():
                b_row = baseline_df.loc[baseline_df["sample_id"] == diff_row["sample_id"]].iloc[0]
                print(f"  Diff on [{diff_row['source']}]: False='{b_row['generated_text']}' vs True='{diff_row['generated_text']}'")

    # Metrics computation
    metrics = {}
    for lang, mask in [("hangul", df_cache_true["hangul"] == True), ("non_hangul", df_cache_true["hangul"] == False)]:
        sub = df_cache_true[mask]
        n_rows = len(sub)
        t1_hits = int(sub["correct_top1"].sum())
        t5_hits = int(sub["correct_top5"].sum())
        t10_hits = int(sub["correct_top10"].sum())
        direct_hits = int(sub["direct_exact_match"].sum())
        metrics[lang] = {
            "rows": n_rows,
            "direct_exact_hits": direct_hits,
            "direct_exact_accuracy": round(direct_hits / n_rows, 5) if n_rows else 0.0,
            "gold_name_in_top_1": round(t1_hits / n_rows, 5) if n_rows else 0.0,
            "gold_name_hits_top_1": t1_hits,
            "gold_name_in_top_5": round(t5_hits / n_rows, 5) if n_rows else 0.0,
            "gold_name_hits_top_5": t5_hits,
            "gold_name_in_top_10": round(t10_hits / n_rows, 5) if n_rows else 0.0,
            "gold_name_hits_top_10": t10_hits,
        }

    h_acc = metrics["hangul"]["gold_name_in_top_1"]
    nh_acc = metrics["non_hangul"]["gold_name_in_top_1"]
    weighted_acc = round(0.7 * h_acc + 0.3 * nh_acc, 5)
    metrics["weighted_name_exact_accuracy"] = weighted_acc

    # Latency comparison
    baseline_inference_seconds = 904.01  # AS-HCX-03 recorded inference time on 128 rows
    speedup = baseline_inference_seconds / cache_true_total_seconds if cache_true_total_seconds else 0.0

    print(f"\n=== Latency Benchmark ===")
    print(f"use_cache=False (AS-HCX-03 recorded): {baseline_inference_seconds:.1f}s ({baseline_inference_seconds/len(pilot_df):.2f}s/row)")
    print(f"use_cache=True  (Current benchmark):  {cache_true_total_seconds:.1f}s ({cache_true_total_seconds/len(pilot_df):.2f}s/row)")
    print(f"Speedup: {speedup:.2f}x faster ({cache_true_total_seconds - baseline_inference_seconds:.1f}s reduction)")

    print(f"\n=== Performance Metrics (Scorer: {args.catalog_scorer}) ===")
    print(f"Hangul Direct Exact: {metrics['hangul']['direct_exact_accuracy']*100:.2f}% ({metrics['hangul']['direct_exact_hits']}/{metrics['hangul']['rows']})")
    print(f"Hangul Top-1: {metrics['hangul']['gold_name_in_top_1']*100:.2f}% ({metrics['hangul']['gold_name_hits_top_1']}/{metrics['hangul']['rows']})")
    print(f"Hangul Top-10: {metrics['hangul']['gold_name_in_top_10']*100:.2f}% ({metrics['hangul']['gold_name_hits_top_10']}/{metrics['hangul']['rows']})")
    print(f"Non-Hangul Direct Exact: {metrics['non_hangul']['direct_exact_accuracy']*100:.2f}% ({metrics['non_hangul']['direct_exact_hits']}/{metrics['non_hangul']['rows']})")
    print(f"Non-Hangul Top-1: {metrics['non_hangul']['gold_name_in_top_1']*100:.2f}% ({metrics['non_hangul']['gold_name_hits_top_1']}/{metrics['non_hangul']['rows']})")
    print(f"Non-Hangul Top-10: {metrics['non_hangul']['gold_name_in_top_10']*100:.2f}% ({metrics['non_hangul']['gold_name_hits_top_10']}/{metrics['non_hangul']['rows']})")
    print(f"Weighted Name Accuracy: {metrics['weighted_name_exact_accuracy']*100:.2f}%")

    finished_at = now_kst()
    report = {
        "benchmark_id": "BENCH-HCX-CACHE-01",
        "started_at_kst": started_at,
        "finished_at_kst": finished_at,
        "threads": args.threads,
        "catalog_scorer": args.catalog_scorer,
        "use_cache_false_seconds": baseline_inference_seconds,
        "use_cache_true_seconds": round(cache_true_total_seconds, 2),
        "speedup_factor": round(speedup, 2),
        "equality_summary": equality_summary,
        "metrics": metrics,
    }

    output_path = ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark report to {output_path}", flush=True)


if __name__ == "__main__":
    main()
