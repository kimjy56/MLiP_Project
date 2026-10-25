"""Benchmark Beam Search (num_beams=4, num_return_sequences=4) vs Greedy on AS-HCX-03.

Evaluates on the exact 128 heldout pilot rows (Fold 0 seed 42):
1. Greedy (beams=1) + ratio
2. Beam 4 (Top-1 sequence) + ratio
3. Beam 4 (4 sequences returned -> multi-sequence candidate aggregation) + ratio
4. Wall-clock latency and throughput comparison on CPU (4 threads)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from rapidfuzz import fuzz, process
from transformers import AutoModelForCausalLM, AutoTokenizer

from name_decoder.catalog import canonical_name
from name_decoder.catalog_search import CatalogNameSearch, SCORERS
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


def extract_top_aggregated(beam_texts, catalog_names, scorer_fn, limit=10):
    """Aggregate candidate scores across multiple beam completions."""
    candidate_scores = defaultdict(float)
    
    for rank_idx, text in enumerate(beam_texts):
        # Discount factor for lower-ranked beam sequences
        rank_weight = 1.0 / (1.0 + 0.1 * rank_idx)
        matches = process.extract(text, catalog_names, scorer=scorer_fn, limit=limit)
        for name, score, _ in matches:
            weighted_score = score * rank_weight
            if weighted_score > candidate_scores[name]:
                candidate_scores[name] = weighted_score

    # Sort descending by score, tie-breaking lexicographically
    sorted_candidates = sorted(candidate_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [c[0] for c in sorted_candidates[:limit]]


def compute_metrics(df_res):
    metrics = {}
    for lang, mask in [("hangul", df_res["hangul"] == True), ("non_hangul", df_res["hangul"] == False)]:
        sub = df_res[mask]
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
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark Beam Search vs Greedy for HyperCLOVA X")
    parser.add_argument("--model-path", default="artifacts/AS-HCX-03/model")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--folds", default="eval_kit/folds_group5_seed42.csv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--eval-per-language", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--catalog-scorer", default="ratio", choices=["ratio", "WRatio"])
    parser.add_argument("--output-json", default="reports/benchmark_beam_search.json")
    args = parser.parse_args()

    started_at = now_kst()
    print(f"=== Starting Beam Search Benchmark at {started_at} ===", flush=True)

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(min(2, args.threads))
    except RuntimeError:
        pass
    print(f"Configured CPU execution with {args.threads} threads", flush=True)

    data_dir = ROOT / args.data_dir
    folds_path = ROOT / args.folds
    pairs, validation, fit_rows, catalog_index = read_frames(data_dir, folds_path, heldout_fold=args.fold)
    pilot_df = select_pilot(validation, per_language=args.eval_per_language, seed=42)
    catalog_names = sorted(catalog_index.ids_by_name.keys())
    scorer_fn = SCORERS[args.catalog_scorer]

    print(f"Loaded {len(pilot_df)} heldout pilot rows", flush=True)

    model_path = ROOT / args.model_path
    print(f"Loading model from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, device_map="cpu")
    model.eval()

    # ----------------------------------------------------
    # Phase 1: Greedy Search (num_beams=1)
    # ----------------------------------------------------
    print("\n--- Phase 1: Greedy Decoding (beams=1) ---", flush=True)
    results_greedy = []
    t_start_greedy = time.perf_counter()

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

        raw = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        clean = raw.strip().split("\n")[0].strip().upper()
        target = r.target
        direct_exact = (clean == target)

        top_matches = [m[0] for m in process.extract(clean, catalog_names, scorer=scorer_fn, limit=10)]
        is_top1 = (top_matches[0] == target) if top_matches else False
        is_top5 = (target in top_matches[:5]) if top_matches else False
        is_top10 = (target in top_matches[:10]) if top_matches else False

        results_greedy.append({
            "sample_id": r.sample_id,
            "hangul": bool(r.hangul),
            "source": r.source,
            "target": target,
            "generated_text": clean,
            "direct_exact_match": bool(direct_exact),
            "predicted_catalog_name": top_matches[0] if top_matches else "",
            "correct_top1": bool(is_top1),
            "correct_top5": bool(is_top5),
            "correct_top10": bool(is_top10),
            "seconds": round(t1 - t0, 3),
        })

        if (idx + 1) % 32 == 0 or (idx + 1) == len(pilot_df):
            elapsed = time.perf_counter() - t_start_greedy
            print(f"[Greedy] Evaluated {idx + 1}/{len(pilot_df)} rows ({elapsed:.1f}s)...", flush=True)

    greedy_total_seconds = time.perf_counter() - t_start_greedy
    df_greedy = pd.DataFrame(results_greedy)
    metrics_greedy = compute_metrics(df_greedy)
    metrics_greedy["total_seconds"] = round(greedy_total_seconds, 2)
    metrics_greedy["avg_seconds_per_row"] = round(greedy_total_seconds / len(pilot_df), 3)

    # ----------------------------------------------------
    # Phase 2: Beam Search (num_beams=4, returns=4)
    # ----------------------------------------------------
    print("\n--- Phase 2: Beam Search (beams=4, returns=4) ---", flush=True)
    results_beam1 = []
    results_beam4 = []
    t_start_beam = time.perf_counter()

    for idx, (_, r) in enumerate(pilot_df.iterrows()):
        prompt = build_prompt(r.source)
        inp = tokenizer(prompt, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=4,
                num_return_sequences=4,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        t1 = time.perf_counter()
        gen_duration = round(t1 - t0, 3)

        gen_texts = []
        for seq in out:
            raw = tokenizer.decode(seq[inp["input_ids"].shape[1]:], skip_special_tokens=True)
            clean = raw.strip().split("\n")[0].strip().upper()
            gen_texts.append(clean)

        target = r.target

        # Mode 2: Beam 4 Top-1 (first sequence only)
        top1_text = gen_texts[0]
        top1_direct = (top1_text == target)
        top1_matches = [m[0] for m in process.extract(top1_text, catalog_names, scorer=scorer_fn, limit=10)]
        results_beam1.append({
            "sample_id": r.sample_id,
            "hangul": bool(r.hangul),
            "source": r.source,
            "target": target,
            "generated_text": top1_text,
            "direct_exact_match": bool(top1_direct),
            "predicted_catalog_name": top1_matches[0] if top1_matches else "",
            "correct_top1": bool(top1_matches[0] == target) if top1_matches else False,
            "correct_top5": bool(target in top1_matches[:5]) if top1_matches else False,
            "correct_top10": bool(target in top1_matches[:10]) if top1_matches else False,
            "seconds": gen_duration,
        })

        # Mode 3: Beam 4 Multi-return (4 sequences aggregated)
        agg_matches = extract_top_aggregated(gen_texts, catalog_names, scorer_fn, limit=10)
        results_beam4.append({
            "sample_id": r.sample_id,
            "hangul": bool(r.hangul),
            "source": r.source,
            "target": target,
            "generated_texts": json.dumps(gen_texts, ensure_ascii=False),
            "direct_exact_match": bool(target in gen_texts),
            "predicted_catalog_name": agg_matches[0] if agg_matches else "",
            "correct_top1": bool(agg_matches[0] == target) if agg_matches else False,
            "correct_top5": bool(target in agg_matches[:5]) if agg_matches else False,
            "correct_top10": bool(target in agg_matches[:10]) if agg_matches else False,
            "top10_candidates": json.dumps(agg_matches, ensure_ascii=False),
            "seconds": gen_duration,
        })

        if (idx + 1) % 32 == 0 or (idx + 1) == len(pilot_df):
            elapsed = time.perf_counter() - t_start_beam
            print(f"[Beam 4] Evaluated {idx + 1}/{len(pilot_df)} rows ({elapsed:.1f}s, {elapsed/(idx+1):.2f}s/row)...", flush=True)

    beam_total_seconds = time.perf_counter() - t_start_beam

    df_beam1 = pd.DataFrame(results_beam1)
    metrics_beam1 = compute_metrics(df_beam1)
    metrics_beam1["total_seconds"] = round(beam_total_seconds, 2)
    metrics_beam1["avg_seconds_per_row"] = round(beam_total_seconds / len(pilot_df), 3)

    df_beam4 = pd.DataFrame(results_beam4)
    metrics_beam4 = compute_metrics(df_beam4)
    metrics_beam4["total_seconds"] = round(beam_total_seconds, 2)
    metrics_beam4["avg_seconds_per_row"] = round(beam_total_seconds / len(pilot_df), 3)

    finished_at = now_kst()
    report = {
        "benchmark_id": "BENCH-HCX-BEAM-01",
        "started_at_kst": started_at,
        "finished_at_kst": finished_at,
        "threads": args.threads,
        "max_new_tokens": args.max_new_tokens,
        "catalog_scorer": args.catalog_scorer,
        "comparison": {
            "greedy": metrics_greedy,
            "beam4_top1": metrics_beam1,
            "beam4_return4_aggregated": metrics_beam4,
        }
    }

    print("\n=======================================================")
    print("=== Beam Search Benchmark Results (Scorer: ratio) ===")
    print("=======================================================")
    print(f"1. Greedy:                 Weighted {metrics_greedy['weighted_name_exact_accuracy']*100:.2f}% | Hangul T1 {metrics_greedy['hangul']['gold_name_in_top_1']*100:.2f}% (T10 {metrics_greedy['hangul']['gold_name_in_top_10']*100:.2f}%) | Non-Hangul T1 {metrics_greedy['non_hangul']['gold_name_in_top_1']*100:.2f}% | Time: {metrics_greedy['total_seconds']:.1f}s ({metrics_greedy['avg_seconds_per_row']:.2f}s/row)")
    print(f"2. Beam 4 (Top-1):         Weighted {metrics_beam1['weighted_name_exact_accuracy']*100:.2f}% | Hangul T1 {metrics_beam1['hangul']['gold_name_in_top_1']*100:.2f}% (T10 {metrics_beam1['hangul']['gold_name_in_top_10']*100:.2f}%) | Non-Hangul T1 {metrics_beam1['non_hangul']['gold_name_in_top_1']*100:.2f}% | Time: {metrics_beam1['total_seconds']:.1f}s ({metrics_beam1['avg_seconds_per_row']:.2f}s/row)")
    print(f"3. Beam 4 (Aggregated 4):  Weighted {metrics_beam4['weighted_name_exact_accuracy']*100:.2f}% | Hangul T1 {metrics_beam4['hangul']['gold_name_in_top_1']*100:.2f}% (T10 {metrics_beam4['hangul']['gold_name_in_top_10']*100:.2f}%) | Non-Hangul T1 {metrics_beam4['non_hangul']['gold_name_in_top_1']*100:.2f}% | Time: {metrics_beam4['total_seconds']:.1f}s ({metrics_beam4['avg_seconds_per_row']:.2f}s/row)")
    print("=======================================================")

    out_file = ROOT / args.output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved report to {out_file}", flush=True)

    # Save detailed per-row predictions for deep inspection and error analysis
    csv_greedy = out_file.with_name(out_file.stem + "_greedy_rows.csv")
    csv_beam1 = out_file.with_name(out_file.stem + "_beam1_rows.csv")
    csv_beam4 = out_file.with_name(out_file.stem + "_beam4_rows.csv")
    df_greedy.to_csv(csv_greedy, index=False, encoding="utf-8-sig")
    df_beam1.to_csv(csv_beam1, index=False, encoding="utf-8-sig")
    df_beam4.to_csv(csv_beam4, index=False, encoding="utf-8-sig")
    print(f"Saved detailed per-row CSVs:\n - {csv_greedy}\n - {csv_beam1}\n - {csv_beam4}", flush=True)


if __name__ == "__main__":
    main()
