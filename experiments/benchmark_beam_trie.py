"""Benchmark Beam 4 + Catalog BPE Token Trie Constrained Decoding on AS-HCX-03.

Enforces that generated tokens strictly follow valid paths in the 78,592 catalog names,
preventing off-catalog tokens and allowing termination (EOS) only on complete registered names.

Evaluates on the exact 128 heldout pilot rows (Fold 0 seed 42):
1. Beam 4 + Trie (Top-1 sequence return)
2. Beam 4 + Trie (4 sequences returned -> multi-sequence candidate aggregation)
3. Direct side-by-side comparison against unconstrained Beam 4 (BENCH-HCX-BEAM-01)
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


class BpeNameTrie:
    """Prefix tree on BPE token sequences for 78,592 unique catalog names."""
    def __init__(self, names, tokenizer):
        self.root = {}
        self.eos_id = tokenizer.eos_token_id
        t0 = time.perf_counter()
        count = 0
        for name in names:
            tokens = tokenizer.encode(name, add_special_tokens=False)
            curr = self.root
            for t in tokens:
                curr = curr.setdefault(t, {})
            curr["__is_leaf__"] = True
            count += 1
        self.build_time = time.perf_counter() - t0
        self.count = count

    def get_allowed_tokens(self, generated_tokens):
        # If EOS was already emitted, only allow EOS to pad/finish
        if self.eos_id in generated_tokens:
            return [self.eos_id]

        curr = self.root
        for t in generated_tokens:
            if t not in curr:
                return [self.eos_id]
            curr = curr[t]

        allowed = [k for k in curr.keys() if k != "__is_leaf__"]
        if curr.get("__is_leaf__", False):
            allowed.append(self.eos_id)
        if not allowed:
            allowed = [self.eos_id]
        return allowed


def extract_top_aggregated(beam_texts, catalog_names, scorer_fn, limit=10):
    """Aggregate candidate scores across multiple beam completions."""
    candidate_scores = defaultdict(float)

    for rank_idx, text in enumerate(beam_texts):
        rank_weight = 1.0 / (1.0 + 0.1 * rank_idx)
        matches = process.extract(text, catalog_names, scorer=scorer_fn, limit=limit)
        for name, score, _ in matches:
            weighted_score = score * rank_weight
            if weighted_score > candidate_scores[name]:
                candidate_scores[name] = weighted_score

    sorted_candidates = sorted(candidate_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [c[0] for c in sorted_candidates[:limit]]


def compute_metrics(df_res, catalog_set):
    metrics = {}
    for lang, mask in [("hangul", df_res["hangul"] == True), ("non_hangul", df_res["hangul"] == False)]:
        sub = df_res[mask]
        n_rows = len(sub)
        t1_hits = int(sub["correct_top1"].sum())
        t5_hits = int(sub["correct_top5"].sum())
        t10_hits = int(sub["correct_top10"].sum())
        direct_hits = int(sub["direct_exact_match"].sum())
        in_catalog_hits = int(sub["in_catalog"].sum())
        metrics[lang] = {
            "rows": n_rows,
            "in_catalog_count": in_catalog_hits,
            "in_catalog_ratio": round(in_catalog_hits / n_rows, 5) if n_rows else 0.0,
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
    parser = argparse.ArgumentParser(description="Benchmark Beam 4 + Trie for HyperCLOVA X")
    parser.add_argument("--model-path", default="artifacts/AS-HCX-03/model")
    parser.add_argument("--baseline-beam-json", default="reports/benchmark_beam_search.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--folds", default="eval_kit/folds_group5_seed42.csv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--eval-per-language", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--catalog-scorer", default="ratio", choices=["ratio", "WRatio"])
    parser.add_argument("--output-json", default="reports/benchmark_beam_trie.json")
    args = parser.parse_args()

    started_at = now_kst()
    print(f"=== Starting Beam 4 + Trie Benchmark at {started_at} ===", flush=True)

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
    catalog_set = set(catalog_names)
    scorer_fn = SCORERS[args.catalog_scorer]

    print(f"Loaded {len(pilot_df)} heldout pilot rows", flush=True)

    model_path = ROOT / args.model_path
    print(f"Loading model from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, device_map="cpu")
    model.eval()

    print(f"Building BPE Token Trie for {len(catalog_names)} unique catalog names...", flush=True)
    trie = BpeNameTrie(catalog_names, tokenizer)
    print(f"Trie built in {trie.build_time:.2f}s for {trie.count} names!", flush=True)

    # ----------------------------------------------------
    # Beam 4 + Trie Constrained Generation
    # ----------------------------------------------------
    print(f"\n--- Running Beam 4 + Trie Constrained Search (max_new_tokens={args.max_new_tokens}) ---", flush=True)
    results_trie_top1 = []
    results_trie_agg = []
    t_start = time.perf_counter()

    for idx, (_, r) in enumerate(pilot_df.iterrows()):
        prompt = build_prompt(r.source)
        inp = tokenizer(prompt, return_tensors="pt")
        prompt_len = inp["input_ids"].shape[1]

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            gen = input_ids[prompt_len:].tolist()
            return trie.get_allowed_tokens(gen)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=4,
                num_return_sequences=4,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        t1 = time.perf_counter()
        gen_duration = round(t1 - t0, 3)

        gen_texts = []
        for seq in out:
            raw = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            clean = raw.strip().split("\n")[0].strip().upper()
            gen_texts.append(clean)

        target = r.target

        # Top-1 Beam sequence
        top1_text = gen_texts[0]
        top1_in_cat = (top1_text in catalog_set)
        top1_direct = (top1_text == target)
        # Because top1_text is guaranteed to be in catalog, exact top match is itself if in catalog
        top1_matches = [m[0] for m in process.extract(top1_text, catalog_names, scorer=scorer_fn, limit=10)]

        results_trie_top1.append({
            "sample_id": r.sample_id,
            "hangul": bool(r.hangul),
            "source": r.source,
            "target": target,
            "generated_text": top1_text,
            "in_catalog": bool(top1_in_cat),
            "direct_exact_match": bool(top1_direct),
            "predicted_catalog_name": top1_matches[0] if top1_matches else "",
            "correct_top1": bool(top1_matches[0] == target) if top1_matches else False,
            "correct_top5": bool(target in top1_matches[:5]) if top1_matches else False,
            "correct_top10": bool(target in top1_matches[:10]) if top1_matches else False,
            "seconds": gen_duration,
        })

        # Aggregated 4 Beams
        all_in_cat = all(t in catalog_set for t in gen_texts)
        agg_matches = extract_top_aggregated(gen_texts, catalog_names, scorer_fn, limit=10)
        results_trie_agg.append({
            "sample_id": r.sample_id,
            "hangul": bool(r.hangul),
            "source": r.source,
            "target": target,
            "generated_texts": json.dumps(gen_texts, ensure_ascii=False),
            "in_catalog": bool(all_in_cat),
            "direct_exact_match": bool(target in gen_texts),
            "predicted_catalog_name": agg_matches[0] if agg_matches else "",
            "correct_top1": bool(agg_matches[0] == target) if agg_matches else False,
            "correct_top5": bool(target in agg_matches[:5]) if agg_matches else False,
            "correct_top10": bool(target in agg_matches[:10]) if agg_matches else False,
            "top10_candidates": json.dumps(agg_matches, ensure_ascii=False),
            "seconds": gen_duration,
        })

        if (idx + 1) % 32 == 0 or (idx + 1) == len(pilot_df):
            elapsed = time.perf_counter() - t_start
            print(f"[Beam 4 + Trie] Evaluated {idx + 1}/{len(pilot_df)} rows ({elapsed:.1f}s, {elapsed/(idx+1):.2f}s/row)...", flush=True)

    total_seconds = time.perf_counter() - t_start

    df_top1 = pd.DataFrame(results_trie_top1)
    metrics_top1 = compute_metrics(df_top1, catalog_set)
    metrics_top1["total_seconds"] = round(total_seconds, 2)
    metrics_top1["avg_seconds_per_row"] = round(total_seconds / len(pilot_df), 3)

    df_agg = pd.DataFrame(results_trie_agg)
    metrics_agg = compute_metrics(df_agg, catalog_set)
    metrics_agg["total_seconds"] = round(total_seconds, 2)
    metrics_agg["avg_seconds_per_row"] = round(total_seconds / len(pilot_df), 3)

    # Load unconstrained Beam 4 baseline for comparison
    baseline_beam_path = ROOT / args.baseline_beam_json
    baseline_data = {}
    if baseline_beam_path.exists():
        baseline_data = json.loads(baseline_beam_path.read_text(encoding="utf-8"))

    finished_at = now_kst()
    report = {
        "benchmark_id": "BENCH-HCX-TRIE-01",
        "started_at_kst": started_at,
        "finished_at_kst": finished_at,
        "threads": args.threads,
        "max_new_tokens": args.max_new_tokens,
        "catalog_scorer": args.catalog_scorer,
        "trie_stats": {
            "catalog_names_count": len(catalog_names),
            "trie_build_seconds": round(trie.build_time, 2),
        },
        "comparison": {
            "unconstrained_beam4_top1": baseline_data.get("comparison", {}).get("beam4_top1", {}),
            "unconstrained_beam4_aggregated": baseline_data.get("comparison", {}).get("beam4_return4_aggregated", {}),
            "trie_constrained_beam4_top1": metrics_top1,
            "trie_constrained_beam4_aggregated": metrics_agg,
        }
    }

    print("\n=========================================================================")
    print("=== Beam 4 + Trie Constrained Benchmark Results ===")
    print("=========================================================================")
    print(f"1. Trie Beam 4 (Top-1):         Weighted {metrics_top1['weighted_name_exact_accuracy']*100:.2f}% | Hangul T1 {metrics_top1['hangul']['gold_name_in_top_1']*100:.2f}% (T10 {metrics_top1['hangul']['gold_name_in_top_10']*100:.2f}%) | Non-Hangul T1 {metrics_top1['non_hangul']['gold_name_in_top_1']*100:.2f}% | In-Catalog: {metrics_top1['hangul']['in_catalog_ratio']*100:.1f}%")
    print(f"2. Trie Beam 4 (Aggregated 4):  Weighted {metrics_agg['weighted_name_exact_accuracy']*100:.2f}% | Hangul T1 {metrics_agg['hangul']['gold_name_in_top_1']*100:.2f}% (T10 {metrics_agg['hangul']['gold_name_in_top_10']*100:.2f}%) | Non-Hangul T1 {metrics_agg['non_hangul']['gold_name_in_top_1']*100:.2f}% | In-Catalog: {metrics_agg['hangul']['in_catalog_ratio']*100:.1f}%")
    print(f"Total Time: {total_seconds:.1f}s ({total_seconds/len(pilot_df):.2f}s/row)")
    print("=========================================================================")

    out_file = ROOT / args.output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved report to {out_file}", flush=True)

    csv_trie_top1 = out_file.with_name(out_file.stem + "_top1_rows.csv")
    csv_trie_agg = out_file.with_name(out_file.stem + "_aggregated_rows.csv")
    df_top1.to_csv(csv_trie_top1, index=False, encoding="utf-8-sig")
    df_agg.to_csv(csv_trie_agg, index=False, encoding="utf-8-sig")
    print(f"Saved detailed per-row CSVs:\n - {csv_trie_top1}\n - {csv_trie_agg}", flush=True)


if __name__ == "__main__":
    main()
