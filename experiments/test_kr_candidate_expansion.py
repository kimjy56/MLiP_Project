"""Korean-only candidate expansion test (Severe noise, heldout Fold 0).

All 239 Korean rows are scored (not only misses) so that both fixes and breaks
are counted. Results reported for all Korean rows and noisy-only Korean rows.
LLM scores are cached on disk (resumable after interruption).
Hardware safety: 2 threads + per-call cooldown.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "AR-06"))

import pandas as pd
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

from script import CatalogIndex, fit_matching_model, predict_rows, is_korean_name
from eval_kit.robustness import RobustnessPerturber, PRESETS
from name_decoder.catalog import canonical_name
from name_decoder.noisy_channel import NoisyChannelScorer, normalize_observed, ChannelScore
from name_decoder.factored_noisy_channel import FactoredNoisyChannelMatcher

MAX_K = 25
K_LIST = [8, 15, 25]
LAMBDAS = [0.20, 0.35, 0.50]
CACHE_PATH = ROOT / "scratch/kr_expansion_llm_cache.jsonl"
OUT_PATH = ROOT / "reports/robustness/kr_candidate_expansion_severe_fold0.json"


class CachedScorer:
    """Exact cache: score_pair applies canonical_name/normalize_observed internally."""

    def __init__(self, scorer, path):
        self.scorer = scorer
        self.path = path
        self.cache = {}
        self.calls = 0
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[(r["c"], r["o"])] = ChannelScore(r["lp"], r["mlp"], r["n"])
        self.fh = path.open("a", encoding="utf-8")

    def score_pair(self, catalog_name, observed_name):
        key = (canonical_name(catalog_name), normalize_observed(observed_name))
        if key not in self.cache:
            s = self.scorer.score_pair(catalog_name, observed_name)
            self.cache[key] = s
            self.fh.write(json.dumps({"c": key[0], "o": key[1], "lp": s.log_probability,
                                      "mlp": s.mean_log_probability, "n": s.target_token_count},
                                     ensure_ascii=False) + "\n")
            self.fh.flush()
            self.calls += 1
            time.sleep(0.01)
        return self.cache[key]


def acc(flags):
    return sum(flags) / len(flags) if flags else float("nan")


def main():
    t_start = time.perf_counter()
    data_dir = ROOT / "data"
    train_df = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
    catalog_df = pd.read_csv(data_dir / "vessel_catalog.csv", encoding="utf-8-sig")
    name_by_id = dict(zip(catalog_df["vessel_id"], catalog_df["vessel_name"]))
    folds_df = pd.read_csv(ROOT / "eval_kit/folds_group5_seed42.csv",
                           dtype={"sample_id": str, "fold": int}, encoding="utf-8-sig")
    merged = train_df.merge(folds_df[["sample_id", "fold"]], on="sample_id")
    fit_df = merged[merged["fold"] != 0].reset_index(drop=True)
    val_df = merged[merged["fold"] == 0].reset_index(drop=True)
    gold_ids = val_df["target_vessel_id"].tolist()

    cat_idx = CatalogIndex(catalog_df)
    model = fit_matching_model(fit_df, catalog_df, kr_mode="AR06", en_mode="EN_V3", use_train_lookup=False)

    perturber = RobustnessPerturber(deepcopy(PRESETS["severe"]), seed=20261022)
    noisy_df, _, audits = perturber.perturb_dataset(val_df, catalog_df, return_row_audits=True)
    audit_df = pd.DataFrame(audits)
    noisy_mask = (audit_df["name_typo_applied"] | audit_df["hangul_spacing_applied"]
                  | audit_df["ton_shift_applied"] | audit_df["ton_missing_applied"]
                  | audit_df["flag_missing_applied"]).tolist()

    preds, gold_ranks, cands = predict_rows(noisy_df, model, cat_idx, use_train_lookup=False,
                                            gold_ids=gold_ids, top_candidates=MAX_K)

    kr_idx = [i for i in range(len(val_df)) if is_korean_name(val_df.iloc[i]["vessel_name"])]
    print(f"Korean rows: {len(kr_idx)} (noisy: {sum(noisy_mask[i] for i in kr_idx)})", flush=True)

    scorer = CachedScorer(NoisyChannelScorer.load(ROOT / "artifacts/AS-HCX-NC-02/model"), CACHE_PATH)
    print(f"Cache loaded: {len(scorer.cache)} entries", flush=True)
    factored = FactoredNoisyChannelMatcher(scorer, catalog_df, model_dict=model)

    # Per-row correctness for each method
    results = {"baseline": {}}
    for k in K_LIST:
        for lmb in LAMBDAS:
            results[f"fused_k{k}_l{lmb:.2f}"] = {}
        results[f"factored_k{k}"] = {}
    gold_in = {k: {} for k in K_LIST}

    t_score = time.perf_counter()
    for n, i in enumerate(kr_idx, start=1):
        obs = str(noisy_df.iloc[i]["vessel_name"])
        obs_row = noisy_df.iloc[i]
        g = gold_ids[i]
        row_cands = cands[i]
        results["baseline"][i] = preds[i] == g
        for k in K_LIST:
            ck = row_cands[:k]
            vids = [v for _, v in ck]
            gold_in[k][i] = g in vids
            for lmb in LAMBDAS:
                fused = sorted(((sc + lmb * scorer.score_pair(name_by_id[v], obs).mean_log_probability, v)
                                for sc, v in ck), key=lambda x: (-x[0], x[1]))
                results[f"fused_k{k}_l{lmb:.2f}"][i] = fused[0][1] == g
            ranked = factored.rank_candidates(obs_row, vids)
            results[f"factored_k{k}"][i] = bool(ranked) and ranked[0][1] == g
        if n % 10 == 0 or n == len(kr_idx):
            el = time.perf_counter() - t_score
            print(f"  {n}/{len(kr_idx)} rows | new LLM calls {scorer.calls} | {el:.0f}s", flush=True)

    summary = {}
    for subset_name, rows in [("all_korean", kr_idx), ("noisy_korean", [i for i in kr_idx if noisy_mask[i]])]:
        base = results["baseline"]
        sub = {"n_rows": len(rows),
               "baseline_correct": sum(base[i] for i in rows),
               "baseline_accuracy": acc([base[i] for i in rows]),
               "gold_in_topk": {k: sum(gold_in[k][i] for i in rows) for k in K_LIST},
               "methods": {}}
        for m, r in results.items():
            if m == "baseline":
                continue
            fixes = [val_df.iloc[i]["sample_id"] for i in rows if r[i] and not base[i]]
            breaks = [val_df.iloc[i]["sample_id"] for i in rows if base[i] and not r[i]]
            sub["methods"][m] = {"correct": sum(r[i] for i in rows),
                                 "accuracy": acc([r[i] for i in rows]),
                                 "fixes": fixes, "breaks": breaks}
        summary[subset_name] = sub

    out = {"benchmark_id": "KR-CAND-EXPANSION-SEVERE-FOLD0-01",
           "timestamp_kst": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
           "preset": "severe", "seed": 20261022, "heldout_fold": 0,
           "model": "AS-HCX-NC-02", "fused_rule": "baseline_score + lambda * mean_logp (ungated, top-K)",
           "factored_weights": "FactoredNoisyChannelMatcher defaults",
           "scope_note": "Korean rows only; English rows unchanged (Top-8 baseline).",
           "total_seconds": round(time.perf_counter() - t_start, 1),
           "summary": summary}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_PATH}", flush=True)
    for sn, sub in summary.items():
        print(f"\n[{sn}] N={sub['n_rows']} baseline={sub['baseline_accuracy']*100:.3f}% gold_in_topk={sub['gold_in_topk']}")
        for m, v in sub["methods"].items():
            print(f"  {m:22s} acc={v['accuracy']*100:.3f}% fix={len(v['fixes'])} break={len(v['breaks'])}")


if __name__ == "__main__":
    main()
