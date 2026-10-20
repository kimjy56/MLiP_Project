"""Local diagnostic only: public inputs, aggregate output, no training or ZIP."""
from pathlib import Path
import argparse
from datetime import datetime, timezone, timedelta
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
import torch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import model.matcher as matching
from model.torch_indexer import TorchNgramIndexer
from model.name_processing import name_view


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


class CachedSearch:
    def __init__(self, queries, results, count):
        self.queries = queries
        self.results = results
        self.count = count
        self.row = 0

    def search(self, query, count):
        assert count == self.count
        assert query == self.queries[self.row]
        return self.results[self.row]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-language", type=int, default=128)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/gpu_batch_comparison.json")
    args = parser.parse_args()
    report = {"status": "running", "started_at_kst": now(), "seed": 42,
              "environment": {"python": platform.python_version(),
                              "platform": platform.platform(),
                              "executable": sys.executable,
                              "packages": {p: importlib.metadata.version(p) for p in
                                           ["torch", "numpy", "pandas", "scikit-learn", "lightgbm"]}},
              "git_revision": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "sha256": {}, "retrieval": {}, "predictions": {},
              "scope": "Local saved-model execution equivalence, not accuracy or server validation"}
    start = time.perf_counter()

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        assert torch.cuda.is_available(), "CUDA is required; no CPU fallback"
        torch.set_num_threads(6)
        torch.manual_seed(42)
        report["environment"].update(
            gpu=torch.cuda.get_device_name(0), cuda_build=torch.version.cuda,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        for rel in ["data/train.csv", "data/vessel_catalog.csv", "model/assets.json",
                    "model/ranker.txt", "model/matcher.py", "model/torch_indexer.py",
                    "model/name_processing.py", "environment/compare_gpu_batch.py"]:
            report["sha256"][rel] = sha(ROOT / rel)
        train = pd.read_csv(ROOT / "data/train.csv", keep_default_na=False)
        catalog = pd.read_csv(ROOT / "data/vessel_catalog.csv", keep_default_na=False)
        mask = train.vessel_name.map(matching.hangul)
        sampled = pd.concat([train[mask].sample(min(args.rows_per_language, int(mask.sum())), random_state=42),
                             train[~mask].sample(min(args.rows_per_language, int((~mask).sum())), random_state=42)])
        report["sample"] = {"rows": len(sampled),
                            "hangul": int(sampled.vessel_name.map(matching.hangul).sum()),
                            "non_hangul": int((~sampled.vessel_name.map(matching.hangul)).sum()),
                            "sample_id_sha256": hashlib.sha256(
                                "\n".join(sampled.sample_id.astype(str)).encode()).hexdigest(),
                            "catalog_rows": len(catalog), "split": None,
                            "seen_unseen": "not measured; saved full-training assets; not holdout accuracy"}
        queries = sampled.loc[:, matching.QUERY_COLUMNS].reset_index(drop=True)
        # Build metadata/features once, then explicitly inject real GPU search results.
        # This avoids keeping four dense catalog tensors on the GPU simultaneously.
        matching.HAS_TORCH = False
        matcher = matching.Matcher(catalog, ROOT / "model", use_train_lookup=False)
        index = matcher.index
        report["model"] = {k: index.assets.get(k) for k in
                           ["name_mode", "phonetic_search", "retrieval_channels"]}
        records = queries.to_dict("records")
        modes = ["single", "batch_2", "batch_16", "batch_64", "batch_256",
                 "reverse_64", "repeat_64", "subset_64"]
        stored = {m: {} for m in modes}
        definitions = [
            ("torch_names", index.names, matching.norm, 24),
            ("torch_romans", index.romans, matching.roman, 60),
            ("torch_phones", index.phones, matching.phonetic, 60),
        ]
        if index.phonetic_search:
            definitions.append(("torch_keys", index.name_index.search_keys,
                                lambda x: name_view(x).search_key, index.phonetic_search))
        for attr, strings, transform, count in definitions:
            print(f"Checking {attr}", flush=True)
            begun = time.perf_counter()
            gpu = TorchNgramIndexer(strings, device="cuda")
            torch.cuda.synchronize()
            q = [transform(x) for x in queries.vessel_name]
            stats = {"index_seconds": time.perf_counter() - begun,
                     "tensor_shape": list(gpu.catalog_tensor.shape), "dtype": str(gpu.dtype),
                     "count": count, "comparisons": {}}
            begun = time.perf_counter()
            single = [gpu.search(x, count) for x in q]
            torch.cuda.synchronize()
            stats["single_search_seconds"] = time.perf_counter() - begun
            stored["single"][attr] = CachedSearch(q, single, count)
            for mode in modes[1:]:
                batch_size = int(mode.rsplit("_", 1)[1])
                order = list(range(len(q)))
                if mode.startswith("reverse"):
                    order.reverse()
                if mode.startswith("subset"):
                    order = order[::2]
                result = {}
                begun = time.perf_counter()
                for offset in range(0, len(order), batch_size):
                    selected = order[offset:offset + batch_size]
                    values = gpu.batch_search([q[i] for i in selected], count)
                    result.update(zip(selected, values))
                torch.cuda.synchronize()
                stats["comparisons"][mode] = {
                    "rows": len(order),
                    "rank_mismatch_rows": sum(single[i] != result[i] for i in order),
                    "set_mismatch_rows": sum(set(single[i]) != set(result[i]) for i in order),
                    "search_seconds": time.perf_counter() - begun}
                # Subset-only result uses single baseline for untested rows.
                stored[mode][attr] = CachedSearch(q, [result.get(i, single[i]) for i in range(len(q))], count)
            # Isolate numerical differences from top-k tie behavior.
            qt = torch.from_numpy(gpu.vectorizer.transform(q[:32]).toarray()).to("cuda", gpu.dtype)
            batched = qt @ gpu.catalog_tensor.T
            score_rows = 0
            max_abs = 0.0
            for i in range(len(qt)):
                separate = (qt[i:i+1] @ gpu.catalog_tensor.T)[0]
                score_rows += int(not torch.equal(separate, batched[i]))
                max_abs = max(max_abs, float((separate - batched[i]).abs().max().item()))
            stats["score_probe"] = {"rows": len(qt), "non_bitwise_equal_rows": score_rows,
                                    "max_absolute_difference": max_abs}
            report["retrieval"][attr] = stats
            save()
            del gpu, qt, batched, separate
            gc.collect()
            torch.cuda.empty_cache()

        # Use identical production feature generation and ranker on each candidate union.
        baseline_candidates, baseline_pred = None, None
        for mode in modes:
            for attr, cached in stored[mode].items():
                setattr(index, attr, cached)
            candidates = []
            for i, row in enumerate(records):
                for cached in stored[mode].values():
                    cached.row = i
                candidates.append(index.candidates(row))
            predictions = []
            for row, ids in zip(records, candidates):
                scores = matcher.model.predict(index.features(row, ids), num_threads=1)
                predictions.append(index.ids[ids[int(np.argmax(scores))]])
            if mode == "single":
                baseline_candidates, baseline_pred = candidates, predictions
            selected = list(range(0, len(records), 2)) if mode.startswith("subset") else list(range(len(records)))
            report["predictions"][mode] = {
                "rows": len(selected),
                "candidate_union_mismatch_rows": sum(candidates[i] != baseline_candidates[i] for i in selected),
                "prediction_mismatch_rows": sum(predictions[i] != baseline_pred[i] for i in selected),
                "hangul_prediction_mismatch_rows": sum(
                    predictions[i] != baseline_pred[i] for i in selected if matching.hangul(records[i]["vessel_name"])),
                "non_hangul_prediction_mismatch_rows": sum(
                    predictions[i] != baseline_pred[i] for i in selected if not matching.hangul(records[i]["vessel_name"]))}
            print(mode, report["predictions"][mode], flush=True)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["finished_at_kst"] = now()
        report["total_seconds"] = time.perf_counter() - start
        report["timing_scope"] = "Data/model loading, index construction, repeated retrieval and feature/ranker comparison; not production latency"
        save()


if __name__ == "__main__":
    with threadpool_limits(limits=6):
        main()
