"""Executed from an isolated public benchmark directory, never in submission."""
import time
start = time.perf_counter()
import importlib.util
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
package = root / "package"
spec = importlib.util.spec_from_file_location("as05_timed_package", package / "script.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sys.path.insert(0, str(package / "model"))
from as05_reranker import NoisyChannelReranker
profile = {"model_load_seconds": 0.0, "pair_seconds": [], "pair_token_counts": [],
           "eligible_per_row_pairs": [], "predict_seconds": None}
load = NoisyChannelReranker._load
select = NoisyChannelReranker.select
predict = module.predict_rows


def timed_load(self):
    if self.scorer is not None:
        return load(self)
    t = time.perf_counter()
    value = load(self)
    profile["model_load_seconds"] += time.perf_counter() - t
    score = self.scorer.score_pair
    def timed_score(*args, **kwargs):
        st = time.perf_counter()
        result = score(*args, **kwargs)
        profile["pair_seconds"].append(time.perf_counter() - st)
        profile["pair_token_counts"].append(result.target_token_count)
        return result
    self.scorer.score_pair = timed_score
    return value


def timed_select(self, *args, **kwargs):
    eligible = self.eligible_rows
    pairs = self.scored_pairs
    result = select(self, *args, **kwargs)
    if self.eligible_rows > eligible:
        profile["eligible_per_row_pairs"].append(self.scored_pairs - pairs)
    return result


def timed_predict(*args, **kwargs):
    t = time.perf_counter()
    result = predict(*args, **kwargs)
    profile["predict_seconds"] = time.perf_counter() - t
    return result


NoisyChannelReranker._load = timed_load
NoisyChannelReranker.select = timed_select
module.predict_rows = timed_predict
module.main()
profile["total_seconds"] = time.perf_counter() - start
profile["fixed_seconds"] = profile["total_seconds"] - profile["predict_seconds"] + profile["model_load_seconds"]
profile["baseline_predict_seconds"] = profile["predict_seconds"] - profile["model_load_seconds"] - sum(profile["pair_seconds"])
profile["note"] = "Exact package functions with timing wrappers; no score changes. CPU 4 threads."
(root / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
print(json.dumps(profile), flush=True)
