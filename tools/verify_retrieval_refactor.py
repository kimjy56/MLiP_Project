"""Verify this refactor against local working-tree snapshots, without fitting or ZIPs.

Only public train and the distributed fold file are read. Reports contain
aggregates and hashes; per-row inputs, candidates and predictions stay in memory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))  # Preserve legacy top-level model imports.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "model")))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ID = "REFACTOR-RETRIEVAL-03"
SOURCE_FILES = ("model/matcher.py", "experiments/train_bi_encoder.py",
                "experiments/evaluate_bi_encoder_oof.py")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def same_nested(left, right):
    """Require bit-exact deterministic values; elapsed times are checked separately."""
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_nested(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same_nested(a, b)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right, (type(left).__name__, type(right).__name__)


class FrozenEncoder:
    def __init__(self, catalog, queries):
        self.catalog, self.queries = catalog, queries

    def eval(self):
        return self

    def encode_catalog(self, names, batch_size=256):
        return self.catalog

    def encode_queries(self, queries, batch_size=256):
        return self.queries[:len(queries)]


def evaluator_parity(old_train, new_train, old_eval, new_eval):
    rng = np.random.default_rng(20261027)
    catalog = rng.normal(size=(40, 8)).astype(np.float32)
    queries = rng.normal(size=(6, 8)).astype(np.float32)
    # Ties are intentional; retain NumPy's native order rather than adding a tie-break.
    catalog[3:7] = catalog[0]
    encoder = FrozenEncoder(catalog, queries)
    ids = [f"V{i:02}" for i in range(40)]
    targets = [ids[0], ids[3], "MISSING", ids[39], ids[20], ids[0]]
    train_old = old_train.evaluate_retrieval(encoder, ["q"] * 6, targets, ids, catalog)
    train_new = new_train.evaluate_retrieval(encoder, ["q"] * 6, targets, ids, catalog)
    same_nested(train_old, train_new)
    frame = pd.DataFrame({"vessel_name": ["한글", "ALPHA", None, "", "BRAVO", "한글"],
                          "target_vessel_id": targets})
    a = old_eval.evaluate_dense_retrieval(encoder, catalog, ids, frame)
    b = new_eval.evaluate_dense_retrieval(encoder, catalog, ids, frame)
    for key in ("encode_time_sec", "dot_time_sec", "latency_per_row_ms"):
        a.pop(key)
        b.pop(key)
    same_nested(a, b)
    # Keep the distinct denominators and empty-output schemas of the two public functions.
    for module in (old_train, new_train):
        same_nested(module.evaluate_retrieval(encoder, [], [], ids, catalog),
                    {"recall@1": 0., "recall@5": 0., "recall@10": 0., "recall@35": 0., "mrr": 0.})
    return {"exact_metric_fields": True, "tie_fixture": True,
            "missing_target_denominators_preserved": True, "elapsed_times_excluded": True}


def public_sample(data_dir, folds_path):
    train = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig", keep_default_na=False,
                        dtype={"sample_id": str, "target_vessel_id": str, "call_sign": str})
    folds = pd.read_csv(folds_path, dtype={"sample_id": str, "target_vessel_id": str})
    assert train.sample_id.is_unique and folds.sample_id.is_unique
    merged = train.merge(folds, on=["sample_id", "target_vessel_id"], validate="one_to_one")
    assert len(merged) == len(train)
    heldout = merged.loc[merged.fold.eq(0)].sort_values("sample_id")
    ko = heldout.vessel_name.str.contains("[가-힣]", na=False)
    selected = pd.concat([heldout[ko].head(32), heldout[~ko].head(32)]).sort_values("sample_id")
    assert len(selected) == 64
    catalog = pd.read_csv(data_dir / "vessel_catalog.csv", encoding="utf-8-sig", keep_default_na=False,
                          dtype={"vessel_id": str})
    # Gold/fold never reach inference.
    return selected.drop(columns=["target_vessel_id", "fold"]), catalog


def step_state(module, index, scorer, queries):
    state = []
    for raw in queries.to_dict("records"):
        row = module.prepare_row(raw, index.korean_expansion)
        direct = index.direct(row)
        candidates = index.candidates(row)
        features = index.features(row, candidates)
        # Dense branch uses a frozen synthetic scorer, separate from saved classifier checks.
        scores = (scorer.predict_groups(features, [len(features)], num_threads=1)
                  if hasattr(scorer, "predict_groups") else scorer.predict(features, num_threads=1))
        rank = np.argsort(-scores, kind="stable")
        state.append((direct, candidates, features, scores, rank,
                      index.ids[candidates[int(np.argmax(scores))]]))
    return state


class TieScorer:
    def predict(self, features, num_threads=None):
        return np.zeros(len(features), dtype=np.float32)


def matcher_from_index(module, index, scorer):
    matcher = module.Matcher.__new__(module.Matcher)
    matcher.index, matcher.model = index, scorer
    return matcher


def ordered_checks(matcher, queries):
    expected = matcher.predict(queries)
    assert matcher.predict(queries.iloc[::-1])[::-1] == expected
    assert matcher.predict(queries.iloc[:17]) + matcher.predict(queries.iloc[17:]) == expected
    assert [matcher.predict(queries.iloc[[i]])[0] for i in range(len(queries))] == expected
    assert matcher.predict(queries.iloc[:0]) == []
    return expected


@threadpool_limits.wrap(limits=1)
def verify(args):
    import torch
    torch.set_num_threads(1)
    from phonetic_encoder.hybrid_matcher import HybridDenseLexicalMatcher
    old = [load(args.before / p, "_before_refactor_" + str(i)) for i, p in enumerate(SOURCE_FILES)]
    new = [load(ROOT / p, "_after_refactor_" + str(i)) for i, p in enumerate(SOURCE_FILES)]
    report = {"experiment_id": ID, "recorded_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
              "python": sys.version, "executable": sys.executable, "rtol": 0, "atol": 0}
    start = time.monotonic()
    report["evaluators"] = evaluator_parity(old[1], new[1], old[2], new[2])
    queries, catalog = public_sample(args.data_dir, args.folds)
    report["data"] = {"sample_rule": "sort sample_id; fold 0; first 32 Hangul and 32 non-Hangul",
                      "rows": len(queries), "hangul_rows": 32, "non_hangul_rows": 32,
                      "sample_ids_sha256": hashlib.sha256("\n".join(queries.sample_id).encode()).hexdigest(),
                      "train_sha256": sha(args.data_dir / "train.csv"),
                      "catalog_sha256": sha(args.data_dir / "vessel_catalog.csv"),
                      "folds_sha256": sha(args.folds)}
    # Real saved classifier; lookup on/off; candidates, feature values, scores, ranks and IDs.
    for lookup in (False, True):
        engines = [m.Matcher(catalog, ROOT / "model", use_train_lookup=lookup) for m in (old[0], new[0])]
        states = [step_state(m, e.index, e.model, queries) for m, e in zip((old[0], new[0]), engines)]
        same_nested(*states)
        predictions = [ordered_checks(e, queries) for e in engines]
        same_nested(*predictions)
        report[f"saved_model_lookup_{lookup}"] = {"exact_step_states": True, "prediction_differences": 0,
                                                  "full_split_reverse_single_empty": True,
                                                  "direct_hits": sum(s[0] is not None for s in states[0])}
    # Changed dense-map branch with the existing real checkpoint, full catalog, and frozen embeddings.
    dense = HybridDenseLexicalMatcher(
        ROOT / "artifacts/DPR-SYMPHON-01/fold_0_encoder.pt",
        ROOT / "artifacts/DPR-SYMPHON-01/catalog_embeds_fp16.npy",
        catalog_df=pd.read_csv(args.data_dir / "vessel_catalog.csv"), device="cpu")
    assets = json.loads((ROOT / "model/assets.json").read_text(encoding="utf-8"))
    assets = deepcopy(assets)
    assets["include_dense"] = True
    dense_queries = queries.copy()
    # Limited edge fixtures are derived only from public rows; no private test inputs.
    dense_queries.loc[dense_queries.index[:4], "vessel_name"] = [None, "", "   ", "한글 ALPHA ２"]
    dense_queries.loc[dense_queries.index[4:8], "gross_tonnage"] = ["", None, "bad", 0]
    dense_queries.loc[:, "call_sign"] = "UNKNOWN_REFACTOR_CALL"
    engines = []
    for m in (old[0], new[0]):
        index = m.CatalogIndex(catalog, assets, use_train_lookup=False, dense_matcher=dense)
        engines.append(matcher_from_index(m, index, TieScorer()))
    states = [step_state(m, e.index, e.model, dense_queries) for m, e in zip((old[0], new[0]), engines)]
    same_nested(*states)
    predictions = [ordered_checks(e, dense_queries) for e in engines]
    same_nested(*predictions)
    report["dense"] = {"rows": len(dense_queries), "real_checkpoint_loaded": True,
                       "exact_candidates_features_scores_ranks": True, "prediction_differences": 0,
                       "scorer": "synthetic all-tie scorer to verify first-candidate tie handling; not accuracy",
                       "full_split_reverse_single_empty": True, "missing_values": True}
    # Original signatures, CLI defaults, and help must remain unchanged.
    import inspect
    for left, right in ((old[0].CatalogIndex, new[0].CatalogIndex),
                        (old[0].Matcher, new[0].Matcher),
                        (old[1].evaluate_retrieval, new[1].evaluate_retrieval),
                        (old[2].evaluate_dense_retrieval, new[2].evaluate_dense_retrieval)):
        assert str(inspect.signature(left)) == str(inspect.signature(right))
    cli = []
    for p in SOURCE_FILES[1:]:
        a = subprocess.run([sys.executable, "-B", "-X", "utf8", str(args.before / p), "--help"],
                           cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
        b = subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / p), "--help"],
                           cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
        assert a.stdout == b.stdout
        module = p[:-3].replace("/", ".")
        c = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", module, "--help"],
                           cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
        assert b.stdout == c.stdout
        cli.append({"path":p, "before_after_help_identical":True, "direct_and_module":True})
    report["cli"] = cli
    report["source_before_sha256"] = {p:sha(args.before / p) for p in SOURCE_FILES}
    report["source_after_sha256"] = {p:sha(ROOT / p) for p in SOURCE_FILES}
    manifest = json.loads(args.protected.read_text(encoding="utf-8"))
    changed = [p for p, item in manifest.items() if sha(ROOT / p) != item["sha256"]]
    assert changed == [], "Protected artifact changed"
    report["protected_files_unchanged"] = len(manifest)
    report["wall_seconds"] = time.monotonic() - start
    report["timing_scope"] = "all verification including loading/index/features/scores/CLI; not inference benchmark"
    report["passed"] = True
    report["limitations"] = ["Public train was previously used for EDA/final fitting; not an independent final test",
                              "Only fold 0 sample; no OOF accuracy or official-server evaluation",
                              "CPU verification; no GPU or Python 3.11.17 parity claim",
                              "Dense prediction probe uses synthetic tie scores; real classifier checked without dense columns",
                              "No model fitting, checkpoint update, ZIP creation or network retrieval"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed":True, "rows":64, "protected_files":len(manifest),
                      "prediction_differences":0, "seconds":report["wall_seconds"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=ROOT / f"artifacts/{ID}/before")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--folds", type=Path, default=ROOT / "eval_kit/folds_group5_seed42.csv")
    parser.add_argument("--protected", type=Path, default=ROOT / f"reports/refactoring/{ID}/protected_before_sha256.json")
    parser.add_argument("--output", type=Path, default=ROOT / f"reports/refactoring/{ID}/verification.json")
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
