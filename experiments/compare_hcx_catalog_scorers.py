"""Re-score a public heldout HCX pilot without loading or retraining the model."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from name_decoder.catalog_search import CatalogNameSearch
from name_decoder.data import read_frames, select_pilot


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def boolean(value):
    check(str(value).lower() in ("true", "false"), f"Invalid boolean: {value}")
    return str(value).lower() == "true"


def metrics(rows, rankings):
    result = {}
    for lang, is_hangul in (("hangul", True), ("non_hangul", False)):
        selected = [(r, top) for r, top in zip(rows, rankings) if r.hangul == is_hangul]
        n = len(selected)
        result[lang] = {"rows": n}
        for k in (1, 5, 10):
            hits = sum(r.target in top[:k] for r, top in selected)
            result[lang][f"top_{k}_hits"] = hits
            result[lang][f"top_{k}_name_accuracy"] = hits / n if n else None
        direct = sum(r.target == r.generated_text for r, _ in selected)
        result[lang]["direct_exact_hits"] = direct
        result[lang]["direct_exact_accuracy"] = direct / n if n else None
    h, nh = [result[l]["top_1_name_accuracy"] for l in ("hangul", "non_hangul")]
    result["weighted_name_accuracy"] = .7 * h + .3 * nh if h is not None and nh is not None else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-experiment", default="AS-HCX-03")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    for value in (args.experiment_id, args.source_experiment):
        check(value and all(c.isalnum() or c in "-_" for c in value), "Invalid experiment ID")
    check(args.repeats >= 2, "Use at least two alternating-order repeats")
    record_path = ROOT / "project_log/experiments" / f"{args.experiment_id}.json"
    output = ROOT / "artifacts" / args.experiment_id
    check(not record_path.exists() and not output.exists(), "Refusing to overwrite an existing experiment")
    source_path = ROOT / "project_log/experiments" / f"{args.source_experiment}.json"
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    record = json.loads((ROOT / "project_log/templates/experiment.json").read_text(encoding="utf-8-sig"))
    record.update(experiment_id=args.experiment_id, status="planned", recorded_at_kst=now(),
                  purpose="Paired comparison of cached HCX generated-name catalog lookup",
                  hypothesis="Full-string ratio reduces false positives from short partial catalog names",
                  baseline_experiment_id=args.source_experiment,
                  changed_factors=["Only catalog scorer: WRatio -> ratio"],
                  fixed_conditions=["Same cached generations, targets, sorted catalog, top-10 and fold-0 pilot", "No training, no generation, no ID disambiguation"],
                  commands=[subprocess.list2cmdline([sys.executable, *sys.argv])])
    write(record_path, record)  # Plan persisted BEFORE any evaluation.
    try:
        record.update(status="running", started_at_kst=now())
        write(record_path, record)
        output.mkdir()
        cached = ROOT / "artifacts" / args.source_experiment / "pilot_predictions.csv"
        source_artifacts = {a["path"]: a for a in source["artifacts"]}
        expected = source_artifacts[cached.relative_to(ROOT).as_posix()]["sha256"]
        check(sha(cached) == expected, "Cached prediction hash differs from source experiment")
        for path, expected in source["data"]["source_files_sha256"].items():
            check(sha(ROOT / path) == expected, f"Data hash mismatch: {path}")
        folds = ROOT / "eval_kit/folds_group5_seed42.csv"
        check(sha(folds) == source["data"]["split_file_sha256"], "Fold hash mismatch")
        check(source["data"]["split_id"] == "supplied-group-fold-0/pilot", "Only known fold-0 pilots supported")
        t0 = time.perf_counter()
        _, validation, fit_rows, catalog = read_frames(ROOT / "data", folds, heldout_fold=0)
        df = pd.read_csv(cached, keep_default_na=False, dtype=str)
        check(not df.sample_id.duplicated().any(), "Duplicate cached sample IDs")
        for col in ("hangul", "direct_exact_match", "correct_top1", "correct_top5", "correct_top10"):
            df[col] = df[col].map(boolean)
        counts = df.groupby("hangul").size()
        check(len(counts) == 2 and counts.nunique() == 1, "Expected equal-language pilot")
        pilot = select_pilot(validation, int(counts.iloc[0]), source["data"]["seed"])
        check(df.sample_id.tolist() == pilot.sample_id.tolist(), "Cached rows differ from seeded pilot")
        for col in ("source", "target", "hangul"):
            check(df[col].tolist() == pilot[col].tolist(), f"Cached {col} mismatch")
        seen = pilot.target_vessel_id.isin(set(fit_rows.target_vessel_id))
        check(not seen.any(), "Heldout vessel appeared in training fold")
        rows = list(df.itertuples(index=False))
        loading_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        searches = {s: CatalogNameSearch(catalog.names, s) for s in ("WRatio", "ratio")}
        index_seconds = time.perf_counter() - t0
        ranks, timings = {}, {s: [] for s in searches}
        for repeat in range(args.repeats):
            order = list(searches) if repeat % 2 == 0 else list(reversed(searches))
            for scorer in order:
                t0 = time.perf_counter()
                ranking = [searches[scorer].top_names(r.generated_text) for r in rows]
                timings[scorer].append(time.perf_counter() - t0)
                if scorer in ranks:
                    check(ranks[scorer] == ranking, "Repeated lookup changed candidates")
                ranks[scorer] = ranking
                print(f"repeat={repeat+1} scorer={scorer} seconds={timings[scorer][-1]:.4f}", flush=True)
        for row, top in zip(rows, ranks["WRatio"]):
            check(json.loads(row.top10_candidates) == top, "Historical top-10 not reproduced")
            check(row.predicted_catalog_name == top[0], "Historical top-1 not reproduced")
            for k in (1, 5, 10):
                check(getattr(row, f"correct_top{k}") == (row.target in top[:k]), "Historical hit flags differ")
            check(row.direct_exact_match == (row.generated_text == row.target), "Direct match flag differs")
        results = {s: metrics(rows, ranking) for s, ranking in ranks.items()}
        paired, changes = [], {}
        for row, old, new in zip(rows, ranks["WRatio"], ranks["ratio"]):
            before, after = old[0] == row.target, new[0] == row.target
            category = "fixed" if after and not before else "regressed" if before and not after else "still_correct" if after else "still_wrong"
            paired.append({"sample_id": row.sample_id, "hangul": row.hangul,
                           "source": row.source, "target": row.target, "generated_text": row.generated_text,
                           "WRatio_top1": old[0], "ratio_top1": new[0], "change": category,
                           "WRatio_top10": json.dumps(old, ensure_ascii=False), "ratio_top10": json.dumps(new, ensure_ascii=False),
                           "ratio_error_type": None if after else "candidate_selection" if row.target in new else "retrieval_miss_at_10"})
        for lang, flag in (("hangul", True), ("non_hangul", False)):
            subset = [r for r in paired if r["hangul"] == flag]
            changes[lang] = {cat: sum(r["change"] == cat for r in subset) for cat in ("fixed", "regressed", "still_correct", "still_wrong")}
        prediction_path = output / "paired_predictions.csv"
        pd.DataFrame(paired).to_csv(prediction_path, index=False, encoding="utf-8-sig")
        medians = {s: statistics.median(t) for s, t in timings.items()}
        record.update(status="completed", finished_at_kst=now(), data=dict(source["data"]),
                      model={"type": source["model"]["type"], "source_experiment": args.source_experiment, "retrained": False},
                      retrieval={"scorers": list(searches), "catalog_names": len(catalog.names), "limit": 10, "tie_break": "lexicographic canonical name"},
                      preprocessing={"generated_text": "Unchanged cached cleaned text", "catalog": "canonical_name; no fitted preprocessing"},
                      results=results, paired_changes=changes,
                      checks={"source_hashes_equal": True, "seeded_pilot_equal": True, "historical_WRatio_top10_equal": True, "repeats_equal": True, "vessel_group_disjoint": True},
                      environment={"python": platform.python_version(), "os": platform.platform(), "packages": {p: importlib.metadata.version(p) for p in ("rapidfuzz", "pandas", "numpy")}, "thread_cap": 4},
                      limitations=["128 previously explored/selected pilot rows; not an untouched final test or full OOF", "Name matching only; official vessel-ID score is unmeasured", "End-to-end generation latency was not remeasured", "Python 3.11.3 differs from target server Python 3.11.17", "No seen-vessel rows; seen-vessel accuracy is unmeasured"])
        record["data"].update(unique_validation_vessels=int(pilot.target_vessel_id.nunique()), cached_predictions_sha256=sha(cached), source_record_sha256=sha(source_path))
        record["metrics"].update(metric_source="local cached public-train heldout name comparison", hangul_rows=64, non_hangul_rows=64, seen_vessel_fraction=float(seen.mean()))
        record["name_accuracy_delta"] = results["ratio"]["weighted_name_accuracy"] - results["WRatio"]["weighted_name_accuracy"]
        record["cost"].update(loading_seconds=loading_seconds, index_seconds=index_seconds, lookup_seconds_by_repeat=timings, lookup_seconds_median=medians,
                              lookup_speedup_median=medians["WRatio"]/medians["ratio"], timing_scope="Lookup only: cached-text loop + RapidFuzz top-10; excludes loading/index/training/generation/metrics/saving")
        record["code"] = {"git_revision_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
                          "file_sha256": {p: sha(ROOT / p) for p in ("experiments/compare_hcx_catalog_scorers.py", "experiments/train_hyperclovax_sft.py", "name_decoder/catalog_search.py", "name_decoder/catalog.py", "name_decoder/data.py", "tests/test_catalog_search.py")}}
        record["artifacts"] = [{"path": prediction_path.relative_to(ROOT).as_posix(), "sha256": sha(prediction_path), "bytes": prediction_path.stat().st_size}]
        record["error_cases_file"] = prediction_path.relative_to(ROOT).as_posix()
        record["decision"] = "Use ratio by default for generated-name lookup; retain WRatio option for reproduction. Official-ID integration requires separate evaluation."
        record["next_actions"] = ["Expand heldout name evaluation with fixed scorer", "Evaluate union of decoder names and existing ID candidates; preserve all same-name IDs", "Measure full ID OOF and runtime before adoption in submission matcher"]
        write(record_path, record)
        print(json.dumps({"results": results, "paired_changes": changes, "timing_medians": medians}, indent=2), flush=True)
    except BaseException as exc:
        record.update(status="failed", finished_at_kst=now(), failure={"type": type(exc).__name__, "message": str(exc), "retry": "Use a new experiment ID"})
        write(record_path, record)
        raise


if __name__ == "__main__":
    main()
