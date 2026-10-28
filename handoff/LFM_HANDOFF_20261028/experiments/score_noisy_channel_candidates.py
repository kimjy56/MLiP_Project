"""Export reverse-channel features on fixed PUBLIC heldout candidate sets.

This script does not calculate accuracy or choose a fusion weight. It rejects
sample IDs outside official train's specified heldout fold, and never builds a
candidate set from gold IDs. Validation belongs to a separate agent/run.
"""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.noisy_channel_common import (
    configure_cpu, now_kst, record_failure, reserve_experiment,
    sha256, source_hashes, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--candidates-csv", type=Path, required=True,
                        help="sample_id,candidate_vessel_id,baseline_score; higher score is better")
    parser.add_argument("--baseline-record", type=Path, required=True,
                        help="Existing baseline run/provenance JSON; retained by path and hash")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--folds-file", type=Path,
                        default=ROOT / "eval_kit/folds_group5_seed42.csv")
    parser.add_argument("--heldout-fold", type=int, choices=range(5), default=0)
    parser.add_argument("--channel-weight", type=float, default=0.0,
                        help="Untuned by this script; zero keeps the supplied baseline ranking")
    parser.add_argument("--normalization", choices=("sum", "mean"), default="mean")
    parser.add_argument("--allow-subset", action="store_true",
                        help="Allow a public heldout subset; explicitly marked as subset in metadata")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if not math.isfinite(args.channel_weight) or args.channel_weight < 0:
        parser.error("channel-weight must be finite and >= 0")
    return args


def score(args, output_dir, record_path, record):
    import json
    import numpy as np
    import pandas as pd
    import torch
    from name_decoder.data import read_frames
    from name_decoder.noisy_channel import MANIFEST_NAME, NoisyChannelScorer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    hashes = source_hashes(args.data_dir, args.folds_file)
    _, heldout, _, catalog = read_frames(args.data_dir, args.folds_file, args.heldout_fold)
    candidates = pd.read_csv(args.candidates_csv, keep_default_na=False,
                             dtype={"sample_id": str, "candidate_vessel_id": str})
    required = ["sample_id", "candidate_vessel_id", "baseline_score"]
    if not set(required).issubset(candidates.columns) or candidates.empty:
        raise ValueError("Candidate CSV must be nonempty and have " + ",".join(required))
    candidates = candidates[required].copy()
    candidates.baseline_score = pd.to_numeric(candidates.baseline_score, errors="raise")
    if not np.isfinite(candidates.baseline_score.to_numpy(dtype=float)).all():
        raise ValueError("Baseline scores must be finite")
    if candidates.duplicated(["sample_id", "candidate_vessel_id"]).any():
        raise ValueError("Duplicate (sample_id,candidate_vessel_id) pair")
    sample_ids = set(candidates.sample_id)
    heldout_ids = set(heldout.sample_id)
    if not sample_ids <= heldout_ids:
        raise ValueError("Only public training rows in the requested heldout fold may be exported")
    if not args.allow_subset and sample_ids != heldout_ids:
        raise ValueError("Missing heldout rows; use --allow-subset to label a deliberate subset")
    if not set(candidates.candidate_vessel_id) <= set(catalog.name_by_id):
        raise ValueError("Candidate ID missing from official catalog")
    baseline_record = json.loads(args.baseline_record.read_text(encoding="utf-8-sig"))
    if not isinstance(baseline_record, dict):
        raise ValueError("Baseline provenance must be a JSON object")
    # Parse and check fold provenance before model loading. The scorer separately
    # checks direction, normalization, tokenizer EOS, and checkpoint content.
    manifest_path = args.model_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("heldout_fold") != args.heldout_fold:
        raise ValueError("Reverse checkpoint was trained for a different heldout fold")
    if manifest.get("data_sha256") != hashes:
        raise ValueError("Reverse checkpoint and scoring data/folds have different hashes")
    scorer = NoisyChannelScorer.load(args.model_path)
    record["data"] = {
        "source_files_sha256": hashes, "split_id": args.folds_file.name,
        "split_file_sha256": hashes["folds"], "heldout_fold": args.heldout_fold,
        "scope": "full_public_heldout_fold" if sample_ids == heldout_ids else "public_heldout_subset",
        "scored_rows": len(sample_ids), "heldout_total_rows": len(heldout),
        "validation_is_independent_final_test": False,
    }
    counts = candidates.groupby("sample_id").size()
    record["retrieval"] = {
        "source": "fixed externally supplied baseline candidates; unchanged by this script",
        "candidates_csv": str(args.candidates_csv.resolve()),
        "candidates_sha256": sha256(args.candidates_csv), "candidate_pairs": len(candidates),
        "count_min": int(counts.min()), "count_max": int(counts.max()),
        "count_mean": float(counts.mean()),
        "baseline_record": str(args.baseline_record.resolve()),
        "baseline_record_sha256": sha256(args.baseline_record),
        "baseline_fitting_provenance_verified": False,
    }
    record["model"] = {
        "type": "teacher-forced reverse name likelihood (merged CPU model)",
        "checkpoint": str(args.model_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "parameters": {"channel_weight": args.channel_weight,
                       "normalization": args.normalization, "pair_batch_size": 1},
        "formula": "combined_score = baseline_score + channel_weight * channel_logp",
        "tie_break": "vessel ID ascending",
    }
    record["environment"]["threads"] = args.threads
    record["features"] = ["log_probability", "mean_log_probability", "target_token_count"]
    record["cost"]["preparation_seconds"] = time.perf_counter() - started
    record["limitations"] = [
        "No accuracy evaluation or weight tuning is performed here.",
        "The baseline record is retained but its fold fitting must be audited externally.",
        "Candidate recall cannot increase because the provided candidate IDs are unchanged.",
        "The ID tie rule must be compared with the original baseline predictor.",
        "Same-name catalog IDs receive the same channel likelihood.",
    ]
    write_json(record_path, record)
    names_by_sample = heldout.set_index("sample_id").vessel_name.to_dict()
    records = []
    predictions = []
    scoring_started = time.perf_counter()
    # No shared name cache across records; no use of target_vessel_id below.
    for index, (sample_id, group) in enumerate(candidates.groupby("sample_id", sort=True), start=1):
        channel = pd.DataFrame(scorer.score_candidates(
            names_by_sample[sample_id], group.candidate_vessel_id.tolist(), catalog))
        rows = group.merge(channel, on="candidate_vessel_id", validate="one_to_one")
        rows = rows.sort_values(["baseline_score", "candidate_vessel_id"],
                                ascending=[False, True]).reset_index(drop=True)
        rows["baseline_rank"] = range(1, len(rows) + 1)
        baseline_id = rows.iloc[0].candidate_vessel_id
        score_column = "mean_log_probability" if args.normalization == "mean" else "log_probability"
        rows["combined_score"] = rows.baseline_score + args.channel_weight * rows[score_column]
        if not np.isfinite(rows.combined_score.to_numpy()).all():
            raise FloatingPointError("Nonfinite fused candidate score")
        rows = rows.sort_values(["combined_score", "candidate_vessel_id"],
                                ascending=[False, True]).reset_index(drop=True)
        rows["combined_rank"] = range(1, len(rows) + 1)
        records.extend(rows.to_dict("records"))
        predictions.append({"sample_id": sample_id, "baseline_vessel_id": baseline_id,
                            "pred_vessel_id": rows.iloc[0].candidate_vessel_id,
                            "fold": args.heldout_fold})
        if index == 1 or index % 25 == 0 or index == len(sample_ids):
            print(f"scored public rows={index}/{len(sample_ids)}", flush=True)
    record["cost"]["inference_seconds"] = time.perf_counter() - scoring_started
    saving_started = time.perf_counter()
    candidate_path = output_dir / "candidate_scores.csv"
    prediction_path = output_dir / "predictions.csv"
    pd.DataFrame(records).sort_values(["sample_id", "combined_rank"]).to_csv(
        candidate_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).sort_values("sample_id").to_csv(
        prediction_path, index=False, encoding="utf-8-sig")
    record["artifact_sha256"] = {path.name: sha256(path)
                                  for path in (candidate_path, prediction_path)}
    record["cost"].update(
        saving_seconds=time.perf_counter() - saving_started,
        total_seconds=time.perf_counter() - started,
        timing_scope="preparation: input/hash checks and model load; inference: likelihood plus fusion; "
                     "saving: CSV export plus CSV hashing", submission_zip_bytes=None)
    record.update(status="scored_unvalidated", finished_at_kst=now_kst(), failure=None,
                  artifacts=[str(output_dir.resolve())], validation_status="pending_external_agent",
                  decision="Separate agent must evaluate official ID accuracy and fixes/breaks.")
    write_json(record_path, record)
    write_json(output_dir / "run_record.json", record)
    print(f"Saved public candidate features: {output_dir}; accuracy has not been evaluated.", flush=True)


def main():
    args = parse_args()
    configure_cpu(args.threads)
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    output_dir, record_path, record = reserve_experiment(
        args.experiment_id, ROOT / "reports/noisy_channel",
        "Public heldout fixed-candidate reverse likelihood feature export", command)
    try:
        score(args, output_dir, record_path, record)
    except BaseException as exc:
        record_failure(record_path, record, exc)
        raise


if __name__ == "__main__":
    main()
