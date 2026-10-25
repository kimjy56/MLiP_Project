"""Train catalog-name prediction on a supplied fold or all official training rows.

This produces name-recall diagnostics, not a submission or complete ID-matching OOF.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
import torch
from unidecode import unidecode

from name_decoder.data import read_frames, select_pilot
from name_decoder.model import ByT5NameDecoder


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def relative(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name  # Shared experiment records omit machine-specific paths.


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def name_metrics(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("Prediction count does not match evaluation rows")
    result = {}
    for label, mask in (("hangul", rows.hangul), ("non_hangul", ~rows.hangul)):
        positions = np.flatnonzero(mask.to_numpy())
        group = {"rows": len(positions)}
        for k in (1, 5, 10):
            hits = sum(rows.iloc[int(i)].target in predictions[int(i)][:k] for i in positions)
            group[f"gold_name_in_top_{k}"] = hits / len(positions) if len(positions) else None
            group[f"gold_name_hits_top_{k}"] = hits
        group["name_exact_accuracy"] = group["gold_name_in_top_1"]
        group["mean_returned_names"] = float(np.mean([len(predictions[int(i)]) for i in positions])) if len(positions) else None
        result[label] = group
    ko, other = result["hangul"]["name_exact_accuracy"], result["non_hangul"]["name_exact_accuracy"]
    result["weighted_name_exact_accuracy"] = .7 * ko + .3 * other if ko is not None and other is not None else None
    result["scope"] = "normalized catalog NAME matching; not official vessel-ID accuracy"
    return result


def evaluate(decoder, rows):
    sync()
    start = time.perf_counter()
    names, detailed = [], []
    for index, source in enumerate(rows.source):
        candidates = decoder.decode(source, top_k=10)
        names.append([item.name for item in candidates])
        detailed.append([asdict(item) for item in candidates])
        if (index + 1) % 16 == 0:
            print(json.dumps({"decoded_rows": index + 1, "total": len(rows)}), flush=True)
    sync()
    return {"metrics": name_metrics(rows, names), "seconds": time.perf_counter() - start,
            "names": names, "candidates": detailed}


def training_steps(pair_count, batch_size, accumulation, steps=None, epochs=None):
    """Epoch-equivalent is a draw budget: weighted sampling uses replacement."""
    if pair_count < 1 or batch_size < 1 or accumulation < 1:
        raise ValueError("Training pairs and batch sizes must be positive")
    if steps is not None and epochs is not None:
        raise ValueError("Specify steps OR epochs")
    if epochs is not None:
        if not math.isfinite(epochs) or epochs <= 0:
            raise ValueError("epochs must be finite and positive")
        return math.ceil(pair_count * epochs / (batch_size * accumulation))
    if steps is not None and steps <= 0:
        raise ValueError("steps must be positive")
    return 200 if steps is None else steps


def sequence_loss(logits, labels, reduction):
    """Ignore padding; optionally give each name pair equal loss weight."""
    valid = labels.ne(-100)
    if not valid.any(dim=1).all():
        raise ValueError("Every training target must contain an unmasked token")
    losses = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=-100, reduction="none").reshape(labels.shape)
    if reduction == "row":
        return (losses.sum(dim=1) / valid.sum(dim=1)).mean()
    if reduction != "token":
        raise ValueError("loss reduction must be token or row")
    return losses.sum() / valid.sum()


def train(decoder, pairs, args):
    model, tokenizer = decoder.model, decoder.tokenizer
    sources, targets = pairs.source.tolist(), pairs.target.tolist()
    if max(len(s.encode("utf-8")) for s in sources) > decoder.max_input_bytes:
        raise ValueError("Training input exceeds the documented input byte limit")
    counts = pairs.hangul.value_counts().to_dict()
    weights = torch.tensor([(.7 if ko else .3) / counts[ko] for ko in pairs.hangul], dtype=torch.double)
    generator = torch.Generator().manual_seed(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    losses = []
    sync()
    start = time.perf_counter()
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for _ in range(args.accumulation):
            ids = torch.multinomial(weights, args.batch_size, replacement=True, generator=generator).tolist()
            inputs = tokenizer([sources[i] for i in ids], padding=True, return_tensors="pt").to(model.device)
            labels = tokenizer(text_target=[targets[i] for i in ids], padding=True, return_tensors="pt").input_ids.to(model.device)
            labels[labels == tokenizer.pad_token_id] = -100
            with decoder.precision_context():
                output = model(**inputs, labels=labels, use_cache=False)
                loss = (output.loss if getattr(args, "loss_reduction", "token") == "token"
                        else sequence_loss(output.logits, labels, "row"))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss_sum += float(loss.detach()) / args.accumulation
            (loss / args.accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(loss_sum)
        if (step + 1) % 20 == 0:
            print(json.dumps({"optimizer_step": step + 1,
                              "mean_recent_loss": float(np.mean(losses[-20:]))}), flush=True)
    sync()
    elapsed = time.perf_counter() - start
    model.eval()
    del optimizer
    return {"seconds": elapsed, "losses": losses,
            "first_20_mean": float(np.mean(losses[:20])), "last_20_mean": float(np.mean(losses[-20:]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--folds", type=Path, default=ROOT / "eval_kit/folds_group5_seed42.csv")
    parser.add_argument("--base-model", type=Path, default=ROOT / "artifacts/pretrained/byt5-small")
    parser.add_argument("--fold", type=int, default=0)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--steps", type=int, help="Optimizer steps; default 200 if epochs omitted")
    budget.add_argument("--epochs", type=float, help="Pair-count-equivalent weighted sampling passes")
    parser.add_argument("--loss-reduction", choices=("token", "row"), default="token")
    parser.add_argument("--final-train", action="store_true",
                        help="Train all official rows without heldout metrics; no fold file needed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-per-language", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    import re
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9-]+", args.experiment_id):
        parser.error("Use an uppercase alphanumeric experiment ID with hyphens")
    if min(args.batch_size, args.accumulation) <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Training settings must be positive")
    try:
        training_steps(1, args.batch_size, args.accumulation, args.steps, args.epochs)
    except ValueError as error:
        parser.error(str(error))
    invocation = [sys.executable, *sys.argv]
    output = ROOT / "artifacts" / args.experiment_id
    record_path = ROOT / "project_log/experiments" / f"{args.experiment_id}.json"
    if output.exists() or record_path.exists():
        parser.error("Experiment already exists; use a new ID rather than overwrite")
    output.mkdir(parents=True)
    record = json.loads((ROOT / "project_log/templates/experiment.json").read_text(encoding="utf-8"))
    record.update(experiment_id=args.experiment_id, plan_id="NAME-DECODER-V1", status="planned",
                  recorded_at_kst=now(), purpose=("Full-data catalog-name training; no heldout evaluation"
                      if args.final_train else "Heldout catalog-name prediction and candidate evaluation"),
                  hypothesis="Official observed/canonical name pairs teach useful byte-level alternatives",
                  changed_factors=["ByT5 fine-tuning", f"loss reduction: {args.loss_reduction}"],
                  fixed_conditions=["official full-catalog trie", "10 beams", f"seed {args.seed}",
                                    "name input only; no augmentation"],
                  commands=[subprocess.list2cmdline(invocation)],
                  limitations=["Name matching only; not full OOF or official ID accuracy",
                               "Known development data, not an untouched final test",
                               "Single-row inference; no server timing claim",
                               "Actual environment versions recorded separately; no automatic server compatibility claim"])
    write_json(record_path, record)
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(6)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        record.update(status="running", started_at_kst=now())
        files = [*sorted((ROOT / "name_decoder").glob("*.py")), Path(__file__).resolve()]
        record["code"] = {"git_revision_at_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "file_sha256": {relative(path): digest(path) for path in files}}
        record["environment"] = {"python": platform.python_version(), "os": platform.system(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torch", "transformers", "safetensors", "pandas", "rapidfuzz", "Unidecode")},
            "device": args.device, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
        pairs, validation, fit_rows, catalog = read_frames(
            args.data_dir, args.folds, args.fold, final_train=args.final_train)
        pilot = validation if args.final_train else select_pilot(validation, args.eval_per_language, args.seed)
        args.steps = training_steps(len(pairs), args.batch_size, args.accumulation, args.steps, args.epochs)
        record["data"].update(source_files_sha256={relative(args.data_dir / n): digest(args.data_dir / n)
            for n in ("train.csv", "vessel_catalog.csv")},
            split_id=("all-official-train/no-validation" if args.final_train else
                      f"supplied-group-fold-{args.fold}/" + ("full" if args.eval_per_language <= 0 else "pilot")),
            split_file_sha256=None if args.final_train else digest(args.folds), seed=args.seed, training_rows=len(fit_rows),
            validation_rows=len(pilot), total_heldout_rows=len(validation),
            training_vessels=int(fit_rows.target_vessel_id.nunique()),
            training_sample_ids_sha256=hashlib.sha256("\n".join(sorted(fit_rows.sample_id)).encode()).hexdigest(),
            selected_sample_ids_sha256=hashlib.sha256("\n".join(pilot.sample_id).encode()).hexdigest())
        record["training_pairs"] = {"positive_rule": "training-fold observed name -> official gold catalog name",
            "negative_rule": None, "count": len(pairs), "hangul_pairs": int(pairs.hangul.sum()),
            "non_hangul_pairs": int((~pairs.hangul).sum()), "sampling_hangul_mass": .7}
        pairs.to_csv(output / "training_pairs.csv", index=False, encoding="utf-8-sig")
        record["preprocessing"] = {"name": "NFC, uppercase, whitespace collapse; retain digits/punctuation",
                                   "augmentation": "none"}
        record["retrieval"] = {"catalog_names": len(catalog.names), "vessel_ids": len(catalog.name_by_id),
            "duplicate_name_groups": sum(len(ids) > 1 for ids in catalog.ids_by_name.values()),
            "method": "full-catalog byte-prefix-constrained beam search", "num_beams": 10,
            "length_penalty": 1., "training_lookup": False}
        record["model"] = {"type": "google/byt5-small", "loss": f"teacher-forced cross entropy/{args.loss_reduction}",
            "parameters": {"steps": args.steps, "batch_size": args.batch_size,
                           "epoch_equivalent_requested": args.epochs,
                           "sampled_pair_draws": args.steps * args.batch_size * args.accumulation,
                           "loss_reduction": args.loss_reduction, "final_train": args.final_train,
                           "accumulation": args.accumulation, "learning_rate": args.learning_rate,
                           "weight_decay": .01, "optimizer": "AdamW", "max_grad_norm": 1.,
                           "training_autocast": "bfloat16 on CUDA", "inference_dtype": "float32",
                           "checkpoint_config": "preserve tie_word_embeddings and scale_decoder_outputs"},
            "pretrained_manifest": json.loads((args.base_model / "download_manifest.json").read_text())}
        record["metrics"].update(hangul_rows=int(pilot.hangul.sum()), non_hangul_rows=int((~pilot.hangul).sum()),
                                seen_vessel_fraction=None if args.final_train else 0.,
                                metric_source="no heldout evaluation" if args.final_train else
                                    "local catalog-name evaluation, not ID accuracy")
        write_json(record_path, record)
        print(json.dumps({"stage": "prepared", "pairs": len(pairs), "pilot_rows": len(pilot),
                          "catalog_names": len(catalog.names)}), flush=True)
        baseline, baseline_seconds = [], None
        if not args.final_train:
            roman_names = [unidecode(name).upper() for name in catalog.names]
            start = time.perf_counter()
            baseline = [[catalog.names[index] for _, _, index in process.extract(
                unidecode(source).upper(), roman_names, scorer=fuzz.WRatio, limit=10)] for source in pilot.source]
            baseline_seconds = time.perf_counter() - start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        decoder = ByT5NameDecoder.load(args.base_model, catalog, args.device)
        sync()
        record["cost"]["loading_and_trie_seconds"] = time.perf_counter() - start
        record["model"]["parameter_count"] = sum(p.numel() for p in decoder.model.parameters())
        before = {"seconds": None, "names": []}
        record["pilot_results"] = None
        if not args.final_train:
            print(json.dumps({"stage": "pretrained_evaluation"}), flush=True)
            before = evaluate(decoder, pilot)
            record["pilot_results"] = {"roman_wratio": name_metrics(pilot, baseline),
                                       "pretrained": before["metrics"]}
        write_json(record_path, record)
        print(json.dumps({"stage": "training", "steps": args.steps}), flush=True)
        history = train(decoder, pairs, args)
        record["cost"]["training_seconds"] = history["seconds"]
        write_json(output / "training_history.json", history)
        model_dir = output / "model"
        decoder.model.config._name_or_path = ""
        decoder.model.save_pretrained(model_dir, safe_serialization=True)
        decoder.tokenizer.init_kwargs.pop("name_or_path", None)
        decoder.tokenizer.save_pretrained(model_dir)
        probe_rows = fit_rows if args.final_train else pilot
        probe_names = probe_rows.source.iloc[:min(4, len(probe_rows))].tolist()
        memory_candidates = [[c.name for c in decoder.decode(name)] for name in probe_names]
        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        decoder = ByT5NameDecoder.load(model_dir, catalog, args.device)
        reloaded_candidates = [[c.name for c in decoder.decode(name)] for name in probe_names]
        record["serialization"] = {"probe_rows": len(probe_names),
                                   "top_10_names_equal": memory_candidates == reloaded_candidates}
        if memory_candidates != reloaded_candidates:
            raise RuntimeError("Saved/reloaded model changed candidate lists")
        after = {"seconds": None, "names": [], "candidates": []}
        if not args.final_train:
            print(json.dumps({"stage": "finetuned_evaluation"}), flush=True)
            after = evaluate(decoder, pilot)
            record["pilot_results"]["finetuned"] = after["metrics"]
        # Compare same per-row names/IDs, not floating-point score representations.
        probe_indices = list(range(len(probe_names)))
        expected = dict(enumerate(reloaded_candidates))
        reordered = {i: [c.name for c in decoder.decode(probe_names[i])] for i in reversed(probe_indices)}
        subset = {i: [c.name for c in decoder.decode(probe_names[i])] for i in probe_indices[::2]}
        independent = expected == reordered and all(subset[i] == expected[i] for i in subset)
        if not independent:
            raise RuntimeError("Candidate lists changed under reordered/subset execution")
        if not args.final_train:
            rows = pilot[["sample_id", "fold", "target_vessel_id", "hangul", "source", "target"]].copy()
            rows["predicted_name"] = [names[0] for names in after["names"]]
            rows["name_correct"] = rows.predicted_name.eq(rows.target)
            for field, values in (("baseline_names", baseline), ("pretrained_names", before["names"]),
                                  ("finetuned_candidates", after["candidates"])):
                rows[field] = [json.dumps(value, ensure_ascii=False) for value in values]
            rows.to_csv(output / "pilot_predictions.csv", index=False, encoding="utf-8-sig")
        record["independence"] = {"probe_rows": len(probe_indices), "reordered_and_subset_equal": independent,
                                  "scope": "top-10 name lists, same process, single-row inference"}
        record["cost"].update(inference_seconds=after["seconds"],
            timing_scope="inference: tokenization + full-catalog constrained decoding + name/ID mapping; excludes loading, saving",
            baseline_seconds=baseline_seconds, pretrained_inference_seconds=before["seconds"],
            peak_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            model_bytes=sum(p.stat().st_size for p in model_dir.iterdir() if p.is_file()))
        write_json(model_dir / "name_decoder_manifest.json", {
            "experiment_id": args.experiment_id, "recorded_at_kst": now(),
            "task": "observed_name_to_catalog_name", "top_k": 10,
            "final_train": args.final_train, "training_data": record["data"],
            "model_parameters": record["model"]["parameters"],
            "catalog_sha256": digest(args.data_dir / "vessel_catalog.csv"),
            "preprocessing": record["preprocessing"], "environment": record["environment"],
            "serialization": record["serialization"], "independence": record["independence"]})
        record["cost"]["model_bytes"] = sum(p.stat().st_size for p in model_dir.iterdir() if p.is_file())
        record["artifacts"] = [{"path": relative(p), "sha256": digest(p), "bytes": p.stat().st_size}
                               for p in sorted(output.rglob("*")) if p.is_file()]
        record.update(status="completed", finished_at_kst=now(),
            observations=["All outputs restricted to official catalog names; duplicate-name IDs retained",
                          f"Mean loss first/last 20 steps: {history['first_20_mean']:.5f}/{history['last_20_mean']:.5f}"],
            decision=("Full-train model saved; generalization unmeasured" if args.final_train else
                      "Heldout name evaluation completed; no model adoption from a single fold"),
            next_actions=["Full G5-s42 name-recall evaluation before selection",
                          "Validate saved model in server environment",
                          "Design metadata-aware ID resolution separately"])
        write_json(record_path, record)
        print(json.dumps({"status": "completed", "results": record["pilot_results"], "cost": record["cost"]}), flush=True)
    except BaseException as error:
        record.update(status="failed", finished_at_kst=now(),
                      failure={"type": type(error).__name__, "message": str(error)},
                      next_actions=["Diagnose failure; retry under a new experiment ID"])
        write_json(record_path, record)
        raise


if __name__ == "__main__":
    main()
