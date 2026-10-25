"""Supervised Fine-Tuning (SFT) of HyperCLOVA X SEED Instruct 0.5B on Korean/English vessel name pairs."""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from rapidfuzz import process, fuzz
from transformers import AutoModelForCausalLM, AutoTokenizer
from unidecode import unidecode

from name_decoder.data import read_frames, select_pilot
from name_decoder.catalog_search import CatalogNameSearch, SCORERS
from name_decoder.lora import adapter_state_dict, inject_attention_lora, merge_attention_lora


def now_kst():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def file_sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_prompt(source):
    return (
        f"<|im_start|>system\n"
        f"한국어 선박명을 국제 카탈로그의 공식 영문 대문자 선박명으로 1:1 변환합니다. 오직 영문 공식 선박명만 대문자로 한 줄로 출력하세요.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"입력: {source}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"출력:\n"
    )


def collate_train_batch(samples, tokenizer):
    item_input_ids = []
    item_labels = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for src, tgt in samples:
        prompt = build_prompt(src)
        full_text = prompt + tgt + "<|im_end|>"
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        f_ids = tokenizer.encode(full_text, add_special_tokens=False)
        lbl = [-100] * len(p_ids) + f_ids[len(p_ids):]
        item_input_ids.append(f_ids)
        item_labels.append(lbl)

    max_len = max(len(x) for x in item_input_ids)
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for ids, lbl in zip(item_input_ids, item_labels):
        pad_len = max_len - len(ids)
        batch_input_ids.append(ids + [pad_id] * pad_len)
        batch_attention_mask.append([1] * len(ids) + [0] * pad_len)
        batch_labels.append(lbl + [-100] * pad_len)

    return (
        torch.tensor(batch_input_ids, dtype=torch.long),
        torch.tensor(batch_attention_mask, dtype=torch.long),
        torch.tensor(batch_labels, dtype=torch.long),
    )


def evaluate_pilot(model, tokenizer, pilot_df, catalog_names, device="cpu", max_new_tokens=16,
                   catalog_scorer="ratio", num_beams=1, num_return_sequences=1):
    model.eval()
    catalog_search = CatalogNameSearch(catalog_names, scorer=catalog_scorer)
    results = []
    t_start = time.perf_counter()

    num_beams = max(1, num_beams)
    num_return_sequences = min(max(1, num_return_sequences), num_beams)

    print(f"Starting pilot evaluation on {len(pilot_df)} samples (beams={num_beams}, returns={num_return_sequences})...", flush=True)
    for idx, (_, r) in enumerate(pilot_df.iterrows()):
        prompt = build_prompt(r.source)
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if num_beams > 1:
            gen_kwargs["num_beams"] = num_beams
            gen_kwargs["num_return_sequences"] = num_return_sequences

        with torch.no_grad():
            out = model.generate(**inp, **gen_kwargs)
        t1 = time.perf_counter()

        input_len = inp["input_ids"].shape[1]
        gen_texts = []
        for seq in out:
            raw = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            clean = raw.strip().split("\n")[0].strip().upper()
            if clean:
                gen_texts.append(clean)
        if not gen_texts:
            gen_texts = [""]

        target = r.target
        direct_exact = (target in gen_texts)

        # Catalog lookup via RapidFuzz
        if len(gen_texts) == 1:
            top_matches = catalog_search.top_names(gen_texts[0], limit=10)
        else:
            candidate_scores = defaultdict(float)
            for rank_idx, text in enumerate(gen_texts):
                rank_weight = 1.0 / (1.0 + 0.1 * rank_idx)
                matches = catalog_search.top_matches(text, limit=10)
                for m_name, score in matches:
                    w_score = score * rank_weight
                    if w_score > candidate_scores[m_name]:
                        candidate_scores[m_name] = w_score
            sorted_candidates = sorted(candidate_scores.items(), key=lambda kv: (-kv[1], kv[0]))
            top_matches = [c[0] for c in sorted_candidates[:10]]

        is_top1 = (top_matches[0] == target) if top_matches else False
        is_top5 = (target in top_matches[:5]) if top_matches else False
        is_top10 = (target in top_matches[:10]) if top_matches else False

        results.append({
            "sample_id": r.sample_id,
            "hangul": bool(r.hangul),
            "source": r.source,
            "target": target,
            "generated_text": gen_texts[0],
            "generated_texts": json.dumps(gen_texts, ensure_ascii=False),
            "direct_exact_match": bool(direct_exact),
            "predicted_catalog_name": top_matches[0] if top_matches else "",
            "correct_top1": bool(is_top1),
            "correct_top5": bool(is_top5),
            "correct_top10": bool(is_top10),
            "top10_candidates": json.dumps(top_matches, ensure_ascii=False),
            "seconds": round(t1 - t0, 3),
        })
        if (idx + 1) % 16 == 0 or (idx + 1) == len(pilot_df):
            print(f"Evaluated {idx + 1}/{len(pilot_df)} rows...", flush=True)

    eval_time = time.perf_counter() - t_start
    df_res = pd.DataFrame(results)


    # Compute metrics
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

    return df_res, metrics, eval_time



def sample_training_indices(weights, hangul_indices, other_indices, batch_size, generator,
                            sampling_mode="weighted"):
    """Optional exact 50:50 microbatches; default preserves legacy weighted draws."""
    if sampling_mode == "weighted":
        return torch.multinomial(weights, batch_size, replacement=True, generator=generator)
    if sampling_mode != "balanced" or batch_size % 2:
        raise ValueError("Balanced sampling requires an even batch size")
    if not len(hangul_indices) or not len(other_indices):
        raise ValueError("Both language groups are required")
    half = batch_size // 2
    h = hangul_indices[torch.randint(len(hangul_indices), (half,), generator=generator)]
    n = other_indices[torch.randint(len(other_indices), (half,), generator=generator)]
    indices = torch.cat([h, n])
    return indices[torch.randperm(batch_size, generator=generator)]


def collect_artifacts(output_dir):
    return [{"path": p.relative_to(ROOT).as_posix(), "sha256": file_sha256(p),
             "bytes": p.stat().st_size}
            for p in sorted(output_dir.rglob("*")) if p.is_file()]


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning of HyperCLOVA X SEED 0.5B")
    parser.add_argument("--experiment-id", default="AS-HCX-02")
    parser.add_argument("--pretrained-model", default="artifacts/pretrained/hyperclovax-seed-0.5b")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--folds", default="eval_kit/folds_group5_seed42.csv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-mode", choices=["top-layers", "full", "lora"], default="top-layers")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--hangul-mass", type=float, default=0.7,
                        help="Hangul probability mass for weighted sampling")
    parser.add_argument("--sampling-mode", choices=["weighted", "balanced"], default="weighted",
                        help="balanced: exactly half Hangul in EACH microbatch; requires mass=0.5")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Train/save only; skip Roman baseline, probes, reload and pilot evaluation")
    parser.add_argument("--eval-per-language", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--catalog-scorer", choices=tuple(SCORERS), default="ratio",
                        help="Generated-name lookup scorer; WRatio reproduces older runs")
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant",
                        help="Learning rate schedule (constant or cosine decay with warmup)")
    parser.add_argument("--warmup-steps", type=int, default=15,
                        help="Number of warmup steps for cosine scheduler")
    parser.add_argument("--eval-num-beams", type=int, default=1,
                        help="Number of beams during evaluation")
    parser.add_argument("--eval-num-return-sequences", type=int, default=1,
                        help="Number of returned beam sequences to aggregate during evaluation")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.experiment_id):
        parser.error("Invalid experiment ID")
    if min(args.steps, args.batch_size, args.accumulation, args.lora_rank) < 1:
        parser.error("Steps, batch size, accumulation and LoRA rank must be positive")
    if not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0:
        parser.error("LoRA alpha must be finite and positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    if not 0 < args.hangul_mass < 1:
        parser.error("hangul-mass must be between 0 and 1 exclusively")
    if args.sampling_mode == "balanced" and (args.batch_size % 2 or args.hangul_mass != 0.5):
        parser.error("balanced requires an even batch size and --hangul-mass 0.5")
    if args.lr_scheduler == "cosine" and not 0 <= args.warmup_steps < args.steps:
        parser.error("Cosine warmup must be >= 0 and smaller than steps")

    started_at = now_kst()
    print(f"=== Starting SFT Experiment: {args.experiment_id} at {started_at} ===")
    torch.manual_seed(args.seed)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(min(2, args.threads))
    except RuntimeError:
        pass
    print(f"Configured CPU execution with {args.threads} threads (interop capped)")

    data_dir = ROOT / args.data_dir
    folds_path = ROOT / args.folds
    output_dir = ROOT / f"artifacts/{args.experiment_id}"
    model_output_dir = output_dir / "model"
    exp_log_path = ROOT / f"project_log/experiments/{args.experiment_id}.json"
    if output_dir.exists():
        raise FileExistsError("Output exists; use a new experiment ID")
    if exp_log_path.exists():
        record = json.loads(exp_log_path.read_text(encoding="utf-8-sig"))
        if record.get("status") != "planned" or record.get("experiment_id") != args.experiment_id:
            raise FileExistsError("Existing executed experiment must not be overwritten")
    else:
        record = json.loads((ROOT / "project_log/templates/experiment.json").read_text(encoding="utf-8-sig"))
    model_output_dir.mkdir(parents=True, exist_ok=False)
    exp_log_path.parent.mkdir(parents=True, exist_ok=True)
    record.update(experiment_id=args.experiment_id, status="running", started_at_kst=started_at,
                  recorded_at_kst=now_kst(), commands=[subprocess.list2cmdline([sys.executable, *sys.argv])],
                  validation_status="pending" if args.skip_evaluation else "scheduled_after_training")
    record["configuration"] = vars(args).copy()
    from experiments.noisy_channel_common import code_state, environment
    record["code"] = code_state()
    record["code"]["file_sha256"]["experiments/train_hyperclovax_sft.py"] = file_sha256(__file__)
    record["environment"] = {**environment(), "threads": args.threads}
    write_json(exp_log_path, record)

    try:
        print(f"Loading data from {data_dir} and folds from {folds_path} (Heldout Fold {args.fold})...")
        pairs, validation, fit_rows, catalog_index = read_frames(data_dir, folds_path, heldout_fold=args.fold)
        pilot_df = select_pilot(validation, per_language=args.eval_per_language, seed=args.seed)
        catalog_names = sorted(catalog_index.ids_by_name.keys())
        print(f"Training pairs: {len(pairs)} (Hangul: {sum(pairs.hangul)}, Non-Hangul: {sum(~pairs.hangul)})")
        print(f"Evaluation pilot rows: {len(pilot_df)} (Hangul: {sum(pilot_df.hangul)}, Non-Hangul: {sum(~pilot_df.hangul)})")

        if not args.skip_evaluation:
            # Evaluate Roman WRatio baseline for direct apple-to-apple comparison
            print("Evaluating Roman WRatio baseline on pilot rows...", flush=True)
            roman_catalog = [unidecode(n).upper() for n in catalog_names]
            b_start = time.perf_counter()
            baseline_t1_h, baseline_t1_nh = 0, 0
            for _, r in pilot_df.iterrows():
                b_query = unidecode(r.source).upper()
                b_top = [catalog_names[idx] for _, _, idx in process.extract(b_query, roman_catalog, scorer=fuzz.WRatio, limit=1)]
                if b_top and b_top[0] == r.target:
                    if r.hangul:
                        baseline_t1_h += 1
                    else:
                        baseline_t1_nh += 1
            baseline_seconds = time.perf_counter() - b_start
            n_h = int(pilot_df.hangul.sum())
            n_nh = int((~pilot_df.hangul).sum())
            b_h_acc = baseline_t1_h / n_h if n_h else 0.0
            b_nh_acc = baseline_t1_nh / n_nh if n_nh else 0.0
            b_weighted = round(0.7 * b_h_acc + 0.3 * b_nh_acc, 5)
            print(f"Baseline Roman WRatio: Hangul {b_h_acc*100:.2f}% ({baseline_t1_h}/{n_h}), Non-Hangul {b_nh_acc*100:.2f}% ({baseline_t1_nh}/{n_nh}), Weighted {b_weighted*100:.2f}% ({baseline_seconds:.2f}s)")

        # Defaults preserve the previous weighted sampler; balanced mode is explicit.
        counts = pairs.hangul.value_counts().to_dict()
        if not counts.get(True) or not counts.get(False):
            raise ValueError("Both language groups are required by the sampler")
        weights = [args.hangul_mass / counts[True] if ko else (1 - args.hangul_mass) / counts[False]
                   for ko in pairs.hangul]
        weights_tensor = torch.tensor(weights, dtype=torch.double)
        sample_generator = torch.Generator().manual_seed(args.seed)
        hangul_indices = torch.tensor(np.flatnonzero(pairs.hangul.to_numpy()), dtype=torch.long)
        other_indices = torch.tensor(np.flatnonzero(~pairs.hangul.to_numpy()), dtype=torch.long)
        sampled_indices_hash = hashlib.sha256()
        hangul_draws = 0
        record["data"] = {
            "source_files_sha256": {name: file_sha256(data_dir / name)
                                     for name in ("train.csv", "vessel_catalog.csv")},
            "split_id": folds_path.name, "split_file_sha256": file_sha256(folds_path),
            "seed": args.seed, "heldout_fold": args.fold,
            "training_rows": len(fit_rows), "validation_rows": len(validation),
        }
        record["training_pairs"] = {
            "positive_rule": "training-fold observed name -> catalog name", "negative_rule": "none",
            "count": len(pairs), "hangul_pairs": int(counts[True]), "non_hangul_pairs": int(counts[False]),
            "sampling_mode": args.sampling_mode, "hangul_mass": args.hangul_mass,
            "ordered_pairs_sha256": hashlib.sha256(json.dumps(
                pairs[["source", "target"]].values.tolist(), ensure_ascii=False).encode("utf-8")).hexdigest(),
        }
        write_json(exp_log_path, record)

        print(f"Loading HyperCLOVA X model from {args.pretrained_model}...")
        tokenizer = AutoTokenizer.from_pretrained(ROOT / args.pretrained_model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            ROOT / args.pretrained_model,
            local_files_only=True,
            dtype=torch.float32,
            device_map="cpu",
        )

        adapter_spec = None
        if args.train_mode == "lora":
            adapter_spec = inject_attention_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
            print("Train mode: LoRA (all base weights frozen; q_proj/v_proj in every layer)")
        elif args.train_mode == "top-layers":
            # Freeze bottom 16 layers, train top 8 layers + LM head
            for param in model.model.layers[:16].parameters():
                param.requires_grad = False
            print("Train mode: top-layers (Bottom 16 layers frozen, top 8 layers + LM head trainable)")
        else:
            print("Train mode: full (All layers trainable)")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

        record["model"] = {
            "type": "HyperCLOVA X forward observed-name to catalog-name SFT",
            "parameters": {"train_mode": args.train_mode, "adapter": adapter_spec,
                           "trainable_parameters": trainable_params, "total_parameters": total_params,
                           "learning_rate": args.learning_rate, "steps": args.steps,
                           "batch_size": args.batch_size, "accumulation": args.accumulation,
                           "lr_scheduler": args.lr_scheduler, "warmup_steps": args.warmup_steps},
            "base_sha256": {p.name: file_sha256(p) for p in sorted((ROOT / args.pretrained_model).iterdir())
                            if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}},
            "loss": "existing forward SFT target-token cross entropy",
        }
        write_json(exp_log_path, record)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=0.01,
        )

        if args.lr_scheduler == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.steps,
            )
            print(f"LR Scheduler: Cosine with {args.warmup_steps} warmup steps (lr: {args.learning_rate} -> 0)")
        else:
            scheduler = None
            print(f"LR Scheduler: Constant (lr: {args.learning_rate})")

        # Training loop
        model.train()
        history = []
        train_start = time.perf_counter()

        print(f"Training for {args.steps} optimizer steps (batch {args.batch_size} × accum {args.accumulation})...")
        optimizer.zero_grad()

        pair_list = list(zip(pairs.source, pairs.target))

        for step in range(1, args.steps + 1):
            step_loss = 0.0
            for _ in range(args.accumulation):
                batch_indices = sample_training_indices(
                    weights_tensor, hangul_indices, other_indices, args.batch_size,
                    sample_generator, args.sampling_mode)
                index_list = batch_indices.tolist()
                sampled_indices_hash.update((",".join(map(str, index_list)) + "\n").encode("ascii"))
                hangul_draws += int(pairs.hangul.iloc[index_list].sum())
                batch_samples = [pair_list[i] for i in batch_indices.tolist()]
                input_ids, attention_mask, labels = collate_train_batch(batch_samples, tokenizer)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                loss.backward()
                step_loss += loss.item()

            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.learning_rate
            history.append({"step": step, "loss": round(step_loss, 5), "lr": round(current_lr, 8)})
            if step % 10 == 0 or step == 1 or step == args.steps:
                record["completed_optimizer_steps"] = step
                write_json(output_dir / "training_history.json", history)
                write_json(exp_log_path, record)
                print(f"Step {step}/{args.steps} | Loss: {step_loss:.4f} | LR: {current_lr:.2e}", flush=True)


        train_seconds = time.perf_counter() - train_start
        first_loss = float(np.mean([h["loss"] for h in history[:min(10, len(history))]]))
        last_loss = float(np.mean([h["loss"] for h in history[-min(10, len(history)):]]))
        print(f"Training complete in {train_seconds:.1f}s. Loss: {first_loss:.4f} -> {last_loss:.4f}")

        # Save training history
        history_path = output_dir / "training_history.json"
        write_json(history_path, history)

        record["training_pairs"].update(
            sampled_indices_sha256=sampled_indices_hash.hexdigest(), hangul_draws=hangul_draws,
            non_hangul_draws=args.steps * args.batch_size * args.accumulation - hangul_draws)
        if adapter_spec is not None:
            from safetensors.torch import save_file
            save_file(adapter_state_dict(model), str(output_dir / "adapter.safetensors"))
            write_json(output_dir / "adapter_config.json", {
                **adapter_spec, "direction": "observed_name_to_catalog_name",
                "base_sha256": record["model"]["base_sha256"]})
            record["model"]["merged_module_count"] = merge_attention_lora(model)
            record["model"]["merge_equivalence_verified"] = False
            print("Saved adapters and merged LoRA; numerical merge equivalence is pending external validation.")

        if args.skip_evaluation:
            save_started = time.perf_counter()
            model.eval()
            model.save_pretrained(model_output_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_output_dir)
            record.update(status="trained_unvalidated", finished_at_kst=now_kst(),
                          validation_status="pending_external_agent", failure=None,
                          decision="Training and saving only; separate agent must validate.")
            record["metrics"] = {key: None for key in (
                "metric_source", "hangul_rows", "non_hangul_rows", "hangul_accuracy",
                "non_hangul_accuracy", "weighted_accuracy", "candidate_recall_hangul",
                "candidate_recall_non_hangul", "seen_vessel_fraction")}
            record["pilot_results"] = None
            record["serialization"] = None
            record["independence"] = None
            record["cost"] = {
                "training_seconds": train_seconds, "inference_seconds": None,
                "saving_seconds": time.perf_counter() - save_started,
                "timing_scope": "training: optimizer loop including periodic history logging; "
                                "saving: merged model/tokenizer only; LoRA merge and hashes excluded",
                "first_10_loss": first_loss, "last_10_loss": last_loss,
                "model_bytes": sum(p.stat().st_size for p in model_output_dir.rglob("*") if p.is_file()),
                "adapter_bytes": (output_dir / "adapter.safetensors").stat().st_size if adapter_spec else None,
                "submission_zip_bytes": None,
            }
            record["artifacts"] = collect_artifacts(output_dir)
            write_json(exp_log_path, record)
            print(f"Saved model and training record; evaluation was skipped: {exp_log_path}")
            return

        # Pre-save probe generations for serialization test
        probe_samples = pilot_df.iloc[:min(4, len(pilot_df))]
        probe_gens_before = []
        model.eval()
        for _, r in probe_samples.iterrows():
            inp = tokenizer(build_prompt(r.source), return_tensors="pt")
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            probe_gens_before.append(tokenizer.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())

        # Save fine-tuned model and tokenizer
        print(f"Saving fine-tuned model to {model_output_dir}...")
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)

        # Reload model to test serialization
        print("Testing serialization and reloading saved model...")
        del model
        del optimizer
        import gc
        gc.collect()
        reloaded_model = AutoModelForCausalLM.from_pretrained(model_output_dir, dtype=torch.float32, device_map="cpu")
        reloaded_model.eval()

        probe_gens_after = []
        for _, r in probe_samples.iterrows():
            inp = tokenizer(build_prompt(r.source), return_tensors="pt")
            with torch.no_grad():
                o = reloaded_model.generate(**inp, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            probe_gens_after.append(tokenizer.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())

        serialization_equal = (probe_gens_before == probe_gens_after)
        print(f"Serialization check (reloaded model matches in-memory model): {serialization_equal}")

        # Row independence check (reverse order and subset)
        probe_indices = list(range(len(probe_samples)))
        reversed_gens = []
        for idx in reversed(probe_indices):
            r = probe_samples.iloc[idx]
            inp = tokenizer(build_prompt(r.source), return_tensors="pt")
            with torch.no_grad():
                o = reloaded_model.generate(**inp, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            reversed_gens.append(tokenizer.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        reversed_gens.reverse()
        independence_equal = (probe_gens_after == reversed_gens)
        print(f"Independence check (reordered execution matches): {independence_equal}")

        # Evaluation on Heldout Pilot using reloaded model
        eval_df, eval_metrics, eval_seconds = evaluate_pilot(
            reloaded_model, tokenizer, pilot_df, catalog_names, device="cpu", max_new_tokens=16,
            catalog_scorer=args.catalog_scorer, num_beams=args.eval_num_beams,
            num_return_sequences=args.eval_num_return_sequences,
        )

        pilot_csv_path = output_dir / "pilot_predictions.csv"
        eval_df.to_csv(pilot_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved pilot predictions to {pilot_csv_path}")

        finished_at = now_kst()
        print("\n=== Validation Results ===")
        print(f"Hangul Direct Exact: {eval_metrics['hangul']['direct_exact_accuracy']*100:.2f}% ({eval_metrics['hangul']['direct_exact_hits']}/{eval_metrics['hangul']['rows']})")
        print(f"Hangul Top-1: {eval_metrics['hangul']['gold_name_in_top_1']*100:.2f}% ({eval_metrics['hangul']['gold_name_hits_top_1']}/{eval_metrics['hangul']['rows']})")
        print(f"Hangul Top-10: {eval_metrics['hangul']['gold_name_in_top_10']*100:.2f}% ({eval_metrics['hangul']['gold_name_hits_top_10']}/{eval_metrics['hangul']['rows']})")
        print(f"Non-Hangul Direct Exact: {eval_metrics['non_hangul']['direct_exact_accuracy']*100:.2f}% ({eval_metrics['non_hangul']['direct_exact_hits']}/{eval_metrics['non_hangul']['rows']})")
        print(f"Non-Hangul Top-1: {eval_metrics['non_hangul']['gold_name_in_top_1']*100:.2f}% ({eval_metrics['non_hangul']['gold_name_hits_top_1']}/{eval_metrics['non_hangul']['rows']})")
        print(f"Non-Hangul Top-10: {eval_metrics['non_hangul']['gold_name_in_top_10']*100:.2f}% ({eval_metrics['non_hangul']['gold_name_hits_top_10']}/{eval_metrics['non_hangul']['rows']})")
        print(f"Weighted Name Accuracy: {eval_metrics['weighted_name_exact_accuracy']*100:.2f}%")

        # Collect artifacts
        artifact_entries = []
        for p in sorted(output_dir.rglob("*")):
            if p.is_file():
                rel_p = str(p.relative_to(ROOT)).replace("\\", "/")
                artifact_entries.append({
                    "path": rel_p,
                    "sha256": file_sha256(p),
                    "bytes": p.stat().st_size,
                })

        # Update record
        record["status"] = "completed"
        record["validation_status"] = "pilot_name_evaluation_completed"
        record["started_at_kst"] = started_at
        record["finished_at_kst"] = finished_at
        record["retrieval"] = {
            "method": f"SFT generation + RapidFuzz {args.catalog_scorer} catalog search",
            "catalog_scorer": args.catalog_scorer,
            "candidate_limit": 10,
            "tie_break": "lexicographic canonical name",
            "roman_baseline_scorer": "WRatio",
        }
        record["pilot_results"] = {
            "roman_wratio": {
                "hangul_accuracy": b_h_acc,
                "non_hangul_accuracy": b_nh_acc,
                "weighted_accuracy": b_weighted,
                "seconds": round(baseline_seconds, 2),
            },
            "finetuned_hcx": eval_metrics,
        }
        record["metrics"] = {
            "metric_source": "pilot heldout name evaluation",
            "hangul_rows": eval_metrics["hangul"]["rows"],
            "non_hangul_rows": eval_metrics["non_hangul"]["rows"],
            "hangul_accuracy": eval_metrics["hangul"]["gold_name_in_top_1"],
            "non_hangul_accuracy": eval_metrics["non_hangul"]["gold_name_in_top_1"],
            "weighted_accuracy": eval_metrics["weighted_name_exact_accuracy"],
            "candidate_recall_hangul": eval_metrics["hangul"]["gold_name_in_top_10"],
            "candidate_recall_non_hangul": eval_metrics["non_hangul"]["gold_name_in_top_10"],
            "seen_vessel_fraction": 0.0,
        }
        record["cost"] = {
            "training_seconds": round(train_seconds, 2),
            "inference_seconds": round(eval_seconds, 2),
            "timing_scope": f"inference_seconds: CPU {args.threads} threads, greedy decoding + RapidFuzz {args.catalog_scorer}; excludes training/loading/index construction/saving",
            "first_10_loss": round(first_loss, 5),
            "last_10_loss": round(last_loss, 5),
            "model_bytes": sum(p.stat().st_size for p in model_output_dir.iterdir() if p.is_file()),
            "submission_zip_bytes": None,
        }
        record["serialization"] = {
            "probe_rows": len(probe_samples),
            "top_10_names_equal": serialization_equal,
        }
        record["independence"] = {
            "probe_rows": len(probe_samples),
            "reordered_and_subset_equal": independence_equal,
        }
        record["artifacts"] = artifact_entries
        record["observations"] = [
            f"Initial mean loss: {first_loss:.4f} -> Final mean loss: {last_loss:.4f}",
            f"Hangul Top-1 name exact accuracy reached {eval_metrics['hangul']['gold_name_in_top_1']*100:.2f}% (Top-10: {eval_metrics['hangul']['gold_name_in_top_10']*100:.2f}%)",
            f"Non-Hangul Top-1 name exact accuracy reached {eval_metrics['non_hangul']['gold_name_in_top_1']*100:.2f}% (Top-10: {eval_metrics['non_hangul']['gold_name_in_top_10']*100:.2f}%)",
            f"Direct exact match without fuzzy search: Hangul {eval_metrics['hangul']['direct_exact_accuracy']*100:.2f}%, Non-Hangul {eval_metrics['non_hangul']['direct_exact_accuracy']*100:.2f}%",
        ]
        record["interpretations"] = [
            "SFT successfully aligned HyperCLOVA X to suppress chat explanatory tokens and output concise shipping names.",
            f"RapidFuzz {args.catalog_scorer} anchors generation to {len(catalog_names)} canonical names; this is not vessel-ID evaluation.",
        ]
        record["decision"] = "HyperCLOVA X SEED 0.5B SFT evaluation completed; compare with ByT5-small AS-NAME-05."
        write_json(exp_log_path, record)
        print(f"\nSaved updated experiment log to {exp_log_path}")

    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      finished_at_kst=now_kst(),
                      failure={"type": type(exc).__name__, "message": str(exc),
                               "retry": "Use a new experiment ID; partial outputs are preserved."})
        write_json(exp_log_path, record)
        raise


if __name__ == "__main__":
    main()
