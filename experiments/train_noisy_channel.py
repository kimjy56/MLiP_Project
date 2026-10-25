"""Train catalog-name -> observed-name LoRA; never evaluate or build a ZIP.

Run with the project's .venv-server Python. All models/data are local. Defaults
are a proposed research configuration, not an established performance result.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
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
    parser.add_argument("--pretrained", type=Path,
                        default=ROOT / "artifacts/pretrained/hyperclovax-seed-0.5b")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--folds-file", type=Path,
                        default=ROOT / "eval_kit/folds_group5_seed42.csv")
    parser.add_argument("--heldout-fold", type=int, choices=range(5), default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=15)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--hangul-mass", type=float, default=0.7)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--augment-factor", type=float, default=0.5,
                        help="Ratio of synthetic noise pairs to generate (default 0.5; 0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.accumulation, args.rank, args.max_length) < 1:
        parser.error("steps/batch-size/accumulation/rank/max-length must be positive")
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("warmup-steps must be >= 0 and smaller than steps")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("alpha must be finite and positive")
    if not 0 < args.hangul_mass < 1:
        parser.error("hangul-mass must be between 0 and 1 exclusively")
    if args.augment_factor < 0:
        parser.error("augment-factor must be >= 0")
    return args


def train(args, output_dir, record_path, record):
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from name_decoder.catalog import has_hangul
    from name_decoder.data import read_frames
    from name_decoder.lora import adapter_state_dict, inject_attention_lora, merge_attention_lora
    from name_decoder.noisy_channel import (
        CATALOG_NORMALIZATION, MANIFEST_NAME, OBSERVED_NORMALIZATION, PROMPT_VERSION,
        collate_channel_examples, encode_channel_example, prepare_channel_pairs,
    )
    from name_decoder.noise_augmenter import generate_augmented_pairs

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    hashes = source_hashes(args.data_dir, args.folds_file)
    _, validation, fit_rows, catalog = read_frames(
        args.data_dir, args.folds_file, args.heldout_fold)
    # All fitted pairs come from fold != heldout_fold. No validation sampler,
    # dictionary fitting, gold candidate injection, or checkpoint selection.
    raw_pairs = prepare_channel_pairs(fit_rows, catalog)
    n_raw_pairs = len(raw_pairs)
    pairs = raw_pairs

    # Apply deterministic domain-specific noise augmentation if requested
    n_augmented = 0
    if args.augment_factor > 0:
        aug_tuples = generate_augmented_pairs(raw_pairs, seed=args.seed, augment_factor=args.augment_factor)
        if aug_tuples:
            aug_df = pd.DataFrame(aug_tuples, columns=["catalog_name", "observed_name"])
            aug_df["hangul"] = aug_df.observed_name.map(has_hangul)
            pairs = pd.concat([raw_pairs, aug_df], ignore_index=True)
            pairs = pairs.drop_duplicates(["catalog_name", "observed_name"]).reset_index(drop=True)
            n_augmented = len(pairs) - n_raw_pairs
    hangul = pairs.hangul.to_numpy(dtype=bool)
    n_hangul = int(hangul.sum())
    n_other = len(pairs) - n_hangul
    if not n_hangul or not n_other:
        raise ValueError("Both language groups are required by the mixture sampler")
    pair_bytes = json.dumps(pairs[["catalog_name", "observed_name"]].values.tolist(),
                            ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    train_ids = sorted(fit_rows.target_vessel_id.unique().tolist())
    write_json(output_dir / "training_vessel_ids.json", train_ids)
    record["data"] = {
        "source_files_sha256": hashes, "split_id": args.folds_file.name,
        "split_file_sha256": hashes["folds"], "seed": args.seed,
        "heldout_fold": args.heldout_fold, "training_rows": len(fit_rows),
        "validation_rows": len(validation), "training_vessels": len(train_ids),
        "validation_is_independent_final_test": False,
        "training_vessel_ids_sha256": sha256(output_dir / "training_vessel_ids.json"),
    }
    record["training_pairs"] = {
        "positive_rule": "training-fold target ID -> catalog name; raw observed name normalized",
        "negative_rule": "none (conditional language modelling)",
        "count": len(pairs),
        "raw_pairs_count": n_raw_pairs,
        "augmented_pairs_count": n_augmented,
        "augment_factor": args.augment_factor,
        "hangul_pairs": n_hangul, "non_hangul_pairs": n_other,
        "ordered_pairs_sha256": hashlib.sha256(pair_bytes).hexdigest(),
        "sampling": "with replacement, uniform within each language group",
        "hangul_probability_mass": args.hangul_mass,
        "draws_planned": args.steps * args.batch_size * args.accumulation,
    }
    record["preprocessing"] = {
        "catalog": CATALOG_NORMALIZATION, "observed": OBSERVED_NORMALIZATION,
        "deduplicate": ["catalog_name", "observed_name"],
        "pair_sort_order": ["catalog_name", "observed_name"],
        "prompt_version": PROMPT_VERSION, "max_length": args.max_length,
        "truncation": "error", "prompt_labels": -100, "padding_labels": -100,
        "target": "separately tokenized observed name plus EOS",
    }
    record["hypothesis"] = "A reverse name likelihood may add candidate-ranking information."
    record["changed_factors"] = ["reverse objective", "all-layer q/v LoRA",
                                  "candidate learning rate 2e-4 (unless overridden)",
                                  "observation normalization preserves case and internal spaces"]
    record["limitations"] = [
        "Not a LoRA-only ablation against AS-HCX-04.",
        "No accuracy, merge equivalence, throughput or memory claim without separate verification.",
        "A name-only model cannot distinguish IDs with the same catalog name.",
        "Public train was previously explored; heldout is not an untouched final test.",
    ]
    record["environment"]["threads"] = args.threads
    write_json(record_path, record)

    if not args.pretrained.is_dir():
        raise FileNotFoundError(f"Local pretrained directory missing: {args.pretrained}")
    if (args.pretrained / MANIFEST_NAME).exists():
        raise ValueError("Start from the original pretrained base, not a reverse trained checkpoint")
    base_hashes = {path.name: sha256(path) for path in sorted(args.pretrained.iterdir())
                   if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}}
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = [encode_channel_example(tokenizer, row.catalog_name, row.observed_name,
                                       args.max_length)
                for row in pairs.itertuples(index=False)]
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager")
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("max-length exceeds model position capacity")
    model.config.use_cache = False
    adapter_spec = inject_attention_lora(model, rank=args.rank, alpha=args.alpha)
    write_json(output_dir / "adapter_config.json", {
        **adapter_spec, "base_path": str(args.pretrained.resolve()),
        "base_sha256": base_hashes, "direction": "catalog_name_to_observed_name",
    })
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.steps)
    record["model"] = {
        "type": "Llama causal LM with plain PyTorch q_proj/v_proj LoRA",
        "base_path": str(args.pretrained.resolve()), "base_sha256": base_hashes,
        "parameters": {**adapter_spec, "steps": args.steps, "batch_size": args.batch_size,
                       "accumulation": args.accumulation, "learning_rate": args.learning_rate,
                       "scheduler": "warmup_cosine", "warmup_steps": args.warmup_steps,
                       "weight_decay": 0.01, "gradient_clip_norm": 1.0},
        "loss": "token-mean cross entropy on observed name and EOS; mean of microbatch losses",
    }
    record["cost"]["preparation_seconds"] = time.perf_counter() - started
    write_json(record_path, record)
    weights = torch.tensor(np.where(hangul, args.hangul_mass / n_hangul,
                                    (1 - args.hangul_mass) / n_other), dtype=torch.double)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    sample_digest = hashlib.sha256()
    hangul_draws = 0
    history = []
    model.train()
    print(f"Training pairs={len(pairs)}; trainable={adapter_spec['trainable_parameters']:,}; "
          f"threads={args.threads}", flush=True)
    optimization_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        lr_used = optimizer.param_groups[0]["lr"]
        for _ in range(args.accumulation):
            indices = torch.multinomial(weights, args.batch_size, replacement=True,
                                        generator=generator).tolist()
            sample_digest.update((",".join(map(str, indices)) + "\n").encode("ascii"))
            hangul_draws += sum(bool(hangul[index]) for index in indices)
            batch = collate_channel_examples([examples[index] for index in indices],
                                             tokenizer.pad_token_id)
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at optimizer step {step}")
            (loss / args.accumulation).backward()
            step_loss += loss.detach().item() / args.accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        time.sleep(0.02)  # Thermal safety cooldown: yield CPU to prevent overheating
        history.append({"step": step, "loss": step_loss, "learning_rate_used": lr_used,
                        "gradient_norm_before_clip": float(grad_norm)})
        record["completed_optimizer_steps"] = step
        if step == 1 or step % 10 == 0 or step == args.steps:
            write_json(output_dir / "training_history.json", history)
            write_json(record_path, record)
            print(f"step={step}/{args.steps} loss={step_loss:.6f} lr={lr_used:.8g}", flush=True)
    record["cost"]["training_seconds"] = time.perf_counter() - optimization_started
    record["training_pairs"].update(
        sampled_indices_sha256=sample_digest.hexdigest(), hangul_draws=hangul_draws,
        non_hangul_draws=args.steps * args.batch_size * args.accumulation - hangul_draws)

    saving_started = time.perf_counter()
    save_file(adapter_state_dict(model), str(output_dir / "adapter.safetensors"))
    merged_count = merge_attention_lora(model)
    model.eval()
    model.config.use_cache = True
    model_dir = output_dir / "model"
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    checkpoint_hashes = {path.relative_to(model_dir).as_posix(): sha256(path)
                         for path in sorted(model_dir.rglob("*")) if path.is_file()}
    manifest = {
        "schema_version": 1, "experiment_id": args.experiment_id,
        "direction": "catalog_name_to_observed_name", "prompt_version": PROMPT_VERSION,
        "catalog_normalization": CATALOG_NORMALIZATION,
        "observed_normalization": OBSERVED_NORMALIZATION,
        "max_length": args.max_length, "eos_token_id": tokenizer.eos_token_id,
        "heldout_fold": args.heldout_fold, "data_sha256": hashes,
        "base_sha256": base_hashes, "adapter": adapter_spec,
        "merged": True, "merged_module_count": merged_count,
        "checkpoint_sha256": checkpoint_hashes,
        "validation_status": "pending_external_agent", "created_at_kst": now_kst(),
    }
    write_json(model_dir / MANIFEST_NAME, manifest)
    record["cost"].update(
        saving_seconds=time.perf_counter() - saving_started,
        total_seconds=time.perf_counter() - started,
        timing_scope="preparation includes hashes/data/tokenization/load; training is optimizer loop; "
                     "saving includes adapters/merge/tokenizer/checkpoint hashing",
        model_bytes=sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file()),
        adapter_bytes=(output_dir / "adapter.safetensors").stat().st_size,
        inference_seconds=None, submission_zip_bytes=None)
    record.update(status="trained_unvalidated", finished_at_kst=now_kst(), failure=None,
                  artifacts=[str(output_dir.resolve())],
                  validation_status="pending_external_agent",
                  decision="Do not promote until the separate validation agent reports.")
    record["artifact_sha256"] = {
        "adapter.safetensors": sha256(output_dir / "adapter.safetensors"),
        "adapter_config.json": sha256(output_dir / "adapter_config.json"),
        f"model/{MANIFEST_NAME}": sha256(model_dir / MANIFEST_NAME),
        "training_history.json": sha256(output_dir / "training_history.json"),
    }
    write_json(record_path, record)
    write_json(output_dir / "run_record.json", record)
    print(f"Saved merged reverse model: {model_dir}; validation is still pending.", flush=True)


def main():
    args = parse_args()
    configure_cpu(args.threads)
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    output_dir, record_path, record = reserve_experiment(
        args.experiment_id, ROOT / "artifacts", "Reverse name-channel LoRA training", command)
    try:
        train(args, output_dir, record_path, record)
    except BaseException as exc:
        record_failure(record_path, record, exc)
        raise


if __name__ == "__main__":
    main()
