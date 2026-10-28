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
                        default=ROOT / "artifacts/pretrained/lfm2.5-350m")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--folds-file", type=Path,
                        default=ROOT / "eval_kit/folds_group5_seed42.csv")
    parser.add_argument("--heldout-fold", type=int, choices=range(5), default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--loss-reduction", choices=["row", "token"], default="row")
    parser.add_argument("--sampling", choices=["cycle", "replacement"], default="cycle")
    parser.add_argument("--rank-beta", type=float, default=0.1)
    parser.add_argument("--rank-temperature", type=float, default=1.0)
    parser.add_argument("--confusions-csv", type=Path)
    parser.add_argument("--confusions-record", type=Path)
    parser.add_argument("--negative-count", type=int, default=2)
    parser.add_argument("--learned-hangul", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--checkpoint-selection',choices=['internal-best','last'],default='internal-best')
    parser.add_argument('--dev-fraction',type=float,default=0.05)
    parser.add_argument('--dev-max-vessels',type=int,default=128,help='0=no cap; cap is vessel count, not pair count')
    parser.add_argument('--dev-every',type=int,default=25)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=15)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument(
        "--lora-targets",
        default="q_proj,v_proj",
        help="Comma-separated Linear projection names (e.g. q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3)",
    )
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
    if not math.isfinite(args.augment_factor) or args.augment_factor < 0:
        parser.error("augment-factor must be >= 0")
    if not math.isfinite(args.rank_beta) or args.rank_beta < 0:
        parser.error("rank-beta must be finite and nonnegative")
    if not math.isfinite(args.rank_temperature) or args.rank_temperature <= 0:
        parser.error("rank-temperature must be finite and positive")
    if args.negative_count < 1:
        parser.error("negative-count must be positive")
    if not math.isfinite(args.dev_fraction) or not 0 < args.dev_fraction < 1:
        parser.error('dev-fraction must be finite and between0 and1')
    if args.dev_max_vessels<0 or args.dev_max_vessels==1 or args.dev_every<1:
        parser.error('dev-max-vessels must be0 or >=2; dev-every must be positive')
    if bool(args.confusions_csv)!=bool(args.confusions_record):
        parser.error('confusions-csv and confusions-record must be supplied together')
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
        collate_channel_examples, encode_channel_example, prepare_channel_pairs, load_channel_tokenizer,
    )
    from name_decoder.noise_augmenter import generate_augmented_pairs

    from name_decoder.channel_training import (
        LanguageCycleSampler, mine_name_negatives, sequence_nll, use_inner_confusions,
        channel_objective, learned_hangul_variants,
    )
    from name_decoder.internal_dev import (
        split_internal_rows, crossfit_confusions, grouped_loss, evaluate_dev, restore_adapter,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU fallback is disabled")
    device = torch.device(args.device)
    record["environment"].update(device=str(device), cuda=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
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
    outer_fit_rows = fit_rows
    dev_rows = fit_rows.iloc[:0].copy()
    if args.checkpoint_selection == 'internal-best':
        fit_rows, dev_rows = split_internal_rows(
            fit_rows,args.dev_fraction,args.dev_max_vessels,args.seed)
    split_info = {
        'training_sample_ids':sorted(fit_rows.sample_id),
        'dev_sample_ids':sorted(dev_rows.sample_id),
        'training_vessel_ids':sorted(fit_rows.target_vessel_id.unique()),
        'dev_vessel_ids':sorted(dev_rows.target_vessel_id.unique()),
        'seed':args.seed,'fraction_requested':args.dev_fraction,
        'max_dev_vessels':args.dev_max_vessels,
    }
    write_json(output_dir/'internal_split.json',split_info)
    record['checkpoint_selection'] = {
        'method':args.checkpoint_selection,'dev_every':args.dev_every,
        'split_sha256':sha256(output_dir/'internal_split.json'),
        'dev_rows':len(dev_rows),'dev_vessels':len(split_info['dev_vessel_ids']),
        'actual_dev_vessel_fraction':len(split_info['dev_vessel_ids'])/outer_fit_rows.target_vessel_id.nunique(),
        'best_step':None,'best_dev_loss':None,
        'criterion':'0.7/0.3 (or hangul-mass) language weighted SFT+rank; matching row/token SFT reduction',
        'dev_augmentation':False,'external_heldout_used_for_selection':False,
    }
    # Fit mappings/augmentation/mining only AFTER the vessel-group split.
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
    learned_count = 0
    if args.learned_hangul:
        learned, learned_count = learned_hangul_variants(raw_pairs)
        extra = pd.DataFrame(learned, columns=["catalog_name", "observed_name"])
        extra["hangul"] = extra.observed_name.map(has_hangul)
        pairs = pd.concat([pairs, extra], ignore_index=True).drop_duplicates(
            ["catalog_name", "observed_name"]).reset_index(drop=True)
        n_augmented = len(pairs) - n_raw_pairs
    # Reject synthetic names colliding with another fit-known catalog identity.
    if len(pairs) > n_raw_pairs:
        known = raw_pairs.groupby("observed_name").catalog_name.agg(set).to_dict()
        tail = pairs.iloc[n_raw_pairs:]
        keep = [not (known.get(r.observed_name, set()) - {r.catalog_name})
                for r in tail.itertuples(index=False)]
        pairs = pd.concat([pairs.iloc[:n_raw_pairs], tail.loc[keep]], ignore_index=True)
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
        "negative_rule": "public catalog lexical confusions; exclude same name and fit-known positives",
        "count": len(pairs),
        "raw_pairs_count": n_raw_pairs,
        "augmented_pairs_count": n_augmented,
        "augment_factor": args.augment_factor,
        "hangul_pairs": n_hangul, "non_hangul_pairs": n_other,
        "ordered_pairs_sha256": hashlib.sha256(pair_bytes).hexdigest(),
        "sampling": args.sampling, "learned_hangul_token_count": learned_count,
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
    record["hypothesis"] = record.get("hypothesis") or "A reverse name likelihood may add candidate-ranking information."
    record["changed_factors"] = record.get("changed_factors") or ["lexical confusion listwise ranking", "language cycle sampling",
                                  "row-mean CE", "fit-learned Hangul augmentation", "explicit CUDA placement",
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
    tokenizer = load_channel_tokenizer(args.pretrained)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = [encode_channel_example(tokenizer, row.catalog_name, row.observed_name,
                                       args.max_length)
                for row in pairs.itertuples(index=False)]
    negatives = mine_name_negatives(pairs, catalog, args.negative_count) if args.rank_beta else [[] for _ in examples]
    if args.confusions_csv and args.rank_beta:
        if not args.confusions_record:
            raise ValueError('confusions-record is required with confusions-csv')
        provenance = json.loads(args.confusions_record.read_text(encoding='utf-8-sig'))
        if (provenance.get('status') != 'mined_untrained' or
            provenance['data']['external_heldout_fold'] != args.heldout_fold or
            provenance['data']['fold_sha256'] != hashes['folds'] or
            provenance['artifact_sha256']['confusions.csv'] != sha256(args.confusions_csv)):
            raise ValueError('Confusion mining provenance/hash mismatch')
        if args.checkpoint_selection == 'internal-best':
            catalog_df = pd.read_csv(args.data_dir/'vessel_catalog.csv',keep_default_na=False,dtype={'vessel_id':str})
            mined, mining_audit = crossfit_confusions(fit_rows,catalog_df,catalog)
            write_json(output_dir/'miner_fit_audit.json',mining_audit)
            mined.to_csv(output_dir/'internal_confusions.csv',index=False)
            record['training_pairs'].update(
                mining_policy='recomputed after internal vessel split; supplied outer-fit CSV NOT reused',
                internal_confusions_sha256=sha256(output_dir/'internal_confusions.csv'),
                miner_fit_audit_sha256=sha256(output_dir/'miner_fit_audit.json'))
            print('Internal-best: rebuilding miners without dev vessels; original CSV not reused',flush=True)
        else:
            mined = pd.read_csv(args.confusions_csv, keep_default_na=False,
                                dtype={'sample_id':str,'negative_vessel_id':str})
        negatives, mining_stats = use_inner_confusions(pairs, catalog, fit_rows, mined, negatives)
        record['training_pairs'].update(negative_rule='inner-cross-fitted AR06 confusions, exact original pair; lexical fallback',
            confusions_sha256=sha256(args.confusions_csv),
            confusions_record_sha256=sha256(args.confusions_record), **mining_stats)
        write_json(output_dir / 'negative_names.json', negatives)
        record['training_pairs']['negative_names_sha256'] = sha256(output_dir / 'negative_names.json')
        write_json(record_path, record)
    groups = [[positive] + [encode_channel_example(tokenizer, name, row.observed_name,
               args.max_length) for name in negative]
              for positive, negative, row in zip(examples, negatives, pairs.itertuples(index=False))]

    dev_groups=[]
    dev_hangul=[]
    if len(dev_rows):
        dev_pairs=prepare_channel_pairs(dev_rows,catalog)  # Original held-out observations only.
        dev_negatives=mine_name_negatives(dev_pairs,catalog,args.negative_count) if args.rank_beta else [[] for _ in range(len(dev_pairs))]
        dev_groups=[[encode_channel_example(tokenizer,name,row.observed_name,args.max_length)
                     for name in [row.catalog_name]+negative]
                    for row,negative in zip(dev_pairs.itertuples(index=False),dev_negatives)]
        dev_hangul=dev_pairs.hangul.tolist()
        write_json(output_dir/'dev_negative_names.json',dev_negatives)
        write_json(output_dir/'dev_pairs.json',dev_pairs[['catalog_name','observed_name']].values.tolist())
        record['checkpoint_selection'].update(dev_pairs=len(dev_pairs),
            dev_pairs_sha256=sha256(output_dir/'dev_pairs.json'),
            dev_negatives_sha256=sha256(output_dir/'dev_negative_names.json'),
            dev_negative_policy='fixed catalog lexical candidates; no fitted dev miner')

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager").to(device)
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("max-length exceeds model position capacity")
    model.config.use_cache = False
    lora_targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
    adapter_spec = inject_attention_lora(
        model, rank=args.rank, alpha=args.alpha, targets=lora_targets
    )
    write_json(output_dir / "adapter_config.json", {
        **adapter_spec, "base_path": str(args.pretrained.resolve()),
        "base_sha256": base_hashes, "direction": "catalog_name_to_observed_name",
    })
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.steps)
    record["model"] = {
        "type": f"{model.config.model_type} causal LM with plain PyTorch linear LoRA",
        "base_path": str(args.pretrained.resolve()), "base_sha256": base_hashes,
        "parameters": {**adapter_spec, "steps": args.steps, "batch_size": args.batch_size,
                       "accumulation": args.accumulation, "learning_rate": args.learning_rate,
                       "scheduler": "warmup_cosine", "warmup_steps": args.warmup_steps,
                       "weight_decay": 0.01, "gradient_clip_norm": 1.0},
        "loss": {"sft_reduction": args.loss_reduction, "rank_beta": args.rank_beta,
                 "rank_temperature": args.rank_temperature, "negative_count": args.negative_count},
    }
    record["cost"]["preparation_seconds"] = time.perf_counter() - started
    write_json(record_path, record)
    weights = torch.tensor(np.where(hangul, args.hangul_mass / n_hangul,
                                    (1 - args.hangul_mass) / n_other), dtype=torch.double)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    sampler = LanguageCycleSampler(hangul, args.hangul_mass, args.seed)
    draw_counts = [0] * len(pairs)
    sample_digest = hashlib.sha256()
    hangul_draws = 0
    history = []
    best_dev_loss = None
    best_step = None
    best_adapter_state = None
    internal_dev_seconds = 0.0
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
                                        generator=generator).tolist() if args.sampling == "replacement" else sampler.draw(args.batch_size)
            for index in indices:
                draw_counts[index] += 1
            sample_digest.update((",".join(map(str, indices)) + "\n").encode("ascii"))
            hangul_draws += sum(bool(hangul[index]) for index in indices)
            batch = collate_channel_examples([example for index in indices for example in groups[index]],
                                             tokenizer.pad_token_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, use_cache=False)
            nll = sequence_nll(outputs.logits, batch["labels"])
            group_size = len(groups[indices[0]])
            loss = grouped_loss(nll,batch['labels'],group_size,args.rank_beta,
                                args.rank_temperature,args.loss_reduction)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at optimizer step {step}")
            (loss / args.accumulation).backward()
            step_loss += loss.detach().item() / args.accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if device.type == "cpu":
            time.sleep(0.02)

        dev_loss = None
        if dev_groups and (step % args.dev_every == 0 or step == args.steps):
            dev_started=time.perf_counter()
            dev_loss=evaluate_dev(model,dev_groups,dev_hangul,tokenizer.pad_token_id,
                                  args.batch_size,device,args.rank_beta,args.rank_temperature,
                                  args.loss_reduction,args.hangul_mass)
            if best_dev_loss is None or dev_loss < best_dev_loss:
                best_dev_loss=dev_loss;best_step=step
                best_adapter_state=adapter_state_dict(model)
                save_file(best_adapter_state,str(output_dir/'best_adapter.safetensors'))
            internal_dev_seconds+=time.perf_counter()-dev_started
            record['checkpoint_selection'].update(best_step=best_step,best_dev_loss=best_dev_loss)
            print(f'internal dev step={step} loss={dev_loss:.6f} best={best_step}',flush=True)

        history.append({"step": step, "loss": step_loss, "learning_rate_used": lr_used,
                        "gradient_norm_before_clip": float(grad_norm),
                        "dev_loss":dev_loss,"best_dev_loss": best_dev_loss, "best_step": best_step})
        record["completed_optimizer_steps"] = step
        if step == 1 or step % 10 == 0 or step == args.steps:
            write_json(output_dir / "training_history.json", history)
            write_json(record_path, record)
            print(f"step={step}/{args.steps} loss={step_loss:.6f} lr={lr_used:.8g}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        record["cost"]["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    record["training_pairs"]["coverage"] = {
        "unique_pairs_seen": sum(count > 0 for count in draw_counts),
        "hangul_unique_seen": sum(count > 0 and bool(hangul[i]) for i, count in enumerate(draw_counts)),
        "non_hangul_unique_seen": sum(count > 0 and not hangul[i] for i, count in enumerate(draw_counts)),
        "max_repetitions": max(draw_counts),
        "target_tokens_seen": sum(count * examples[i].target_count for i, count in enumerate(draw_counts)),
    }
    record['cost']['optimization_loop_seconds'] = time.perf_counter()-optimization_started
    record['cost']['internal_dev_seconds'] = internal_dev_seconds
    record["cost"]["training_seconds"] = record['cost']['optimization_loop_seconds']-internal_dev_seconds
    record["training_pairs"].update(
        sampled_indices_sha256=sample_digest.hexdigest(), hangul_draws=hangul_draws,
        non_hangul_draws=args.steps * args.batch_size * args.accumulation - hangul_draws)

    saving_started = time.perf_counter()
    if args.checkpoint_selection == 'internal-best':
        if best_adapter_state is None:
            raise RuntimeError('No finite internal best checkpoint available')
        restore_adapter(model,best_adapter_state)
        print(f'Restored best adapter step={best_step} dev_loss={best_dev_loss:.6f}',flush=True)
    record['checkpoint_selection'].update(saved_step=best_step if dev_groups else args.steps)

    save_file(adapter_state_dict(model), str(output_dir / "adapter.safetensors"))

    model.to("cpu")
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
        "checkpoint_selection":record["checkpoint_selection"],
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
        timing_scope="preparation includes hashes/data/tokenization/load; training is optimizer loop excluding internal dev/snapshot blocks; internal_dev_seconds includes validation and best snapshots; "
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
