"""SPEC-DPR-03 original-snapshot Stage-B-only contrastive training.

Training excludes outer-fold vessel IDs. The user-requested composite selection
uses that outer fold, so its results are explicitly development-selected.
Historical Stage-A implementations are preserved in source snapshots.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phonetic_encoder.dataset import PhoneticPairDataset, make_collate_fn, make_multiview_collate_fn
from phonetic_encoder.loss import HybridContrastiveLoss, SupConCosFaceHardLoss
from phonetic_encoder.model import CharPhoneticBiEncoder
from phonetic_encoder.retrieval import retrieval_metrics
from experiments.dpr03_common import (ROOT, start_run, finish_run, write_json, sha256,
                                      object_hash, load_fold_data, composite_score)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_retrieval(
    model: CharPhoneticBiEncoder,
    val_queries: List[str],
    val_targets: List[str],
    catalog_names: List[str],
    catalog_embeds: Optional[np.ndarray] = None,
    device: str = "cpu",
    batch_size: int = 256,
    catalog_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate Top-K Retrieval Recalls and MRR on heldout validation pairs.
    """
    model.eval()
    if catalog_embeds is None:
        catalog_embeds = model.encode_catalog(catalog_names, batch_size=batch_size)

    query_embeds = model.encode_queries(val_queries, batch_size=batch_size)

    # Matrix dot product (N_val, N_catalog)
    sims = np.dot(query_embeds, catalog_embeds.T)

    return retrieval_metrics(sims, catalog_ids if catalog_ids is not None else catalog_names, val_targets)


def mine_hard_negatives_for_dataset(
    model: CharPhoneticBiEncoder,
    train_pairs: List[Dict[str, Any]],
    catalog_names: List[str],
    catalog_ids: List[str],
    device: str = "cpu",
    k_sister: int = 2,
    k_global: int = 2,
) -> None:
    """
    Mine 2 sister-vessel negatives + 2 global ANCE confusion negatives per sample (SPEC-DPR-02 Section 2.4).
    """
    print("  [Mining] Encoding 78,595 catalog vessels for hard negative extraction...")
    model.eval()
    t0 = time.time()
    cat_embs = model.encode_catalog(catalog_names, batch_size=512)
    cat_tensor = torch.tensor(cat_embs, device=device)  # (78595, 128)

    # Prefix index for sister vessels (first word >= 3 chars)
    prefix_to_cat_indices: Dict[str, List[int]] = {}
    for idx, cname in enumerate(catalog_names):
        w0 = cname.upper().split()[0] if cname.strip() else ""
        if len(w0) >= 3:
            prefix_to_cat_indices.setdefault(w0, []).append(idx)

    # Extract unique queries
    unique_queries = sorted({p["query"] for p in train_pairs})
    positives_by_query = {}
    for p in train_pairs:
        positives_by_query.setdefault(p['query'], set()).add(p['catalog'])
    print(f"  [Mining] Encoding {len(unique_queries):,} unique training queries...")
    q_embs = model.encode_queries(unique_queries, batch_size=256)
    q_tensor = torch.tensor(q_embs, device=device)  # (N_q, 128)

    print("  [Mining] Computing global similarity matrix & extracting Top-15 confusions...")
    with torch.no_grad():
        all_sims = torch.matmul(q_tensor, cat_tensor.T)  # (N_q, 78595)
        top15_scores, top15_indices = torch.topk(all_sims, k=15, dim=1)

    top_confusions_by_q: Dict[str, List[int]] = {}
    for i, q in enumerate(unique_queries):
        top_confusions_by_q[q] = top15_indices[i].cpu().tolist()

    q_emb_map = {q: q_embs[i] for i, q in enumerate(unique_queries)}

    # Assign 4 negatives to each pair
    total_assigned = 0
    for pair in train_pairs:
        q_name = pair["query"]
        c_pos = pair["catalog"]
        v_pos = pair.get("vessel_id")
        q_vec = q_emb_map[q_name]

        # 1. Sister vessels (same leading prefix)
        w0 = c_pos.upper().split()[0] if c_pos.strip() else ""
        sister_indices = [
            idx for idx in prefix_to_cat_indices.get(w0, [])
            if catalog_ids[idx] != v_pos and catalog_names[idx] not in positives_by_query[q_name]
        ]
        sister_negs: List[str] = []
        if sister_indices:
            sister_vecs = cat_embs[sister_indices]
            s_sims = np.dot(sister_vecs, q_vec)
            top_s_local = np.argsort(-s_sims)[:k_sister]
            sister_negs = [catalog_names[sister_indices[loc_idx]] for loc_idx in top_s_local]

        # 2. Global ANCE negatives
        global_negs: List[str] = []
        for g_idx in top_confusions_by_q.get(q_name, []):
            if catalog_ids[g_idx] != v_pos and catalog_names[g_idx] not in positives_by_query[q_name] and catalog_names[g_idx] not in sister_negs + global_negs:
                global_negs.append(catalog_names[g_idx])
                if len(global_negs) >= k_global:
                    break

        # 3. Combine to ensure exactly 4 negatives
        combined = list(sister_negs) + list(global_negs)
        if len(combined) < 4:
            for g_idx in top_confusions_by_q.get(q_name, []):
                cand_name = catalog_names[g_idx]
                if catalog_ids[g_idx] != v_pos and cand_name not in positives_by_query[q_name] and cand_name not in combined:
                    combined.append(cand_name)
                    if len(combined) >= 4:
                        break
        if len(combined) < 4:
            for cand_name in catalog_names:
                if cand_name not in positives_by_query[q_name] and cand_name not in combined:
                    combined.append(cand_name)
                    if len(combined) == 4:
                        break
        if len(combined) < 4:
            raise ValueError('Fewer than four valid hard negatives')

        pair["hard_negs"] = combined[:4]
        total_assigned += len(pair["hard_negs"])

    elapsed = time.time() - t0
    print(f"  [Mining] Assigned {total_assigned:,} hard negatives across {len(train_pairs):,} samples in {elapsed:.1f}s.")


def train_one_epoch(
    model: CharPhoneticBiEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    cooldown: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()
        is_multiview = "query_primary" in batch

        if is_multiview:
            q_prim = {k: v.to(device) for k, v in batch["query_primary"].items()}
            q_sec = {k: v.to(device) for k, v in batch["query_secondary"].items()}
            q_tail = {k: v.to(device) for k, v in batch["query_tail"].items()}

            c_prim = {k: v.to(device) for k, v in batch["catalog_primary"].items()}
            c_sec = {k: v.to(device) for k, v in batch["catalog_secondary"].items()}
            c_tail = {k: v.to(device) for k, v in batch["catalog_tail"].items()}

            hard_negs_info = batch["hard_negs"]
            vessel_ids = batch.get("vessel_ids")
            B = batch["batch_size"]

            if use_amp:
                with torch.amp.autocast("cuda"):
                    q_emb = model.forward_multiview(q_prim, q_sec, q_tail)
                    c_emb = model.forward_multiview(c_prim, c_sec, c_tail)

                    neg_embs = None
                    if hard_negs_info is not None:
                        h_prim = {k: v.to(device) for k, v in hard_negs_info["primary"].items()}
                        h_sec = {k: v.to(device) for k, v in hard_negs_info["secondary"].items()}
                        h_tail = {k: v.to(device) for k, v in hard_negs_info["tail"].items()}
                        h_flat = model.forward_multiview(h_prim, h_sec, h_tail)
                        K = h_flat.shape[0] // B
                        neg_embs = h_flat.view(B, K, -1)

                    if isinstance(loss_fn, SupConCosFaceHardLoss):
                        loss = loss_fn(q_emb, c_emb, hard_neg_embs=neg_embs, vessel_ids=vessel_ids)
                    else:
                        loss = loss_fn(q_emb, c_emb, hard_neg_embs=neg_embs)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                q_emb = model.forward_multiview(q_prim, q_sec, q_tail)
                c_emb = model.forward_multiview(c_prim, c_sec, c_tail)

                neg_embs = None
                if hard_negs_info is not None:
                    h_prim = {k: v.to(device) for k, v in hard_negs_info["primary"].items()}
                    h_sec = {k: v.to(device) for k, v in hard_negs_info["secondary"].items()}
                    h_tail = {k: v.to(device) for k, v in hard_negs_info["tail"].items()}
                    h_flat = model.forward_multiview(h_prim, h_sec, h_tail)
                    K = h_flat.shape[0] // B
                    neg_embs = h_flat.view(B, K, -1)

                if isinstance(loss_fn, SupConCosFaceHardLoss):
                    loss = loss_fn(q_emb, c_emb, hard_neg_embs=neg_embs, vessel_ids=vessel_ids)
                else:
                    loss = loss_fn(q_emb, c_emb, hard_neg_embs=neg_embs)

                loss.backward()
                optimizer.step()
        else:
            q_tensors = {k: v.to(device) for k, v in batch["query"].items()}
            c_tensors = {k: v.to(device) for k, v in batch["catalog"].items()}
            hard_negs_info = batch["hard_negs"]
            neg_embs = None

            if use_amp:
                with torch.amp.autocast("cuda"):
                    q_emb = model.encoder(**q_tensors)
                    c_emb = model.encoder(**c_tensors)
                    if hard_negs_info is not None:
                        neg_batch = {k: v.to(device) for k, v in hard_negs_info["neg_batch"].items()}
                        neg_embs = model.encoder(**neg_batch)
                    loss = loss_fn(q_emb, c_emb, hard_neg_embs=neg_embs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                q_emb = model.encoder(**q_tensors)
                c_emb = model.encoder(**c_tensors)
                if hard_negs_info is not None:
                    neg_batch = {k: v.to(device) for k, v in hard_negs_info["neg_batch"].items()}
                    neg_embs = model.encoder(**neg_batch)
                loss = loss_fn(q_emb, c_emb, hard_neg_embs=neg_embs)
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if cooldown > 0:
            time.sleep(cooldown)

    return total_loss / max(1, num_batches)


def build_optimizer(model, backbone_lr=3e-5, new_lr=1.5e-4):
    # The actual SyllableConvBlock attribute is conv_block (not syllable_conv).
    tags = ('conv_block', 'syllable_conv', 'multi_query_pool', 'mq_proj', 'mq_gate', 'fusion')
    groups = {'backbone': [], 'new_modules': []}
    names = {'backbone': [], 'new_modules': []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = 'new_modules' if any(tag in name for tag in tags) else 'backbone'
            groups[key].append(parameter)
            names[key].append(name)
    if not all(groups.values()):
        raise ValueError('Both differential LR groups must be nonempty')
    optimizer = torch.optim.AdamW([
        {'params': groups['backbone'], 'lr': backbone_lr, 'weight_decay': 1e-4, 'name': 'backbone'},
        {'params': groups['new_modules'], 'lr': new_lr, 'weight_decay': 1e-4, 'name': 'new_modules'},
    ])
    return optimizer, {key: {'parameter_names': names[key], 'numel': sum(p.numel() for p in values)}
                       for key, values in groups.items()}


def main():
    parser = argparse.ArgumentParser(description='SPEC-DPR-03: original-snapshot Stage-B-only training')
    parser.add_argument('--experiment-id', default='DPR-SYMPHON-03')
    parser.add_argument('--train-csv', default='data/train.csv')
    parser.add_argument('--catalog-csv', default='data/vessel_catalog.csv')
    parser.add_argument('--folds-csv', default='eval_kit/folds_group5_seed42.csv')
    parser.add_argument('--epochs-pretrain', type=int, choices=[0], default=0, help='Stage A is disabled; never reuse its checkpoint')
    parser.add_argument('--epochs-finetune', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr-backbone', '--lr-finetune', dest='lr_backbone', type=float, default=3e-5)
    parser.add_argument('--lr-new', type=float, default=1.5e-4)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output-dir')
    parser.add_argument('--pretrained-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--cooldown', type=float, default=0.)
    args = parser.parse_args()
    if min(args.epochs_finetune, args.batch_size, args.threads) < 1:
        parser.error('Epochs, batch size and threads must be positive')
    out, record_path, record = start_run(args.experiment_id, args.output_dir)
    started = time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=args.threads)
        torch.set_num_threads(args.threads)
        set_seed(args.seed)
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_str)
        use_amp = device_str == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        record['environment'].update(torch=torch.__version__, device=device_str,
                                      gpu=torch.cuda.get_device_name(0) if use_amp else None,
                                      threads=args.threads, amp=use_amp)
        catalog, fit, val, data = load_fold_data(args.train_csv, args.catalog_csv, args.folds_csv, args.fold)
        data['seed'] = args.seed
        record['data'] = data
        write_json(out/'data_manifest.json', data)
        # Always initialize from the original public snapshot. No output-directory fallback.
        snapshots = sorted((Path.home()/'.cache/huggingface/hub/models--docuracy--symphonym-v8/snapshots').glob('*'))
        if args.pretrained_dir is None and len(snapshots) != 1:
            raise FileNotFoundError('Supply --pretrained-dir for one unambiguous original symphonym-v8 snapshot')
        base = args.pretrained_dir or snapshots[0]
        required = ['model.safetensors', 'config.json', 'vocab/char_vocab.json', 'vocab/script_vocab.json', 'vocab/lang_vocab.json']
        base_hashes = {s: sha256(base/s) for s in required}
        record['initialization'] = {'source': 'docuracy/symphonym-v8 original snapshot', 'path': str(base),
                                    'revision': base.name, 'file_sha256': base_hashes,
                                    'stage_a_executed': False, 'stage_a_checkpoint_loaded': False}
        print('Loading original symphonym-v8:', base, flush=True)
        bi_encoder = CharPhoneticBiEncoder.from_symphonym_pretrained(base, device=device_str).float()
        tokenizer = bi_encoder.tokenizer
        names = catalog.vessel_name.tolist()
        ids = catalog.vessel_id.tolist()
        name_map = dict(zip(ids, names))
        train_pairs = [{'query': str(r.vessel_name).strip(), 'catalog': name_map[r.target_vessel_id],
                        'vessel_id': r.target_vessel_id, 'hard_negs': []} for r in fit.itertuples(index=False)]
        record['training_pairs'].update(count=len(train_pairs), positive_rule='official fit rows only',
                                        negative_rule='2 sister + 2 ANCE; all fit-query positive names excluded')
        val_kr = val.loc[val.vessel_name.str.contains('[가-힣]', regex=True)]
        val_queries = val_kr.vessel_name.tolist()
        val_targets = val_kr.target_vessel_id.tolist()
        if args.fold >= 0 and not val_queries:
            raise ValueError('Empty Hangul selection subset')
        optimizer, parameter_groups = build_optimizer(bi_encoder, args.lr_backbone, args.lr_new)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_finetune, eta_min=5e-6)
        loss_fn = SupConCosFaceHardLoss(temperature=.05, margin=.08, hard_gamma=.20, hard_weight=.75).to(device)
        record['model'].update(type='CharPhoneticBiEncoder', parameters=vars(args) | {'pretrained_dir': str(base)},
                               loss='SupConCosFaceHardLoss', parameter_groups=parameter_groups,
                               checkpoint_dtype='float32', selection='outer-fold Hangul composite; explicitly development-selected',
                               composite_formula='0.45*R35 + 0.30*R10 + 0.25*R1 + 10*full_catalog_MRR')
        record['metrics']['metric_source'] = 'training and outer-fold checkpoint selection; not untouched validation'
        record['observations'] = ['Stage A data and checkpoints are never read',
                                  'Composite fractions are in [0,1]; the user-specified MRR coefficient dominates',
                                  'Full-catalog ID ranking differs from historical name-deduplicated selection; MRR@35 also retained']
        write_json(record_path, record)
        if val_queries:
            baseline = evaluate_retrieval(bi_encoder, val_queries, val_targets, names, device=device_str, catalog_ids=ids)
            write_json(out/'initial_retrieval.json', baseline)
            print('Original-snapshot baseline:', baseline, flush=True)
        best = None
        history = []
        mining = []
        checkpoint = out/('full_encoder.pt' if args.fold < 0 else f'fold_{args.fold}_encoder.pt')
        dataset = PhoneticPairDataset(train_pairs, format_inputs=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=make_multiview_collate_fn(tokenizer, max_length=96),
                            num_workers=0, drop_last=False)
        for epoch_index in range(args.epochs_finetune):
            if epoch_index in (0, 4):
                tick = time.perf_counter()
                mine_hard_negatives_for_dataset(bi_encoder, train_pairs, names, ids, device=device_str)
                negative_hash = object_hash([p['hard_negs'] for p in train_pairs])
                mining.append({'before_epoch_zero_based': epoch_index, 'seconds': time.perf_counter()-tick,
                               'negative_pairs_sha256': negative_hash, 'fit_pairs_only': True})
                # Dataset owns the same mutable records; the refreshed negatives are visible.
            tick = time.perf_counter()
            learning_rates = [g['lr'] for g in optimizer.param_groups]
            loss = train_one_epoch(bi_encoder, loader, optimizer, loss_fn, device,
                                   use_amp=use_amp, scaler=scaler, cooldown=args.cooldown)
            if not math.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            train_seconds = time.perf_counter()-tick
            scheduler.step()
            metrics = evaluate_retrieval(bi_encoder, val_queries, val_targets, names,
                                         device=device_str, catalog_ids=ids) if val_queries else None
            composite = composite_score(metrics) if metrics else None
            improved = metrics is None or best is None or composite > best['composite']
            epoch_file = out/'epochs'/f'epoch_{epoch_index+1}.pt'
            bi_encoder.save_checkpoint(epoch_file, as_fp16=False)
            if improved:
                bi_encoder.save_checkpoint(checkpoint, as_fp16=False)
                best = {'epoch': epoch_index+1, 'composite': composite, 'metrics': metrics,
                        'checkpoint_sha256': sha256(checkpoint)}
            history.append({'epoch': epoch_index+1, 'learning_rates': learning_rates,
                            'loss': loss, 'training_seconds': train_seconds, 'retrieval': metrics,
                            'composite': composite, 'saved_as_best': improved})
            write_json(out/'training_history.json', {'epochs': history, 'mining': mining, 'best': best})
            print(f'Epoch {epoch_index+1}/{args.epochs_finetune}: loss={loss:.5f}, composite={composite}; best={best["epoch"]}', flush=True)
        record['selection'] = best
        record['history'] = history
        record['mining'] = mining
        record['artifacts'] = [str(checkpoint), str(out/'training_history.json'), str(out/'data_manifest.json')]
        record['cost'].update(training_seconds=time.perf_counter()-started,
                              timing_scope='data/base loading, initial evaluation, mining, training, epoch evaluation and checkpoint writes',
                              model_bytes=checkpoint.stat().st_size,
                              peak_memory_bytes=torch.cuda.max_memory_allocated() if use_amp else None)
        record['metrics'].update(selected_epoch=best['epoch'], selected_composite=best['composite'])
        manifest = {'experiment_id': args.experiment_id, 'checkpoint_sha256': sha256(checkpoint),
                    'data': data, 'initialization': record['initialization'], 'selection': best,
                    'code': record['code'], 'config': record['model']['parameters']}
        write_json(checkpoint.with_suffix('.manifest.json'), manifest)
        finish_run(record_path, record, 'trained_development_selected')
        print('Saved selected checkpoint:', checkpoint, flush=True)
    except BaseException as exc:
        finish_run(record_path, record, 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', exc)
        raise


if __name__ == '__main__':
    main()
