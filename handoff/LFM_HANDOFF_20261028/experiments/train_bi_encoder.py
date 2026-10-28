"""
Two-Stage Curriculum Contrastive Training Script (SPEC-DPR-01).

Experiment ID: DPR-SYMPHON-01

Stages:
  Stage A: Phonetic Pre-training on 100k synthetic pairs (data/synthetic_phonetic_100k.json.gz)
  Stage B: Group 5-Fold Contrastive Fine-tuning (eval_kit/folds_group5_seed42.csv)
           - Holdout validation on fold k (Strict zero leakage)
           - Evaluation of Recall@1, Recall@5, Recall@10, Recall@35 & MRR
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
from phonetic_encoder.retrieval import top_score_indices


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
) -> Dict[str, float]:
    """
    Evaluate Top-K Retrieval Recalls and MRR on heldout validation pairs.
    """
    model.eval()
    if catalog_embeds is None:
        catalog_embeds = model.encode_catalog(catalog_names, batch_size=batch_size)

    # Build target name to catalog indices mapping
    name_to_idx = {name: i for i, name in enumerate(catalog_names)}

    query_embeds = model.encode_queries(val_queries, batch_size=batch_size)

    # Matrix dot product (N_val, N_catalog)
    sims = np.dot(query_embeds, catalog_embeds.T)

    recalls = {1: 0, 5: 0, 10: 0, 35: 0}
    mrr_sum = 0.0
    valid_count = 0

    for i, target in enumerate(val_targets):
        if target not in name_to_idx:
            continue
        valid_count += 1
        gold_idx = name_to_idx[target]
        row_sims = sims[i]

        # Top 35 indices
        top35_sorted = top_score_indices(row_sims, 35)

        # Check rank
        for k in [1, 5, 10, 35]:
            if gold_idx in top35_sorted[:k]:
                recalls[k] += 1

        if gold_idx in top35_sorted:
            rank = np.where(top35_sorted == gold_idx)[0][0] + 1
            mrr_sum += 1.0 / rank

    if valid_count == 0:
        return {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "recall@35": 0.0, "mrr": 0.0}

    return {
        "recall@1": recalls[1] / valid_count,
        "recall@5": recalls[5] / valid_count,
        "recall@10": recalls[10] / valid_count,
        "recall@35": recalls[35] / valid_count,
        "mrr": mrr_sum / valid_count,
        "eval_count": valid_count,
    }


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
    unique_queries = list({p["query"] for p in train_pairs})
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
            if catalog_ids[idx] != v_pos
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
            if catalog_ids[g_idx] != v_pos and catalog_names[g_idx] not in sister_negs:
                global_negs.append(catalog_names[g_idx])
                if len(global_negs) >= k_global:
                    break

        # 3. Combine to ensure exactly 4 negatives
        combined = list(sister_negs) + list(global_negs)
        if len(combined) < 4:
            for g_idx in top_confusions_by_q.get(q_name, []):
                cand_name = catalog_names[g_idx]
                if catalog_ids[g_idx] != v_pos and cand_name not in combined:
                    combined.append(cand_name)
                    if len(combined) >= 4:
                        break
        while len(combined) < 4:
            combined.append(catalog_names[0])

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


def main():
    parser = argparse.ArgumentParser(description="Train lightweight Char-Phonetic Bi-Encoder (DPR-SYMPHON-02).")
    parser.add_argument("--experiment-id", default="DPR-SYMPHON-02", help="Experiment identifier")
    parser.add_argument("--synthetic-data", default="data/synthetic_phonetic_100k.json.gz", help="Path to synthetic pairs")
    parser.add_argument("--train-csv", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--catalog-csv", default="data/vessel_catalog.csv", help="Path to vessel_catalog.csv")
    parser.add_argument("--folds-csv", default="eval_kit/folds_group5_seed42.csv", help="Path to folds CSV")
    parser.add_argument("--epochs-pretrain", type=int, default=3, help="Stage A pretraining epochs")
    parser.add_argument("--epochs-finetune", type=int, default=12, help="Stage B fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr-pretrain", type=float, default=2e-4, help="Stage A learning rate")
    parser.add_argument("--lr-finetune", type=float, default=5e-5, help="Stage B learning rate")
    parser.add_argument("--fold", type=int, default=0, help="Target fold to holdout for Stage B evaluation")
    parser.add_argument("--output-dir", default="artifacts/DPR-SYMPHON-02", help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cooldown", type=float, default=0.0, help="Per-batch cooldown delay (seconds)")
    args = parser.parse_args()

    set_seed(args.seed)

    # Hardware setup
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    use_amp = (device_str == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    if device_str == "cpu":
        torch.set_num_threads(2)
        print("Running on CPU with 2 threads capped.")
    else:
        print(f"Running on GPU: {torch.cuda.get_device_name(0)} with AMP enabled.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Bi-Encoder from local Symphonym-v8 HuggingFace snapshot
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    matches = list(hf_cache_dir.glob("models--docuracy--symphonym-v8/snapshots/*"))
    if not matches:
        raise FileNotFoundError("docuracy/symphonym-v8 snapshot not found in HuggingFace cache.")

    pretrain_ckpt_path = out_dir / "pretrain_synthetic_encoder.pt"
    if args.epochs_pretrain == 0 and pretrain_ckpt_path.exists():
        print(f"[1/4] Loading existing Stage A pretrained checkpoint from {pretrain_ckpt_path}...")
        bi_encoder = CharPhoneticBiEncoder.load_checkpoint(pretrain_ckpt_path, device=device_str)
        bi_encoder = bi_encoder.float()  # ensure both encoder and fusion are FP32 for training with AMP
    else:
        snapshot_path = matches[0]
        print(f"[1/4] Loading Symphonym-v8 pretrained weights from {snapshot_path}...")
        bi_encoder = CharPhoneticBiEncoder.from_symphonym_pretrained(snapshot_path, device=device_str)

    tokenizer = bi_encoder.tokenizer
    print(f"Model ready: Vocab size {tokenizer.vocab_size}, Output dim 128.")

    pretrain_collate_fn = make_collate_fn(tokenizer, max_length=96)
    loss_fn = SupConCosFaceHardLoss(temperature=0.05, margin=0.08, hard_gamma=0.20, hard_weight=0.75).to(device)

    # =========================================================================
    # Stage A: Pretraining on Synthetic Data
    # =========================================================================
    if args.epochs_pretrain > 0 and Path(args.synthetic_data).exists():
        print(f"\n[2/4] Stage A: Pre-training on {args.synthetic_data} ({args.epochs_pretrain} epochs)...")
        with gzip.open(args.synthetic_data, "rt", encoding="utf-8") as f:
            synthetic_records = json.load(f)

        pretrain_dataset = PhoneticPairDataset(synthetic_records, format_inputs=False)
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=pretrain_collate_fn,
            num_workers=0,
            drop_last=True,
        )

        optimizer_a = torch.optim.AdamW(bi_encoder.parameters(), lr=args.lr_pretrain, weight_decay=1e-4)

        for epoch in range(1, args.epochs_pretrain + 1):
            t0 = time.time()
            avg_loss = train_one_epoch(
                bi_encoder, pretrain_loader, optimizer_a, loss_fn, device,
                use_amp=use_amp, scaler=scaler, cooldown=args.cooldown
            )
            elapsed = time.time() - t0
            print(f"  Pretrain Epoch {epoch}/{args.epochs_pretrain} - Loss: {avg_loss:.4f} ({elapsed:.1f}s)")

        # Save pretrained base checkpoint
        pretrain_ckpt_path = out_dir / "pretrain_synthetic_encoder.pt"
        bi_encoder.save_checkpoint(pretrain_ckpt_path, as_fp16=True)
        print(f"Saved pretrained checkpoint to {pretrain_ckpt_path}")

    # =========================================================================
    # Stage B: Group 5-Fold Contrastive Fine-tuning
    # =========================================================================
    print(f"\n[3/4] Stage B: Fine-tuning on Fold {args.fold} (Holdout Validation)...")

    # Load catalog mapping
    df_cat = pd.read_csv(args.catalog_csv, usecols=["vessel_id", "vessel_name"]).dropna()
    catalog_map = dict(zip(df_cat["vessel_id"], df_cat["vessel_name"]))
    all_catalog_names = df_cat["vessel_name"].astype(str).tolist()
    all_catalog_ids = df_cat["vessel_id"].astype(str).tolist()
    eval_catalog_names = df_cat["vessel_name"].drop_duplicates().tolist()

    # Load train data and folds
    df_train = pd.read_csv(args.train_csv)
    df_folds = pd.read_csv(args.folds_csv)
    df_merged = df_train.merge(df_folds, on=["sample_id", "target_vessel_id"])

    # Separate train and validation folds
    if args.fold >= 0:
        df_val = df_merged[df_merged["fold"] == args.fold].reset_index(drop=True)
        df_trn = df_merged[df_merged["fold"] != args.fold].reset_index(drop=True)
        print(f"Fold {args.fold} Split: Train {len(df_trn):,} rows, Val {len(df_val):,} rows.")
        re_hangul = __import__("re").compile(r"[가-힣]")
        val_kr = df_val[df_val["vessel_name"].astype(str).apply(lambda x: bool(re_hangul.search(x)))]
        val_queries = val_kr["vessel_name"].tolist()
        val_targets = [catalog_map.get(vid) for vid in val_kr["target_vessel_id"]]
        print(f"Validation Korean Queries: {len(val_queries):,} rows.")
    else:
        df_val = pd.DataFrame()
        df_trn = df_merged.reset_index(drop=True)
        print(f"Full Training Mode: Training on all {len(df_trn):,} rows.")
        val_queries = []
        val_targets = []

    # Prepare training pairs with vessel IDs
    train_pairs: List[Dict[str, Any]] = []
    for _, row in df_trn.iterrows():
        q_name = str(row["vessel_name"]).strip()
        v_id = row["target_vessel_id"]
        c_name = catalog_map.get(v_id)
        if c_name:
            train_pairs.append({
                "query": q_name,
                "catalog": c_name,
                "vessel_id": str(v_id),
                "hard_negs": [],
            })

    # Initial hard negative mining (2 sister + 2 global ANCE)
    print("\n[Stage B Setup] Mining initial ANCE & Sister-Vessel Hard Negatives (SPEC-DPR-02)...")
    mine_hard_negatives_for_dataset(
        bi_encoder, train_pairs, all_catalog_names, all_catalog_ids, device=device_str, k_sister=2, k_global=2
    )

    finetune_collate_fn = make_multiview_collate_fn(tokenizer, max_length=96)
    finetune_dataset = PhoneticPairDataset(train_pairs, format_inputs=False)
    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=finetune_collate_fn,
        num_workers=0,
        drop_last=False,
    )

    optimizer_b = torch.optim.AdamW(bi_encoder.parameters(), lr=args.lr_finetune, weight_decay=1e-4)

    best_recall35 = 0.0
    best_mrr = 0.0
    best_ckpt_path = out_dir / ("full_encoder.pt" if args.fold < 0 else f"fold_{args.fold}_encoder.pt")

    if len(val_queries) > 0:
        # Pre-fine-tuning zero-shot baseline evaluation
        print("\n--- Zero-Shot Baseline Evaluation on Heldout Validation Set ---")
        baseline_metrics = evaluate_retrieval(bi_encoder, val_queries, val_targets, eval_catalog_names, device=device_str)
        print(f"  Recall@1:  {baseline_metrics['recall@1']*100:.2f}%")
        print(f"  Recall@5:  {baseline_metrics['recall@5']*100:.2f}%")
        print(f"  Recall@10: {baseline_metrics['recall@10']*100:.2f}%")
        print(f"  Recall@35: {baseline_metrics['recall@35']*100:.2f}%")
        print(f"  MRR:       {baseline_metrics['mrr']:.4f}")

    for epoch in range(1, args.epochs_finetune + 1):
        if epoch == 6:
            print(f"\n[Refresh] Epoch 6: Re-mining ANCE & Sister-Vessel Hard Negatives with updated encoder...")
            mine_hard_negatives_for_dataset(
                bi_encoder, train_pairs, all_catalog_names, all_catalog_ids, device=device_str, k_sister=2, k_global=2
            )

        t0 = time.time()
        avg_loss = train_one_epoch(
            bi_encoder, finetune_loader, optimizer_b, loss_fn, device,
            use_amp=use_amp, scaler=scaler, cooldown=args.cooldown
        )
        elapsed = time.time() - t0

        if len(val_queries) > 0:
            # Evaluate on validation fold
            metrics = evaluate_retrieval(bi_encoder, val_queries, val_targets, eval_catalog_names, device=device_str)
            rec35 = metrics["recall@35"]
            mrr_val = metrics["mrr"]
            print(f"  Epoch {epoch:02d}/{args.epochs_finetune:02d} - Loss: {avg_loss:.4f} - Val Recall@35: {rec35*100:.2f}% (R@1: {metrics['recall@1']*100:.1f}%, MRR: {mrr_val:.4f}) [{elapsed:.1f}s]")

            if rec35 > best_recall35 or (rec35 == best_recall35 and mrr_val > best_mrr):
                best_recall35 = rec35
                best_mrr = mrr_val
                bi_encoder.save_checkpoint(best_ckpt_path, as_fp16=True)
                print(f"  --> Saved new best checkpoint (Recall@35: {rec35*100:.2f}%, MRR: {mrr_val:.4f}) to {best_ckpt_path}")
        else:
            print(f"  Epoch {epoch:02d}/{args.epochs_finetune:02d} - Loss: {avg_loss:.4f} [{elapsed:.1f}s]")
            bi_encoder.save_checkpoint(best_ckpt_path, as_fp16=True)

    if len(val_queries) > 0:
        print(f"\n[4/4] Training Complete! Best Heldout Recall@35: {best_recall35*100:.2f}%")
    else:
        print(f"\n[4/4] Full Dataset Training Complete!")
    print(f"Model checkpoint saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()
