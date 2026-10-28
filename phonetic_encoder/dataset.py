"""
Dataset and Collation for Bi-Encoder Contrastive Training (SPEC-DPR-01).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from .tokenizer import (
    PhoneticTokenizer,
    clean_vessel_name,
    en_to_phonetic,
    extract_tail_core_text,
    format_hangul_query,
    format_latin_catalog,
    hangul_to_rom,
)


class PhoneticPairDataset(Dataset):
    """
    Dataset storing (query_text, catalog_text, Optional[hard_negatives], Optional[vessel_id]).
    """

    def __init__(
        self,
        pairs: List[Dict[str, Any]],  # list of dicts: {'query': str, 'catalog': str, 'hard_negs': List[str]}
        format_inputs: bool = True,
    ):
        self.pairs = pairs
        self.format_inputs = format_inputs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.pairs[idx]
        q_raw = item["query"]
        c_raw = item["catalog"]
        hard_negs = item.get("hard_negs", [])
        v_id = str(item.get("vessel_id", ""))

        if self.format_inputs:
            q_formatted = format_hangul_query(q_raw)
            c_formatted = format_latin_catalog(c_raw)
            negs_formatted = [format_latin_catalog(n) for n in hard_negs]
        else:
            q_formatted = q_raw
            c_formatted = c_raw
            negs_formatted = list(hard_negs)

        return {
            "query_text": q_formatted,
            "catalog_text": c_formatted,
            "hard_neg_texts": negs_formatted,
            "query_raw": q_raw,
            "catalog_raw": c_raw,
            "hard_negs_raw": list(hard_negs),
            "vessel_id": v_id,
        }


def make_collate_fn(tokenizer: PhoneticTokenizer, max_length: int = 96):
    """
    Creates a DataLoader collate function with batch-padding.
    """

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        q_items = [(item["query_text"], "HANGUL", "ko") for item in batch]
        c_items = [(item["catalog_text"], "LATIN", "en") for item in batch]

        q_tensors = tokenizer.batch_encode(q_items, max_length=max_length)
        c_tensors = tokenizer.batch_encode(c_items, max_length=max_length)

        # Check if hard negatives exist
        has_negs = any(len(item["hard_neg_texts"]) > 0 for item in batch)
        neg_tensors = None
        if has_negs:
            # Flatten hard negatives while tracking indices
            all_neg_items: List[Tuple[str, str, str]] = []
            neg_counts: List[int] = []
            for item in batch:
                negs = item["hard_neg_texts"]
                neg_counts.append(len(negs))
                for n in negs:
                    all_neg_items.append((n, "LATIN", "en"))

            if all_neg_items:
                neg_batch = tokenizer.batch_encode(all_neg_items, max_length=max_length)
                neg_tensors = {
                    "neg_batch": neg_batch,
                    "neg_counts": neg_counts,
                }

        return {
            "query": q_tensors,
            "catalog": c_tensors,
            "hard_negs": neg_tensors,
            "batch_size": len(batch),
        }

    return collate_fn


def make_multiview_collate_fn(tokenizer: PhoneticTokenizer, max_length: int = 96):
    """
    Creates a DataLoader collate function preparing multi-view batches (primary, secondary, tail)
    for End-to-End GatedMultiViewFusion training (SPEC-DPR-02).
    """

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        vids = [item.get("vessel_id", "") for item in batch]
        q_raws = [item.get("query_raw", item["query_text"]) for item in batch]
        c_raws = [item.get("catalog_raw", item["catalog_text"]) for item in batch]

        # Multi-views for queries (Hangul or English)
        re_ko = re.compile(r"[가-힣]")
        q_prim_items = []
        q_sec_items = []
        q_tail_items = []
        for q in q_raws:
            q_clean = clean_vessel_name(q)
            if re_ko.search(q_clean):
                q_prim_items.append((q_clean, "HANGUL", "ko"))
                q_sec_items.append((hangul_to_rom(q_clean), "LATIN", "en"))
                tail_txt = extract_tail_core_text(q_clean)
                q_tail_items.append((hangul_to_rom(tail_txt) if re_ko.search(tail_txt) else tail_txt, "LATIN", "en"))
            else:
                q_prim_items.append((q_clean.upper(), "LATIN", "en"))
                q_sec_items.append((en_to_phonetic(q_clean.upper()), "LATIN", "en"))
                q_tail_items.append((extract_tail_core_text(q_clean.upper()), "LATIN", "en"))

        # Multi-views for catalogs (Latin)
        c_prim_items = [(clean_vessel_name(c).upper(), "LATIN", "en") for c in c_raws]
        c_sec_items = [(en_to_phonetic(clean_vessel_name(c).upper()), "LATIN", "en") for c in c_raws]
        c_tail_items = [(extract_tail_core_text(clean_vessel_name(c).upper()), "LATIN", "en") for c in c_raws]

        q_prim = tokenizer.batch_encode(q_prim_items, max_length=max_length)
        q_sec = tokenizer.batch_encode(q_sec_items, max_length=max_length)
        q_tail = tokenizer.batch_encode(q_tail_items, max_length=max_length)

        c_prim = tokenizer.batch_encode(c_prim_items, max_length=max_length)
        c_sec = tokenizer.batch_encode(c_sec_items, max_length=max_length)
        c_tail = tokenizer.batch_encode(c_tail_items, max_length=max_length)

        # Check hard negatives
        has_negs = any(len(item.get("hard_negs_raw", item.get("hard_neg_texts", []))) > 0 for item in batch)
        neg_data = None
        if has_negs:
            h_prim_items, h_sec_items, h_tail_items = [], [], []
            neg_counts: List[int] = []
            for item in batch:
                negs = item.get("hard_negs_raw", item.get("hard_neg_texts", []))
                neg_counts.append(len(negs))
                for n in negs:
                    h_clean = clean_vessel_name(n).upper()
                    h_prim_items.append((h_clean, "LATIN", "en"))
                    h_sec_items.append((en_to_phonetic(h_clean), "LATIN", "en"))
                    h_tail_items.append((extract_tail_core_text(h_clean), "LATIN", "en"))

            if h_prim_items:
                neg_data = {
                    "primary": tokenizer.batch_encode(h_prim_items, max_length=max_length),
                    "secondary": tokenizer.batch_encode(h_sec_items, max_length=max_length),
                    "tail": tokenizer.batch_encode(h_tail_items, max_length=max_length),
                    "neg_counts": neg_counts,
                }

        return {
            "query_primary": q_prim,
            "query_secondary": q_sec,
            "query_tail": q_tail,
            "catalog_primary": c_prim,
            "catalog_secondary": c_sec,
            "catalog_tail": c_tail,
            "hard_negs": neg_data,
            "vessel_ids": vids,
            "batch_size": len(batch),
        }

    return collate_fn
