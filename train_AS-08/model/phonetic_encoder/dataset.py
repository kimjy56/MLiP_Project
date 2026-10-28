"""
Dataset and Collation for Bi-Encoder Contrastive Training (SPEC-DPR-01).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from .tokenizer import PhoneticTokenizer, format_hangul_query, format_latin_catalog


class PhoneticPairDataset(Dataset):
    """
    Dataset storing (query_text, catalog_text, Optional[hard_negatives]).
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
