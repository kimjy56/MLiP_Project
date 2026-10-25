"""Offline, deterministic, one-record-at-a-time ByT5 decoding."""
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from transformers import ByT5Tokenizer, T5Config, T5ForConditionalGeneration

from .catalog import ByteNameTrie, NameCandidate, NamePrediction, canonical_name


class ByT5NameDecoder:
    def __init__(self, model, tokenizer, catalog, max_input_bytes=256):
        self.model = model
        self.tokenizer = tokenizer
        self.catalog = catalog
        self.max_input_bytes = max_input_bytes
        if (model.config.decoder_start_token_id, model.config.eos_token_id,
                model.config.pad_token_id) != (0, 1, 0):
            raise ValueError("This decoder requires the ByT5 special-token layout")
        if tokenizer.encode("A", add_special_tokens=False) != [68]:
            raise ValueError("This decoder requires ByT5 byte tokenization")
        self.trie = ByteNameTrie(catalog.names)

    @classmethod
    def load(cls, checkpoint, catalog, device="cpu"):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_dir():
            raise ValueError("A local checkpoint directory is required")
        tokenizer = ByT5Tokenizer.from_pretrained(checkpoint, local_files_only=True)
        raw_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        config = T5Config.from_pretrained(checkpoint, local_files_only=True)
        # Transformers 5.20 rewrites T5's tie flag. Preserve the checkpoint's
        # untied ByT5 head and its original output scaling across save/reload.
        config.tie_word_embeddings = raw_config.get("tie_word_embeddings", True)
        config.scale_decoder_outputs = raw_config.get(
            "scale_decoder_outputs", config.tie_word_embeddings)
        model = T5ForConditionalGeneration.from_pretrained(
            checkpoint, config=config, local_files_only=True, attn_implementation="eager")
        return cls(model.to(device), tokenizer, catalog)

    def precision_context(self):
        if self.model.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.inference_mode()
    def decode(self, name, top_k=10):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        source = canonical_name(name)
        if not source:
            raise ValueError("Empty input name")
        if len(source.encode("utf-8")) > self.max_input_bytes:
            raise ValueError("Input exceeds max_input_bytes; no silent truncation")
        count = min(top_k, len(self.catalog.names))
        self.model.eval()
        inputs = self.tokenizer(source, return_tensors="pt").to(self.model.device)
        # Use FP32 for ranking reproducibility; BF16 autocast is training-only.
        output = self.model.generate(
            **inputs, do_sample=False, num_beams=max(2, count),
            num_return_sequences=count, max_new_tokens=self.trie.max_new_tokens,
            prefix_allowed_tokens_fn=self.trie, early_stopping=True,
            length_penalty=1.0, forced_eos_token_id=None,
            return_dict_in_generate=True, output_scores=True,
        )
        best = {}
        for ids, score in zip(output.sequences, output.sequences_scores):
            candidate = self.tokenizer.decode(ids, skip_special_tokens=True)
            if candidate not in self.catalog.ids_by_name:
                raise RuntimeError("Incomplete or out-of-catalog decoded name")
            value = float(score)
            best[candidate] = max(value, best.get(candidate, -float("inf")))
        # Scores are length-normalized beam scores, not calibrated probabilities.
        return [NameCandidate(name, score, self.catalog.ids_by_name[name])
                for name, score in sorted(best.items(), key=lambda item: (-item[1], item[0]))[:top_k]]

    def decode_many(self, names, top_k=10):
        # Single-row execution makes batch/order independence explicit. Optimize later.
        return [self.decode(name, top_k=top_k) for name in names]

    def predict(self, name, top_k=10):
        """Select rank 1 from the same beam search used for candidate evaluation.

        Keep top_k=10 for the trained workflow: beam width changes can change
        rank 1. Duplicate catalog IDs are deliberately not resolved by ID order.
        """
        candidates = tuple(self.decode(name, top_k=top_k))
        if not candidates:
            raise RuntimeError("Decoder returned no catalog name")
        return NamePrediction(candidates[0].name, candidates[0].vessel_ids, candidates)

    def predict_many(self, names, top_k=10):
        return [self.predict(name, top_k=top_k) for name in names]
