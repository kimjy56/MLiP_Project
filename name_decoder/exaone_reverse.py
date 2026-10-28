"""EXAONE reverse transliteration (KO -> EN); fold-isolated and legacy-free."""
from __future__ import annotations
import json
from pathlib import Path
import re
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from .deterministic_ko_rules import canonical_name, clean_korean
from .exaone_translit import (
    MODEL_ID,
    load_base,
    inject_lora,
    restore_adapter,
    adapter_state,
    collate_examples,
    resolve_base,
)

PROMPT_VERSION = "ko-en-raw-v1"


def build_reverse_prompt(korean: str) -> str:
    korean = clean_korean(korean)
    if not korean:
        raise ValueError("Empty reverse transliteration source")
    return f"[KO] {korean} => [EN] "


def encode_reverse_example(tokenizer, source: str, target: str, max_length: int = 256):
    target = canonical_name(target)
    if not target:
        raise ValueError("Invalid English training target")
    prompt = tokenizer.encode(build_reverse_prompt(source), add_special_tokens=False)
    answer = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer EOS is required")
    ids = prompt + answer
    if len(ids) > max_length:
        raise ValueError("Example exceeds max length; do not silently truncate")
    return {"input_ids": ids, "labels": [-100] * len(prompt) + answer}


class EnglishOnlyLogitsProcessor(LogitsProcessor):
    """Allow uppercase English/digits/space/punctuation and tokenizer EOS."""

    def __init__(self, tokenizer, prompt_length: int):
        self.prompt_length = prompt_length
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("EOS required")
        self.allowed = []
        self.initial = []
        self._masks = {}
        specials = set(tokenizer.all_special_ids)
        for i in range(len(tokenizer)):
            if i in specials:
                continue
            value = tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if value and re.fullmatch(r"[A-Za-z0-9 \-.'&/]+", value):
                self.allowed.append(i)
                if value.strip():
                    self.initial.append(i)
        if not self.initial:
            raise ValueError("No valid target tokens")
        self.allowed.append(self.eos_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        first = input_ids.shape[1] <= self.prompt_length
        key = (scores.device, scores.dtype, scores.shape[-1], first)
        if key not in self._masks:
            mask = torch.full((scores.shape[-1],), float("-inf"), device=scores.device, dtype=scores.dtype)
            ids = self.initial if first else self.allowed
            valid = [i for i in ids if i < scores.shape[-1]]
            if not valid:
                raise ValueError("Allowed vocabulary does not fit logits")
            mask[valid] = 0
            self._masks[key] = mask
        return scores + self._masks[key]


class ExaoneReverseDecoder:
    def __init__(self, model, tokenizer, manifest=None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.manifest = manifest or {}
        self._allowed_processor = None

    @classmethod
    def from_adapter(cls, checkpoint, device="cuda", local_model=None):
        checkpoint = Path(checkpoint)
        manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("model_id") != MODEL_ID:
            raise ValueError("Wrong model identity")
        base = Path(local_model or manifest["base_path"])
        from experiments.exaone_common import sha256

        for name, expected in manifest["base_file_sha256"].items():
            if sha256(base / name) != expected:
                raise ValueError("Base/tokenizer files changed")
        if sha256(checkpoint / "adapter.pt") != manifest["adapter_sha256"]:
            raise ValueError("Adapter hash mismatch")
        model, tokenizer = load_base(base, device)
        inject_lora(model, **{k: manifest["lora"][k] for k in ["rank", "alpha", "dropout"]})
        restore_adapter(model, torch.load(checkpoint / "adapter.pt", map_location="cpu", weights_only=True))
        return cls(model, tokenizer, manifest)

    @torch.inference_mode()
    def decode(self, korean_names: list[str], num_beams: int = 4, num_return_sequences: int = 3, max_new_tokens: int = 48):
        if not 1 <= num_return_sequences <= num_beams:
            raise ValueError("return_sequences must be 1..num_beams")
        if not korean_names:
            return []
        batch = self.tokenizer(
            [build_reverse_prompt(n) for n in korean_names],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        length = batch["input_ids"].shape[1]
        if self._allowed_processor is None:
            self._allowed_processor = EnglishOnlyLogitsProcessor(self.tokenizer, length)
        self._allowed_processor.prompt_length = length
        output = self.model.generate(
            **batch,
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([self._allowed_processor]),
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )
        decoded = self.tokenizer.batch_decode(
            output.sequences[:, length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results = []
        for i in range(len(korean_names)):
            choices = []
            for j in range(num_return_sequences):
                k = i * num_return_sequences + j
                tail = output.sequences[k, length:].tolist()
                value = canonical_name(decoded[k])
                if value and self.tokenizer.eos_token_id in tail:
                    choices.append({"text": value, "score": float(-j), "score_type": "beam_rank"})
            seen = set()
            results.append([v for v in choices if not (v["text"] in seen or seen.add(v["text"]))])
        return results
