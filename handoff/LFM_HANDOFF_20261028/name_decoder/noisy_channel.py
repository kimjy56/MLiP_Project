"""Reverse name channel: log P(observed name | official catalog name).

Scores only the observed tokens and EOS, never the conditioning prompt. This
module does not select candidates, consult labels, or interpret a name as an ID.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import unicodedata

import torch
from torch.nn import functional as F

from .catalog import canonical_name, has_hangul

PROMPT_VERSION_V1 = "catalog-to-observed-name-v1"
PROMPT_VERSION_V2 = "catalog-to-observed-name-v2-task-prefix"
PROMPT_VERSION = PROMPT_VERSION_V2  # Default for new training
OBSERVED_NORMALIZATION = "NFC-strip-preserve-case-internal-space-digits-punctuation"
CATALOG_NORMALIZATION = "NFC-uppercase-collapse-whitespace"
MANIFEST_NAME = "noisy_channel_manifest.json"


def load_channel_tokenizer(checkpoint):
    """Load local fast-tokenizer metadata across the v4/v5 backend rename."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'tokenizer_config.json').read_text(encoding='utf-8-sig'))
    if config.get('tokenizer_class') == 'TokenizersBackend':
        # v5 renamed PreTrainedTokenizerFast; use the same tokenizer.json,
        # special tokens and chat template without editing downloaded files.
        return PreTrainedTokenizerFast.from_pretrained(checkpoint, local_files_only=True, extra_special_tokens={})
    return AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)


def normalize_observed(value):
    if not isinstance(value, str):
        raise TypeError("Observed name must be a string")
    value = unicodedata.normalize("NFC", value).strip()
    if not value:
        raise ValueError("Observed name must not be empty")
    return value


def prepare_channel_pairs(fit_rows, catalog):
    """Call only with training-fold rows; deduplicate AFTER normalization."""
    import pandas as pd

    pairs = pd.DataFrame({
        "catalog_name": fit_rows.target_vessel_id.map(catalog.name_by_id),
        "observed_name": fit_rows.vessel_name.map(normalize_observed),
    })
    if pairs.catalog_name.isna().any():
        raise ValueError("Training ID is missing from the catalog")
    pairs = pairs.drop_duplicates(["catalog_name", "observed_name"])
    pairs = pairs.sort_values(["catalog_name", "observed_name"]).reset_index(drop=True)
    pairs["hangul"] = pairs.observed_name.map(has_hangul)
    if pairs.empty:
        raise ValueError("No training pairs")
    return pairs


@dataclass(frozen=True)
class ChannelExample:
    input_ids: tuple[int, ...]
    prompt_length: int

    @property
    def target_count(self):
        return len(self.input_ids) - self.prompt_length


def encode_channel_example(tokenizer, catalog_name, observed_name, max_length=256, prompt_version=PROMPT_VERSION):
    name = canonical_name(catalog_name)
    observed = normalize_observed(observed_name)
    if not name:
        raise ValueError("Catalog name must not be empty")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    # Separate encoding is intentional: every candidate sees the same observed
    # tokens, independent of prompt length or boundary-dependent token merging.
    target_ids = tokenizer.encode(observed, add_special_tokens=False)
    name_ids = tokenizer.encode(name, add_special_tokens=False)
    special_ids = set(tokenizer.all_special_ids)
    if not target_ids or special_ids.intersection(target_ids + name_ids):
        raise ValueError("Names must contain ordinary text tokens only")

    # Construct prompt payload based on version
    if prompt_version == PROMPT_VERSION_V2:
        task = "KO_TRANSLITERATION" if has_hangul(observed) else "EN_TYPO_RECOVERY"
        payload = {"task": task, "catalog_name": name}
    else:
        payload = {"catalog_name": name}

    prompt_ids = tokenizer.apply_chat_template([
        {"role": "user", "content": (
            "Write one possible observed vessel name from a port record for "
            "this official catalog name. Output only the observed name.\n"
            + json.dumps(payload, ensure_ascii=False))},
    ], tokenize=True, add_generation_prompt=True)
    if not prompt_ids:
        raise ValueError("Empty conditioning prompt")
    tokens = tuple(prompt_ids) + tuple(target_ids) + (tokenizer.eos_token_id,)
    if len(tokens) > max_length:
        raise ValueError(f"Channel example has {len(tokens)} tokens; limit={max_length}. "
                         "Truncation is forbidden; raise the limit explicitly.")
    return ChannelExample(tokens, len(prompt_ids))


def collate_channel_examples(examples, pad_token_id):
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    length = max(len(example.input_ids) for example in examples)
    inputs = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(inputs)
    labels = torch.full_like(inputs, -100)
    for row, example in enumerate(examples):
        size = len(example.input_ids)
        inputs[row, :size] = torch.tensor(example.input_ids, dtype=torch.long)
        mask[row, :size] = 1
        labels[row, example.prompt_length:size] = inputs[row, example.prompt_length:size]
    return {"input_ids": inputs, "attention_mask": mask, "labels": labels}


@dataclass(frozen=True)
class ChannelScore:
    log_probability: float
    mean_log_probability: float
    target_token_count: int  # Includes EOS.


class NoisyChannelScorer:
    """Teacher-forced likelihood for fixed candidates; no free text generation."""

    def __init__(self, model, tokenizer, max_length=256, manifest=None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.manifest = manifest
        self.device = next(model.parameters()).device
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad or EOS")

    @classmethod
    def load(cls, checkpoint):
        """Load only a locally trained, merged reverse-channel checkpoint on CPU."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint = Path(checkpoint)
        with (checkpoint / MANIFEST_NAME).open(encoding="utf-8-sig") as stream:
            manifest = json.load(stream)
        expected = {"schema_version": 1, "direction": "catalog_name_to_observed_name",
                    "observed_normalization": OBSERVED_NORMALIZATION,
                    "catalog_normalization": CATALOG_NORMALIZATION,
                    "merged": True}
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Incompatible reverse checkpoint: {key}")
        prompt_ver = manifest.get("prompt_version")
        if prompt_ver not in (PROMPT_VERSION_V1, PROMPT_VERSION_V2):
            raise ValueError(f"Unsupported reverse checkpoint prompt_version: {prompt_ver}")
        # Check file content before use; an AS-HCX-04 forward model is not a
        # substitute for reverse training, even if its architecture is identical.
        import hashlib
        files = manifest.get("checkpoint_sha256", {})
        if not files:
            raise ValueError("Checkpoint manifest has no content hashes")
        for relative, expected_hash in files.items():
            file_path = (checkpoint / relative).resolve()
            if not file_path.is_relative_to(checkpoint.resolve()):
                raise ValueError("Invalid checkpoint hash path")
            with file_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected_hash:
                raise ValueError(f"Checkpoint hash mismatch: {relative}")
        tokenizer = load_channel_tokenizer(checkpoint)
        if tokenizer.eos_token_id != manifest.get("eos_token_id"):
            raise ValueError("Checkpoint EOS differs from its training manifest")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation="eager")
        return cls(model, tokenizer, max_length=manifest["max_length"], manifest=manifest)

    @torch.inference_mode()
    def score_pair(self, catalog_name, observed_name):
        prompt_ver = self.manifest.get("prompt_version", PROMPT_VERSION) if self.manifest else PROMPT_VERSION
        example = encode_channel_example(
            self.tokenizer, catalog_name, observed_name, self.max_length,
            prompt_version=prompt_ver)
        batch = collate_channel_examples([example], self.pad_token_id)
        # Batch size 1 avoids cross-record padding/numerical dependence.
        outputs = self.model(input_ids=batch["input_ids"].to(self.device),
                             attention_mask=batch["attention_mask"].to(self.device),
                             use_cache=False)
        target_logits = outputs.logits[0, example.prompt_length - 1:-1].float()
        targets = batch["input_ids"][0, example.prompt_length:].to(self.device)
        token_logp = -F.cross_entropy(target_logits, targets, reduction="none")
        if not torch.isfinite(token_logp).all():
            raise FloatingPointError("Nonfinite channel likelihood")
        total = token_logp.double().sum().item()
        return ChannelScore(total, total / example.target_count, example.target_count)

    def score_candidates(self, observed_name, vessel_ids, catalog):
        """Return every supplied ID; same-name IDs receive identical LM scores.

        Deduplication by name is confined to ONE observed record. Returned order
        is ID-sorted and no other record or held-out gold label enters scoring.
        """
        vessel_ids = list(vessel_ids)
        if not vessel_ids or len(set(vessel_ids)) != len(vessel_ids):
            raise ValueError("Candidate IDs must be nonempty and unique")
        unknown = set(vessel_ids) - set(catalog.name_by_id)
        if unknown:
            raise ValueError("Candidate ID is missing from official catalog")
        names = sorted({catalog.name_by_id[vessel_id] for vessel_id in vessel_ids})
        scores = {name: self.score_pair(name, observed_name) for name in names}
        return [{"candidate_vessel_id": vessel_id,
                 "catalog_name": catalog.name_by_id[vessel_id],
                 **asdict(scores[catalog.name_by_id[vessel_id]])}
                for vessel_id in sorted(vessel_ids)]
