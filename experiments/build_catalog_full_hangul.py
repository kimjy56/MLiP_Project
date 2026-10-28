# -*- coding: utf-8 -*-
"""Offline GPU generator: Official English catalog name -> Top-2 Synthetic Hangul names."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from name_decoder.catalog import canonical_name, has_hangul
from name_decoder.noisy_channel import (
    MANIFEST_NAME,
    PROMPT_VERSION_V2,
    load_channel_tokenizer,
)


def _patch_lfm2_reorder_cache() -> None:
    """Fix transformers LFM2 reorder_cache IndexError on empty conv-layer key_cache tensors."""
    try:
        from transformers.models.lfm2 import modeling_lfm2
    except ImportError:
        return

    for attr_name in dir(modeling_lfm2):
        cls = getattr(modeling_lfm2, attr_name)
        if isinstance(cls, type) and hasattr(cls, "reorder_cache"):
            def _safe_reorder_cache(self, beam_idx: torch.LongTensor) -> None:
                for layer_idx in range(len(getattr(self, "key_cache", []))):
                    k_tensor = self.key_cache[layer_idx]
                    if isinstance(k_tensor, torch.Tensor) and k_tensor.numel() > 0:
                        dev = k_tensor.device
                        self.key_cache[layer_idx] = k_tensor.index_select(0, beam_idx.to(dev))
                    v_tensor = self.value_cache[layer_idx]
                    if isinstance(v_tensor, torch.Tensor) and v_tensor.numel() > 0:
                        dev = v_tensor.device
                        self.value_cache[layer_idx] = v_tensor.index_select(0, beam_idx.to(dev))
                for attr in ("conv_cache", "conv_states"):
                    c_list = getattr(self, attr, None)
                    if isinstance(c_list, list):
                        for layer_idx in range(len(c_list)):
                            c_tensor = c_list[layer_idx]
                            if isinstance(c_tensor, torch.Tensor) and c_tensor.numel() > 0:
                                dev = c_tensor.device
                                c_list[layer_idx] = c_tensor.index_select(0, beam_idx.to(dev))

            cls.reorder_cache = _safe_reorder_cache


class HangulOnlyLogitsProcessor(LogitsProcessor):
    """Mask out tokens containing Latin alphabet characters so generation only emits Hangul/digits/spaces."""

    def __init__(self, tokenizer, eos_token_id: int) -> None:
        vocab_size = max(len(tokenizer), max(tokenizer.get_vocab().values()) + 1)
        banned = torch.zeros(vocab_size, dtype=torch.bool)
        re_latin = re.compile(r"[A-Za-z]")
        for tok_id in range(len(tokenizer)):
            if tok_id == eos_token_id:
                continue
            piece = tokenizer.decode([tok_id], skip_special_tokens=False)
            if re_latin.search(piece):
                banned[tok_id] = True
        self.banned_mask = banned

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = self.banned_mask.to(scores.device)
        if mask.shape[0] <= scores.shape[-1]:
            scores[:, : mask.shape[0]] = scores[:, : mask.shape[0]].masked_fill(mask, float("-inf"))
        else:
            scores = scores.masked_fill(mask[: scores.shape[-1]], float("-inf"))
        return scores


def encode_ko_generation_prompt(tokenizer, catalog_name: str, prompt_version: str) -> List[int]:
    name = canonical_name(catalog_name)
    if prompt_version == PROMPT_VERSION_V2:
        payload = {"task": "KO_TRANSLITERATION", "catalog_name": name}
    else:
        payload = {"catalog_name": name}
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "Write one possible observed vessel name from a port record for "
                    "this official catalog name. Output only the observed name.\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def main() -> None:
    _patch_lfm2_reorder_cache()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "artifacts/LM-LFM25-FULL-R24-01/model",
        help="Path to merged reverse noisy-channel model directory",
    )
    parser.add_argument("--catalog-csv", type=Path, default=ROOT / "data/vessel_catalog.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "AR-06/catalog_full_ko_map.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=2, help="Top-K candidates via first-token branching")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    manifest_path = args.model_dir / MANIFEST_NAME
    prompt_version = PROMPT_VERSION_V2
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        prompt_version = manifest.get("prompt_version", PROMPT_VERSION_V2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )

    tokenizer = load_channel_tokenizer(args.model_dir)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=dtype
    ).to(device)
    model.eval()

    hangul_proc = HangulOnlyLogitsProcessor(tokenizer, eos_id)
    logits_processors = LogitsProcessorList([hangul_proc])

    cat_df = pd.read_csv(args.catalog_csv, encoding="utf-8-sig")
    unique_names = sorted(
        {canonical_name(str(n)) for n in cat_df["vessel_name"].dropna() if canonical_name(str(n))}
    )

    name_to_ko: Dict[str, List[str]] = {}
    if args.output.exists():
        name_to_ko = json.loads(args.output.read_text(encoding="utf-8"))

    pending = [n for n in unique_names if n not in name_to_ko]
    k_branch = max(1, args.num_candidates)
    print(
        f"[Device={device}, dtype={dtype}, prompt={prompt_version}, top_k={k_branch}] "
        f"Total unique catalog names={len(unique_names)}, Pending={len(pending)}",
        flush=True,
    )

    for start in range(0, len(pending), args.batch_size):
        batch_names = pending[start : start + args.batch_size]
        bsz = len(batch_names)
        prompt_seqs = [encode_ko_generation_prompt(tokenizer, n, prompt_version) for n in batch_names]
        max_p_len = max(len(seq) for seq in prompt_seqs)

        input_ids = torch.full((bsz, max_p_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for r_idx, seq in enumerate(prompt_seqs):
            input_ids[r_idx, max_p_len - len(seq) :] = torch.tensor(seq, dtype=torch.long, device=device)
            attention_mask[r_idx, max_p_len - len(seq) :] = 1

        if k_branch == 1:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                logits_processor=logits_processors,
            )
            gen_Slices = outputs[:, max_p_len:]
        else:
            # First-token Top-K branching + Greedy continuation (avoids LFM2 beam reorder_cache & 2x faster)
            first_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            next_logits = hangul_proc(input_ids, first_out.logits[:, -1, :].float())
            topk_ids = torch.topk(next_logits, k=k_branch, dim=-1).indices  # [bsz, k_branch]

            rep_inputs = input_ids.repeat_interleave(k_branch, dim=0)
            rep_mask = attention_mask.repeat_interleave(k_branch, dim=0)
            first_tokens = topk_ids.reshape(bsz * k_branch, 1)

            branched_ids = torch.cat([rep_inputs, first_tokens], dim=1)
            branched_mask = torch.cat([rep_mask, torch.ones((bsz * k_branch, 1), dtype=torch.long, device=device)], dim=1)

            outputs = model.generate(
                input_ids=branched_ids,
                attention_mask=branched_mask,
                max_new_tokens=max(1, args.max_new_tokens - 1),
                num_beams=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                logits_processor=logits_processors,
            )
            gen_Slices = outputs[:, max_p_len:]

        for b_idx, cname in enumerate(batch_names):
            cands: List[str] = []
            for k in range(k_branch):
                gen_ids = gen_Slices[b_idx * k_branch + k]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                text = text.splitlines()[0].strip() if text else ""
                clean_ko = re.sub(r"[^가-힣0-9\s]", "", text).strip()
                clean_ko = re.sub(r"\s+", " ", clean_ko)
                if clean_ko and has_hangul(clean_ko) and clean_ko not in cands:
                    cands.append(clean_ko)
            if cands:
                name_to_ko[cname] = cands

        if (start // args.batch_size) % 20 == 0 or start + args.batch_size >= len(pending):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(name_to_ko, ensure_ascii=False), encoding="utf-8")
            print(f"Generated {min(start + args.batch_size, len(pending))}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()