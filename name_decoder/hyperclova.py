"""Offline HyperCLOVA X vessel name decoder with Beam Search and Catalog Search."""
from collections import defaultdict
from pathlib import Path
import unicodedata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .catalog import NameCandidate, NamePrediction, canonical_name
from .catalog_search import CatalogNameSearch, SCORERS


def build_chatml_prompt(source: str) -> str:
    """Build standard ChatML prompt for HyperCLOVA X vessel transliteration."""
    return (
        f"<|im_start|>system\n"
        f"한국어 선박명을 국제 카탈로그의 공식 영문 대문자 선박명으로 1:1 변환합니다. "
        f"오직 영문 공식 선박명만 대문자로 한 줄로 출력하세요.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"입력: {source}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"출력:\n"
    )


class HyperCLOVANameDecoder:
    """Decodes vessel names using fine-tuned HyperCLOVA X with beam search and catalog lookup."""

    def __init__(
        self,
        model,
        tokenizer,
        catalog,
        num_beams: int = 4,
        num_return_sequences: int = 4,
        max_new_tokens: int = 20,
        scorer: str = "ratio",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.catalog = catalog
        self.num_beams = max(1, num_beams)
        self.num_return_sequences = min(max(1, num_return_sequences), self.num_beams)
        self.max_new_tokens = max_new_tokens
        self.scorer = scorer
        self.catalog_search = CatalogNameSearch(catalog.names, scorer=scorer)

    @classmethod
    def load(
        cls,
        checkpoint,
        catalog,
        num_beams: int = 4,
        num_return_sequences: int = 4,
        max_new_tokens: int = 20,
        scorer: str = "ratio",
        device: str = "cpu",
    ):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_dir():
            raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.float32, device_map=device, local_files_only=True
        )
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            catalog=catalog,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            scorer=scorer,
        )

    def _clean_text(self, raw_text: str) -> str:
        clean = raw_text.strip().split("\n")[0].strip().upper()
        return " ".join(unicodedata.normalize("NFC", clean).split())

    @torch.inference_mode()
    def decode(self, name: str, top_k: int = 10) -> list[NameCandidate]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        source = canonical_name(name)
        if not source:
            raise ValueError("Empty input name")

        prompt = build_chatml_prompt(source)
        inp = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.num_beams > 1:
            gen_kwargs["num_beams"] = self.num_beams
            gen_kwargs["num_return_sequences"] = self.num_return_sequences

        out = self.model.generate(**inp, **gen_kwargs)

        input_len = inp["input_ids"].shape[1]
        gen_texts = []
        for seq in out:
            raw = self.tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            clean = self._clean_text(raw)
            if clean:
                gen_texts.append(clean)
        if not gen_texts:
            gen_texts = [""]

        # If single sequence, direct top_matches
        if len(gen_texts) == 1:
            matches = self.catalog_search.top_matches(gen_texts[0], limit=top_k)
            return [
                NameCandidate(m_name, m_score, self.catalog.ids_by_name[m_name])
                for m_name, m_score in matches
            ]

        # Multiple beam sequences: score aggregation with rank discount factor
        candidate_scores = defaultdict(float)
        for rank_idx, text in enumerate(gen_texts):
            rank_weight = 1.0 / (1.0 + 0.1 * rank_idx)
            matches = self.catalog_search.top_matches(text, limit=top_k)
            for m_name, m_score in matches:
                weighted_score = m_score * rank_weight
                if weighted_score > candidate_scores[m_name]:
                    candidate_scores[m_name] = weighted_score

        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda kv: (-kv[1], kv[0])
        )[:top_k]

        return [
            NameCandidate(c_name, c_score, self.catalog.ids_by_name[c_name])
            for c_name, c_score in sorted_candidates
        ]

    def decode_many(self, names: list[str], top_k: int = 10) -> list[list[NameCandidate]]:
        return [self.decode(n, top_k=top_k) for n in names]

    def predict(self, name: str, top_k: int = 10) -> NamePrediction:
        candidates = tuple(self.decode(name, top_k=top_k))
        if not candidates:
            raise RuntimeError(f"Decoder returned no catalog candidates for: {name}")
        return NamePrediction(
            predicted_name=candidates[0].name,
            vessel_ids=candidates[0].vessel_ids,
            candidates=candidates,
        )

    def predict_many(self, names: list[str], top_k: int = 10) -> list[NamePrediction]:
        return [self.predict(n, top_k=top_k) for n in names]
