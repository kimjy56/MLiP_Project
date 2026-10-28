"""
Char-Phonetic Bi-Encoder Model Architecture (SPEC-DPR-01).

Based on Symphonym-v8 UniversalEncoder:
  - Input: Character, Script, Language, Length Embeddings (Input dim 104 -> Hidden dim 128)
  - Backbone: 2-Layer Bidirectional LSTM (256-d output)
  - Contextual: Residual Multi-Head Self-Attention
  - Aggregation: Parametric Attention Pooling
  - Projection Head: Linear(256 -> 128) -> ReLU -> Linear(128 -> 128) -> LayerNorm(128) -> L2 Norm
  - 100% Batch-Independent: Uses ONLY LayerNorm (Zero BatchNorm).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import (
    PhoneticTokenizer,
    clean_vessel_name,
    format_hangul_query,
    format_latin_catalog,
    hangul_to_rom,
    canonicalize_numerals,
    extract_acronym_view,
    extract_initial_view,
    extract_semantic_view,
    expand_latin_acronym,
    expand_latin_initials,
    expand_latin_semantics,
)


class SelfAttention(nn.Module):
    """Multi-Head Self-Attention over sequence representation."""

    def __init__(self, hidden_dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, H = x.shape

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        Q = reshape(self.q_proj(x))
        K = reshape(self.k_proj(x))
        V = reshape(self.v_proj(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        w = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(w, V).transpose(1, 2).contiguous().view(B, L, H)
        return self.out_proj(out), w


class AttentionPooling(nn.Module):
    """Parametric Attention-based sequence pooling into a single vector."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=-1)
        pooled = torch.bmm(w.unsqueeze(1), x).squeeze(1)
        return pooled, w


class UniversalEncoder(nn.Module):
    """
    Symphonym UniversalEncoder adapted for 100% row-independent inference.
    """

    def __init__(
        self,
        vocab_size: int = 200,
        num_scripts: int = 3,
        num_langs: int = 3,
        char_embed_dim: int = 64,
        script_embed_dim: int = 16,
        lang_embed_dim: int = 16,
        length_embed_dim: int = 8,
        num_length_buckets: int = 16,
        hidden_dim: int = 128,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_attention_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_length_buckets = num_length_buckets

        self.char_embed = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        self.script_embed = nn.Embedding(num_scripts, script_embed_dim)
        self.lang_embed = nn.Embedding(num_langs, lang_embed_dim, padding_idx=0)
        self.length_embed = nn.Embedding(num_length_buckets, length_embed_dim)

        input_dim = char_embed_dim + script_embed_dim + lang_embed_dim + length_embed_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.bilstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.self_attention = SelfAttention(hidden_dim * 2, num_attention_heads, dropout)
        self.pooling = AttentionPooling(hidden_dim * 2, dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _length_bucket(self, lengths: torch.Tensor) -> torch.Tensor:
        buckets = (lengths.to(torch.long) - 1) // 2
        return buckets.clamp(0, self.num_length_buckets - 1)

    def forward(
        self,
        char_ids: torch.Tensor,
        script_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, L = char_ids.shape
        device = char_ids.device
        mask = torch.arange(L, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)

        c_emb = self.char_embed(char_ids)
        s_emb = self.script_embed(script_ids).unsqueeze(1).expand(-1, L, -1)
        l_emb = self.lang_embed(lang_ids).unsqueeze(1).expand(-1, L, -1)
        lb = self._length_bucket(lengths)
        len_emb = self.length_embed(lb.to(device)).unsqueeze(1).expand(-1, L, -1)

        x = torch.cat([c_emb, s_emb, l_emb, len_emb], dim=-1)
        x = self.input_norm(self.input_proj(x))

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=L)

        attended, _ = self.self_attention(lstm_out, mask)
        attended = attended + lstm_out
        pooled, _ = self.pooling(attended, mask)
        emb = self.output_proj(pooled)
        return F.normalize(emb, p=2, dim=-1)


class CharPhoneticBiEncoder(nn.Module):
    """
    High-level Bi-Encoder wrapper managing weights, tokenization, and inference.
    """

    def __init__(self, tokenizer: Optional[PhoneticTokenizer] = None, model: Optional[UniversalEncoder] = None):
        super().__init__()
        self.tokenizer = tokenizer if tokenizer is not None else PhoneticTokenizer()
        if model is not None:
            self.encoder = model
        else:
            self.encoder = UniversalEncoder(
                vocab_size=self.tokenizer.vocab_size,
                num_scripts=self.tokenizer.num_scripts,
                num_langs=self.tokenizer.num_langs,
            )

    @classmethod
    def from_symphonym_pretrained(
        cls,
        model_dir_or_snapshot: Union[str, Path],
        device: str = "cpu",
    ) -> "CharPhoneticBiEncoder":
        """
        Load pretrained weights from Symphonym-v8 with automatic vocabulary trimming.
        """
        model_path = Path(model_dir_or_snapshot)
        safetensors_file = model_path / "model.safetensors"
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_file))

        # Load original vocabularies
        with open(model_path / "vocab" / "char_vocab.json", "r", encoding="utf-8") as f:
            raw_cv = json.load(f)
        orig_c2i = raw_cv.get("char_to_id", raw_cv)

        with open(model_path / "vocab" / "lang_vocab.json", "r", encoding="utf-8") as f:
            raw_lv = json.load(f)
        orig_l2i = raw_lv.get("lang_to_id", raw_lv)

        with open(model_path / "vocab" / "script_vocab.json", "r", encoding="utf-8") as f:
            raw_sv = json.load(f)
        orig_s2i = raw_sv.get("script_to_id", raw_sv)

        # Create trimmed tokenizer
        tokenizer = PhoneticTokenizer()
        encoder = UniversalEncoder(
            vocab_size=tokenizer.vocab_size,
            num_scripts=tokenizer.num_scripts,
            num_langs=tokenizer.num_langs,
            dropout=0.0,
        )

        # Copy non-embedding weights directly
        filtered_sd = {
            k: v for k, v in state_dict.items()
            if k not in ("char_embed.weight", "script_embed.weight", "lang_embed.weight")
        }
        encoder.load_state_dict(filtered_sd, strict=False)

        # Sliced transfer for character embedding table
        encoder.char_embed.weight.data[0] = state_dict["char_embed.weight"][0]  # PAD
        encoder.char_embed.weight.data[1] = state_dict["char_embed.weight"][1]  # UNK
        encoder.char_embed.weight.data[2] = state_dict["char_embed.weight"][2]  # SPACE

        for ch, new_id in tokenizer.char_to_id.items():
            if new_id > 2:
                orig_id = orig_c2i.get(ch, 1)
                encoder.char_embed.weight.data[new_id] = state_dict["char_embed.weight"][orig_id]

        # Sliced transfer for lang embedding table
        for lang, new_id in tokenizer.lang_to_id.items():
            orig_id = orig_l2i.get(lang, 0)
            encoder.lang_embed.weight.data[new_id] = state_dict["lang_embed.weight"][orig_id]

        # Sliced transfer for script embedding table
        for script, new_id in tokenizer.script_to_id.items():
            orig_id = orig_s2i.get(script, 0)
            encoder.script_embed.weight.data[new_id] = state_dict["script_embed.weight"][orig_id]

        wrapper = cls(tokenizer=tokenizer, model=encoder)
        wrapper.to(device)
        wrapper.eval()
        return wrapper

    def forward(
        self,
        char_ids: torch.Tensor,
        script_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(char_ids, script_ids, lang_ids, lengths)

    def encode_batch(
        self,
        items: List[Tuple[str, str, str]],  # (text, script, lang)
        batch_size: int = 256,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Encode items in batches and return L2-normalized 128-d tensor.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(items), batch_size):
                chunk = items[i : i + batch_size]
                batch_tensors = self.tokenizer.batch_encode(chunk, device=device)
                emb = self.encoder(**batch_tensors)
                embeddings.append(emb.cpu())

        if not embeddings:
            return torch.empty((0, self.encoder.embed_dim))
        return torch.cat(embeddings, dim=0)

    def encode_queries(
        self,
        names: List[str],
        mode: str = "fused",  # "fused", "raw", or "rom"
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Encode Korean vessel names into L2-normalized (N, 128) numpy array.
        When mode="fused", fuses the native Hangul phonetic embedding and Romanized embedding.
        """
        clean_names = [clean_vessel_name(n) for n in names]
        if mode == "raw":
            items = [(n, "HANGUL", "ko") for n in clean_names]
            embs = self.encode_batch(items, batch_size=batch_size)
            return embs.numpy()
        elif mode == "rom":
            items = [(hangul_to_rom(n), "LATIN", "en") for n in clean_names]
            embs = self.encode_batch(items, batch_size=batch_size)
            return embs.numpy()
        else:
            # Multi-view fusion with exception views (SPEC-DPR-01 Multi-View Expansion)
            canon_names = [canonicalize_numerals(clean_vessel_name(n)) for n in names]
            items_ko = [(n, "HANGUL", "ko") for n in canon_names]
            items_rom = [(hangul_to_rom(n), "LATIN", "en") for n in canon_names]
            embs_ko = self.encode_batch(items_ko, batch_size=batch_size)
            embs_rom = self.encode_batch(items_rom, batch_size=batch_size)
            total_emb = embs_ko + embs_rom

            # Additional multi-views
            acr_items = [extract_acronym_view(n) for n in canon_names]
            if any(acr_items):
                batch_acr = [(a if a else " ", "LATIN", "en") for a in acr_items]
                embs_acr = self.encode_batch(batch_acr, batch_size=batch_size)
                mask = torch.tensor([bool(a) for a in acr_items], dtype=torch.float32, device=embs_ko.device).unsqueeze(1)
                total_emb = total_emb + 0.8 * (embs_acr * mask)

            init_items = [extract_initial_view(n) for n in canon_names]
            if any(init_items):
                batch_init = [(a if a else " ", "LATIN", "en") for a in init_items]
                embs_init = self.encode_batch(batch_init, batch_size=batch_size)
                mask = torch.tensor([bool(a) for a in init_items], dtype=torch.float32, device=embs_ko.device).unsqueeze(1)
                total_emb = total_emb + 0.8 * (embs_init * mask)

            sem_items = [extract_semantic_view(n) for n in canon_names]
            if any(sem_items):
                batch_sem = [(a if a else " ", "LATIN", "en") for a in sem_items]
                embs_sem = self.encode_batch(batch_sem, batch_size=batch_size)
                mask = torch.tensor([bool(a) for a in sem_items], dtype=torch.float32, device=embs_ko.device).unsqueeze(1)
                total_emb = total_emb + 0.8 * (embs_sem * mask)

            fused = F.normalize(total_emb, p=2, dim=-1)
            return fused.numpy()

    def encode_catalog(self, names: List[str], batch_size: int = 256) -> np.ndarray:
        """
        Encode English catalog vessel names into L2-normalized (N, 128) numpy array
        with multi-view reverse expansions (acronyms, company initials, semantics, numerals).
        """
        clean_names = [canonicalize_numerals(clean_vessel_name(n).upper()) for n in names]
        items_lat = [(n, "LATIN", "en") for n in clean_names]
        embs_lat = self.encode_batch(items_lat, batch_size=batch_size)
        total_emb = embs_lat

        acr_items = [expand_latin_acronym(n) for n in clean_names]
        if any(acr_items):
            batch_acr = [(a if a else " ", "HANGUL", "ko") for a in acr_items]
            embs_acr = self.encode_batch(batch_acr, batch_size=batch_size)
            mask = torch.tensor([bool(a) for a in acr_items], dtype=torch.float32, device=embs_lat.device).unsqueeze(1)
            total_emb = total_emb + 0.8 * (embs_acr * mask)

        init_items = [expand_latin_initials(n) for n in clean_names]
        if any(init_items):
            batch_init = [(a if a else " ", "HANGUL", "ko") for a in init_items]
            embs_init = self.encode_batch(batch_init, batch_size=batch_size)
            mask = torch.tensor([bool(a) for a in init_items], dtype=torch.float32, device=embs_lat.device).unsqueeze(1)
            total_emb = total_emb + 0.8 * (embs_init * mask)

        sem_items = [expand_latin_semantics(n) for n in clean_names]
        if any(sem_items):
            batch_sem = [(a if a else " ", "HANGUL", "ko") for a in sem_items]
            embs_sem = self.encode_batch(batch_sem, batch_size=batch_size)
            mask = torch.tensor([bool(a) for a in sem_items], dtype=torch.float32, device=embs_lat.device).unsqueeze(1)
            total_emb = total_emb + 0.8 * (embs_sem * mask)

        fused = F.normalize(total_emb, p=2, dim=-1)
        return fused.numpy()

    def save_checkpoint(self, path: Union[str, Path], as_fp16: bool = True):
        """Save trimmed model weights and tokenizer configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if as_fp16:
            saved_state_dict = {k: v.half() for k, v in self.encoder.state_dict().items()}
        else:
            saved_state_dict = {k: v.float() for k, v in self.encoder.state_dict().items()}

        payload = {
            "encoder_state_dict": saved_state_dict,
            "vocab": {
                "char_to_id": self.tokenizer.char_to_id,
                "script_to_id": self.tokenizer.script_to_id,
                "lang_to_id": self.tokenizer.lang_to_id,
            },
            "as_fp16": as_fp16,
            "embed_dim": self.encoder.embed_dim,
        }
        torch.save(payload, str(path))

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], device: str = "cpu") -> "CharPhoneticBiEncoder":
        """Load trimmed model from saved checkpoint."""
        payload = torch.load(str(path), map_location=device)
        tokenizer = PhoneticTokenizer()
        tokenizer.char_to_id = payload["vocab"]["char_to_id"]
        tokenizer.id_to_char = {v: k for k, v in tokenizer.char_to_id.items()}
        tokenizer.script_to_id = payload["vocab"]["script_to_id"]
        tokenizer.lang_to_id = payload["vocab"]["lang_to_id"]

        encoder = UniversalEncoder(
            vocab_size=tokenizer.vocab_size,
            num_scripts=tokenizer.num_scripts,
            num_langs=tokenizer.num_langs,
            embed_dim=payload.get("embed_dim", 128),
            dropout=0.0,
        )
        encoder.load_state_dict(payload["encoder_state_dict"])
        if payload.get("as_fp16", False):
            encoder = encoder.half()
        else:
            encoder = encoder.float()

        wrapper = cls(tokenizer=tokenizer, model=encoder)
        wrapper.to(device)
        wrapper.eval()
        return wrapper
