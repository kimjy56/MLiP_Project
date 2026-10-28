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
import re
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
    en_to_phonetic,
    extract_tail_core_text,
    canonicalize_numerals,
    extract_acronym_view,
    extract_initial_view,
    extract_semantic_view,
    expand_latin_acronym,
    expand_latin_initials,
    expand_latin_semantics,
)


class SyllableConvBlock(nn.Module):
    """
    한글 1음절(2~4자모) 및 영문 이중자음(GHT, PH, SYMP) 국소 패턴 포착 블록 (SPEC-DPR-02).
    패딩 토큰이 유효 토큰에 영향을 주지 않도록 매 스텝 마스크를 곱해 행 독립성(Row Independence)을 100% 보장함.
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Depthwise 1D convolutions (kernel=3: 한글 초중종성 1음절, kernel=5: 영문 자음군/이중모음)
        self.dw_conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dw_conv5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.pw_proj = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()
        # Zero-init gate: Step 0에서 기존 symphonym-v8 가중치 출력과 100% 동일하게 시작
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D), mask: (B, L) bool (True = valid token, False = pad)
        mask_f = mask.unsqueeze(-1).type_as(x)  # (B, L, 1)
        h = self.norm(x) * mask_f
        h_t = h.transpose(1, 2)                 # (B, D, L)
        c3 = self.dw_conv3(h_t).transpose(1, 2) * mask_f
        c5 = self.dw_conv5(h_t).transpose(1, 2) * mask_f
        out = self.pw_proj(self.act(torch.cat([c3, c5], dim=-1))) * mask_f
        return x + self.gate * out


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
    Enhanced with SyllableConvBlock and Multi-Query Attention Pooling (SPEC-DPR-02).
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

        # Zero-Init 1D-CNN SyllableConvBlock (SPEC-DPR-02)
        self.conv_block = SyllableConvBlock(dim=hidden_dim)

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

        # Multi-Query Attention Pooling (4-Head Contextual Pooling)
        self.multi_query_pool = nn.Linear(hidden_dim * 2, 4)
        self.mq_proj = nn.Linear(hidden_dim * 2 * 4, hidden_dim * 2)
        self.mq_gate = nn.Parameter(torch.zeros(1))

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
        x = self.conv_block(x, mask)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=L)

        attended, _ = self.self_attention(lstm_out, mask)
        attended = attended + lstm_out
        base_pooled, _ = self.pooling(attended, mask)

        # Multi-Query Attention Pooling residual (SPEC-DPR-02)
        mq_scores = self.multi_query_pool(attended)  # (B, L, 4)
        if mask is not None:
            mq_scores = mq_scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        mq_weights = F.softmax(mq_scores, dim=1)  # (B, L, 4)
        mq_pooled = torch.einsum("blh,bld->bhd", mq_weights, attended).reshape(B, -1)
        mq_out = self.mq_proj(mq_pooled)
        pooled = base_pooled + self.mq_gate * mq_out

        emb = self.output_proj(pooled)
        return F.normalize(emb, p=2, dim=-1)


class GatedMultiViewFusion(nn.Module):
    """
    End-to-End Learnable Gated Multi-View and Tail Fusion (SPEC-DPR-02).
    Dynamically fuses primary view, secondary view, and core tail token view.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        # 초기값 0.35: 전체 선박명 벡터에 고유 끝단어(Tail) 벡터를 0.35 비중으로 합산
        self.tail_weight = nn.Parameter(torch.tensor(0.35))

    def forward(
        self,
        e_primary: torch.Tensor,   # (B, 128) 한글 자모 뷰 or 영문 원문 뷰
        e_secondary: torch.Tensor, # (B, 128) 한글 로마자/약어 뷰 or 영문 음소 골격 뷰
        e_tail: torch.Tensor,      # (B, 128) 선사 접두사 제거 후 고유 끝단어(Tail) 뷰
    ) -> torch.Tensor:
        g = self.gate_net(torch.cat([e_primary, e_secondary], dim=-1))
        e_mv = F.normalize(g * e_primary + (1.0 - g) * e_secondary, p=2, dim=-1)
        w_tail = torch.clamp(self.tail_weight, 0.1, 0.7)
        return F.normalize(e_mv + w_tail * e_tail, p=2, dim=-1)


class CharPhoneticBiEncoder(nn.Module):
    """
    High-level Bi-Encoder wrapper managing weights, tokenization, and inference.
    """

    def __init__(
        self,
        tokenizer: Optional[PhoneticTokenizer] = None,
        model: Optional[UniversalEncoder] = None,
        fusion: Optional[GatedMultiViewFusion] = None,
    ):
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
        self.fusion = fusion if fusion is not None else GatedMultiViewFusion(embed_dim=self.encoder.embed_dim)

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

    def forward_multiview(
        self,
        primary_batch: Dict[str, torch.Tensor],
        secondary_batch: Dict[str, torch.Tensor],
        tail_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        End-to-End learnable forward pass through UniversalEncoder and GatedMultiViewFusion (SPEC-DPR-02).
        """
        e_prim = self.encoder(**primary_batch)
        e_sec = self.encoder(**secondary_batch)
        e_tail = self.encoder(**tail_batch)
        return self.fusion(e_prim, e_sec, e_tail)

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
        Encode Korean vessel names into L2-normalized (N, 128) numpy array
        using GatedMultiViewFusion with Tail-Token decoupling (SPEC-DPR-02).
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
            re_ko = re.compile(r"[가-힣]")
            canon_names = [canonicalize_numerals(clean_vessel_name(n)) for n in names]
            items_prim = []
            items_sec = []
            items_tail = []
            for n in canon_names:
                if re_ko.search(n):
                    items_prim.append((n, "HANGUL", "ko"))
                    items_sec.append((hangul_to_rom(n), "LATIN", "en"))
                    tail_txt = extract_tail_core_text(n)
                    items_tail.append((hangul_to_rom(tail_txt) if re_ko.search(tail_txt) else tail_txt, "LATIN", "en"))
                else:
                    items_prim.append((n.upper(), "LATIN", "en"))
                    items_sec.append((en_to_phonetic(n.upper()), "LATIN", "en"))
                    items_tail.append((extract_tail_core_text(n.upper()), "LATIN", "en"))

            combined = self.encode_batch(items_prim + items_sec + items_tail, batch_size=batch_size * 3)
            n = len(names)
            embs_prim, embs_sec, embs_tail = combined[:n], combined[n:2*n], combined[2*n:]

            dev = next(self.parameters()).device
            self.eval()
            with torch.no_grad():
                fused = self.fusion(embs_prim.to(dev), embs_sec.to(dev), embs_tail.to(dev))
            return fused.cpu().numpy()

    def encode_catalog(self, names: List[str], batch_size: int = 256) -> np.ndarray:
        """
        Encode English catalog vessel names into L2-normalized (N, 128) numpy array
        using GatedMultiViewFusion with Tail-Token decoupling (SPEC-DPR-02).
        """
        clean_names = [canonicalize_numerals(clean_vessel_name(n).upper()) for n in names]
        items_lat = [(n, "LATIN", "en") for n in clean_names]
        items_phon = [(en_to_phonetic(n), "LATIN", "en") for n in clean_names]
        items_tail = [(extract_tail_core_text(n), "LATIN", "en") for n in clean_names]

        combined = self.encode_batch(items_lat + items_phon + items_tail, batch_size=batch_size * 3)
        n = len(names)
        embs_lat, embs_phon, embs_tail = combined[:n], combined[n:2*n], combined[2*n:]

        dev = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            fused = self.fusion(embs_lat.to(dev), embs_phon.to(dev), embs_tail.to(dev))
        return fused.cpu().numpy()

    def save_checkpoint(self, path: Union[str, Path], as_fp16: bool = True):
        """Save trimmed model weights and tokenizer configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if as_fp16:
            saved_state_dict = {k: v.half() for k, v in self.encoder.state_dict().items()}
            fusion_state_dict = {k: v.half() for k, v in self.fusion.state_dict().items()}
        else:
            saved_state_dict = {k: v.float() for k, v in self.encoder.state_dict().items()}
            fusion_state_dict = {k: v.float() for k, v in self.fusion.state_dict().items()}

        payload = {
            "encoder_state_dict": saved_state_dict,
            "fusion_state_dict": fusion_state_dict,
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
        # TF32 cuDNN accumulation exceeded the requested 1e-6 batch-invariance
        # tolerance on the trained DPR03 model. Inference uses IEEE FP32 math.
        # Training from the original snapshot retains its separately logged settings.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
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
        encoder.load_state_dict(payload["encoder_state_dict"], strict=False)
        if payload.get("as_fp16", False):
            encoder = encoder.half()
        else:
            encoder = encoder.float()

        fusion = GatedMultiViewFusion(embed_dim=encoder.embed_dim)
        if "fusion_state_dict" in payload:
            fusion.load_state_dict(payload["fusion_state_dict"])
        if payload.get("as_fp16", False):
            fusion = fusion.half()
        else:
            fusion = fusion.float()

        wrapper = cls(tokenizer=tokenizer, model=encoder, fusion=fusion)
        wrapper.to(device)
        wrapper.eval()
        return wrapper
