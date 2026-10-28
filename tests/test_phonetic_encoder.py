"""
Unit Tests for phonetic_encoder package (SPEC-DPR-01).

Tests:
  1. Tokenizer & Multi-View String Formatting
  2. Tokenizer Batch Encoding & Tensor Padding
  3. Model Forward Pass & L2 Normalization (128-d)
  4. Row Independence (batch_size=1 vs batch_size=8 numerical invariance)
  5. Trimmed Weight Transfer from Symphonym-v8
  6. Hybrid Contrastive Loss & Backward Gradient Propagation
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from phonetic_encoder.dataset import PhoneticPairDataset, make_collate_fn
from phonetic_encoder.loss import HybridContrastiveLoss, SymmetricInfoNCELoss
from phonetic_encoder.model import CharPhoneticBiEncoder, UniversalEncoder
from phonetic_encoder.tokenizer import (
    PhoneticTokenizer,
    decompose_hangul,
    en_to_phonetic,
    format_hangul_query,
    format_latin_catalog,
    hangul_to_rom,
    syllable_to_rom,
)


def test_multi_view_formatting():
    # 1. Hangul decomposition
    jamo = decompose_hangul("할사이언")
    assert jamo == "ㅎㅏㄹㅅㅏㅇㅣㅇㅓㄴ"

    # 2. Romanization
    rom = syllable_to_rom("할")
    assert rom == "HAL"

    full_rom = hangul_to_rom("할사이언")
    assert "HAL" in full_rom

    # 3. English phonetic skeleton
    phon = en_to_phonetic("HALCYON")
    assert len(phon) > 0 and phon == "HARSYON"

    # 4. Multi-view strings
    q_fmt = format_hangul_query("할사이언 2호")
    assert "[KOR]" in q_fmt and "[JAMO]" in q_fmt and "[ROM]" in q_fmt
    assert "ㅎㅏㄹㅅㅏㅇㅣㅇㅓㄴ" in q_fmt

    c_fmt = format_latin_catalog("HALCYON TWO")
    assert "[ENG]" in c_fmt and "[PHON]" in c_fmt


def test_tokenizer_batch_encoding():
    tok = PhoneticTokenizer()
    items = [
        ("[KOR] 한나 [JAMO] ㅎㅏㄴㄴㅏ [ROM] HANNA", "HANGUL", "ko"),
        ("[ENG] HANNAH [PHON] HN", "LATIN", "en"),
        ("[KOR] 나이트 [JAMO] ㄴㅏㅇㅣㅌㅡ [ROM] NAITEU", "HANGUL", "ko"),
    ]
    batch = tok.batch_encode(items, max_length=64)

    assert batch["char_ids"].shape[0] == 3
    assert batch["lengths"].shape[0] == 3
    assert (batch["lengths"] > 0).all()
    assert batch["script_ids"].shape[0] == 3
    assert batch["lang_ids"].shape[0] == 3


def test_model_forward_and_l2_norm():
    tok = PhoneticTokenizer()
    model = UniversalEncoder(
        vocab_size=tok.vocab_size,
        num_scripts=tok.num_scripts,
        num_langs=tok.num_langs,
        embed_dim=128,
        dropout=0.0,
    )
    model.eval()

    items = [
        ("[KOR] 동방 [JAMO] ㄷㅗㅇㅂㅏㅇ [ROM] DONGBANG", "HANGUL", "ko"),
        ("[ENG] DONG BANG [PHON] TNK PNK", "LATIN", "en"),
    ]
    batch = tok.batch_encode(items)

    with torch.no_grad():
        out = model(**batch)

    assert out.shape == (2, 128)
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_row_independence():
    """
    CRITICAL TEST for eval_kit/check_submission.py compliance:
    Sample evaluated alone must produce IDENTICAL output to sample evaluated within a batch.
    """
    tok = PhoneticTokenizer()
    model = UniversalEncoder(
        vocab_size=tok.vocab_size,
        num_scripts=tok.num_scripts,
        num_langs=tok.num_langs,
        embed_dim=128,
        dropout=0.0,
    )
    model.eval()

    sample_target = ("[KOR] 모닝 레나 [JAMO] ㅁㅗㄴㅣㅇ ㄹㅔㄴㅏ [ROM] MONING RENA", "HANGUL", "ko")
    distractor_samples = [
        ("[ENG] MORNING LENA [PHON] MRNNK LN", "LATIN", "en"),
        ("[ENG] MORNING CARINA [PHON] MRNNK KRN", "LATIN", "en"),
        ("[KOR] 씨 울프 [JAMO] ㅆㅣ ㅇㅜㄹㅍㅡ [ROM] SSI ULPEU", "HANGUL", "ko"),
        ("[ENG] SEA WOLF [PHON] S WLF", "LATIN", "en"),
    ]

    # Run single row
    single_batch = tok.batch_encode([sample_target])
    with torch.no_grad():
        emb_single = model(**single_batch)

    # Run as element 0 in batch
    multi_batch = tok.batch_encode([sample_target] + distractor_samples)
    with torch.no_grad():
        emb_multi = model(**multi_batch)

    max_abs_diff = (emb_single[0] - emb_multi[0]).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(emb_single[0:1], emb_multi[0:1]).item()

    assert max_abs_diff < 1e-6, f"Row independence failed: max diff = {max_abs_diff}"
    assert cos_sim > 0.999999, f"Row independence cosine similarity degraded: {cos_sim}"


def test_loss_backward():
    loss_fn = HybridContrastiveLoss(temperature=0.05, margin=0.15, margin_weight=0.5)

    B, D = 4, 128
    q_embs = torch.randn(B, D, requires_grad=True)
    c_embs = torch.randn(B, D, requires_grad=True)
    q_norm = torch.nn.functional.normalize(q_embs, p=2, dim=-1)
    c_norm = torch.nn.functional.normalize(c_embs, p=2, dim=-1)

    loss = loss_fn(q_norm, c_norm)
    assert loss.item() > 0.0
    loss.backward()
    assert q_embs.grad is not None and c_embs.grad is not None


def test_trimmed_weight_loading_from_hf_cache():
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    matches = list(hf_cache_dir.glob("models--docuracy--symphonym-v8/snapshots/*"))
    if not matches:
        print("Local HuggingFace snapshot for docuracy/symphonym-v8 not found; skipping.")
        return

    snapshot_path = matches[0]
    bi_encoder = CharPhoneticBiEncoder.from_symphonym_pretrained(snapshot_path)
    assert bi_encoder.encoder.embed_dim == 128

    # Check zero-shot query vs catalog
    q_vec = bi_encoder.encode_queries(["나이트"])
    c_vec = bi_encoder.encode_catalog(["KNIGHT"])
    sim = float(np.dot(q_vec[0], c_vec[0]))
    assert sim > 0.5, f"Expected strong phonetic similarity for 나이트/KNIGHT, got {sim}"


def test_spec_dpr_02_components():
    """
    SPEC-DPR-02 Enhancements Verification:
      1. Letter-Spell auto-conversion & Semivowel W alignment
      2. Core/Tail token extraction
      3. SyllableConvBlock & Multi-Query Attention Pooling zero-init invariance
      4. GatedMultiViewFusion forward & gradient backward
      5. SupConCosFaceHardLoss with False Negative Masking & ANCE hard negatives
    """
    from phonetic_encoder.loss import SupConCosFaceHardLoss
    from phonetic_encoder.model import GatedMultiViewFusion, SyllableConvBlock
    from phonetic_encoder.tokenizer import extract_tail_core_text

    # 1. Letter-Spell auto-conversion
    assert hangul_to_rom("씨엔씨 드림") == "CNC DEURIM"
    assert hangul_to_rom("에스케이 프론티어") == "SK PEURONTIEO"

    # 2. Semivowel W alignment
    assert en_to_phonetic("WOLF") == "URP"
    assert "URP" in en_to_phonetic(hangul_to_rom("울프"))

    # 3. Core/Tail token extraction
    assert extract_tail_core_text("글로비스 심포니") == "심포니"
    assert extract_tail_core_text("GLOVIS SYMPHONY") == "SYMPHONY"
    assert extract_tail_core_text("씨 울프") == "울프"
    assert extract_tail_core_text("SEA WOLF") == "WOLF"
    assert extract_tail_core_text("모닝 레나 2호") == "레나"
    assert extract_tail_core_text("한나") == "한나"

    # 4. Zero-Init SyllableConvBlock invariance
    conv = SyllableConvBlock(dim=128)
    dummy_x = torch.randn(2, 10, 128)
    dummy_mask = torch.ones(2, 10, dtype=torch.bool)
    dummy_mask[1, 7:] = False
    with torch.no_grad():
        out = conv(dummy_x, dummy_mask)
    assert torch.allclose(out, dummy_x, atol=1e-7), "Zero-init gate must produce identical tensor"

    # 5. GatedMultiViewFusion
    fusion = GatedMultiViewFusion(embed_dim=128)
    e_p = torch.randn(4, 128, requires_grad=True)
    e_s = torch.randn(4, 128, requires_grad=True)
    e_t = torch.randn(4, 128, requires_grad=True)
    fused = fusion(
        torch.nn.functional.normalize(e_p, p=2, dim=-1),
        torch.nn.functional.normalize(e_s, p=2, dim=-1),
        torch.nn.functional.normalize(e_t, p=2, dim=-1),
    )
    assert fused.shape == (4, 128)
    fused_norm = torch.norm(fused, p=2, dim=-1)
    assert torch.allclose(fused_norm, torch.ones_like(fused_norm), atol=1e-5)

    # 6. SupConCosFaceHardLoss with False Negative Masking & Hard Negatives
    loss_fn = SupConCosFaceHardLoss(temperature=0.05, margin=0.08, hard_gamma=0.20, hard_weight=0.75)
    B, K, D = 4, 4, 128
    q_leaf = torch.randn(B, D, requires_grad=True)
    c_leaf = torch.randn(B, D, requires_grad=True)
    h_leaf = torch.randn(B, K, D, requires_grad=True)
    q_dummy = torch.nn.functional.normalize(q_leaf, p=2, dim=-1)
    c_dummy = torch.nn.functional.normalize(c_leaf, p=2, dim=-1)
    h_dummy = torch.nn.functional.normalize(h_leaf, p=2, dim=-1)
    vids = ["vessel_A", "vessel_B", "vessel_C", "vessel_A"]  # 0 and 3 are identical vessel

    loss = loss_fn(q_dummy, c_dummy, hard_neg_embs=h_dummy, vessel_ids=vids)
    assert loss.item() > 0.0
    loss.backward()
    assert q_leaf.grad is not None and c_leaf.grad is not None and h_leaf.grad is not None
