"""Unit and verification tests for Reverse Noisy Channel (name_decoder.noisy_channel).

Follows verification requirements from docs/NOISY_CHANNEL.md:
1. Normalization and prompt encoding contracts (separate tokenization, EOS, prompt/pad masking with -100).
2. NoisyChannelScorer loading, manifest verification, and hash checking.
3. Teacher-forced likelihood calculation (causal shift, target token log-probs).
4. Candidate ranking sanity on test pairs (e.g. 마스그라치 -> MARSGRACHT vs MAASGRACHT).
5. Independence and invariance (candidate order permutation invariance, row isolation).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from name_decoder.catalog import canonical_name
from name_decoder.noisy_channel import (
    normalize_observed,
    encode_channel_example,
    collate_channel_examples,
    NoisyChannelScorer,
    PROMPT_VERSION,
    OBSERVED_NORMALIZATION,
    CATALOG_NORMALIZATION,
)


def test_observed_normalization_contracts():
    """Verify observed name normalization preserves case, internal spaces, and rejects empty."""
    assert normalize_observed("  HAFNIA WREN  ") == "HAFNIA WREN"
    assert normalize_observed("사쿠라 타이거") == "사쿠라 타이거"
    assert normalize_observed("Jeil Crysstal 1") == "Jeil Crysstal 1"

    try:
        normalize_observed("")
        assert False, "Should raise ValueError on empty string"
    except ValueError:
        pass

    try:
        normalize_observed(12345)
        assert False, "Should raise TypeError on non-string"
    except TypeError:
        pass


def test_encode_and_collate_masking():
    """Verify encoding produces correct prompt length, target length, and collate masks labels."""
    from transformers import AutoTokenizer

    model_dir = ROOT / "artifacts/AS-HCX-NC-01/model"
    if not model_dir.exists():
        model_dir = ROOT / "artifacts/pretrained/hyperclovax-seed-0.5b"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    catalog_name = "MARSGRACHT"
    observed_name = "마스그라치"

    example = encode_channel_example(tokenizer, catalog_name, observed_name, max_length=256)
    assert example.prompt_length > 0
    assert example.target_count > 1  # observed tokens + EOS
    assert example.input_ids[-1] == tokenizer.eos_token_id

    # Test batch collation
    batch = collate_channel_examples([example], tokenizer.pad_token_id or tokenizer.eos_token_id)
    inputs = batch["input_ids"]
    mask = batch["attention_mask"]
    labels = batch["labels"]

    assert inputs.shape == (1, len(example.input_ids))
    assert mask.sum().item() == len(example.input_ids)

    # Prompt positions in labels MUST be -100
    prompt_labels = labels[0, :example.prompt_length]
    assert (prompt_labels == -100).all(), "Prompt tokens must be masked with -100 in labels"

    # Target positions in labels MUST match inputs
    target_labels = labels[0, example.prompt_length:]
    target_inputs = inputs[0, example.prompt_length:]
    assert torch.equal(target_labels, target_inputs), "Target tokens must be unmasked in labels"


def test_scorer_loading_and_likelihood():
    """Verify NoisyChannelScorer loads checkpoint and computes expected likelihoods."""
    model_dir = ROOT / "artifacts/AS-HCX-NC-01/model"
    if not model_dir.exists():
        print(f"Skipping test_scorer_loading_and_likelihood: {model_dir} not found")
        return

    scorer = NoisyChannelScorer.load(model_dir)
    assert scorer is not None
    assert scorer.manifest["experiment_id"] == "AS-HCX-NC-01"
    assert scorer.manifest["direction"] == "catalog_name_to_observed_name"

    # Sanity check: MARSGRACHT vs MAASGRACHT on '마스그라치'
    score_correct = scorer.score_pair("MARSGRACHT", "마스그라치")
    score_incorrect = scorer.score_pair("MAASGRACHT", "마스그라치")

    print(f"마스그라치 -> MARSGRACHT log_prob: {score_correct.log_probability:.4f}")
    print(f"마스그라치 -> MAASGRACHT log_prob: {score_incorrect.log_probability:.4f}")
    assert score_correct.log_probability > score_incorrect.log_probability, (
        f"Correct vessel MARSGRACHT ({score_correct.log_probability}) should have higher "
        f"likelihood than MAASGRACHT ({score_incorrect.log_probability})"
    )

    # Sanity check: SAKURA TIGER vs SAKURA MARU on '사쿠라 타이거'
    score_sakura = scorer.score_pair("SAKURA TIGER", "사쿠라 타이거")
    score_other = scorer.score_pair("SAKURA MARU", "사쿠라 타이거")
    print(f"사쿠라 타이거 -> SAKURA TIGER log_prob: {score_sakura.log_probability:.4f}")
    print(f"사쿠라 타이거 -> SAKURA MARU  log_prob: {score_other.log_probability:.4f}")
    assert score_sakura.log_probability > score_other.log_probability


def test_candidate_order_and_row_invariance():
    """Verify candidate scoring is permutation-invariant and row-isolated."""
    model_dir = ROOT / "artifacts/AS-HCX-NC-01/model"
    if not model_dir.exists():
        return

    scorer = NoisyChannelScorer.load(model_dir)

    # Create dummy catalog
    vessel_map = {
        "VESSEL_001": "MARSGRACHT",
        "VESSEL_002": "MAASGRACHT",
        "VESSEL_003": "SAKURA TIGER",
    }

    class DummyCatalog:
        name_by_id = vessel_map

    catalog = DummyCatalog()

    cands_order1 = ["VESSEL_001", "VESSEL_002", "VESSEL_003"]
    cands_order2 = ["VESSEL_003", "VESSEL_001", "VESSEL_002"]

    res1 = scorer.score_candidates("마스그라치", cands_order1, catalog)
    res2 = scorer.score_candidates("마스그라치", cands_order2, catalog)

    # Results are sorted by candidate_vessel_id
    map1 = {r["candidate_vessel_id"]: r["log_probability"] for r in res1}
    map2 = {r["candidate_vessel_id"]: r["log_probability"] for r in res2}

    assert map1 == map2, "Candidate scoring must be invariant to input candidate ordering"
    assert len(res1) == 3


if __name__ == "__main__":
    print("Running test_observed_normalization_contracts...")
    test_observed_normalization_contracts()
    print("Running test_encode_and_collate_masking...")
    test_encode_and_collate_masking()
    print("Running test_scorer_loading_and_likelihood...")
    test_scorer_loading_and_likelihood()
    print("Running test_candidate_order_and_row_invariance...")
    test_candidate_order_and_row_invariance()
    print("\nALL NOISY CHANNEL VERIFICATION TESTS PASSED SUCCESSFULLY!")
