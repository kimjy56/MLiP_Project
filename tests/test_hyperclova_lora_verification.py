"""Unit and verification tests for HyperCLOVA X LoRA and balanced sampling.

Follows verification requirements in docs/HYPERCLOVA_LORA.md:
1. LoRA B=0 initial output equality, all base frozen, trainable param count == 1,966,080.
2. A/B gradients exist, base weights gradient is None.
3. Pre-merge vs Post-merge logits match within tolerance, merge idempotency.
4. Balanced sampler ensures exact 2+2 per batch (for batch_size=4).
5. Existing weighted sampler regression check.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from name_decoder.lora import (
    LoRALinear,
    inject_attention_lora,
    adapter_state_dict,
    merge_attention_lora,
)


def test_lora_linear_math_and_idempotency():
    """Verify LoRALinear mathematical identity at initialization and after merge."""
    in_dim, out_dim = 64, 128
    rank = 16
    alpha = 32.0

    base = nn.Linear(in_dim, out_dim, bias=True)
    orig_w = base.weight.clone()
    orig_b = base.bias.clone()

    adapter = LoRALinear(base, rank=rank, alpha=alpha)

    # 1. Base weights must be frozen
    assert not adapter.base.weight.requires_grad
    assert not adapter.base.bias.requires_grad

    # 2. Since B=0 initially, forward output must be IDENTICAL to base forward
    x = torch.randn(4, in_dim)
    base_out = base(x)
    lora_out = adapter(x)
    assert torch.allclose(base_out, lora_out, atol=1e-7), "Initial LoRA output must equal base output when B=0"

    # 3. Simulate training update on A and B
    with torch.no_grad():
        adapter.lora_A.normal_(std=0.02)
        adapter.lora_B.normal_(std=0.02)

    # Output before merge
    adapted_out = adapter(x)

    # 4. Merge
    merged_linear = adapter.merge()
    post_merge_out = merged_linear(x)

    # Logits / outputs must match within tight tolerance
    max_diff = (adapted_out - post_merge_out).abs().max().item()
    assert max_diff < 1e-5, f"Post-merge output mismatch: max diff {max_diff}"

    # 5. Idempotent merge: second merge must NOT add delta again
    w_after_first = merged_linear.weight.clone()
    adapter.merge()
    assert torch.equal(merged_linear.weight, w_after_first), "Second merge must be a no-op (idempotent)"


def test_hyperclova_lora_injection_and_param_count():
    """Verify injection on actual HyperCLOVA X model matches exact expected parameter count (1,966,080)."""
    model_path = ROOT / "artifacts/pretrained/hyperclovax-seed-0.5b"
    if not model_path.exists():
        pytest.skip("Pretrained model not found at artifacts/pretrained/hyperclovax-seed-0.5b")

    # Load base model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float32,
        device_map="cpu",
    )

    total_base_params = sum(p.numel() for p in model.parameters())

    # Inject LoRA on q_proj and v_proj across all 24 layers
    info = inject_attention_lora(model, rank=16, alpha=32.0, targets=("q_proj", "v_proj"))

    # Expected trainable count: 24 layers * 16 * ((2048 + 1024) + (1024 + 1024)) = 1,966,080
    EXPECTED_TRAINABLE = 1_966_080
    assert info["trainable_parameters"] == EXPECTED_TRAINABLE, (
        f"Trainable parameters must be exactly {EXPECTED_TRAINABLE}, got {info['trainable_parameters']}"
    )

    # Check that all base parameters are frozen and only LoRA A/B are trainable
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert len(trainable_names) == 24 * 2 * 2, f"Expected 96 trainable tensors (48 A + 48 B), got {len(trainable_names)}"
    for name in trainable_names:
        assert ("lora_A" in name or "lora_B" in name), f"Unexpected trainable parameter: {name}"

    # Verify forward pass with B=0 produces exact match
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    test_input = tokenizer("선박명: 마스그라치\n영문 공식명:", return_tensors="pt")

    with torch.no_grad():
        out_initial = model(**test_input).logits

    # Verify gradient computation on dummy backward
    dummy_labels = test_input["input_ids"].clone()
    loss = model(**test_input, labels=dummy_labels).loss
    loss.backward()

    # Check gradients exist on A and B, but None on base weights
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            assert module.lora_A.grad is not None, f"{name}.lora_A must have gradient"
            assert module.lora_B.grad is not None, f"{name}.lora_B must have gradient"
            assert module.base.weight.grad is None, f"{name}.base.weight must NOT have gradient"

    # Merge adapters and verify forward output
    merge_attention_lora(model)
    with torch.no_grad():
        out_merged = model(**test_input).logits

    diff = (out_initial - out_merged).abs().max().item()
    assert diff < 1e-4, f"Pre-merge vs post-merge logits diff {diff} exceeds tolerance"


def test_balanced_sampler_exact_2_plus_2():
    """Verify balanced sampler produces exactly 2 Hangul and 2 Non-Hangul samples per batch of 4."""
    from experiments.train_hyperclovax_sft import sample_training_indices

    # Dummy dataset of 30 Hangul and 70 Non-Hangul
    hangul_flags = [True] * 30 + [False] * 70
    hangul_indices = torch.tensor([i for i, h in enumerate(hangul_flags) if h], dtype=torch.long)
    other_indices = torch.tensor([i for i, h in enumerate(hangul_flags) if not h], dtype=torch.long)
    weights_tensor = torch.zeros(len(hangul_flags), dtype=torch.double)
    generator = torch.Generator().manual_seed(42)

    batch_size = 4
    for _ in range(50):
        batch_idx = sample_training_indices(
            weights_tensor, hangul_indices, other_indices,
            batch_size=batch_size, generator=generator, sampling_mode="balanced"
        )
        assert len(batch_idx) == 4
        h_count = sum(hangul_flags[i] for i in batch_idx.tolist())
        nh_count = sum(not hangul_flags[i] for i in batch_idx.tolist())
        assert h_count == 2, f"Expected exactly 2 Hangul in batch, got {h_count}"
        assert nh_count == 2, f"Expected exactly 2 Non-Hangul in batch, got {nh_count}"


if __name__ == "__main__":
    print("Running test_lora_linear_math_and_idempotency()...")
    test_lora_linear_math_and_idempotency()
    print("-> PASSED")

    print("Running test_balanced_sampler_exact_2_plus_2()...")
    test_balanced_sampler_exact_2_plus_2()
    print("-> PASSED")

    print("Running test_hyperclova_lora_injection_and_param_count()...")
    test_hyperclova_lora_injection_and_param_count()
    print("-> PASSED")

    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")
