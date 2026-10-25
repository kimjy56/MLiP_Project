"""Minimal plain-PyTorch LoRA for local Llama attention projections."""
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen Linear plus alpha/r * B(A(x)); B starts at zero."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA requires an nn.Linear")
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not base.weight.is_floating_point():
            raise TypeError("Quantized base weights are not supported")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False

    def forward(self, inputs):
        result = self.base(inputs)
        if self.merged:
            return result
        return result + ((inputs @ self.lora_A.T) @ self.lora_B.T) * self.scaling

    @torch.no_grad()
    def merge(self):
        """Idempotent merge; the returned Linear has no adapter forward cost."""
        if not self.merged:
            delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
            self.base.weight.add_(delta.to(self.base.weight.dtype))
            self.merged = True
            self.lora_A.requires_grad_(False)
            self.lora_B.requires_grad_(False)
        return self.base


def inject_attention_lora(model, rank=16, alpha=32.0, targets=("q_proj", "v_proj")):
    """Freeze ALL base parameters, then adapt every selected attention projection."""
    targets = tuple(targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("targets must be nonempty and unique")
    if not set(targets) <= {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError("Only Llama attention projections are supported")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("Model already has LoRA adapters")
    selected = [(name, module) for name, module in model.named_modules()
                if ".self_attn." in name and name.rsplit(".", 1)[-1] in targets]
    layers = getattr(model.config, "num_hidden_layers", None)
    for target in targets:
        if sum(name.endswith("." + target) for name, _ in selected) != layers:
            raise ValueError(f"Expected one {target} per model layer")
    if not all(isinstance(module, nn.Linear) for _, module in selected):
        raise TypeError("Selected projections must be unquantized Linear layers")
    model.requires_grad_(False)
    entries = []
    for name, module in selected:
        parent, _, attr = name.rpartition(".")
        adapter = LoRALinear(module, rank=rank, alpha=alpha)
        setattr(model.get_submodule(parent), attr, adapter)
        entries.append({"name": name, "in_features": module.in_features,
                        "out_features": module.out_features,
                        "adapter_parameters": rank * (module.in_features + module.out_features)})
    expected = sum(entry["adapter_parameters"] for entry in entries)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if expected != trainable:
        raise RuntimeError("Unexpected trainable base parameters")
    return {"rank": rank, "alpha": alpha, "targets": list(targets), "modules": entries,
            "trainable_parameters": trainable,
            "total_parameters_with_adapters": sum(p.numel() for p in model.parameters())}


def adapter_state_dict(model):
    """Small portable state; loading it requires the exact base and injection spec."""
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if module.merged:
                raise ValueError("Save adapters before merging")
            for attr in ("lora_A", "lora_B"):
                result[f"{name}.{attr}"] = getattr(module, attr).detach().cpu().contiguous()
    if not result:
        raise ValueError("No unmerged adapters found")
    return result


def merge_attention_lora(model):
    selected = [(name, module) for name, module in model.named_modules()
                if isinstance(module, LoRALinear)]
    for name, module in selected:
        parent, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent), attr, module.merge())
    return len(selected)
