"""Lossless serialization of a local phonetic-encoder checkpoint; no training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_checkpoint(source, target):
    source, target = Path(source), Path(target)
    if target.exists() or target.with_suffix(".config.json").exists():
        raise FileExistsError("Export target/config already exists; choose a new path")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    groups = {"encoder_state_dict": "encoder", "fusion_state_dict": "fusion"}
    tensors = {}
    for group, prefix in groups.items():
        for name, tensor in payload[group].items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Non-tensor in {group}: {name}")
            tensors[f"{prefix}.{name}"] = tensor.detach().cpu().contiguous().clone()
    config = {key: value for key, value in payload.items() if key not in groups}
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target), metadata={
        "format": "pt", "schema": "phonetic-encoder-v1",
        "checkpoint_config": json.dumps(config, ensure_ascii=False, sort_keys=True),
        "source_sha256": sha256(source),
        "tensor_groups": json.dumps(groups, sort_keys=True),
    })
    target.with_suffix(".config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    restored = read_checkpoint(target)
    assert set(restored) == set(payload), "Checkpoint fields changed"
    for group in groups:
        assert set(restored[group]) == set(payload[group]), "Tensor keys changed"
        for name, original in payload[group].items():
            saved = restored[group][name]
            assert original.dtype == saved.dtype and torch.equal(original, saved), name
    assert {key: restored[key] for key in config} == config, "Metadata changed"
    return {"source_sha256": sha256(source), "output_sha256": sha256(target),
            "source_bytes": source.stat().st_size, "output_bytes": target.stat().st_size,
            "tensor_count": len(tensors), "tensor_elements": sum(t.numel() for t in tensors.values()),
            "exact_tensor_roundtrip": True, "exact_metadata_roundtrip": True}


def read_checkpoint(path):
    """Recover the original payload structure without using pickle."""
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        metadata = stream.metadata()
    if metadata.get("schema") != "phonetic-encoder-v1":
        raise ValueError("Unsupported safetensors schema")
    payload = json.loads(metadata["checkpoint_config"])
    tensors = load_file(str(path), device="cpu")
    for group, prefix in json.loads(metadata["tensor_groups"]).items():
        payload[group] = {key[len(prefix) + 1:]: value for key, value in tensors.items()
                          if key.startswith(prefix + ".")}
    return payload


def load_encoder_safetensors(path, device="cpu"):
    """Use the bundled AS-09 phonetic_encoder package, not a different version."""
    from phonetic_encoder.model import CharPhoneticBiEncoder, UniversalEncoder, GatedMultiViewFusion
    from phonetic_encoder.tokenizer import PhoneticTokenizer

    payload = read_checkpoint(path)
    tokenizer = PhoneticTokenizer()
    for key, value in payload["vocab"].items():
        setattr(tokenizer, key, value)
    tokenizer.id_to_char = {value: key for key, value in tokenizer.char_to_id.items()}
    encoder = UniversalEncoder(vocab_size=tokenizer.vocab_size,
        num_scripts=tokenizer.num_scripts, num_langs=tokenizer.num_langs,
        embed_dim=payload["embed_dim"], dropout=0.0)
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    fusion = GatedMultiViewFusion(embed_dim=encoder.embed_dim)
    fusion.load_state_dict(payload["fusion_state_dict"], strict=True)
    model = CharPhoneticBiEncoder(tokenizer=tokenizer, model=encoder, fusion=fusion)
    model = model.half() if payload["as_fp16"] else model.float()
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
