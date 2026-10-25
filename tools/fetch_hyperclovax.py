"""Download HyperCLOVAX-SEED-Text-Instruct-0.5B checkpoint cleanly; records manifest and sha256."""
import hashlib
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
REVISION = "3da5046fb0195d14f2497de198136987d35fd644"


def main():
    destination = ROOT / "artifacts/pretrained/hyperclovax-seed-0.5b"
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} (rev: {REVISION}) to {destination}...")
    snapshot_download(
        REPO_ID, revision=REVISION, token=False,
        local_dir=destination, cache_dir=ROOT / ".cache/huggingface",
        allow_patterns=[
            "config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "README.md", "LICENSE"
        ],
        max_workers=2,
    )
    files = {}
    for path in sorted(destination.iterdir()):
        if path.is_file() and path.name != "download_manifest.json":
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
    manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "recorded_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "files": files,
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    total_bytes = sum(v["bytes"] for v in files.values())
    print(json.dumps({
        "downloaded": list(files),
        "revision": REVISION,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024**2, 2),
    }), flush=True)


if __name__ == "__main__":
    main()
