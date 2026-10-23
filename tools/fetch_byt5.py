"""Download only the official public checkpoint; never reads competition data."""
import hashlib
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"


def main():
    destination = ROOT / "artifacts/pretrained/byt5-small"
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        "google/byt5-small", revision=REVISION, token=False,
        local_dir=destination, cache_dir=ROOT / ".cache/huggingface",
        allow_patterns=["config.json", "generation_config.json", "pytorch_model.bin",
                        "tokenizer_config.json", "special_tokens_map.json", "README.md"],
        max_workers=2,
    )
    files = {}
    for path in sorted(destination.iterdir()):
        if path.is_file() and path.name != "download_manifest.json":
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
    manifest = {"repo_id": "google/byt5-small", "revision": REVISION,
                "recorded_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
                "files": files}
    (destination / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"downloaded": list(files), "revision": REVISION,
                      "total_bytes": sum(v["bytes"] for v in files.values())}), flush=True)


if __name__ == "__main__":
    main()
