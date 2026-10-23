"""Inspect interpretations of one explicitly supplied name, without writing logs."""
import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .catalog import CatalogNames
from .model import ByT5NameDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--top-k", default=10, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    catalog = pd.read_csv(args.catalog, keep_default_na=False, dtype={"vessel_id": str})
    decoder = ByT5NameDecoder.load(args.model, CatalogNames(catalog.to_dict("records")), args.device)
    import json
    print(json.dumps([asdict(item) for item in decoder.decode(args.name, args.top_k)],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
