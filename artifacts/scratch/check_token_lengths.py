import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from transformers import AutoTokenizer
from name_decoder.catalog import canonical_name

cat_df = pd.read_csv("data/vessel_catalog.csv")
catalog_names = sorted(set(cat_df["vessel_name"].dropna().map(canonical_name)))

print(f"Total canonical catalog names: {len(catalog_names)}")

tokenizer = AutoTokenizer.from_pretrained("artifacts/AS-HCX-03/model")
lengths = [len(tokenizer.encode(name, add_special_tokens=False)) for name in catalog_names]

max_len = max(lengths)
p99_len = pd.Series(lengths).quantile(0.99)
p999_len = pd.Series(lengths).quantile(0.999)

print(f"Token length stats for 78,592 catalog names:")
print(f"  Max token length: {max_len}")
print(f"  99.9th percentile: {p999_len}")
print(f"  99.0th percentile: {p99_len}")
print(f"  Names with length > 16: {(pd.Series(lengths) > 16).sum()}")
print(f"  Names with length > 24: {(pd.Series(lengths) > 24).sum()}")
print(f"  Names with length > 32: {(pd.Series(lengths) > 32).sum()}")

longest_idx = lengths.index(max_len)
print(f"Longest name: '{catalog_names[longest_idx]}' ({max_len} tokens)")
