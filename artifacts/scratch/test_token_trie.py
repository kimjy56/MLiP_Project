import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from transformers import AutoTokenizer
from name_decoder.catalog import canonical_name

class BpeNameTrie:
    def __init__(self, names, tokenizer):
        self.root = {}
        self.eos_id = tokenizer.eos_token_id
        t0 = time.perf_counter()
        count = 0
        for name in names:
            tokens = tokenizer.encode(name, add_special_tokens=False)
            curr = self.root
            for t in tokens:
                curr = curr.setdefault(t, {})
            curr["__is_leaf__"] = True
            count += 1
        self.build_time = time.perf_counter() - t0
        self.count = count

    def get_allowed_tokens(self, generated_tokens):
        curr = self.root
        for t in generated_tokens:
            if t not in curr:
                # Fell off trie or finished
                return [self.eos_id]
            curr = curr[t]
        
        allowed = [k for k in curr.keys() if k != "__is_leaf__"]
        if curr.get("__is_leaf__", False):
            allowed.append(self.eos_id)
        return allowed

def main():
    cat_df = pd.read_csv("data/vessel_catalog.csv")
    catalog_names = sorted(set(cat_df["vessel_name"].dropna().map(canonical_name)))
    tokenizer = AutoTokenizer.from_pretrained("artifacts/AS-HCX-03/model")

    print(f"Building BPE Token Trie for {len(catalog_names)} names...")
    trie = BpeNameTrie(catalog_names, tokenizer)
    print(f"Trie built in {trie.build_time:.2f}s for {trie.count} names!")

    # Test allowed tokens at root
    root_allowed = trie.get_allowed_tokens([])
    print(f"Allowed first tokens count: {len(root_allowed)}")

    # Test walking down a known vessel name
    test_name = "HANJIN PIONEER"
    tokens = tokenizer.encode(test_name, add_special_tokens=False)
    print(f"Tokens for '{test_name}': {tokens}")

    for step in range(len(tokens) + 1):
        prefix = tokens[:step]
        allowed = trie.get_allowed_tokens(prefix)
        is_leaf = trie.eos_id in allowed
        print(f"Step {step} (prefix len {len(prefix)}): allowed tokens count: {len(allowed)}, can end (EOS allowed): {is_leaf}")

if __name__ == "__main__":
    main()
