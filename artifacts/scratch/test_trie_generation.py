import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from name_decoder.catalog import canonical_name

class BpeNameTrie:
    def __init__(self, names, tokenizer):
        self.root = {}
        self.eos_id = tokenizer.eos_token_id
        for name in names:
            tokens = tokenizer.encode(name, add_special_tokens=False)
            curr = self.root
            for t in tokens:
                curr = curr.setdefault(t, {})
            curr["__is_leaf__"] = True

    def get_allowed_tokens(self, generated_tokens):
        curr = self.root
        for t in generated_tokens:
            if t not in curr:
                return [self.eos_id]
            curr = curr[t]
        
        allowed = [k for k in curr.keys() if k != "__is_leaf__"]
        if curr.get("__is_leaf__", False):
            allowed.append(self.eos_id)
        if not allowed:
            allowed = [self.eos_id]
        return allowed

def build_prompt(source):
    return (
        f"<|im_start|>system\n"
        f"한국어 선박명을 국제 카탈로그의 공식 영문 대문자 선박명으로 1:1 변환합니다. 오직 영문 공식 선박명만 대문자로 한 줄로 출력하세요.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"입력: {source}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"출력:\n"
    )

def test_trie_generation():
    cat_df = pd.read_csv("data/vessel_catalog.csv")
    catalog_names = sorted(set(cat_df["vessel_name"].dropna().map(canonical_name)))
    tokenizer = AutoTokenizer.from_pretrained("artifacts/AS-HCX-03/model")
    trie = BpeNameTrie(catalog_names, tokenizer)
    
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained("artifacts/AS-HCX-03/model", dtype=torch.float32, device_map="cpu")
    model.eval()

    prompt = build_prompt("한진 파이오니아")
    inp = tokenizer(prompt, return_tensors="pt")
    prompt_len = inp["input_ids"].shape[1]

    def prefix_fn(batch_id, input_ids):
        gen = input_ids[prompt_len:].tolist()
        return trie.get_allowed_tokens(gen)

    print("Generating with Beam 4 + Trie Constrained Decoding...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=20,
            num_beams=4,
            num_return_sequences=4,
            prefix_allowed_tokens_fn=prefix_fn,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    t1 = time.perf_counter()
    print(f"Generated 4 beam sequences in {t1-t0:.2f}s:")
    for i, seq in enumerate(out):
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
        in_catalog = text in catalog_names
        print(f"  Beam {i+1}: '{text}' (in catalog: {in_catalog})")

if __name__ == "__main__":
    test_trie_generation()
