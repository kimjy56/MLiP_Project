import os
import sys
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from name_decoder.data import read_frames, select_pilot

def build_prompt(source):
    return (
        f"<|im_start|>system\n"
        f"한국어 선박명을 국제 카탈로그의 공식 영문 대문자 선박명으로 1:1 변환합니다. 오직 영문 공식 선박명만 대문자로 한 줄로 출력하세요.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"입력: {source}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"출력:\n"
    )

def test_cache_small():
    torch.set_num_threads(4)
    model_path = ROOT / "artifacts/AS-HCX-03/model"
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, device_map="cpu")
    model.eval()

    test_sources = ["한진 파이오니아", "골든 치에", "HATTHYAA NAREE", "CHEMROAD SEA"]
    
    print("\n--- Testing use_cache=False vs use_cache=True on 4 samples ---")
    for src in test_sources:
        prompt = build_prompt(src)
        inp = tokenizer(prompt, return_tensors="pt")
        
        # use_cache=False
        t0 = time.perf_counter()
        with torch.no_grad():
            out_false = model.generate(
                **inp,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        t_false = time.perf_counter() - t0
        text_false = tokenizer.decode(out_false[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # use_cache=True
        t0 = time.perf_counter()
        with torch.no_grad():
            out_true = model.generate(
                **inp,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        t_true = time.perf_counter() - t0
        text_true = tokenizer.decode(out_true[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        match = torch.equal(out_false, out_true)
        print(f"[{src}]")
        print(f"  False: '{text_false}' ({t_false:.3f}s)")
        print(f"  True : '{text_true}' ({t_true:.3f}s)")
        print(f"  Speedup: {t_false/t_true:.2f}x | Exact Token Match: {match}")
        assert match, f"Mismatch on {src}!"

if __name__ == "__main__":
    test_cache_small()
