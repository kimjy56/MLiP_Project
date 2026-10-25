"""Test HyperCLOVA X SEED 0.5B inference on a small pilot batch."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from rapidfuzz import process, fuzz
from name_decoder.data import read_frames, select_pilot

sys.stdout.reconfigure(encoding="utf-8")

model_path = "artifacts/pretrained/hyperclovax-seed-0.5b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
torch.set_num_threads(12)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, device_map="cpu")

pairs, validation, fit_rows, index = read_frames(ROOT / "data", ROOT / "eval_kit/folds_group5_seed42.csv", heldout_fold=0)
pilot = select_pilot(validation, per_language=4, seed=42)
catalog_names = sorted(index.ids_by_name.keys())

few_shot = """한국어 선박명을 국제 카탈로그의 공식 영문 대문자 선박명으로 1:1 변환합니다. 오직 영문 공식 선박명만 대문자로 한 줄로 출력하세요.

입력: 글로비스 챔피언
출력: GLOVIS CHAMPION

입력: 한진 파이오니아
출력: HANJIN PIONEER

입력: 그랜드 에이스9
출력: GRAND ACE 9

입력: 흥아 파이오니아호
출력: HEUNG-A PIONEER

입력: 닝화 421
출력: NING HUA 421
"""

for idx, r in pilot.iterrows():
    prompt = f"<|im_start|>system\n{few_shot}<|im_end|>\n<|im_start|>user\n입력: {r.source}<|im_end|>\n<|im_start|>assistant\n출력:"
    inp = tokenizer(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    t1 = time.time()
    gen = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].upper()
    top_matches = [m[0] for m in process.extract(gen, catalog_names, scorer=fuzz.WRatio, limit=5)]
    is_top1 = (top_matches[0] == r.target)
    is_top5 = (r.target in top_matches)
    print(f"[{r.sample_id}] {r.source} -> Gen: '{gen}' | Target: {r.target} | Top1 match: {top_matches[0]} (Top1: {is_top1}, Top5: {is_top5}) ({t1-t0:.2f}s)", flush=True)
