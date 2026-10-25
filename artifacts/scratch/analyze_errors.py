import json
import pandas as pd

df = pd.read_csv("artifacts/AS-HCX-03/pilot_predictions.csv")
h = df[df["hangul"] == True]
err_top1 = h[~h["correct_top1"]]
print(f"Hangul Top-1 errors: {len(err_top1)} / {len(h)}")

categories = {
    "in_top10": [],
    "completely_wrong_gen": [],
    "close_gen_but_catalog_fuzzy_miss": [],
}

for i, (_, r) in enumerate(err_top1.iterrows()):
    src = r["source"]
    gen = str(r["generated_text"])
    pred = str(r["predicted_catalog_name"])
    gold = str(r["target"])
    in_top10 = bool(r["correct_top10"])
    
    print(f"{i+1:2d}. [{src}] -> Gen: [{gen}] -> Pred: [{pred}] (Gold: [{gold}]) [InTop10: {in_top10}]")

print("\n--- Non-Hangul Top-1 errors ---")
nh = df[df["hangul"] == False]
err_nh = nh[~nh["correct_top1"]]
print(f"Non-Hangul Top-1 errors: {len(err_nh)} / {len(nh)}")
for i, (_, r) in enumerate(err_nh.iterrows()):
    print(f"{i+1:2d}. [{r['source']}] -> Gen: [{r['generated_text']}] -> Pred: [{r['predicted_catalog_name']}] (Gold: [{r['target']}]) [InTop10: {r['correct_top10']}]")
