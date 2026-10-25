import sys
sys.path.insert(0, ".")
import time
import pandas as pd
from rapidfuzz import process, fuzz
from name_decoder.catalog import canonical_name

df = pd.read_csv("artifacts/AS-HCX-03/pilot_predictions.csv")
cat_df = pd.read_csv("data/vessel_catalog.csv")
catalog_names = sorted(set(cat_df["vessel_name"].dropna().map(canonical_name)))

print(f"Total canonical catalog names: {len(catalog_names)}")

scorers = {
    "WRatio (current)": fuzz.WRatio,
    "ratio (simple)": fuzz.ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "QRatio": fuzz.QRatio,
}

for name, sc in scorers.items():
    t0 = time.perf_counter()
    h_hits_t1, h_hits_t10 = 0, 0
    nh_hits_t1, nh_hits_t10 = 0, 0
    
    for _, r in df.iterrows():
        gen = str(r["generated_text"])
        target = str(r["target"])
        is_hangul = bool(r["hangul"])
        
        matches = [m[0] for m in process.extract(gen, catalog_names, scorer=sc, limit=10)]
        is_t1 = (matches[0] == target)
        is_t10 = (target in matches)
        
        if is_hangul:
            if is_t1: h_hits_t1 += 1
            if is_t10: h_hits_t10 += 1
        else:
            if is_t1: nh_hits_t1 += 1
            if is_t10: nh_hits_t10 += 1
            
    t1 = time.perf_counter()
    n_h = df["hangul"].sum()
    n_nh = (~df["hangul"]).sum()
    h_acc = h_hits_t1 / n_h
    h_t10 = h_hits_t10 / n_h
    nh_acc = nh_hits_t1 / n_nh
    nh_t10 = nh_hits_t10 / n_nh
    weighted = 0.7 * h_acc + 0.3 * nh_acc
    print(f"[{name}] (Time: {t1-t0:.1f}s)")
    print(f"  Hangul Top-1: {h_acc*100:.2f}% ({h_hits_t1}/{n_h}), Top-10: {h_t10*100:.2f}% ({h_hits_t10}/{n_h})")
    print(f"  Non-Hangul Top-1: {nh_acc*100:.2f}% ({nh_hits_t1}/{n_nh}), Top-10: {nh_t10*100:.2f}% ({nh_hits_t10}/{n_nh})")
    print(f"  Weighted Accuracy: {weighted*100:.2f}%\n")
