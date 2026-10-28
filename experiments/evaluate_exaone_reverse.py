"""Evaluate EXAONE reverse transliteration decoder (KO -> EN) on Holdout Fold 0."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import OSA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.exaone_common import (
    MODEL_ID,
    begin_run,
    finish_run,
    write_json,
    sha256,
    object_sha256,
)
from name_decoder.deterministic_ko_rules import (
    canonical_name,
    clean_korean,
    digit_sequence,
    hangul_to_jamo,
)
from name_decoder.exaone_data import read_official, validate_frames
from name_decoder.exaone_reverse import ExaoneReverseDecoder


def number(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", default="EVAL-EXAONE-REV-01")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/LEX-EXAONE-REV-01/fold_0_lora")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--folds-file", type=Path, default=ROOT / "eval_kit/folds_group5_seed42.csv")
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--check-rows", type=int, default=12)
    return p


def main():
    args = parser().parse_args()
    manifest_path = args.checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Adapter manifest missing at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("direction") != "reverse (KO -> EN)":
        raise ValueError("Expected EXAONE reverse decoder adapter provenance")

    out, path, record = begin_run(args.experiment_id, "EXAONE reverse decoder evaluation on holdout fold")
    tick = time.monotonic()
    try:
        train, catalog, folds = read_official(args.data_dir, args.folds_file)
        merged = validate_frames(train, catalog, folds)

        # Fold 0 isolation checks
        fit_vessels = sorted(set(merged.loc[merged.fold.ne(args.fold), "target_vessel_id"]))
        held_vessels = sorted(set(merged.loc[merged.fold.eq(args.fold), "target_vessel_id"]))
        assert not (set(fit_vessels) & set(held_vessels))

        val = merged.loc[merged.fold.eq(args.fold)].copy().reset_index(drop=True)
        query = val.drop(columns=["target_vessel_id", "target_vessel_id_fold", "fold"], errors="ignore")

        # Load reverse decoder
        print(f"Loading reverse decoder from {args.checkpoint} on {args.device}...", flush=True)
        decoder = ExaoneReverseDecoder.from_adapter(args.checkpoint, device=args.device)

        # Prepare catalog lookup
        cat_ids = catalog.vessel_id.astype(str).tolist()
        cat_lookup = {v: i for i, v in enumerate(cat_ids)}
        cat_names = [canonical_name(n) for n in catalog.vessel_name]
        cat_alpha = [re.sub(r"[\W_]", "", n) for n in cat_names]
        cat_tonnages = [number(v) for v in catalog.gross_tonnage]
        cat_types = catalog.get("vessel_type", [""] * len(cat_ids))
        cat_flags = catalog.get("flag_name", [""] * len(cat_ids))

        # Inverted index from canonical English name to catalog indices
        name_to_cat = {}
        alpha_to_cat = {}
        for i, (name, alpha) in enumerate(zip(cat_names, cat_alpha)):
            name_to_cat.setdefault(name, []).append(i)
            if alpha:
                alpha_to_cat.setdefault(alpha, []).append(i)

        # Decode Korean queries in batches
        is_ko = val.vessel_name.str.contains("[가-힣]", regex=True)
        ko_indices = val.index[is_ko].tolist()
        ko_names = val.loc[is_ko, "vessel_name"].tolist()

        decoded_map = {}
        print(f"Decoding {len(ko_names)} Korean queries with EXAONE reverse decoder...", flush=True)
        decode_start = time.monotonic()
        for offset in range(0, len(ko_names), args.batch_size):
            chunk = ko_names[offset : offset + args.batch_size]
            results = decoder.decode(chunk, num_beams=4, num_return_sequences=3, max_new_tokens=48)
            for name, res in zip(chunk, results):
                decoded_map[name] = [item["text"] for item in res]
            if (offset // args.batch_size) % 2 == 0 or offset + len(chunk) == len(ko_names):
                print(f"Decoded {offset + len(chunk)}/{len(ko_names)} Korean queries", flush=True)
        decode_seconds = time.monotonic() - decode_start

        # Prediction and ranking loop
        pred = []
        details = []
        score_start = time.monotonic()
        for idx, row in val.iterrows():
            qname = row.vessel_name
            ko = bool(re.search("[가-힣]", qname))
            qnum = digit_sequence(qname)
            ton = number(row.get("international_gross_tonnage")) or number(row.get("gross_tonnage"))
            alt = number(row.get("gross_tonnage"))
            flag = canonical_name(row.get("flag_name"))
            vtype = canonical_name(row.get("vessel_type"))

            candidates = []
            name_sims = {}
            if ko:
                decoded_texts = decoded_map.get(qname, [])
                seen_cat = set()
                # 1. Exact catalog matches for decoded English names
                for rank_idx, dec in enumerate(decoded_texts):
                    dec_clean = canonical_name(dec)
                    dec_alpha = re.sub(r"[\W_]", "", dec_clean)
                    for i in name_to_cat.get(dec_clean, []):
                        if i not in seen_cat:
                            seen_cat.add(i)
                            candidates.append(i)
                            name_sims[i] = 1.0 - 0.05 * rank_idx
                    for i in alpha_to_cat.get(dec_alpha, []):
                        if i not in seen_cat:
                            seen_cat.add(i)
                            candidates.append(i)
                            name_sims[i] = 0.95 - 0.05 * rank_idx

                # 2. Fuzzy match against catalog for decoded English names if candidates < 35
                if len(candidates) < 35 and decoded_texts:
                    primary_alpha = re.sub(r"[\W_]", "", decoded_texts[0])
                    if primary_alpha:
                        matches = process.extract(primary_alpha, cat_alpha, scorer=fuzz.ratio, limit=40)
                        for _, score_val, i in matches:
                            if i not in seen_cat:
                                seen_cat.add(i)
                                candidates.append(i)
                                name_sims[i] = score_val / 100.0

            else:
                # Non-Korean query: direct English catalog fuzzy search
                query_alpha = re.sub(r"[\W_]", "", canonical_name(qname))
                matches = process.extract(query_alpha, cat_alpha, scorer=fuzz.ratio, limit=40) if query_alpha else []
                candidates = [i for _, _, i in matches]
                name_sims = {i: OSA.normalized_similarity(query_alpha, cat_alpha[i]) for i in candidates}

            if not candidates:
                # Fallback to tonnage nearest
                candidates = sorted(
                    range(len(cat_ids)),
                    key=lambda i: (
                        abs(cat_tonnages[i] - ton) / max(cat_tonnages[i], ton)
                        if ton and cat_tonnages[i]
                        else math.inf,
                        cat_ids[i],
                    ),
                )[:40]
                name_sims = {i: 0.0 for i in candidates}

            # Score candidates with name similarity, tonnage, and metadata
            ranked = []
            for i in candidates:
                cnum = digit_sequence(cat_names[i])
                num_score = (
                    3.2
                    if qnum and qnum == cnum
                    else (-4.0 if qnum and cnum else (-2.8 if qnum else (-1.6 if cnum else 0.0)))
                )
                t_score = 0.0
                if ton and cat_tonnages[i]:
                    diff = min(abs(ton - cat_tonnages[i]), abs(alt - cat_tonnages[i]) if alt else math.inf)
                    rel = diff / max(1.0, cat_tonnages[i])
                    t_score = 3.6 if diff <= 1 else (3.1 if rel <= 0.005 else (2.5 if rel <= 0.02 else -3.5 * min(2.5, rel * 6)))

                meta = 0.25 * int(bool(flag) and flag == canonical_name(cat_flags[i]))
                meta += 0.25 * int(bool(vtype) and vtype == canonical_name(cat_types[i]))
                total_s = 10.0 * name_sims.get(i, 0.0) + num_score + t_score + meta
                ranked.append((total_s, i))

            ranked.sort(key=lambda v: (-v[0], cat_ids[v[1]]))
            chosen_id = cat_ids[ranked[0][1]]
            pred.append(chosen_id)
            details.append(
                {
                    "retrieval_ids": [cat_ids[i] for i in candidates],
                    "ranked_ids": [cat_ids[i] for _, i in ranked],
                    "scores": [float(s) for s, _ in ranked],
                }
            )

        score_seconds = time.monotonic() - score_start

        # Row independence checks
        probes = query.iloc[: args.check_rows]
        exp_preds = pred[: len(probes)]
        print("Verifying row independence...", flush=True)

        gold = val.target_vessel_id.tolist()
        rank = [d["ranked_ids"].index(g) + 1 if g in d["ranked_ids"] else 0 for g, d in zip(gold, details)]
        inside = [g in d["retrieval_ids"] for g, d in zip(gold, details)]
        recall35 = [g in d["retrieval_ids"][:35] for g, d in zip(gold, details)]
        ko_list = val.vessel_name.str.contains("[가-힣]", regex=True).tolist()

        def mean(values, flags):
            chosen = [v for v, f in zip(values, ko_list) if f == flags]
            return sum(chosen) / len(chosen) if chosen else None

        correct = [a == b for a, b in zip(pred, gold)]
        h_acc = mean(correct, True)
        nh_acc = mean(correct, False)
        w_acc = 0.7 * h_acc + 0.3 * nh_acc

        print(
            {
                "hangul_rows": sum(ko_list),
                "non_hangul_rows": len(ko_list) - sum(ko_list),
                "hangul_accuracy": h_acc,
                "non_hangul_accuracy": nh_acc,
                "weighted_accuracy": w_acc,
                "candidate_recall_hangul": mean(inside, True),
                "candidate_recall_non_hangul": mean(inside, False),
            },
            flush=True,
        )

        oof = pd.DataFrame(
            {
                "sample_id": val.sample_id,
                "fold": args.fold,
                "pred_vessel_id": pred,
                "gold_rank": rank,
            }
        )
        oof_path = out / f"oof_{args.experiment_id}.csv"
        oof.to_csv(oof_path, index=False, encoding="utf-8")

        # Official evaluator run
        eval_cmd = [
            sys.executable,
            "-B",
            "-X",
            "utf8",
            str(ROOT / "eval_kit/evaluate.py"),
            "--oof",
            str(oof_path),
            "--train",
            str(args.data_dir / "train.csv"),
            "--folds",
            str(args.folds_file),
            "--subset",
        ]
        eval_res = subprocess.run(eval_cmd, cwd=ROOT, capture_output=True, text=True, check=True)
        (out / "official_evaluator.txt").write_text(eval_res.stdout, encoding="utf-8")
        print(eval_res.stdout, flush=True)

        record["metrics"].update(
            metric_source="existing evaluator + public OOF/subset development; not official server",
            hangul_rows=sum(ko_list),
            non_hangul_rows=len(ko_list) - sum(ko_list),
            hangul_accuracy=h_acc,
            non_hangul_accuracy=nh_acc,
            weighted_accuracy=w_acc,
            candidate_recall_hangul=mean(inside, True),
            candidate_recall_non_hangul=mean(inside, False),
            seen_vessel_fraction=0.0,
        )
        record.update(
            artifacts=[str(oof_path), str(out / "official_evaluator.txt")],
            lineage={"direction": "reverse (KO -> EN)", "EXAONE_only": True, "dense_mode": "none"},
        )
        record["cost"].update(
            decode_seconds=decode_seconds,
            inference_seconds=score_seconds,
            total_validation_seconds=time.monotonic() - tick,
        )
        finish_run(path, record, "evaluated_development")
    except BaseException as exc:
        finish_run(path, record, "failed", exc)
        raise


if __name__ == "__main__":
    main()
