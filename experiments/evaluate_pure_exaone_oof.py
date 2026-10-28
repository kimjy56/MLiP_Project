"""Evaluate a fold-bound new EXAONE map; never read old dictionaries or build ZIPs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.exaone_common import MODEL_ID,begin_run,finish_run,write_json,sha256,object_sha256
from name_decoder.exaone_data import read_official,validate_frames


def checked_map(path,train,catalog,folds,fold,data_dir,folds_file,allow_partial=False):
    if path.name=="catalog_full_ko_map.json":raise ValueError("Legacy catalog map forbidden")
    sidecar=path.with_suffix(".manifest.json")
    manifest=json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("model_id")!=MODEL_ID or manifest.get("generator")!="EXAONE-only":
        raise ValueError("Expected new EXAONE-only map provenance")
    if manifest["catalog_map_sha256"]!=sha256(path):raise ValueError("Map hash mismatch")
    if not manifest["full_catalog"] and not allow_partial:raise ValueError("Partial map is a pilot, not full OOF")
    policy=manifest["config"]["legacy_policy"]
    if any(policy.values()):raise ValueError("Legacy data policy violation")
    data=manifest["training_data"]
    if data["split_id"]!=f"G5-s42 fold{fold}":raise ValueError("Wrong fold map; full-train maps cannot evaluate OOF")
    hashes={"train.csv":sha256(data_dir/"train.csv"),"vessel_catalog.csv":sha256(data_dir/"vessel_catalog.csv"),
            "folds":sha256(folds_file)}
    if data["source_files_sha256"]!=hashes:raise ValueError("Map data/fold hashes differ")
    merged=validate_frames(train,catalog,folds)
    fit=sorted(set(merged.loc[merged.fold.ne(fold),"target_vessel_id"]))
    hold=sorted(set(merged.loc[merged.fold.eq(fold),"target_vessel_id"]))
    if data["fit_vessel_ids_sha256"]!=object_sha256(fit) or data["heldout_vessel_ids_sha256"]!=object_sha256(hold):
        raise ValueError("Fit/heldout membership differs")
    return json.loads(path.read_text(encoding="utf-8")),manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id",default="EVAL-EXAONE-PURE-01")
    p.add_argument("--ko-map",type=Path)
    p.add_argument("--map-pattern",help="For --all-folds, a path containing {fold}; each map must have its own trained provenance")
    p.add_argument("--fold",type=int,choices=range(5),default=0)
    p.add_argument("--all-folds",action="store_true")
    p.add_argument("--data-dir",type=Path,default=ROOT/"data")
    p.add_argument("--folds-file",type=Path,default=ROOT/"eval_kit/folds_group5_seed42.csv")
    p.add_argument("--allow-partial-map",action="store_true")
    p.add_argument("--check-rows",type=int,default=12)
    p.add_argument("--dense-mode",choices=["none","hybrid"],default="none")
    p.add_argument("--dense-checkpoint",type=Path)
    p.add_argument("--dense-embeddings",type=Path)
    p.add_argument("--dense-provenance",type=Path,help="Audited checkpoint/embedding/fold/fit-membership manifest")
    p.add_argument("--allow-existing-dpr",action="store_true",help="Acknowledge external DPR lineage; results are not EXAONE-only")
    args=p.parse_args()
    if args.all_folds and (not args.map_pattern or "{fold}" not in args.map_pattern):
        raise SystemExit("--all-folds needs a different --map-pattern for each fold")
    if not args.all_folds and args.ko_map is None:raise SystemExit("--ko-map required")
    if args.dense_mode=="hybrid" and (not args.allow_existing_dpr or args.all_folds or not args.dense_checkpoint or not args.dense_embeddings or not args.dense_provenance):
        raise SystemExit("Existing DPR requires explicit lineage acknowledgement, single-fold scope, checkpoint/embedding paths and provenance")
    out,path,record=begin_run(args.experiment_id,"New EXAONE static-map matching on distributed outer folds")
    tick=time.monotonic()
    try:
        import pandas as pd
        from experimental_matchers.exaone_lexicon import ExaoneLexiconMatcher
        train,catalog,folds=read_official(args.data_dir,args.folds_file)
        merged=validate_frames(train,catalog,folds)
        parts=[];diagnostics=[];map_hashes={}
        for fold in range(5) if args.all_folds else [args.fold]:
            map_path=Path(args.map_pattern.format(fold=fold)) if args.all_folds else args.ko_map
            ko_map,manifest=checked_map(map_path,train,catalog,folds,fold,args.data_dir,args.folds_file,args.allow_partial_map)
            val=merged.loc[merged.fold.eq(fold)].copy()
            languages=val.vessel_name.str.contains("[가-힣]",regex=True)
            if val.empty or not languages.any() or languages.all():
                raise ValueError("Official weighted evaluation needs both language groups")
            query=val.drop(columns=["target_vessel_id","target_vessel_id_fold","fold"],errors="ignore")
            dense=None
            if args.dense_mode=="hybrid":
                dense_manifest=json.loads(args.dense_provenance.read_text(encoding="utf-8"))
                expected={"checkpoint_sha256":sha256(args.dense_checkpoint),
                          "embeddings_sha256":sha256(args.dense_embeddings),
                          "folds_sha256":manifest["training_data"]["split_file_sha256"],
                          "fit_vessel_ids_sha256":manifest["training_data"]["fit_vessel_ids_sha256"],
                          "heldout_fold":fold}
                if any(dense_manifest.get(k)!=v for k,v in expected.items()):
                    raise ValueError("DPR provenance does not certify this checkpoint/embedding/fit split")
                from phonetic_encoder.hybrid_matcher import HybridDenseLexicalMatcher
                dense=HybridDenseLexicalMatcher(args.dense_checkpoint,args.dense_embeddings,catalog_df=pd.read_csv(args.data_dir/"vessel_catalog.csv"),device="cpu")
            index_start=time.monotonic()
            matcher=ExaoneLexiconMatcher(catalog,ko_map,dense_matcher=dense)
            index_seconds=time.monotonic()-index_start
            start=time.monotonic()
            pred,details=matcher.predict_details(query)
            score_seconds=time.monotonic()-start
            probes=query.iloc[:args.check_rows]
            expected=pred[:len(probes)]
            assert matcher.predict(probes.iloc[::-1])[::-1]==expected
            assert matcher.predict(probes.iloc[::2])==expected[::2]
            assert [matcher.predict(probes.iloc[[i]])[0] for i in range(len(probes))]==expected
            assert matcher.predict(probes.iloc[:0])==[]
            # Gold is read only AFTER candidates, scores and predictions are finalized.
            gold=val.target_vessel_id.tolist()
            rank=[d["ranked_ids"].index(g)+1 if g in d["ranked_ids"] else 0 for g,d in zip(gold,details)]
            inside=[g in d["retrieval_ids"] for g,d in zip(gold,details)]
            recall35=[g in d["retrieval_ids"][:35] for g,d in zip(gold,details)]
            ko=val.vessel_name.str.contains("[가-힣]",regex=True).tolist()
            def mean(values,flags):
                chosen=[v for v,f in zip(values,ko) if f==flags]
                return sum(chosen)/len(chosen) if chosen else None
            correct=[a==b for a,b in zip(pred,gold)]
            diagnostics.append({"fold":fold,"rows":len(val),"hangul_rows":sum(ko),"non_hangul_rows":len(ko)-sum(ko),
                "hangul_accuracy":mean(correct,True),"non_hangul_accuracy":mean(correct,False),
                "union_inclusion_hangul":mean(inside,True),"union_inclusion_non_hangul":mean(inside,False),
                "recall_at_35_hangul":mean(recall35,True),"recall_at_35_non_hangul":mean(recall35,False),
                "candidate_sizes":{"min":min(len(d["retrieval_ids"]) for d in details),"max":max(len(d["retrieval_ids"]) for d in details)},
                "index_seconds":index_seconds,"score_seconds":score_seconds,"row_independence":True})
            parts.append(pd.DataFrame({"sample_id":val.sample_id,"fold":fold,"pred_vessel_id":pred,"gold_rank":rank}))
            map_hashes[str(fold)]={"map":sha256(map_path),"manifest":sha256(map_path.with_suffix(".manifest.json"))}
        oof=pd.concat(parts).sort_values("sample_id").reset_index(drop=True)
        output=out/f"oof_{args.experiment_id}.csv"
        oof.to_csv(output,index=False,encoding="utf-8")
        command=[sys.executable,"-B","-X","utf8",str(ROOT/"eval_kit/evaluate.py"),"--oof",str(output),
                 "--train",str(args.data_dir/"train.csv"),"--folds",str(args.folds_file)]
        if not args.all_folds:command.append("--subset")
        evaluated=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=240,check=True)
        (out/"official_evaluator.txt").write_text(evaluated.stdout,encoding="utf-8")
        h_n=sum(d["hangul_rows"] for d in diagnostics);n_n=sum(d["non_hangul_rows"] for d in diagnostics)
        h=sum(d["hangul_accuracy"]*d["hangul_rows"] for d in diagnostics)/h_n
        n=sum(d["non_hangul_accuracy"]*d["non_hangul_rows"] for d in diagnostics)/n_n
        record["metrics"].update(metric_source="existing evaluator + public OOF/subset development; not official server",
            hangul_rows=h_n,non_hangul_rows=n_n,hangul_accuracy=h,non_hangul_accuracy=n,weighted_accuracy=.7*h+.3*n,
            candidate_recall_hangul=sum(d["union_inclusion_hangul"]*d["hangul_rows"] for d in diagnostics)/h_n,
            candidate_recall_non_hangul=sum(d["union_inclusion_non_hangul"]*d["non_hangul_rows"] for d in diagnostics)/n_n,
            seen_vessel_fraction=0.)
        record.update(diagnostics=diagnostics,map_hashes=map_hashes,
            retrieval={"jamo_top_k":250,"dense_top_k":35,"top35_ranking":"RRF of Jamo and optional Dense ranks",
                       "union_inclusion_is_not_recall35":True},
            lineage={"legacy_lexicon_reads":0,"external_existing_dpr":args.dense_mode=="hybrid",
                     "EXAONE_only":args.dense_mode=="none","full_5fold_oof":args.all_folds},
            artifacts=[str(output),str(out/"official_evaluator.txt")],
            limitations=["Public train previously used for EDA; not an independent final test",
                         "Known fold0 cases are diagnostics, never training/candidate injections",
                         "Existing DPR lineage is separate and not certified clean-room" if dense else "No existing DPR model used"])
        record["cost"].update(inference_seconds=sum(d["score_seconds"] for d in diagnostics),
                              timing_scope="Per-fold score_seconds only; index loading/build and evaluator timings separately recorded",
                              total_validation_seconds=time.monotonic()-tick)
        finish_run(path,record,"evaluated_development")
        print(record["metrics"],flush=True)
    except BaseException as exc:
        finish_run(path,record,"blocked_missing_map" if isinstance(exc,FileNotFoundError) else
                   ("interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"),exc)
        raise


if __name__=="__main__":
    main()
