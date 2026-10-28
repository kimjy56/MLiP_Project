"""Generate new EXAONE word/full-catalog lexicons with resumable, hash-bound batches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.exaone_common import begin_run,finish_run,write_json,sha256,object_sha256,configure,MODEL_ID,code_state
from name_decoder.exaone_data import catalog_names,catalog_words
from name_decoder.exaone_lexicon import select_word_candidates,merge_catalog_candidates


def generate_batches(generator,names,path,config,batch_size,returns,max_new_tokens):
    """Resume only exact matching source/order/config/checkpoint hashes."""
    path.mkdir(parents=True,exist_ok=True)
    results={};actual_batch=batch_size
    offset=0
    while offset<len(names):
        chunk=names[offset:offset+batch_size]
        filename=path/f"{offset:08d}.json"
        key=object_sha256({"config":config,"names":chunk,"returns":returns,"max_new_tokens":max_new_tokens})
        if filename.exists():
            saved=json.loads(filename.read_text(encoding="utf-8"))
            if saved["batch_key"]!=key:raise ValueError("Resume cache provenance mismatch")
            samples=saved["samples"]
        else:
            # OOM changes physical microbatch only; all source order/seeds/policy are recorded.
            samples=[]
            inner=0
            while inner<len(chunk):
                portion=chunk[inner:inner+actual_batch]
                try:
                    generated=generator.generate(portion,num_beams=4,num_return_sequences=returns,max_new_tokens=max_new_tokens)
                except Exception as exc:
                    import torch
                    if not isinstance(exc,torch.cuda.OutOfMemoryError) or actual_batch<=1:raise
                    actual_batch=max(1,actual_batch//2)
                    torch.cuda.empty_cache()
                    continue
                if len(generated)!=len(portion):raise ValueError("Generator dropped a source")
                samples.extend(generated);inner+=len(portion)
            write_json(filename,{"batch_key":key,"source_names":chunk,"samples":samples,"actual_max_microbatch":actual_batch})
        if len(samples)!=len(chunk):raise ValueError("Invalid generation batch cache")
        results.update(zip(chunk,samples))
        offset+=len(chunk)
        print({"phase":path.name,"completed":offset,"total":len(names),"physical_batch":actual_batch},flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id",default="LEX-EXAONE-GENERATE-01")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--catalog-csv",type=Path,default=ROOT/"data/vessel_catalog.csv")
    p.add_argument("--output-word-lex",type=Path)
    p.add_argument("--output-catalog-map",type=Path)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--max-new-tokens",type=int,default=48)
    p.add_argument("--max-cands",type=int,default=4)
    p.add_argument("--limit-names",type=int,default=0,help="Pilot scope only; never claim full catalog coverage")
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--local-model",type=Path,help="Optional path to local EXAONE base model if relocated")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if not 1<=args.max_cands<=4 or args.batch_size<1 or args.limit_names<0:raise SystemExit("Invalid generation settings")
    record_path=ROOT/f"project_log/experiments/{args.experiment_id}.json"
    out=ROOT/"artifacts"/args.experiment_id
    if args.resume:
        if not record_path.is_file() or not out.is_dir():raise SystemExit("No matching partial run")
        record=json.loads(record_path.read_text(encoding="utf-8"))
        if record["status"] not in ["running","failed","interrupted"]:raise SystemExit("Cannot resume a completed/blocked run")
        if record["code"]["file_sha256"]!=code_state()["file_sha256"]:
            raise SystemExit("Source changed since partial generation; use a new experiment ID")
        record["status"]="running";write_json(record_path,record)
    else:
        out,record_path,record=begin_run(args.experiment_id,"New EXAONE-only word + full-name catalog generation")
    tick=time.monotonic()
    try:
        import pandas as pd
        manifest_path=args.checkpoint/"manifest.json"
        if not manifest_path.is_file():raise FileNotFoundError("EXAONE adapter manifest missing; train the requested base first")
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["model_id"]!=MODEL_ID:raise ValueError("Non-EXAONE generator rejected")
        catalog=pd.read_csv(args.catalog_csv,dtype=str,keep_default_na=False,encoding="utf-8-sig")
        names=catalog_names(catalog)
        if args.limit_names:names=names[:args.limit_names]
        subset=catalog[catalog.vessel_name.map(__import__("name_decoder.deterministic_ko_rules",fromlist=["canonical_name"]).canonical_name).isin(names)]
        words=catalog_words(subset)
        config={"checkpoint_manifest_sha256":sha256(manifest_path),"adapter_sha256":sha256(args.checkpoint/"adapter.pt"),
                "catalog_sha256":sha256(args.catalog_csv),"source_names_sha256":object_sha256(names),
                "source_words_sha256":object_sha256(words),"source_count":len(names),
                "batch_size":args.batch_size,"beam_count":4,"word_returns":3,"full_returns":2,
                "max_new_tokens":args.max_new_tokens,"max_cands":args.max_cands,"seed":args.seed,
                "code_sha256":record["code"]["file_sha256"],
                "legacy_policy":{"manual_IDEA_B":False,"old_catalog_map":False,"old_learned_lexicons":False}}
        if (out/"generation_config.json").exists():
            if json.loads((out/"generation_config.json").read_text(encoding="utf-8"))!=config:
                raise ValueError("Cannot resume after source/model/config changes")
        else:write_json(out/"generation_config.json",config)
        from name_decoder.exaone_translit import ExaoneTransliterator
        configure(args.seed)
        generator=ExaoneTransliterator.from_adapter(args.checkpoint,args.device,local_model=args.local_model)
        generated_words=generate_batches(generator,words,out/"word_batches",config,args.batch_size,3,args.max_new_tokens)
        word_lex={word:select_word_candidates(word,generated_words[word]) for word in words}
        generated_full=generate_batches(generator,names,out/"full_name_batches",config,args.batch_size,2,args.max_new_tokens)
        catalog_map={name:merge_catalog_candidates(name,generated_full[name],word_lex,args.max_cands) for name in names}
        word_path=args.output_word_lex or out/"word_exaone_lexicon.json"
        map_path=args.output_catalog_map or out/"catalog_exaone_ko_map.json"
        if word_path.resolve()==map_path.resolve():raise ValueError("Word and catalog output paths must differ")
        if word_path.exists() or map_path.exists() or map_path.with_suffix(".manifest.json").exists():
            raise FileExistsError("Final lexicons/manifest already exist; never overwrite")
        write_json(word_path,word_lex);write_json(map_path,catalog_map)
        summary={"schema_version":1,"generator":"EXAONE-only","model_id":MODEL_ID,
                 "config":config,"training_data":manifest["data"],"checkpoint":str(args.checkpoint),
                 "checkpoint_manifest_sha256":sha256(manifest_path),"catalog_rows":len(catalog),
                 "unique_catalog_names":len(catalog_names(catalog)),"generated_names":len(names),
                 "full_catalog":len(names)==len(catalog_names(catalog)),
                 "word_lexicon_sha256":sha256(word_path),"catalog_map_sha256":sha256(map_path),
                 "word_lexicon_bytes":word_path.stat().st_size,"catalog_map_bytes":map_path.stat().st_size,
                 "word_missing":sum(not v for v in word_lex.values()),"catalog_missing":sum(not v for v in catalog_map.values()),
                 "max_candidates":max((len(v) for v in catalog_map.values()),default=0),
                 "legacy_assets_read":[],"sequence_score_type":"beam rank; no probabilistic confidence claim"}
        write_json(map_path.with_suffix(".manifest.json"),summary)
        record.update(data=manifest["data"],generation=summary,artifacts=[str(map_path),str(word_path),str(map_path.with_suffix(".manifest.json"))])
        record["metrics"]["metric_source"]="Generation coverage/quality only; no matching accuracy"
        record["cost"].update(inference_seconds=time.monotonic()-tick,timing_scope="offline model load, word/full generation, combination, hash and saving")
        finish_run(record_path,record,"generated_unvalidated")
    except BaseException as exc:
        finish_run(record_path,record,"blocked_missing_adapter" if isinstance(exc,FileNotFoundError) else
                   ("interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"),exc)
        raise


if __name__=="__main__":
    main()
