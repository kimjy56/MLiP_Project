"""Train fold-isolated EXAONE transliteration adapters; never build submission ZIPs."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.exaone_common import MODEL_ID,begin_run,finish_run,write_json,sha256,object_sha256,configure,now_kst
from name_decoder.exaone_data import read_official,prepare_pairs,catalog_names,catalog_words


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id",required=True)
    p.add_argument("--model-id",default=MODEL_ID)
    p.add_argument("--local-model",type=Path)
    p.add_argument("--offline",action="store_true")
    p.add_argument("--data-dir",type=Path,default=ROOT/"data")
    p.add_argument("--folds-file",type=Path,default=ROOT/"eval_kit/folds_group5_seed42.csv")
    p.add_argument("--fold",type=int,choices=range(5),default=0)
    p.add_argument("--final-train",action="store_true")
    p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=64,help="Effective batch size, not physical GPU microbatch")
    p.add_argument("--microbatch-size",type=int,default=4)
    p.add_argument("--rank",type=int,default=24)
    p.add_argument("--alpha",type=float,default=48.)
    p.add_argument("--lora-dropout",type=float,default=.05)
    p.add_argument("--learning-rate",type=float,default=2e-4)
    p.add_argument("--weight-decay",type=float,default=.01)
    p.add_argument("--warmup-ratio",type=float,default=.05)
    p.add_argument("--max-length",type=int,default=256)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--prepare-only",action="store_true",help="Validate fold isolation and count data; no model resolution/load")
    return p


def main():
    args=parser().parse_args()
    if args.model_id!=MODEL_ID:raise SystemExit("Only requested EXAONE model is permitted")
    if min(args.epochs,args.batch_size,args.microbatch_size,args.rank,args.max_length)<1:
        raise SystemExit("Positive training settings required")
    if not 0<=args.warmup_ratio<1 or args.microbatch_size>args.batch_size:
        raise SystemExit("Invalid warmup/microbatch settings")
    out,path,record=begin_run(args.experiment_id,"EXAONE-only fold-isolated forward transliteration")
    tick=time.monotonic()
    try:
        train,catalog,folds=read_official(args.data_dir,args.folds_file)
        pairs,fit,val,audit=prepare_pairs(train,catalog,folds,args.fold,args.final_train)
        if pairs.empty:raise ValueError("No trainable transliteration pairs")
        assert not (set(audit["fit_vessel_ids"]) & set(audit["heldout_vessel_ids"]))
        source_hashes={"train.csv":sha256(args.data_dir/"train.csv"),"vessel_catalog.csv":sha256(args.data_dir/"vessel_catalog.csv"),
                       "folds":sha256(args.folds_file)}
        data={"source_files_sha256":source_hashes,"split_id":"full_train" if args.final_train else f"G5-s42 fold{args.fold}",
              "split_file_sha256":source_hashes["folds"],"seed":args.seed,"training_rows":len(fit),"validation_rows":len(val),
              "fit_vessel_ids_sha256":object_sha256(audit["fit_vessel_ids"]),
              "heldout_vessel_ids_sha256":object_sha256(audit["heldout_vessel_ids"]),
              "fit_pairs_sha256":object_sha256(pairs.to_dict("records")),
              "pair_counts":{k:int(v) for k,v in pairs.kind.value_counts().items()},
              "catalog_rows":len(catalog),"catalog_unique_names":len(catalog_names(catalog)),
              "catalog_unique_words":len(catalog_words(catalog)),"legacy_assets_read":[]}
        record.update(data=data,model={"type":MODEL_ID,"parameters":vars(args)|{"data_dir":str(args.data_dir),"folds_file":str(args.folds_file),
                         "local_model":str(args.local_model) if args.local_model else None},
                       "loss":"causal CE; target tokens only","checkpoint_selection":"fixed last epoch; no outerfold early stopping"})
        # Only descendants of fit rows are saved; no validation rows/labels in training assets.
        write_json(out/"fit_membership.json",{"fit_vessel_ids":audit["fit_vessel_ids"],"heldout_vessel_ids_sha256":data["heldout_vessel_ids_sha256"]})
        write_json(out/"data_manifest.json",data)
        if args.prepare_only:
            record["metrics"]["metric_source"]="Data isolation/count audit only; no model training or accuracy"
            record["observations"]=["Outer heldout vessel IDs absent from all fit rows/derived pairs",audit["alignment_caveat"]]
            record["artifacts"]=[str((out/"data_manifest.json").relative_to(ROOT))]
            finish_run(path,record,"completed_data_audit")
            print({"data_audit":True,"pairs":len(pairs),"catalog_words":data["catalog_unique_words"],"fit_rows":len(fit),"validation_rows":len(val)},flush=True)
            return
        from name_decoder.exaone_translit import resolve_base,load_base,inject_lora,encode_example,collate_examples,adapter_state
        base,base_provenance=resolve_base(args.model_id,args.local_model,args.offline)
        import torch
        from transformers import get_cosine_schedule_with_warmup
        configure(args.seed)
        model,tokenizer=load_base(base,args.device,training=True)
        spec=inject_lora(model,args.rank,args.alpha,args.lora_dropout)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.enable_input_require_grads()
        encoded=[];rejected=0
        for row in pairs.itertuples(index=False):
            try:encoded.append(encode_example(tokenizer,row.source,row.target,args.max_length))
            except ValueError:rejected+=1
        if not encoded:raise ValueError("No tokenized pairs remain")
        steps_per_epoch=math.ceil(len(encoded)/args.batch_size)
        steps=steps_per_epoch*args.epochs
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=args.weight_decay)
        warmup=math.ceil(steps*args.warmup_ratio)
        scheduler=get_cosine_schedule_with_warmup(optimizer,warmup,steps)
        fp16=next(model.parameters()).dtype==torch.float16
        scaler=torch.amp.GradScaler("cuda",enabled=fp16)
        record["training_pairs"].update(count=len(encoded),positive_rule="official fit full-name/aligned-word and generic rules",negative_rule=None)
        record["model"].update(lora=spec,total_optimizer_steps=steps,warmup_steps=warmup,rejected_overlength=rejected)
        record["environment"].update(device=args.device,torch=torch.__version__,
                                     gpu=torch.cuda.get_device_name(0) if args.device=="cuda" else None)
        write_json(path,record)
        model.train()
        import random
        history=[];step=0
        for epoch in range(args.epochs):
            order=list(range(len(encoded)));random.Random(args.seed+epoch).shuffle(order)
            epoch_loss=0.;epoch_tokens=0
            for offset in range(0,len(order),args.batch_size):
                group=[encoded[i] for i in order[offset:offset+args.batch_size]]
                group_tokens=sum(sum(v!=-100 for v in e["labels"][1:]) for e in group)
                optimizer.zero_grad(set_to_none=True)
                group_loss=0.
                for micro in range(0,len(group),args.microbatch_size):
                    examples=group[micro:micro+args.microbatch_size]
                    batch={k:v.to(args.device) for k,v in collate_examples(examples,tokenizer.pad_token_id).items()}
                    n_tokens=int(batch["labels"][:,1:].ne(-100).sum())
                    loss=model(**batch,use_cache=False).loss
                    if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite EXAONE loss")
                    scaled=loss*(n_tokens/group_tokens)
                    scaler.scale(scaled).backward()
                    group_loss+=float(loss.detach())*n_tokens
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not bool(torch.isfinite(norm)):raise FloatingPointError("Nonfinite adapter gradients")
                scaler.step(optimizer);scaler.update();scheduler.step();step+=1
                epoch_loss+=group_loss;epoch_tokens+=group_tokens
                if step==1 or step%10==0:
                    print({"epoch":epoch+1,"step":step,"steps":steps,"target_ce":group_loss/group_tokens},flush=True)
            history.append({"epoch":epoch+1,"target_ce":epoch_loss/epoch_tokens})
            # Separate immutable epoch checkpoints, never overwrite prior experiments.
            epoch_path=out/"epochs"/f"epoch_{epoch+1}.pt"
            epoch_path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(adapter_state(model),epoch_path)
        checkpoint=out/("full_lora" if args.final_train else f"fold_{args.fold}_lora")
        checkpoint.mkdir()
        torch.save(adapter_state(model),checkpoint/"adapter.pt")
        base_hashes={p.name:sha256(p) for p in sorted(base.iterdir()) if p.is_file() and p.suffix in [".json",".safetensors",".model"]}
        manifest={"model_id":MODEL_ID,"prompt_version":"en-ko-raw-v1","base_path":str(base),
                  "base_provenance":base_provenance,"base_file_sha256":base_hashes,
                  "adapter_sha256":sha256(checkpoint/"adapter.pt"),"lora":spec,"data":data,
                  "fit_membership_sha256":sha256(out/"fit_membership.json"),
                  "code":record["code"],"config":record["model"]["parameters"],
                  "selected_epoch":args.epochs,"selection":"predeclared last epoch","history":history}
        write_json(checkpoint/"manifest.json",manifest)
        record["observations"]=["No outerfold selection/tuning; fixed last epoch selected",audit["alignment_caveat"]]
        record["artifacts"]=[str(checkpoint.relative_to(ROOT)),str((out/"data_manifest.json").relative_to(ROOT))]
        record["cost"].update(training_seconds=time.monotonic()-tick,timing_scope="data/base loading, tokenization, training and adapter/checkpoint saving")
        record["metrics"]["metric_source"]="training loss only; no heldout accuracy measurement"
        finish_run(path,record,"trained_unvalidated")
    except BaseException as exc:
        status="blocked_missing_base" if isinstance(exc,FileNotFoundError) else ("interrupted" if isinstance(exc,KeyboardInterrupt) else "failed")
        finish_run(path,record,status,exc)
        raise


if __name__=="__main__":
    main()
