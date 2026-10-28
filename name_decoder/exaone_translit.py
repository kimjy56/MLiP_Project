"""EXAONE-only LoRA and constrained offline transliteration; no legacy lexicon imports."""
from __future__ import annotations
import json
import math
from pathlib import Path
import re
import torch
from torch import nn
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,LogitsProcessor,LogitsProcessorList
from .deterministic_ko_rules import canonical_name,clean_korean

MODEL_ID="LGAI-EXAONE/EXAONE-4.0-1.2B"
TARGETS=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
PROMPT_VERSION="en-ko-raw-v1"


def build_prompt(english):
    english=canonical_name(english)
    if not english:raise ValueError("Empty transliteration source")
    return f"[EN] {english} => [KO] "


def encode_example(tokenizer,source,target,max_length=256):
    target=clean_korean(target)
    if not target:raise ValueError("Invalid Korean training target")
    prompt=tokenizer.encode(build_prompt(source),add_special_tokens=False)
    answer=tokenizer.encode(target,add_special_tokens=False)+[tokenizer.eos_token_id]
    if tokenizer.eos_token_id is None:raise ValueError("Tokenizer EOS is required")
    ids=prompt+answer
    if len(ids)>max_length:raise ValueError("Example exceeds max length; do not silently truncate")
    return {"input_ids":ids,"labels":[-100]*len(prompt)+answer}


def collate_examples(examples,pad_token_id):
    if not examples:raise ValueError("Empty batch")
    size=max(len(e["input_ids"]) for e in examples)
    inputs=torch.full((len(examples),size),pad_token_id,dtype=torch.long)
    labels=torch.full_like(inputs,-100)
    mask=torch.zeros_like(inputs)
    for i,e in enumerate(examples):
        n=len(e["input_ids"])
        inputs[i,:n]=torch.tensor(e["input_ids"])
        labels[i,:n]=torch.tensor(e["labels"])
        mask[i,:n]=1
    return {"input_ids":inputs,"attention_mask":mask,"labels":labels}


class ExaoneLoRALinear(nn.Module):
    def __init__(self,base,rank=24,alpha=48.,dropout=.05):
        super().__init__()
        if not isinstance(base,nn.Linear):raise TypeError("Expected Linear")
        if rank<1 or not math.isfinite(alpha) or alpha<=0 or not 0<=dropout<1:
            raise ValueError("Invalid adapter settings")
        self.base=base.requires_grad_(False)
        self.rank=int(rank);self.alpha=float(alpha);self.dropout=nn.Dropout(dropout)
        self.lora_A=nn.Parameter(base.weight.new_empty(rank,base.in_features))
        self.lora_B=nn.Parameter(base.weight.new_zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))

    def forward(self,value):
        return self.base(value)+((self.dropout(value)@self.lora_A.T)@self.lora_B.T)*(self.alpha/self.rank)


def inject_lora(model,rank=24,alpha=48.,dropout=.05):
    if getattr(model.config,"model_type",None)!="exaone4":
        raise ValueError("Only EXAONE4 architecture is accepted")
    if any(isinstance(m,ExaoneLoRALinear) for m in model.modules()):
        raise ValueError("Adapters already injected")
    selected=[(n,m) for n,m in model.named_modules() if isinstance(m,nn.Linear) and n.rsplit(".",1)[-1] in TARGETS]
    layers=model.config.num_hidden_layers
    for leaf in TARGETS:
        if sum(n.endswith("."+leaf) for n,_ in selected)!=layers:
            raise ValueError(f"Expected {layers} instances of {leaf}; refuse partial LoRA")
    model.requires_grad_(False)
    for name,base in selected:
        parent,_,attr=name.rpartition(".")
        setattr(model.get_submodule(parent),attr,ExaoneLoRALinear(base,rank,alpha,dropout))
    return {"rank":rank,"alpha":alpha,"dropout":dropout,"targets":list(TARGETS),
            "module_names":[n for n,_ in selected],
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)}


def adapter_state(model):
    return {n:p.detach().cpu().contiguous().clone() for n,p in model.named_parameters()
            if n.endswith(".lora_A") or n.endswith(".lora_B")}


def restore_adapter(model,state):
    params={n:p for n,p in model.named_parameters() if n.endswith(".lora_A") or n.endswith(".lora_B")}
    if set(state)!=set(params):raise ValueError("Adapter keys mismatch")
    with torch.no_grad():
        for name,value in state.items():
            if value.shape!=params[name].shape:raise ValueError("Adapter shape mismatch")
            params[name].copy_(value.to(params[name]))


def resolve_base(model_id,local_model=None,offline=False):
    """Resolve only the requested public base. No remote ship names/data are sent."""
    from huggingface_hub import HfApi,snapshot_download
    if model_id!=MODEL_ID:raise ValueError(f"This pipeline requires {MODEL_ID}")
    root=Path(__file__).resolve().parents[1]
    candidate=Path(local_model) if local_model else root/"artifacts/pretrained/exaone-4.0-1.2b"
    if (candidate/"config.json").is_file():
        config=AutoConfig.from_pretrained(candidate,local_files_only=True,trust_remote_code=False)
        if config.model_type!="exaone4" or config.num_hidden_layers!=30 or config.vocab_size!=102400:
            raise ValueError("Local base configuration does not match EXAONE4 1.2B")
        if not list(candidate.glob("*.safetensors")):raise FileNotFoundError("EXAONE weights missing")
        return candidate.resolve(),{"model_id":model_id,"revision":getattr(config,"_commit_hash",None),"source":"local_files"}
    if offline:
        try:
            result=snapshot_download(model_id,local_files_only=True)
        except Exception as exc:
            raise FileNotFoundError(f"EXAONE base is not cached; expected {candidate}. No fallback model used.") from exc
        return Path(result),{"model_id":model_id,"revision":Path(result).name,"source":"existing_cache"}
    info=HfApi().model_info(model_id,timeout=15)
    revision=info.sha
    if not revision:raise ValueError("Cannot pin base revision")
    result=snapshot_download(model_id,revision=revision,local_dir=candidate,
        allow_patterns=["*.json","*.safetensors","tokenizer.model","README.md","LICENSE*"])
    return Path(result).resolve(),{"model_id":model_id,"revision":revision,"source":"pinned_public_download"}


def load_base(path,device="cuda",training=False):
    if device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    config=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    if config.model_type!="exaone4" or config.num_hidden_layers!=30 or config.vocab_size!=102400:
        raise ValueError("Base configuration does not match EXAONE4 1.2B")
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    if tokenizer.eos_token_id is None:raise ValueError("EOS missing")
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="left"
    dtype=torch.bfloat16 if device=="cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device=="cuda" else torch.float32)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
                                               torch_dtype=dtype,attn_implementation="sdpa").to(device)
    model.config.use_cache=not training
    return model,tokenizer


class HangulOnlyLogitsProcessor(LogitsProcessor):
    """Allow whole decoded Hangul/digit/space tokens and the tokenizer's EOS.

    Byte-fragment tokens decoding to replacement characters are excluded.
    Final decoded sequences are independently checked again.
    """
    def __init__(self,tokenizer,prompt_length):
        self.prompt_length=prompt_length
        self.eos_id=tokenizer.eos_token_id
        if self.eos_id is None:raise ValueError("EOS required")
        self.allowed=[];self.initial=[];self._masks={}
        specials=set(tokenizer.all_special_ids)
        for i in range(len(tokenizer)):
            if i in specials:continue
            value=tokenizer.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if value and re.fullmatch("[가-힣0-9 ]+",value):
                self.allowed.append(i)
                if value.strip():self.initial.append(i)
        if not self.initial:raise ValueError("No valid target tokens")
        self.allowed.append(self.eos_id)

    def __call__(self,input_ids,scores):
        first=input_ids.shape[1]<=self.prompt_length
        key=(scores.device,scores.dtype,scores.shape[-1],first)
        if key not in self._masks:
            mask=torch.full((scores.shape[-1],),float("-inf"),device=scores.device,dtype=scores.dtype)
            ids=self.initial if first else self.allowed
            valid=[i for i in ids if i<scores.shape[-1]]
            if not valid:raise ValueError("Allowed vocabulary does not fit logits")
            mask[valid]=0
            self._masks[key]=mask
        return scores+self._masks[key]


class ExaoneTransliterator:
    def __init__(self,model,tokenizer,manifest=None):
        self.model=model.eval();self.tokenizer=tokenizer
        self.manifest=manifest or {}
        self._allowed_processor=None

    @classmethod
    def from_adapter(cls,checkpoint,device="cuda",local_model=None):
        checkpoint=Path(checkpoint)
        manifest=json.loads((checkpoint/"manifest.json").read_text(encoding="utf-8"))
        if manifest.get("model_id")!=MODEL_ID:raise ValueError("Wrong model identity")
        base=Path(local_model or manifest["base_path"])
        from experiments.exaone_common import sha256
        for name,expected in manifest["base_file_sha256"].items():
            if sha256(base/name)!=expected:raise ValueError("Base/tokenizer files changed")
        if sha256(checkpoint/"adapter.pt")!=manifest["adapter_sha256"]:
            raise ValueError("Adapter hash mismatch")
        model,tokenizer=load_base(base,device)
        inject_lora(model,**{k:manifest["lora"][k] for k in ["rank","alpha","dropout"]})
        restore_adapter(model,torch.load(checkpoint/"adapter.pt",map_location="cpu",weights_only=True))
        return cls(model,tokenizer,manifest)

    @torch.inference_mode()
    def generate(self,names,num_beams=4,num_return_sequences=3,max_new_tokens=48):
        if not 1<=num_return_sequences<=num_beams:raise ValueError("return_sequences must be1..num_beams")
        if not names:return []
        batch=self.tokenizer([build_prompt(n) for n in names],padding=True,add_special_tokens=False,return_tensors="pt")
        device=next(self.model.parameters()).device
        batch={k:v.to(device) for k,v in batch.items()}
        length=batch["input_ids"].shape[1]
        if self._allowed_processor is None:
            self._allowed_processor=HangulOnlyLogitsProcessor(self.tokenizer,length)
        self._allowed_processor.prompt_length=length
        output=self.model.generate(**batch,do_sample=False,num_beams=num_beams,
            num_return_sequences=num_return_sequences,max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([self._allowed_processor]),
            return_dict_in_generate=True,output_scores=False,use_cache=True)
        decoded=self.tokenizer.batch_decode(output.sequences[:,length:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        results=[]
        for i in range(len(names)):
            choices=[]
            for j in range(num_return_sequences):
                k=i*num_return_sequences+j
                tail=output.sequences[k,length:].tolist()
                value=clean_korean(decoded[k])
                # Truncated/empty strings are not silently promoted to pronunciations.
                if value and self.tokenizer.eos_token_id in tail:
                    choices.append({"text":value,"score":float(-j),"score_type":"beam_rank"})
            seen=set()
            results.append([v for v in choices if not (v["text"] in seen or seen.add(v["text"]))])
        return results
