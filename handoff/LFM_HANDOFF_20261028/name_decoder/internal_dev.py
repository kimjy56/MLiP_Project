"""Vessel-isolated internal checkpoint selection and training-only mining."""
import math
import random
import torch
from torch.nn import functional as F


def split_internal_rows(rows, fraction=0.05, max_vessels=128, seed=42):
    if not 0 < fraction < 1:
        raise ValueError('dev fraction must be between0 and1')
    # All observations/descendants of an ID stay together. Stratify by whether
    # the vessel has any Hangul observation, without fitting name mappings.
    languages=rows.groupby('target_vessel_id').hangul.any()
    groups={flag:sorted(languages[languages.eq(flag)].index) for flag in (False,True)}
    if any(len(v)<2 for v in groups.values()):
        raise ValueError('Internal split needs at least2 vessels in each language stratum')
    if max_vessels and max_vessels<2:
        raise ValueError('dev max vessels must be0 or >=2')
    counts={flag:min(len(ids)-1,max(1,math.ceil(len(ids)*fraction))) for flag,ids in groups.items()}
    while max_vessels and sum(counts.values())>max_vessels:
        flag=max(counts,key=lambda k:counts[k] if counts[k]>1 else -1)
        counts[flag]-=1
    rng=random.Random(seed);dev_ids=set()
    for flag,ids in groups.items():
        dev_ids.update(rng.sample(ids,counts[flag]))
    dev=rows[rows.target_vessel_id.isin(dev_ids)].copy()
    train=rows[~rows.target_vessel_id.isin(dev_ids)].copy()
    assert not set(train.target_vessel_id)&set(dev.target_vessel_id)
    if set(train.hangul)!={False,True} or set(dev.hangul)!={False,True}:
        raise ValueError('Both sides must contain both observation languages')
    return train,dev


def crossfit_confusions(train_rows, catalog_df, catalog):
    """Refit every miner using only internal-train groups, never internal dev."""
    import importlib.util
    import pandas as pd
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'AR-06/script.py'
    spec=importlib.util.spec_from_file_location('_internal_ar06',path)
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    index=engine.CatalogIndex(catalog_df);result=[];fit_audit=[]
    folds=sorted(train_rows.fold.unique())
    if len(folds)<2:raise ValueError('Need >=2 internal-train folds for mining')
    for fold in folds:
        learning=train_rows[train_rows.fold.ne(fold)]
        probe=train_rows[train_rows.fold.eq(fold)]
        if set(learning.target_vessel_id)&set(probe.target_vessel_id):
            raise ValueError('Vessel crosses miner folds')
        model=engine.fit_matching_model(learning,catalog_df,kr_mode='AR06',en_mode='EN_V3',use_train_lookup=False)
        _,_,candidates=engine.predict_rows(probe.drop(columns=['target_vessel_id','target','fold'],errors='ignore'),model,index,use_train_lookup=False,top_candidates=8)
        fit_audit.append({'probe_fold':int(fold),'fit_sample_ids':sorted(learning.sample_id),'probe_sample_ids':sorted(probe.sample_id)})
        for row,items in zip(probe.itertuples(index=False),candidates):
            for rank,(score,vid) in enumerate(items,1):
                if catalog.name_by_id[vid]==catalog.name_by_id[row.target_vessel_id]:continue
                result.append(dict(sample_id=row.sample_id,inner_fold=int(fold),observed_name=row.vessel_name,catalog_name=catalog.name_by_id[row.target_vessel_id],negative_name=catalog.name_by_id[vid],negative_vessel_id=vid,baseline_rank=rank,baseline_score=score))
    return pd.DataFrame(result),fit_audit


def grouped_loss(nll, labels, group_size, beta, temperature, reduction):
    scores=-nll.reshape(-1,group_size)
    rank=F.cross_entropy(scores/temperature,torch.zeros(len(scores),device=scores.device,dtype=torch.long))
    positive=nll[::group_size]
    if reduction=='token':
        counts=labels[::group_size,1:].ne(-100).sum(1)
        sft=(positive*counts).sum()/counts.sum()
    else:sft=positive.mean()
    return sft+beta*rank


@torch.no_grad()
def evaluate_dev(model, groups, hangul, pad_token_id, batch_size, device,
                 beta, temperature, reduction, hangul_mass):
    from .noisy_channel import collate_channel_examples
    from .channel_training import sequence_nll
    # Sum row/token sufficient statistics, avoiding a short last batch bias.
    totals={flag:[0.,0.,0.,0] for flag in (False,True)}
    was_training=model.training;model.eval()
    try:
        for offset in range(0,len(groups),batch_size):
            chunk=groups[offset:offset+batch_size];size=len(chunk[0])
            batch={k:v.to(device) for k,v in collate_channel_examples([e for g in chunk for e in g],pad_token_id).items()}
            output=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
            nll=sequence_nll(output.logits,batch['labels']);scores=-nll.reshape(-1,size)
            rank=F.cross_entropy(scores/temperature,torch.zeros(len(chunk),device=device,dtype=torch.long),reduction='none')
            counts=batch['labels'][::size,1:].ne(-100).sum(1)
            for i in range(len(chunk)):
                flag=bool(hangul[offset+i]);stat=totals[flag]
                weight=int(counts[i]) if reduction=='token' else 1
                stat[0]+=float(nll[i*size])*weight;stat[1]+=weight
                stat[2]+=float(rank[i]);stat[3]+=1
        if any(v[3]==0 for v in totals.values()):raise ValueError('Dev requires both languages')
        value=sum((hangul_mass if flag else 1-hangul_mass)*(v[0]/v[1]+beta*v[2]/v[3]) for flag,v in totals.items())
        if not math.isfinite(value):raise FloatingPointError('Nonfinite internal dev loss')
        return value
    finally:model.train(was_training)


@torch.no_grad()
def restore_adapter(model,state):
    from .lora import LoRALinear
    expected={f'{name}.{attr}':getattr(module,attr) for name,module in model.named_modules() if isinstance(module,LoRALinear) for attr in ('lora_A','lora_B')}
    if set(state)!=set(expected):raise ValueError('Adapter snapshot keys mismatch')
    for key,param in expected.items():
        if param.shape!=state[key].shape:raise ValueError('Adapter shape mismatch')
        param.copy_(state[key].to(device=param.device,dtype=param.dtype))
