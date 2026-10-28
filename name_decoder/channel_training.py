"""Fit-fold-only lexical mining, coverage sampling and reverse LM losses."""
import random
import re
import torch
from torch.nn import functional as F

class LanguageCycleSampler:
    def __init__(self, hangul, mass=0.7, seed=42):
        self.rng = random.Random(seed)
        self.mass = mass
        self.groups = {f: [i for i,v in enumerate(hangul) if bool(v)==f] for f in (False,True)}
        if not all(self.groups.values()):
            raise ValueError("Both languages required")
        self.queues = {False: [], True: []}
        self.counts = [0]*len(hangul)
    def draw(self, size):
        result=[]
        for _ in range(size):
            flag=self.rng.random()<self.mass
            if not self.queues[flag]:
                self.queues[flag]=self.groups[flag].copy()
                self.rng.shuffle(self.queues[flag])
            i=self.queues[flag].pop()
            self.counts[i]+=1
            result.append(i)
        return result

def mine_name_negatives(pairs, catalog, count=2):
    """Catalog lexical confusions; exclude all fit-known positives and same names."""
    from rapidfuzz import fuzz, process
    names=sorted(set(catalog.name_by_id.values()))
    positives={}
    for row in pairs.itertuples(index=False):
        positives.setdefault(row.observed_name,set()).add(row.catalog_name)
    result=[]
    for row in pairs.itertuples(index=False):
        forbidden=positives[row.observed_name]
        matches=process.extract(row.catalog_name,names,scorer=fuzz.ratio,limit=count+len(forbidden))
        result.append([name for name,_,_ in matches if name not in forbidden][:count])
    if any(len(items)!=count for items in result):
        raise ValueError("Insufficient unambiguous negative names")
    return result

def sequence_nll(logits, labels):
    labels=labels[:,1:]
    counts=labels.ne(-100).sum(1)
    if (counts==0).any():
        raise ValueError("Every row requires target tokens")
    losses=F.cross_entropy(logits[:,:-1].float().transpose(1,2),labels,ignore_index=-100,reduction="none")
    return losses.sum(1)/counts

def channel_objective(nll, group_size, beta=0.1, temperature=1.0):
    scores=-nll.reshape(-1,group_size)
    sft=-scores[:,0].mean()
    rank=F.cross_entropy(scores/temperature,torch.zeros(len(scores),device=scores.device,dtype=torch.long))
    return sft+beta*rank,sft,rank

def learned_hangul_variants(pairs):
    """Unambiguous token correspondences learned only from aligned fit names.

    Keep digits and punctuation intact; require complete alphabetic coverage.
    """
    mapping={}
    for row in pairs.itertuples(index=False):
        en,ko=row.catalog_name.split(),row.observed_name.split()
        if not row.hangul or len(en)!=len(ko):
            continue
        for a,b in zip(en,ko):
            if re.fullmatch(r"[A-Z]+",a) and re.fullmatch(r"[\uac00-\ud7a3]+",b):
                mapping.setdefault(a,set()).add(b)
    mapping={k:next(iter(v)) for k,v in mapping.items() if len(v)==1}
    variants=set()
    for row in pairs.itertuples(index=False):
        words=row.catalog_name.split()
        if any(re.fullmatch(r"[A-Z]+",w) and w not in mapping for w in words):
            continue
        translated=[mapping.get(w,w) for w in words]
        # Do not collapse boundaries around identifying digits/punctuation.
        separators=(" ", "") if all(re.fullmatch(r"[A-Z]+",w) for w in words) else (" ",)
        for sep in separators:
            observed=sep.join(translated)
            if observed!=row.observed_name and re.search(r"[\uac00-\ud7a3]",observed):
                variants.add((row.catalog_name,observed))
    return sorted(variants),len(mapping)


def use_inner_confusions(pairs, catalog, fit_rows, mined, fallback):
    """Use only audited fit-row observations, excluding all known positives.

    Augmented observations without an exact original pair retain lexical
    negatives. Do not propagate mining results across differing observations.
    """
    from .catalog import canonical_name
    from .noisy_channel import normalize_observed
    required={'sample_id','inner_fold','observed_name','catalog_name',
              'negative_name','negative_vessel_id','baseline_rank','baseline_score'}
    if not required <= set(mined.columns):
        raise ValueError('Invalid confusion CSV columns')
    sources=fit_rows.set_index('sample_id')
    if not set(mined.sample_id)<=set(sources.index):
        raise ValueError('Confusions include samples outside fit folds')
    valid_names=set(catalog.name_by_id.values())
    positives={}
    for row in pairs.itertuples(index=False):
        positives.setdefault(row.observed_name,set()).add(row.catalog_name)
    pools={}
    for row in mined.itertuples(index=False):
        source=sources.loc[row.sample_id]
        correct=catalog.name_by_id[source.target_vessel_id]
        observed=normalize_observed(row.observed_name)
        negative=canonical_name(row.negative_name)
        if (int(row.inner_fold)!=int(source.fold) or
            observed!=normalize_observed(source.vessel_name) or
            canonical_name(row.catalog_name)!=correct or
            catalog.name_by_id.get(row.negative_vessel_id)!=negative):
            raise ValueError('Confusion row provenance mismatch')
        if negative not in valid_names or negative in positives.get(observed,set()):
            continue
        pools.setdefault((correct,observed),[]).append((float(row.baseline_score),int(row.baseline_rank),negative))
    result=[];counts=[]
    for row,back in zip(pairs.itertuples(index=False),fallback):
        selected=[]
        for _,_,name in sorted(pools.get((row.catalog_name,row.observed_name),[]),key=lambda x:(x[1],-x[0],x[2])):
            if name not in selected and name not in positives[row.observed_name]:
                selected.append(name)
            if len(selected)==len(back):break
        counts.append(len(selected))
        for name in back:
            if len(selected)==len(back):break
            if name not in selected:selected.append(name)
        result.append(selected)
    return result, {'pairs_with_mined_negatives':sum(c>0 for c in counts),
                    'mined_negative_slots':sum(counts),
                    'lexical_fallback_slots':sum(map(len,result))-sum(counts)}
