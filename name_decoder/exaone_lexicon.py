"""Two-track combination policy; consumes only EXAONE output and general rules."""
from __future__ import annotations
from itertools import product
import re
from .deterministic_ko_rules import canonical_name,word_tokens,clean_korean,deterministic_variants,digit_sequence,EN_NUMBERS,ORDINALS,ROMAN_NUMBERS


def expected_numbers(name):
    words=word_tokens(name)
    values=[]
    for i,w in enumerate(words):
        if w.isascii() and w.isdigit():values.append(int(w))
        elif i==len(words)-1:
            table=EN_NUMBERS|ORDINALS|ROMAN_NUMBERS
            if w in table:values.append(table[w])
    return tuple(values)


def valid_candidate(name,value):
    value=clean_korean(value)
    if not value:return ""
    # Preserve explicit digits; allow numeric words to stay phonetic words.
    digits=digit_sequence(value)
    expected=expected_numbers(name)
    if digits and digits!=expected:return ""
    if digit_sequence(name) and digits!=digit_sequence(name):return ""
    return value


def select_word_candidates(word,generated,max_candidates=3):
    if max_candidates<1:raise ValueError("max_candidates must be positive")
    values=deterministic_variants(word)+[v["text"] if isinstance(v,dict) else v for v in generated]
    result=[]
    for value in values:
        value=valid_candidate(word,value)
        if value and value not in result:result.append(value)
    return result[:max_candidates]


def compose_word_candidates(name,word_lexicon,limit=4,per_word=2):
    """Bounded product beam: no exponential materialization and no heldout labels."""
    if limit<1:return []
    options=[word_lexicon.get(w,deterministic_variants(w))[:per_word] for w in word_tokens(name)]
    if not options or any(not o for o in options):return []
    beam=[(0,(),())]
    for values in options:
        pool=[]
        for penalty,ranks,parts in beam:
            for rank,value in enumerate(values):
                pool.append((penalty+rank,ranks+(rank,),parts+(value,)))
        # Lexicographic rank tuples deterministically break equal rank sums.
        beam=sorted(pool,key=lambda v:(v[0],v[1],v[2]))[:limit]
    return list(dict.fromkeys(" ".join(v[2]) for v in beam))


def merge_catalog_candidates(name,generated,word_lexicon,max_candidates=4):
    direct=[]
    for sample in generated:
        text=sample["text"] if isinstance(sample,dict) else sample
        value=valid_candidate(name,text)
        if value and value not in direct:direct.append(value)
    composition=[v for x in compose_word_candidates(name,word_lexicon,max(8,max_candidates))
                 if (v:=valid_candidate(name,x))]
    # At most two contextual candidates; remaining slots cover distinct word combinations.
    result=direct[:min(2,max_candidates)]
    pool=list(dict.fromkeys(composition+direct[2:]))
    while pool and len(result)<max_candidates:
        pool=[p for p in pool if p not in result]
        if not pool:break
        if result:
            def distance(p):
                tokens=p.split()
                return min(sum(a!=b for a,b in zip(tokens,r.split()))+abs(len(tokens)-len(r.split())) for r in result)
            chosen=max(enumerate(pool),key=lambda pair:(distance(pair[1]),-pair[0]))[1]
        else:chosen=pool[0]
        result.append(chosen)
    return result
