"""Independent static EXAONE lexicon matcher. No LLM, legacy map or learned lookup loads."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import math
import re
from rapidfuzz import fuzz,process
from rapidfuzz.distance import OSA
from name_decoder.deterministic_ko_rules import canonical_name,hangul_to_jamo,digit_sequence,deterministic_variants


def number(value):
    try:value=float(value)
    except (ValueError,TypeError):return None
    return value if math.isfinite(value) and value>0 else None


def normalize_query(value):
    text=canonical_name(value)
    text=re.sub(r"\s+(?:호|선)$","",text)
    text=re.sub(r"(?<=\d)\s*호$","",text)
    def letters(match):
        choices=deterministic_variants(match.group())
        return choices[0] if choices else match.group()
    return re.sub(r"\b[B-DF-HJ-NP-TV-Z]{2,4}\b",letters,text)


class ExaoneLexiconMatcher:
    def __init__(self,catalog,ko_map,dense_matcher=None,jamo_top_k=250,dense_top_k=35):
        if min(jamo_top_k,dense_top_k)<1:raise ValueError("Candidate counts must be positive")
        if not {"vessel_id","vessel_name","gross_tonnage"}.issubset(catalog):
            raise ValueError("Incomplete catalog")
        if catalog.empty or not catalog.vessel_id.is_unique or catalog.vessel_id.astype(str).eq("").any():
            raise ValueError("Catalog must be nonempty with unique nonempty IDs")
        self.catalog=catalog.copy().sort_values("vessel_id",kind="stable").reset_index(drop=True)
        self.ids=self.catalog.vessel_id.astype(str).tolist()
        self.lookup={v:i for i,v in enumerate(self.ids)}
        self.names=[canonical_name(n) for n in self.catalog.vessel_name]
        self.alpha=[re.sub(r"[\W_]","",n) for n in self.names]
        self.tonnages=[number(v) for v in self.catalog.gross_tonnage]
        self.types=self.catalog.get("vessel_type",[""]*len(self.ids))
        self.flags=self.catalog.get("flag_name",[""]*len(self.ids))
        self.raw=[];self.normalized=[]
        self.exact=defaultdict(set);self.prefix=defaultdict(set);self.suffix=defaultdict(set)
        self.words=defaultdict(set);self.ngrams=defaultdict(set)
        unknown=set(ko_map)-set(self.names)
        if unknown:raise ValueError("Map contains names outside this catalog")
        for i,name in enumerate(self.names):
            variants=ko_map.get(name,[])
            if not isinstance(variants,list) or len(variants)>4 or any(not isinstance(v,str) for v in variants):
                raise ValueError("Malformed EXAONE variants")
            raw=[hangul_to_jamo(v,False) for v in variants]
            norm=[hangul_to_jamo(v,True) for v in variants]
            self.raw.append(raw);self.normalized.append(norm)
            for text in norm:
                if not text:continue
                self.exact[text].add(i);self.prefix[text[:3]].add(i);self.suffix[text[-3:]].add(i)
                for start in range(max(0,len(text)-2)):self.ngrams[text[start:start+3]].add(i)
            for v in variants:
                for word in v.split():
                    key=hangul_to_jamo(word,True)
                    if key:self.words[key[:4]].add(i)
        self.dense=dense_matcher;self.jamo_top_k=jamo_top_k;self.dense_top_k=dense_top_k

    def _jamo_similarity(self,qraw,qnorm,i):
        return max((.60*OSA.normalized_similarity(qraw,a)+.40*OSA.normalized_similarity(qnorm,b)
                    for a,b in zip(self.raw[i],self.normalized[i])),default=0.)

    def retrieve(self,row):
        name=normalize_query(row.get("vessel_name"))
        ko=bool(re.search("[가-힣]",name))
        dense_features={}
        if ko:
            qraw=hangul_to_jamo(name,False);qnorm=hangul_to_jamo(name,True)
            votes=Counter()
            for i in self.exact.get(qnorm,()):votes[i]+=1000
            for index,key in [(self.prefix,qnorm[:3]),(self.suffix,qnorm[-3:])]:
                for i in index.get(key,()):votes[i]+=2
            for word in name.split():
                for i in self.words.get(hangul_to_jamo(word,True)[:4],()):votes[i]+=2
            for gram in sorted({qnorm[i:i+3] for i in range(max(0,len(qnorm)-2))}):
                for i in self.ngrams.get(gram,()):votes[i]+=1
            shortlist=sorted(votes,key=lambda i:(-votes[i],self.ids[i]))[:self.jamo_top_k]
            jamo={i:self._jamo_similarity(qraw,qnorm,i) for i in shortlist}
            lexical=sorted(shortlist,key=lambda i:(-jamo[i],self.ids[i]))
            if self.dense is not None:
                for item in self.dense.retrieve_dense_candidates(row.get("vessel_name",""),self.dense_top_k):
                    if item["vessel_id"] in self.lookup:
                        dense_features[self.lookup[item["vessel_id"]]]=item
            union=set(lexical)|set(dense_features)
            # Retrieval rank for fixed Recall@35 is explicitly RRF, separate from final score.
            rrf={i:1/(60+r) for r,i in enumerate(lexical,1)}
            for i,item in dense_features.items():rrf[i]=rrf.get(i,0)+1/(60+item["dense_rank"])
            retrieved=sorted(union,key=lambda i:(-rrf[i],self.ids[i]))
            # All union candidates receive a Jamo score, even Dense-only candidates.
            for i in retrieved:
                if i not in jamo:jamo[i]=self._jamo_similarity(qraw,qnorm,i)
        else:
            query=re.sub(r"[\W_]","",name)
            matches=process.extract(query,self.alpha,scorer=fuzz.ratio,limit=self.jamo_top_k) if query else []
            retrieved=[i for _,_,i in matches]
            jamo={i:OSA.normalized_similarity(query,self.alpha[i]) for i in retrieved}
        if not retrieved:
            ton=number(row.get("international_gross_tonnage")) or number(row.get("gross_tonnage"))
            retrieved=sorted(range(len(self.ids)),key=lambda i:(
                abs(self.tonnages[i]-ton)/max(self.tonnages[i],ton) if ton and self.tonnages[i] else math.inf,self.ids[i]))[:min(40,len(self.ids))]
            jamo={i:0. for i in retrieved}
        return retrieved,jamo,dense_features

    def predict_details(self,rows):
        required={"sample_id","vessel_name"}
        if not required.issubset(rows):raise ValueError("Missing query columns")
        predictions=[];details=[]
        for row in rows.to_dict("records"):
            candidates,name_scores,dense=self.retrieve(row)
            qnum=digit_sequence(row.get("vessel_name",""))
            ton=number(row.get("international_gross_tonnage")) or number(row.get("gross_tonnage"))
            alt=number(row.get("gross_tonnage"))
            ranked=[]
            for i in candidates:
                cnum=digit_sequence(self.names[i])
                num=3.2 if qnum and qnum==cnum else (-4. if qnum and cnum else (-2.8 if qnum else (-1.6 if cnum else 0.)))
                t=0.
                if ton and self.tonnages[i]:
                    diff=min(abs(ton-self.tonnages[i]),abs(alt-self.tonnages[i]) if alt else math.inf)
                    rel=diff/max(1.,self.tonnages[i])
                    t=3.6 if diff<=1 else (3.1 if rel<=.005 else (2.5 if rel<=.02 else -3.5*min(2.5,rel*6)))
                # Exact same-string metadata only; no learned legacy transfer tables.
                flag=canonical_name(row.get("flag_name"));vtype=canonical_name(row.get("vessel_type"))
                meta=.25*int(bool(flag) and flag==canonical_name(self.flags[i]))
                meta+=.25*int(bool(vtype) and vtype==canonical_name(self.types[i]))
                d=dense.get(i,{})
                dense_score=2.5*d.get("dense_sim",0.)+d.get("dense_rank_recip",0.)
                ranked.append((10*name_scores.get(i,0.)+num+t+meta+dense_score,i))
            ranked.sort(key=lambda v:(-v[0],self.ids[v[1]]))
            predictions.append(self.ids[ranked[0][1]])
            details.append({"retrieval_ids":[self.ids[i] for i in candidates],
                            "ranked_ids":[self.ids[i] for _,i in ranked],
                            "scores":[float(s) for s,_ in ranked]})
        return predictions,details

    def predict(self,rows):
        return self.predict_details(rows)[0]

def main():
    """Static JSON inference only; no neural model imports or ZIP creation."""
    import argparse
    import hashlib
    import json
    import pandas as pd
    p=argparse.ArgumentParser(description="Row-independent EXAONE static-lexicon matching")
    p.add_argument("--ko-map",type=Path,required=True)
    p.add_argument("--test-csv",type=Path,required=True)
    p.add_argument("--catalog-csv",type=Path,default=ROOT/"data/vessel_catalog.csv")
    p.add_argument("--output-csv",type=Path,default=ROOT/"output/submission.csv")
    args=p.parse_args()
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream,"sha256").hexdigest()
    manifest=json.loads(args.ko_map.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_id")!="LGAI-EXAONE/EXAONE-4.0-1.2B" or manifest.get("generator")!="EXAONE-only":
        raise ValueError("Expected new EXAONE dictionary provenance")
    if not manifest.get("full_catalog"):
        raise ValueError("Runtime needs full-catalog generation, not a pilot")
    if any(manifest["config"]["legacy_policy"].values()):
        raise ValueError("Legacy dictionary policy violation")
    if manifest["catalog_map_sha256"]!=digest(args.ko_map) or manifest["config"]["catalog_sha256"]!=digest(args.catalog_csv):
        raise ValueError("Dictionary/catalog identity mismatch")
    catalog=pd.read_csv(args.catalog_csv,keep_default_na=False,dtype=str,encoding="utf-8-sig")
    rows=pd.read_csv(args.test_csv,keep_default_na=False,dtype=str,encoding="utf-8-sig")
    if not rows.sample_id.is_unique or rows.sample_id.eq("").any():
        raise ValueError("sample_id must be nonempty and unique")
    matcher=ExaoneLexiconMatcher(catalog,json.loads(args.ko_map.read_text(encoding="utf-8")))
    pred=matcher.predict(rows)
    if len(pred)!=len(rows) or not set(pred).issubset(matcher.lookup):
        raise ValueError("Invalid output predictions")
    args.output_csv.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"sample_id":rows.sample_id,"target_vessel_id":pred}).to_csv(args.output_csv,index=False,encoding="utf-8")
    print(f"Saved {len(pred)} predictions.")


if __name__=="__main__":
    main()
