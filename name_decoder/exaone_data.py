"""Fold-isolated EXAONE full-name/word training pairs from official inputs only."""
from __future__ import annotations
import re
import pandas as pd
from .deterministic_ko_rules import canonical_name,observed_korean,word_tokens,rule_training_pairs


def read_official(data_dir,folds_file):
    train=pd.read_csv(data_dir/"train.csv",keep_default_na=False,dtype=str,encoding="utf-8-sig")
    catalog=pd.read_csv(data_dir/"vessel_catalog.csv",keep_default_na=False,dtype=str,encoding="utf-8-sig")
    folds=pd.read_csv(folds_file,keep_default_na=False,dtype={"sample_id":str,"target_vessel_id":str},encoding="utf-8-sig")
    return train,catalog,folds


def validate_frames(train,catalog,folds):
    for frame,required in [(train,{"sample_id","target_vessel_id","vessel_name"}),
                           (catalog,{"vessel_id","vessel_name"}),
                           (folds,{"sample_id","target_vessel_id","fold"})]:
        if not required.issubset(frame):raise ValueError(f"Missing columns: {sorted(required-set(frame))}")
    for frame,column in [(train,"sample_id"),(catalog,"vessel_id"),(folds,"sample_id")]:
        if frame[column].eq("").any() or not frame[column].is_unique:
            raise ValueError(f"{column} must be nonempty and unique")
    if set(train.sample_id)!=set(folds.sample_id):raise ValueError("Fold IDs must cover train exactly")
    folds=folds.copy()
    numeric=pd.to_numeric(folds["fold"],errors="raise")
    if not numeric.isin(range(5)).all():raise ValueError("Fold values must be integers0..4")
    folds["fold"]=numeric.astype(int)
    merged=train.merge(folds,on="sample_id",suffixes=("","_fold"),validate="one_to_one")
    if not merged.target_vessel_id.eq(merged.target_vessel_id_fold).all():
        raise ValueError("Fold target IDs mismatch")
    if merged.groupby("target_vessel_id").fold.nunique().max()!=1:
        raise ValueError("A vessel spans outer folds")
    if not set(train.target_vessel_id).issubset(set(catalog.vessel_id)):
        raise ValueError("Unknown training target")
    return merged


def prepare_pairs(train,catalog,folds,heldout_fold=0,final_train=False,include_rules=True):
    merged=validate_frames(train,catalog,folds)
    if not final_train and heldout_fold not in range(5):raise ValueError("heldout_fold must be0..4")
    fit=merged.copy() if final_train else merged.loc[merged.fold.ne(heldout_fold)].copy()
    val=merged.iloc[:0].copy() if final_train else merged.loc[merged.fold.eq(heldout_fold)].copy()
    names={str(r.vessel_id):canonical_name(r.vessel_name) for r in catalog.itertuples(index=False)}
    pairs=set()
    aligned_rejected=0
    for row in fit.sort_values("sample_id").itertuples(index=False):
        target=observed_korean(row.vessel_name)
        if not target or not re.search("[가-힣]",target):continue
        english=names[row.target_vessel_id]
        pairs.add((english,target,"full_name"))
        en=word_tokens(english)
        ko=target.split()
        if len(en)!=len(ko):
            aligned_rejected+=1
            continue
        # Equal token counts are only a conservative alignment heuristic.
        # Restrict each aligned output to valid Korean/numeric text.
        for source,answer in zip(en,ko):
            if observed_korean(answer):
                pairs.add((source,answer,"aligned_word"))
    if include_rules:
        pairs.update((source,target,"general_rule") for source,target in rule_training_pairs())
    rows=[{"source":s,"target":t,"kind":k} for s,t,k in sorted(pairs)]
    # Identical source/target pairs from several sources teach only once.
    pairs_df=pd.DataFrame(rows,columns=["source","target","kind"]).drop_duplicates(["source","target"]).reset_index(drop=True)
    return pairs_df,fit,val,{"aligned_rows_rejected":aligned_rejected,
                            "fit_vessel_ids":sorted(set(fit.target_vessel_id)),
                            "heldout_vessel_ids":sorted(set(val.target_vessel_id)),
                            "selection":"all official fit rows; no historical dictionaries",
                            "alignment_caveat":"Equal token counts do not prove semantic alignment"}


def catalog_names(catalog):
    names=[canonical_name(n) for n in catalog["vessel_name"]]
    if any(not n for n in names):raise ValueError("Empty catalog name")
    return sorted(set(names))


def catalog_words(catalog):
    return sorted({word for name in catalog_names(catalog) for word in word_tokens(name)})
