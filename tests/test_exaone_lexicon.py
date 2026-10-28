"""Functional tests, using synthetic fixtures/tiny random EXAONE config, not model quality."""
import ast
import builtins
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from name_decoder.deterministic_ko_rules import deterministic_variants,romanized_hangul,number_variants,hangul_to_jamo
from name_decoder.exaone_data import prepare_pairs,validate_frames
from name_decoder.exaone_lexicon import select_word_candidates,compose_word_candidates,merge_catalog_candidates
from name_decoder.exaone_translit import HangulOnlyLogitsProcessor,encode_example,collate_examples,inject_lora,adapter_state,restore_adapter
from experimental_matchers.exaone_lexicon import ExaoneLexiconMatcher
from experiments.build_exaone_ko_lexicon import generate_batches


def fixture():
    catalog=pd.DataFrame({"vessel_id":[f"V{i}" for i in range(5)],
       "vessel_name":["ALPHA ONE","BETA TWO","GAMMA THREE","DELTA FOUR","OMEGA FIVE"],
       "gross_tonnage":["100","200","300","400","500"],
       "vessel_type":["CARGO"]*5,"flag_name":["KR"]*5})
    train=pd.DataFrame({"sample_id":[f"S{i}" for i in range(5)],
       "target_vessel_id":catalog.vessel_id,"vessel_name":["외부 비밀","베타 투","감마 쓰리","델타 포","오메가 파이브"],
       "gross_tonnage":catalog.gross_tonnage,"call_sign":["C"+str(i) for i in range(5)]})
    folds=train[["sample_id","target_vessel_id"]].copy()
    folds["fold"]=range(5)
    return train,catalog,folds


class RulesAndData(unittest.TestCase):
    def test_general_spelling_and_not_company_aliases(self):
        self.assertIn("씨엔씨",deterministic_variants("CNC"))
        self.assertIn("에이치디",deterministic_variants("HD"))
        self.assertNotIn("현대",deterministic_variants("HD"))
        self.assertIn("거",deterministic_variants("KEO"))
        self.assertIn("동",deterministic_variants("TONG"))
        self.assertIn("광",deterministic_variants("KWANG"))
        self.assertIn("인천",deterministic_variants("INCHON"))
        self.assertIn("청",deterministic_variants("CHUNG"))
        self.assertIn("2",number_variants("II"))

    def test_fit_only_descendants_and_rule_pairs(self):
        train,cat,fold=fixture()
        pairs,fit,val,audit=prepare_pairs(train,cat,fold,0)
        self.assertEqual(set(val.target_vessel_id),{"V0"})
        self.assertFalse(set(fit.target_vessel_id)&set(val.target_vessel_id))
        self.assertNotIn("ALPHA ONE",set(pairs.source))
        self.assertNotIn("외부 비밀",set(pairs.target))
        self.assertIn("A",set(pairs.source))
        changed=train.copy();changed.loc[0,"vessel_name"]="정답 교체"
        after=prepare_pairs(changed,cat,fold,0)[0]
        pd.testing.assert_frame_equal(pairs,after)

    def test_group_mismatch_duplicate_and_full_train(self):
        train,cat,fold=fixture()
        wrong=fold.copy();wrong.loc[0,"target_vessel_id"]="V1"
        with self.assertRaises(ValueError):validate_frames(train,cat,wrong)
        duplicate=pd.concat([fold,fold.iloc[[0]]])
        with self.assertRaises(ValueError):validate_frames(train,cat,duplicate)
        pairs,fit,val,_=prepare_pairs(train,cat,fold,0,True)
        self.assertEqual(len(val),0);self.assertEqual(len(fit),5)
        self.assertIn("ALPHA ONE",set(pairs.source))

    def test_bounded_composition_and_diversity(self):
        lex={"ALPHA":["알파","알퍼"],"BETA":["베타","비타"]}
        candidates=compose_word_candidates("ALPHA BETA",lex,4)
        self.assertEqual(set(candidates),{"알파 베타","알파 비타","알퍼 베타","알퍼 비타"})
        merged=merge_catalog_candidates("ALPHA BETA",[{"text":"알파 베타"},{"text":"알퍼 베타"}],lex,4)
        self.assertEqual(len(merged),4)
        self.assertIn("알파 비타",merged)
        self.assertEqual(select_word_candidates("CNC",[{"text":"ABC"}]),["씨엔씨"])
        self.assertEqual(merge_catalog_candidates("ALPHA 3",[{"text":"알파 4"}],{}),[])


class FakeTokenizer:
    eos_token_id=1;pad_token_id=0;all_special_ids=[0,1]
    def __len__(self):return 10
    def decode(self,ids,**kwargs):
        return ["[PAD]","[EOS]","가"," 나","A","가A","\n","\ufffd"," ","12"][ids[0]]
    def encode(self,text,**kwargs):return [2]*len(text)


class DecoderContracts(unittest.TestCase):
    def test_hangul_mask_and_eos(self):
        processor=HangulOnlyLogitsProcessor(FakeTokenizer(),2)
        scores=torch.zeros((1,10))
        first=processor(torch.tensor([[4,4]]),scores)
        for i in [0,1,4,5,6,7,8]:self.assertTrue(torch.isneginf(first[0,i]))
        for i in [2,3,9]:self.assertEqual(first[0,i],0)
        later=processor(torch.tensor([[4,4,2]]),scores)
        self.assertEqual(later[0,1],0)
        self.assertTrue(torch.isneginf(later[0,4]))

    def test_supervised_mask_and_padding(self):
        tok=FakeTokenizer()
        a=encode_example(tok,"A","가",64)
        b=encode_example(tok,"ALPHA","가 나",64)
        self.assertEqual(a["labels"][-2:],[2,1])
        self.assertTrue(all(v==-100 for v in a["labels"][:-2]))
        batch=collate_examples([a,b],0)
        self.assertTrue(torch.all(batch["labels"][0,len(a["input_ids"]):]==-100))
        self.assertEqual(int(batch["attention_mask"][0].sum()),len(a["input_ids"]))
        with self.assertRaises(ValueError):encode_example(tok,"A","English",64)

    def test_tiny_exaone_lora_gradient_and_reload(self):
        from transformers import Exaone4Config,Exaone4ForCausalLM
        torch.set_num_threads(2);torch.manual_seed(42)
        config=Exaone4Config(vocab_size=32,hidden_size=32,intermediate_size=64,
                            num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
                            head_dim=8,max_position_embeddings=128)
        config._attn_implementation="eager"
        model=Exaone4ForCausalLM(config)
        spec=inject_lora(model,rank=2,alpha=4,dropout=0.)
        self.assertEqual(len(spec["module_names"]),7)
        ids=torch.randint(2,32,(2,8))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        loss=model(input_ids=ids,labels=ids,use_cache=False).loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters() if ".lora_" not in n))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.endswith("lora_B")))
        optimizer.step();model.eval()
        state=adapter_state(model)
        with torch.no_grad():expected=model(input_ids=ids,use_cache=False).logits.clone()
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n.endswith("lora_B"):p.zero_()
        restore_adapter(model,state)
        with torch.no_grad():actual=model(input_ids=ids,use_cache=False).logits
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


class StaticMatcherContracts(unittest.TestCase):
    def test_no_legacy_files_labels_and_row_independence(self):
        train,cat,_=fixture()
        ko_map={name:[f"가 {i}"] for i,name in enumerate(cat.vessel_name)}
        original_open=builtins.open
        def guard(file,*args,**kwargs):
            if any(t in str(file) for t in ["catalog_full_ko_map","IDEA_B_WORD_VARS"]):
                raise AssertionError("Legacy asset read")
            return original_open(file,*args,**kwargs)
        with patch("builtins.open",guard):
            matcher=ExaoneLexiconMatcher(cat,ko_map)
            rows=train.drop(columns=["target_vessel_id"]).copy()
            rows["vessel_name"]=[f"가 {i}" for i in range(5)]
            pred,details=matcher.predict_details(rows)
            self.assertEqual(pred,cat.vessel_id.tolist())
            self.assertEqual(matcher.predict(rows.iloc[::-1])[::-1],pred)
            self.assertEqual(matcher.predict(rows.iloc[:2])+matcher.predict(rows.iloc[2:]),pred)
            self.assertEqual([matcher.predict(rows.iloc[[i]])[0] for i in range(5)],pred)
            self.assertEqual(matcher.predict(rows.iloc[:0]),[])
            rows["target_vessel_id"]="WRONG"
            self.assertEqual(matcher.predict(rows),pred)
            self.assertTrue(all(len(d["retrieval_ids"])<=250 for d in details))

    def test_runtime_import_is_llm_free_and_legacy_api_lazy(self):
        code="import sys; from experimental_matchers.exaone_lexicon import ExaoneLexiconMatcher; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; from name_decoder import CatalogNames; assert CatalogNames.__module__=='name_decoder.catalog'"
        subprocess.run([sys.executable,"-B","-c",code],cwd=ROOT,check=True,capture_output=True)

    def test_generation_cache_resume_rejects_changed_inputs(self):
        class Generator:
            calls=0
            def generate(self,names,**kwargs):
                self.calls+=1
                return [[{"text":"가","score":0}] for _ in names]
        base=(ROOT/"scratch/exaone-tests").resolve()
        assert base.is_relative_to(ROOT.resolve());base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            cache=Path(directory)/"batches"
            gen=Generator()
            a=generate_batches(gen,["ALPHA","BETA"],cache,{"seed":42},2,3,12)
            b=generate_batches(gen,["ALPHA","BETA"],cache,{"seed":42},2,3,12)
            self.assertEqual(a,b);self.assertEqual(gen.calls,1)
            with self.assertRaises(ValueError):
                generate_batches(gen,["GAMMA","BETA"],cache,{"seed":42},2,3,12)


class ProvenanceContracts(unittest.TestCase):
    def test_fold_map_manifest_and_tamper_rejection(self):
        from experiments.evaluate_pure_exaone_oof import checked_map
        from experiments.exaone_common import MODEL_ID,sha256,object_sha256,write_json
        train,cat,folds=fixture()
        base=(ROOT/"scratch/exaone-manifest-tests").resolve()
        assert base.is_relative_to(ROOT.resolve());base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            data=Path(directory)
            train.to_csv(data/"train.csv",index=False)
            cat.to_csv(data/"vessel_catalog.csv",index=False)
            folds.to_csv(data/"folds.csv",index=False)
            hashes={k:sha256(data/name) for k,name in [
                ("train.csv","train.csv"),("vessel_catalog.csv","vessel_catalog.csv"),("folds","folds.csv")]}
            path=data/"catalog_exaone_ko_map.json"
            write_json(path,{n:["가"] for n in cat.vessel_name})
            manifest={"generator":"EXAONE-only","model_id":MODEL_ID,"catalog_map_sha256":sha256(path),
                "full_catalog":True,"config":{"legacy_policy":{"manual_IDEA_B":False,"old_catalog_map":False}},
                "training_data":{"split_id":"G5-s42 fold0","source_files_sha256":hashes,
                    "fit_vessel_ids_sha256":object_sha256(["V1","V2","V3","V4"]),
                    "heldout_vessel_ids_sha256":object_sha256(["V0"])}}
            sidecar=path.with_suffix(".manifest.json")
            write_json(sidecar,manifest)
            result,_=checked_map(path,train,cat,folds,0,data,data/"folds.csv")
            self.assertEqual(len(result),5)
            with self.assertRaises(ValueError):checked_map(path,train,cat,folds,1,data,data/"folds.csv")
            write_json(path,{"ALPHA ONE":["다름"]})
            with self.assertRaises(ValueError):checked_map(path,train,cat,folds,0,data,data/"folds.csv")

    def test_dense_only_candidates_still_get_lexicon_score(self):
        train,cat,_=fixture()
        mapping={n:["아주 다른 이름"] for n in cat.vessel_name}
        mapping["OMEGA FIVE"]=["가나다"]
        class Dense:
            def retrieve_dense_candidates(self,*a,**kw):
                return [{"vessel_id":"V4","dense_rank":1,"dense_sim":.8,"dense_rank_recip":1.}]
        matcher=ExaoneLexiconMatcher(cat,mapping,Dense(),jamo_top_k=1)
        ids,scores,dense=matcher.retrieve({"vessel_name":"가나다"})
        self.assertIn(4,ids)
        self.assertEqual(scores[4],1.)
        self.assertIn(4,dense)

    def test_generation_groups_and_truncation_rejection(self):
        from name_decoder.exaone_translit import ExaoneTransliterator
        class Tokenizer(FakeTokenizer):
            def __call__(self,texts,**kwargs):
                return {"input_ids":torch.tensor([[4,4]]*len(texts)),
                        "attention_mask":torch.ones((len(texts),2),dtype=torch.long)}
            def batch_decode(self,sequences,**kwargs):
                return ["가" for _ in sequences]
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.parameter=torch.nn.Parameter(torch.ones(1))
            def generate(self,input_ids,**kwargs):
                from types import SimpleNamespace
                n=kwargs["num_return_sequences"]
                repeated=input_ids.repeat_interleave(n,0)
                tails=torch.tensor([[2,1]]*len(repeated))
                return SimpleNamespace(sequences=torch.cat([repeated,tails],1))
        decoder=ExaoneTransliterator(Model(),Tokenizer())
        results=decoder.generate(["ALPHA","BETA"],num_beams=4,num_return_sequences=3)
        self.assertEqual(len(results),2)
        self.assertEqual([r[0]["text"] for r in results],["가","가"])
        self.assertEqual([len(r) for r in results],[1,1])


class RuntimeCLIContracts(unittest.TestCase):
    def test_static_entrypoint_schema_and_input_order(self):
        from experiments.exaone_common import MODEL_ID,sha256,write_json
        train,cat,_=fixture()
        base=(ROOT/"scratch/exaone-runtime-tests").resolve()
        assert base.is_relative_to(ROOT.resolve());base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            folder=Path(directory)
            catalog=folder/"catalog.csv";query=folder/"query.csv";mapping=folder/"catalog_exaone_ko_map.json"
            output=folder/"result.csv"
            cat.to_csv(catalog,index=False)
            rows=train.drop(columns="target_vessel_id").iloc[::-1].copy()
            rows["vessel_name"]=[f"가 {i}" for i in range(4,-1,-1)]
            rows.to_csv(query,index=False)
            write_json(mapping,{n:[f"가 {i}"] for i,n in enumerate(cat.vessel_name)})
            write_json(mapping.with_suffix(".manifest.json"),{
                "model_id":MODEL_ID,"generator":"EXAONE-only","full_catalog":True,
                "catalog_map_sha256":sha256(mapping),
                "config":{"catalog_sha256":sha256(catalog),"legacy_policy":{"manual_IDEA_B":False,"old_catalog_map":False}}})
            command=[sys.executable,"-B","-X","utf8",str(ROOT/"experimental_matchers/exaone_lexicon.py"),
                     "--ko-map",str(mapping),"--test-csv",str(query),"--catalog-csv",str(catalog),"--output-csv",str(output)]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.strip(),"Saved 5 predictions.")
            saved=pd.read_csv(output,dtype=str)
            self.assertEqual(saved.columns.tolist(),["sample_id","target_vessel_id"])
            self.assertEqual(saved.sample_id.tolist(),rows.sample_id.tolist())
            self.assertEqual(saved.target_vessel_id.tolist(),cat.vessel_id.iloc[::-1].tolist())


if __name__=="__main__":
    unittest.main()
