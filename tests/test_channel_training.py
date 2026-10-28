import unittest
from types import SimpleNamespace
import pandas as pd
import torch
from name_decoder.channel_training import (LanguageCycleSampler, sequence_nll,
    channel_objective, mine_name_negatives, learned_hangul_variants)

class TrainingTests(unittest.TestCase):
    def test_json_numpy_counters_and_failure_record(self):
        import json
        import tempfile
        import numpy as np
        from pathlib import Path
        from experiments.noisy_channel_common import write_json, record_failure
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "scratch") as directory:
            path = Path(directory) / "record.json"
            record = {"coverage": np.int32(7), "ratio": np.float32(0.5)}
            write_json(path, record)
            self.assertEqual(json.loads(path.read_text())["coverage"], 7)
            record_failure(path, record, RuntimeError("test"))
            self.assertEqual(json.loads(path.read_text())["status"], "failed")
            with self.assertRaises(ValueError):
                write_json(path, {"bad": np.float32(float("nan"))})

    def test_inner_confusions_exclusions_and_provenance(self):
        from name_decoder.channel_training import use_inner_confusions
        catalog=SimpleNamespace(name_by_id={'a':'SEA A','b':'SEA B','c':'SEA C','d':'SEA D'})
        fit=pd.DataFrame([{'sample_id':'fit1','target_vessel_id':'a','fold':1,'vessel_name':'sea'}])
        pairs=pd.DataFrame({'catalog_name':['SEA A','SEA B'],'observed_name':['sea','sea']})
        mined=pd.DataFrame([
            {'sample_id':'fit1','inner_fold':1,'observed_name':'sea','catalog_name':'SEA A','negative_name':'SEA B','negative_vessel_id':'b','baseline_rank':1,'baseline_score':4.},
            {'sample_id':'fit1','inner_fold':1,'observed_name':'sea','catalog_name':'SEA A','negative_name':'SEA C','negative_vessel_id':'c','baseline_rank':2,'baseline_score':3.}])
        selected,stats=use_inner_confusions(pairs,catalog,fit,mined,[['SEA D','SEA C'],['SEA C','SEA D']])
        self.assertEqual(selected[0],['SEA C','SEA D'])
        self.assertEqual(stats['pairs_with_mined_negatives'],1)
        mined.loc[0,'sample_id']='external'
        with self.assertRaises(ValueError):
            use_inner_confusions(pairs,catalog,fit,mined,[['SEA D','SEA C'],['SEA C','SEA D']])

    def test_cycle_coverage_and_seed(self):
        flags=[True]*3+[False]*7
        a=LanguageCycleSampler(flags,0.7,42)
        b=LanguageCycleSampler(flags,0.7,42)
        draws=a.draw(80)
        self.assertEqual(draws,b.draw(80))
        for flag in (True,False):
            group=[i for i in draws if flags[i]==flag]
            size=sum(v==flag for v in flags)
            self.assertEqual(len(set(group[:size])),size)
    def test_mask_and_ranking_gradient(self):
        logits=torch.randn(3,6,9,requires_grad=True)
        labels=torch.tensor([[-100,-100,2,3,4,-100]]*3)
        nll=sequence_nll(logits,labels)
        loss,_,_=channel_objective(nll,3)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[:,-1].abs().sum()),0)
        better=channel_objective(torch.tensor([0.1,2.,3.]),3)[2]
        worse=channel_objective(torch.tensor([3.,2.,0.1]),3)[2]
        self.assertLess(better,worse)
    def test_known_positive_exclusion(self):
        pairs=pd.DataFrame({'catalog_name':['SEA A','SEA B'],'observed_name':['same','same']})
        catalog=SimpleNamespace(name_by_id={'1':'SEA A','2':'SEA B','3':'SEA C','4':'SEA D'})
        negatives=mine_name_negatives(pairs,catalog,2)
        self.assertEqual(set(negatives[0]),{'SEA C','SEA D'})
    def test_learned_digits_and_ambiguity(self):
        pairs=pd.DataFrame([
            ('SEA STAR','씨 스타',True),('SEA STAR 12','씨 스타 12',True),
            ('BLUE STAR','블루 스타',True)],columns=['catalog_name','observed_name','hangul'])
        variants,n=learned_hangul_variants(pairs)
        self.assertIn(('SEA STAR','씨스타'),variants)
        self.assertFalse(any(cat.endswith('12') and not obs.endswith('12') for cat,obs in variants))
        self.assertGreater(n,0)

if __name__=='__main__':
    unittest.main()
