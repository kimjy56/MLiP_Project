"""Behavioral checks for clean training, composite selection and row-local inference."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import unittest
import tempfile
from functools import wraps
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phonetic_encoder.model import CharPhoneticBiEncoder
from phonetic_encoder.retrieval import retrieval_metrics, stable_top_indices
from phonetic_encoder.hybrid_matcher import HybridDenseLexicalMatcher
from experiments.train_bi_encoder import build_optimizer
from experiments.dpr03_common import composite_score, load_fold_data
from experimental_matchers.dpr_hybrid.script import (
    calibrated_dense_similarity, tail_token_evidence, apply_sister_tail_penalty,
    rare_tail_bonus, CatalogIndex, fit_matching_model, predict_rows,
)


def workspace_temp(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        if args or kwargs:
            return fn(*args, **kwargs)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'scratch') as directory:
            return fn(Path(directory))
    return run


def test_packed_views_one_forward_and_independence():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = CharPhoneticBiEncoder().eval()
    names = ['동방 스타 3호', 'ALPHA WIND', '짧은 배', 'VERY LONG COMPANY MARINE CARRIER 27', '']
    calls = []
    handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    single = model.encode_queries(names[:1], batch_size=1)
    assert len(calls) == 1
    combined = model.encode_queries(names, batch_size=5)
    np.testing.assert_allclose(single[0], combined[0], atol=1e-6, rtol=0)
    for i, name in enumerate(names):
        np.testing.assert_allclose(model.encode_queries([name])[0], combined[i], atol=1e-6, rtol=0)
    np.testing.assert_allclose(model.encode_queries(names[::-1])[::-1], combined, atol=1e-6, rtol=0)
    assert model.encode_queries([]).shape == (0, 128)
    handle.remove()


def test_differential_groups_include_actual_conv_attribute_and_scheduler():
    model = CharPhoneticBiEncoder()
    optimizer, groups = build_optimizer(model)
    assert any('conv_block' in name for name in groups['new_modules']['parameter_names'])
    assert any('bilstm' in name for name in groups['backbone']['parameter_names'])
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.parameters() if p.requires_grad}
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 8, eta_min=5e-6)
    assert [g['lr'] for g in optimizer.param_groups] == [3e-5, 1.5e-4]
    for _ in range(8):
        optimizer.step()
        scheduler.step()
    assert all(abs(g['lr'] - 5e-6) < 1e-12 for g in optimizer.param_groups)


def test_id_rank_ties_full_mrr_and_top60():
    ids = [f'V{i:03}' for i in range(70)]
    scores = np.arange(70, 0, -1, dtype=np.float32)[None, :]
    metrics = retrieval_metrics(scores, ids, [ids[49]])
    assert metrics['recall@35'] == 0 and metrics['recall@60'] == 1
    assert abs(metrics['mrr'] - 1/50) < 1e-12 and metrics['mrr@35'] == 0
    tied = stable_top_indices(np.ones(4), 2, ['D', 'B', 'A', 'C'])
    assert tied.tolist() == [2, 1]
    # Distinct IDs with identical names must still be ranked as separate ships.
    assert retrieval_metrics(np.ones((1, 2)), ['A', 'B'], ['B'])['recall@1'] == 0


def test_composite_preserves_a_peak_across_remine_boundary():
    values = [dict(zip(['recall@35', 'recall@10', 'recall@1', 'mrr'], v)) for v in
              [(.96, .90, .70, .77), (.97, .90, .64, .73), (.98, .91, .63, .72)]]
    best = None
    for epoch, metrics in enumerate(values):
        score = composite_score(metrics)
        if best is None or score > best[1]:
            best = (epoch, score)
        # A mining refresh does not replace the running maximum.
    assert best[0] == 0
    with unittest.TestCase().assertRaises(ValueError):
        composite_score({**values[0], 'mrr': float('nan')})


@workspace_temp
def test_row_context_reuses_encoder_and_full_catalog_dot(tmp_path):
    torch.set_num_threads(2)
    model = CharPhoneticBiEncoder()
    checkpoint = tmp_path/'encoder.pt'
    model.save_checkpoint(checkpoint, as_fp16=False)
    cat = pd.DataFrame({'vessel_id': ['A','B','C'], 'vessel_name': ['FIRST WIND','FIRST STAR','OTHER STAR'],
                        'gross_tonnage': [100,200,300], 'vessel_type':['Cargo']*3,'flag_name':['X']*3})
    npy = tmp_path/'catalog.npy'
    np.save(npy, model.encode_catalog(cat.vessel_name.tolist()).astype(np.float16))
    matcher = HybridDenseLexicalMatcher(checkpoint, npy, catalog_df=cat)
    with patch.object(matcher.model, 'encode_queries', wraps=matcher.model.encode_queries) as encode, \
         patch('phonetic_encoder.hybrid_matcher.np.dot', wraps=np.dot) as dot:
        context = matcher.score_query('퍼스트 윈드', top_k=2)
        candidates = matcher.retrieve_dense_candidates('퍼스트 윈드', 2, result=context)
        features = matcher.extract_stage2_features(context, cat.vessel_id.tolist())
        assert encode.call_count == 1 and dot.call_count == 1
        assert len(candidates) == 2 and len(features) == 3
        assert sum(v['dense_rank_recip'] == 0 for v in features.values()) == 1
    with unittest.TestCase().assertRaises(ValueError):
        matcher.retrieve_dense_candidates('DIFFERENT', 2, result=context)
    query = pd.DataFrame({'sample_id':['Q'], 'vessel_name':['퍼스트 윈드'],
                          'gross_tonnage':[100], 'international_gross_tonnage':[100],
                          'vessel_type':['Cargo'], 'flag_name':['X'], 'arrival_year':[2020],
                          'call_sign':[''], 'target_vessel_id':['A']})
    lexical = fit_matching_model(query,cat,use_train_lookup=False)
    index = CatalogIndex(cat,full_ko_map={})
    with patch.object(matcher.model,'encode_queries',wraps=matcher.model.encode_queries) as encode, \
         patch('phonetic_encoder.hybrid_matcher.np.dot',wraps=np.dot) as dot:
        predicted = predict_rows(query.drop(columns='target_vessel_id'),lexical,index,
                                 use_train_lookup=False,gold_ids=['A'],dense_matcher=matcher,
                                 dense_mode='hybrid',dense_top_k=3)
        assert encode.call_count == 1 and dot.call_count == 1
    changed_gold = predict_rows(query.drop(columns='target_vessel_id'),lexical,index,
                               use_train_lookup=False,gold_ids=['C'],dense_matcher=matcher,
                               dense_mode='hybrid',dense_top_k=3)
    assert predicted[0] == changed_gold[0]


def test_calibration_tail_competition_and_rare_tail_boundaries():
    assert calibrated_dense_similarity(.35) == 0
    assert abs(calibrated_dense_similarity(.95) - 1) < 1e-12
    assert calibrated_dense_similarity(-1) == 0
    assert calibrated_dense_similarity(2) == 1
    evidence = {1: tail_token_evidence([['FLEET'],['SYMPHONY']], ['FLEET','CHAMPION']),
                2: tail_token_evidence([['FLEET'],['SYMPHONY']], ['FLEET','SYMPHONY'])}
    assert apply_sister_tail_penalty([(10.,1),(9.5,2)], evidence) == [(8.5,1),(9.5,2)]
    assert apply_sister_tail_penalty([(10.,1)], evidence) == [(10.,1)]
    assert rare_tail_bonus(['OLD','BIRD'],['NEW','BIRD'],29765,-1,29768,{'BIRD':2}) == 2.5
    assert rare_tail_bonus(['OLD','BIRD'],['NEW','BIRD'],29765,-1,29768,{'BIRD':11}) == 0
    assert rare_tail_bonus(['OLD','BIRD'],['NEW','BIRD'],0,-1,29768,{'BIRD':2}) == 0


@workspace_temp
def test_fold_validation_rejects_label_mismatch_and_group_overlap(tmp_path):
    train = pd.DataFrame({'sample_id':['Q1','Q2'], 'target_vessel_id':['A','B'], 'vessel_name':['하나','둘']})
    catalog = pd.DataFrame({'vessel_id':['A','B'], 'vessel_name':['ONE','TWO']})
    folds = pd.DataFrame({'sample_id':['Q1','Q2'], 'target_vessel_id':['A','B'], 'fold':[0,1]})
    paths = [tmp_path/n for n in ['train.csv','catalog.csv','folds.csv']]
    for frame,path in zip([train,catalog,folds],paths):frame.to_csv(path,index=False)
    _, fit, val, manifest = load_fold_data(*paths,0)
    assert set(fit.target_vessel_id).isdisjoint(val.target_vessel_id)
    folds.loc[0,'target_vessel_id']='B'
    folds.to_csv(paths[2],index=False)
    with unittest.TestCase().assertRaises(ValueError):load_fold_data(*paths,0)
    train.loc[1,'target_vessel_id']='A'
    folds.loc[:,'target_vessel_id']='A'
    train.to_csv(paths[0],index=False)
    folds.to_csv(paths[2],index=False)
    with unittest.TestCase().assertRaises(ValueError):load_fold_data(*paths,0)


def test_real_numpy_verdict_is_json_serializable():
    import json
    from experiments.evaluate_bi_encoder_oof import verify_row_independence
    class Probe:
        def encode_queries(self,names,batch_size):
            return np.full((len(names),128),1e-7 if len(names)>1 else 0,dtype=np.float32)
    verdict=verify_row_independence(Probe(),['A','B'],tolerance=1e-6)
    assert type(verdict) is bool and verdict
    assert json.dumps({'independent':verdict}) == '{"independent": true}'
