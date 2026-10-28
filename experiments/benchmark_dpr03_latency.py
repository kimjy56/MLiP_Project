"""Matched-checkpoint CPU latency comparison with the preserved pre-change sources."""
from pathlib import Path
import argparse
import importlib.util
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phonetic_encoder.model import CharPhoneticBiEncoder
from phonetic_encoder.hybrid_matcher import HybridDenseLexicalMatcher
from experiments.dpr03_common import write_json, sha256, now_kst, start_run, finish_run


def old_module(name, relative):
    file = ROOT/'artifacts/IMPL-DPR-SYMPHON-03/before'/relative
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(fn, names, encoder):
    for name in names[:3]:
        fn(name)
    calls = []
    handle = encoder.register_forward_hook(lambda *args: calls.append(1))
    samples = []
    for name in names:
        tick = time.perf_counter()
        fn(name)
        samples.append((time.perf_counter()-tick)*1000)
    handle.remove()
    return {'rows':len(names),'mean_ms':float(np.mean(samples)), 'median_ms':float(np.median(samples)),
            'p95_ms':float(np.percentile(samples,95)), 'encoder_forwards':len(calls),
            'encoder_forwards_per_row':len(calls)/len(names)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-id', default='BENCH-DPR03-LATENCY-01')
    p.add_argument('--checkpoint', default='artifacts/DPR-SYMPHON-03/fold_0_encoder.pt')
    p.add_argument('--catalog-embeds', default='artifacts/DPR-SYMPHON-03/catalog_embeds_fp16.npy')
    p.add_argument('--output-json', default='reports/dpr03_latency.json')
    p.add_argument('--rows', type=int, default=32)
    args = p.parse_args()
    if Path(args.output_json).exists():
        raise FileExistsError('Preserve prior latency measurement')
    out,path,record=start_run(args.experiment_id)
    started=time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=6)
        torch.set_num_threads(6)
        train = pd.read_csv(ROOT/'data/train.csv',dtype=str,keep_default_na=False)
        folds = pd.read_csv(ROOT/'eval_kit/folds_group5_seed42.csv',dtype={'sample_id':str})
        val = train.merge(folds[['sample_id','fold']],on='sample_id')
        names = val.loc[val.fold.eq(0)&val.vessel_name.str.contains('[가-힣]',regex=True),'vessel_name'].head(args.rows).tolist()
        old_model_module = old_module('phonetic_encoder._dpr03_old_model','phonetic_encoder/model.py')
        old_hybrid_module = old_module('phonetic_encoder._dpr03_old_hybrid','phonetic_encoder/hybrid_matcher.py')
        old_hybrid_module.CharPhoneticBiEncoder = old_model_module.CharPhoneticBiEncoder
        old = old_hybrid_module.HybridDenseLexicalMatcher(args.checkpoint,args.catalog_embeds,
                                                        catalog_csv=ROOT/'data/vessel_catalog.csv',device='cpu')
        new = HybridDenseLexicalMatcher(args.checkpoint,args.catalog_embeds,
                                       catalog_csv=ROOT/'data/vessel_catalog.csv',device='cpu')
        old.model.float()
        reference_embeddings = old.model.encode_queries(names)
        current_embeddings = new.model.encode_queries(names)
        max_diff = float(np.max(np.abs(reference_embeddings-current_embeddings)))
        if max_diff >= 1e-6:
            raise AssertionError(f'Packing changed embeddings by {max_diff}')
        before_encoding = measure(lambda q:old.model.encode_queries([q],batch_size=1),names,old.model.encoder)
        after_encoding = measure(lambda q:new.model.encode_queries([q],batch_size=1),names,new.model.encoder)
        before_retrieval = measure(lambda q:old.retrieve_dense_candidates(q,35),names,old.model.encoder)
        after_retrieval = measure(lambda q:new.retrieve_dense_candidates(q,60),names,new.model.encoder)
        report = {'recorded_at_kst':now_kst(),'device':'cpu','threads':6,'rows':len(names),
                  'checkpoint_sha256':sha256(args.checkpoint), 'source_parity_max_abs_diff':max_diff,
                  'encoding_before':before_encoding,'encoding_after':after_encoding,
                  'retrieval_before_top35':before_retrieval,'retrieval_after_top60':after_retrieval,
                  'encoding_mean_speedup':before_encoding['mean_ms']/after_encoding['mean_ms'],
                  'retrieval_mean_speedup':before_retrieval['mean_ms']/after_retrieval['mean_ms'],
                  'scope':'same FP32 model/checkpoint/catalog on local CPU; loading excluded; warmup3; public fold0 first Hangul rows',
                  'limitations':['Not an L4/6vCPU server reproduction; no extrapolation to the 497s server display.',
                                 'Retrieval also removes unused per-row tail-candidate re-encoding; K changes35 to60.']}
        write_json(args.output_json,report)
        record['metrics']['metric_source']='matched-checkpoint local CPU latency; no accuracy or server inference'
        record['latency']=report
        record['artifacts']=[args.output_json]
        record['cost'].update(inference_seconds=time.perf_counter()-started,timing_scope='entire matched latency experiment including loading and parity check')
        finish_run(path,record,'benchmarked_local_cpu')
        print(report,flush=True)
    except BaseException as exc:
        finish_run(path,record,'failed',exc)
        raise



if __name__ == '__main__':
    main()
