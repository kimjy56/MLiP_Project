"""
Stage-1 Recall & Row-Independence Evaluation Script (SPEC-DPR-01 Section 6).

Evaluates:
  1. Standalone Dense Phonetic Retrieval on Unseen Korean ship names (Ablation: Manual Lexicon OFF)
  2. Standalone Recall@1, Recall@5, Recall@10, Recall@35 and MRR
  3. Row-Independence check (single-sample vs batch numerical invariance)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phonetic_encoder.model import CharPhoneticBiEncoder
from phonetic_encoder.retrieval import retrieval_metrics
from experiments.dpr03_common import start_run, finish_run, load_fold_data, write_json, sha256, object_hash, now_kst


def evaluate_dense_retrieval(
    bi_encoder: CharPhoneticBiEncoder,
    catalog_embeds: np.ndarray,
    catalog_ids: List[str],
    eval_df: pd.DataFrame,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Evaluates Top-K retrieval accuracy against catalog.
    """
    queries = eval_df["vessel_name"].fillna("").astype(str).tolist()
    target_ids = eval_df["target_vessel_id"].tolist()

    t0 = time.time()
    query_embeds = bi_encoder.encode_queries(queries, batch_size=batch_size)
    encode_time = time.time() - t0

    # Fast Matrix dot product: (N_query, N_catalog)
    t1 = time.time()
    sim_matrix = np.dot(query_embeds, catalog_embeds.T)
    dot_time = time.time() - t1

    metrics = retrieval_metrics(sim_matrix, catalog_ids, target_ids)
    metrics.update(encode_time_sec=encode_time, dot_time_sec=dot_time,
                   latency_per_row_ms=(encode_time+dot_time)/max(1,len(queries))*1000,
                   timing_scope='batched throughput per row; not single-row latency')
    return metrics



def verify_row_independence(
    bi_encoder: CharPhoneticBiEncoder,
    test_queries: List[str],
    tolerance: float = 1e-6,
) -> bool:
    """
    Verifies that single row prediction outputs match batch prediction outputs within float tolerance.
    """
    if not test_queries:
        return True

    # 1. Batch encode
    batch_embs = bi_encoder.encode_queries(test_queries, batch_size=len(test_queries))

    # 2. Single encode each row
    max_diff = 0.0
    for i, q in enumerate(test_queries):
        single_emb = bi_encoder.encode_queries([q], batch_size=1)
        diff = np.max(np.abs(batch_embs[i] - single_emb[0]))
        if diff > max_diff:
            max_diff = diff

    print(f"Row independence test: max abs diff = {max_diff:.2e} (tolerance = {tolerance})")
    return bool(max_diff < tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id', default='DPR-SYMPHON-03', help='Training ID; evaluation record uses EVAL- prefix')
    parser.add_argument('--evaluation-id')
    parser.add_argument('--checkpoint',default='artifacts/DPR-SYMPHON-03/fold_0_encoder.pt')
    parser.add_argument('--catalog-embeds',default='artifacts/DPR-SYMPHON-03/catalog_embeds_fp16.npy')
    parser.add_argument('--catalog-csv',default='data/vessel_catalog.csv')
    parser.add_argument('--train-csv',default='data/train.csv')
    parser.add_argument('--folds-csv',default='eval_kit/folds_group5_seed42.csv')
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--output-json',default='reports/dpr_symphon_03_evaluation.json')
    parser.add_argument('--device',default='auto')
    args=parser.parse_args()
    if Path(args.output_json).exists():raise FileExistsError('Existing report; use new evaluation ID and output')
    out,path,record=start_run(args.evaluation_id or 'EVAL-'+args.experiment_id)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=6)
        torch.set_num_threads(6)
        device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
        model=CharPhoneticBiEncoder.load_checkpoint(args.checkpoint,device=device).float().eval()
        catalog,fit,val,data=load_fold_data(args.train_csv,args.catalog_csv,args.folds_csv,args.fold)
        ids=catalog.vessel_id.tolist()
        metadata=json.loads(Path(args.catalog_embeds).with_suffix('.json').read_text(encoding='utf-8'))
        expected={'checkpoint_sha256':sha256(args.checkpoint),'embeddings_sha256':sha256(args.catalog_embeds),
                  'catalog_sha256':sha256(args.catalog_csv),'catalog_ids_sha256':object_hash(ids),
                  'folds_sha256':data['split_file_sha256'],'fit_vessel_ids_sha256':data['fit_vessel_ids_sha256'],
                  'heldout_fold':args.fold}
        if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError('Embedding/checkpoint/fold provenance mismatch')
        embeddings=np.load(args.catalog_embeds).astype(np.float32)
        mask=val.vessel_name.str.contains('[가-힣]',regex=True)
        kr=evaluate_dense_retrieval(model,embeddings,ids,val.loc[mask])
        en=evaluate_dense_retrieval(model,embeddings,ids,val.loc[~mask])
        independent=verify_row_independence(model,val.vessel_name.head(16).tolist(),tolerance=1e-6)
        if not independent:raise AssertionError('Embedding batch invariance exceeds 1e-6')
        report={'experiment_id':args.experiment_id,'evaluation_id':record['experiment_id'],
                'heldout_fold':args.fold,'evaluated_at_kst':now_kst(),
                'checkpoint':args.checkpoint,'checkpoint_sha256':sha256(args.checkpoint),
                'korean_metrics':kr,'english_metrics':en,
                'weighted_recall@35':.7*kr['recall@35']+.3*en['recall@35'],
                'weighted_recall@60':.7*kr['recall@60']+.3*en['recall@60'],
                'row_independence_verified':independent,'embedding_tolerance':1e-6,
                'scope':'external fold0 used for epoch selection; development-selected retrieval, not official server',
                'metric_change':'full-catalog vessel-ID ranking; full MRR and MRR@35 separate; historical selection deduplicated names',
                'quantization':'FP32 checkpoint, exported catalog embeddings rounded to FP16'}
        write_json(args.output_json,report)
        record.update(data=data,retrieval=report,artifacts=[args.output_json])
        record['metrics'].update(metric_source='development-selected standalone retrieval',hangul_rows=len(val.loc[mask]),
                                  non_hangul_rows=len(val.loc[~mask]),candidate_recall_hangul=kr['recall@60'],
                                  candidate_recall_non_hangul=en['recall@60'])
        finish_run(path,record,'evaluated_development_selected')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        finish_run(path,record,'failed',exc)
        raise


if __name__=='__main__':
    main()
