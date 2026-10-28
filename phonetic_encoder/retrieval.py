"""Dense score ordering shared by training and evaluation.

The historical NumPy partial-sort order is part of this contract. In particular,
ties use NumPy's default argsort and zero/invalid counts keep native behavior.
Callers retain their own count limits, target lookup, denominators and schemas.
"""
import numpy as np


def top_score_indices(scores, count):
    """Return selected indices in descending score order without mutating scores."""
    indices = np.argpartition(scores, -count)[-count:]
    return indices[np.argsort(-scores[indices])]


def stable_top_indices(scores, count, vessel_ids):
    """SPEC-DPR-03 ordering: score descending, vessel ID ascending, including cutoff ties."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or len(scores) != len(vessel_ids) or not np.isfinite(scores).all():
        raise ValueError('Invalid catalog scores')
    k = min(max(int(count), 0), len(scores))
    if not k:
        return np.empty(0, dtype=np.int64)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    selected = np.flatnonzero(scores >= threshold)
    ids = np.asarray(vessel_ids, dtype=str)
    return selected[np.lexsort((ids[selected], -scores[selected]))[:k]]


def retrieval_metrics(similarities, vessel_ids, targets):
    """Exact ID-based ranks and full-catalog MRR; preserve MRR@35 separately."""
    sims = np.asarray(similarities)
    ids = np.asarray(vessel_ids, dtype=str)
    if sims.shape != (len(targets), len(ids)) or len(set(ids)) != len(ids):
        raise ValueError('Invalid retrieval dimensions or duplicate vessel IDs')
    positions = {v: i for i, v in enumerate(ids)}
    ranks = []
    for scores, target in zip(sims, targets):
        if target not in positions or not np.isfinite(scores).all():
            raise ValueError('Missing gold ID or nonfinite scores')
        gold = positions[target]
        rank = 1 + int(np.count_nonzero(scores > scores[gold]))
        rank += int(np.count_nonzero((scores == scores[gold]) & (ids < target)))
        ranks.append(rank)
    ranks = np.asarray(ranks, dtype=np.int64)
    metrics = {f'recall@{k}': float(np.mean(ranks <= k)) if len(ranks) else None for k in [1, 5, 10, 35, 60]}
    metrics.update(num_samples=len(ranks), eval_count=len(ranks),
                   mrr=float(np.mean(1. / ranks)) if len(ranks) else None,
                   **{'mrr@35': float(np.mean(np.where(ranks <= 35, 1. / ranks, 0.))) if len(ranks) else None})
    metrics['correct_counts'] = {f'recall@{k}': int(np.count_nonzero(ranks <= k)) for k in [1, 5, 10, 35, 60]}
    metrics['rank_definition'] = 'full catalog by score descending, vessel_id ascending; exact ID match'
    return metrics
