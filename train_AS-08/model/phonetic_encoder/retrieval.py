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
