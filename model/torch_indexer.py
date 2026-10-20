"""PyTorch GPU-accelerated Character N-gram Indexer for fast candidate retrieval.

Evaluates queries independently against a static catalog matrix.
Complies with DACON 236771 rules: row-independent, offline, deterministic.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch


class TorchNgramIndexer:
    """Precomputes character n-gram TF-IDF representations as a PyTorch GPU tensor.

    Allows candidate search with ~0.1ms latency on modern NVIDIA GPUs.
    Falls back gracefully to CPU if CUDA is unavailable.
    """

    def __init__(
        self,
        strings: List[str],
        ngram_range: Tuple[int, int] = (2, 3),
        min_df: int = 2,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=False,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        clean_docs = [s if isinstance(s, str) and s else "" for s in strings]
        matrix = self.vectorizer.fit_transform(clean_docs)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.catalog_tensor = torch.from_numpy(matrix.toarray()).to(
            device=self.device, dtype=self.dtype
        )

    def search(self, query: Optional[str], count: int = 60) -> List[int]:
        """Search top-K candidates for a single query. Fully row-independent."""
        if not query or count <= 0 or not isinstance(query, str) or not query.strip():
            return []
        q_mat = self.vectorizer.transform([query])
        q_tensor = torch.from_numpy(q_mat.toarray()).to(
            device=self.device, dtype=self.dtype
        )
        scores = torch.matmul(q_tensor, self.catalog_tensor.T).squeeze(0)
        non_zero = scores > 0
        if not non_zero.any():
            return []
        k = min(count, int(non_zero.sum().item()))
        topk = torch.topk(scores, k=k)
        return topk.indices.cpu().tolist()

    def batch_search(self, queries: List[Optional[str]], count: int = 60) -> List[List[int]]:
        """Batch search top-K candidates for multiple queries while preserving row independence."""
        if not queries or count <= 0:
            return [[] for _ in queries]
        clean_queries = [q if isinstance(q, str) and q.strip() else "" for q in queries]
        q_mat = self.vectorizer.transform(clean_queries)
        q_tensor = torch.from_numpy(q_mat.toarray()).to(
            device=self.device, dtype=self.dtype
        )
        scores = torch.matmul(q_tensor, self.catalog_tensor.T)
        topk = torch.topk(scores, k=count, dim=-1)
        indices = topk.indices.cpu().numpy()
        values = topk.values.cpu().numpy()
        results: List[List[int]] = []
        for i in range(len(queries)):
            mask = values[i] > 0
            results.append(indices[i, mask].tolist())
        return results
