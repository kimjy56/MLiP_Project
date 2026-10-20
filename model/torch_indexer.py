"""Bounded-memory TF-IDF search with explicit device and stable catalog-order ties."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch


class TorchNgramIndexer:
    def __init__(self, strings, ngram_range=(2, 3), min_df=2, device=None):
        # CPU is deliberate: GPU use must be selected and validated, not auto-enabled.
        self.device = device or 'cpu'
        if self.device not in {'cpu', 'cuda'}:
            raise ValueError('device must be cpu or cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly requested but unavailable')
        self.vectorizer = TfidfVectorizer(
            analyzer='char', ngram_range=ngram_range, min_df=min_df,
            lowercase=False, sublinear_tf=True, norm='l2', dtype=np.float32)
        self.matrix = self.vectorizer.fit_transform(
            [s if isinstance(s, str) else '' for s in strings]).tocsr()
        self.catalog_tensor = None
        if self.device == 'cuda':
            coo = self.matrix.tocoo()
            self.catalog_tensor = torch.sparse_coo_tensor(
                torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)),
                torch.from_numpy(coo.data), size=coo.shape, device='cuda').coalesce()
        self.memory_info = {
            'catalog_rows': self.matrix.shape[0], 'vocabulary_size': self.matrix.shape[1],
            'csr_bytes': self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes,
            'gpu_tensor_bytes': (self.catalog_tensor.indices().numel() * 8 +
                                 self.catalog_tensor.values().numel() * 4) if self.catalog_tensor is not None else 0}

    def search(self, query, count=60):
        return self.batch_search([query], count)[0]

    def batch_search(self, queries, count=60):
        if not queries or count <= 0:
            return [[] for _ in queries]
        queries = [q if isinstance(q, str) and q.strip() else '' for q in queries]
        results = []
        # Bounded transformation and fixed single-query arithmetic preserve order independence.
        for start in range(0, len(queries), 64):
            matrix = self.vectorizer.transform(queries[start:start + 64]).tocsr()
            gpu_queries = (torch.from_numpy(matrix.toarray()).to('cuda')
                           if self.catalog_tensor is not None else None)
            for i in range(matrix.shape[0]):
                if gpu_queries is None:
                    scores = (self.matrix @ matrix.getrow(i).T).toarray().ravel()
                else:
                    with torch.inference_mode():
                        scores = torch.sparse.mm(self.catalog_tensor, gpu_queries[i:i+1].T).flatten().cpu().numpy()
                positive = np.flatnonzero(scores > 0)
                k = min(count, len(positive))
                if k and k < len(positive):
                    threshold = np.partition(scores[positive], len(positive)-k)[len(positive)-k]
                    positive = positive[scores[positive] >= threshold]
                order = np.lexsort((positive, -scores[positive]))[:k]
                results.append(positive[order].tolist())
        return results
