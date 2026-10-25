"""Fixed-catalog character n-gram retrieval; never fit on queries."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from name_processing import name_view


class CharacterNgramIndex:
    def __init__(self, names):
        documents = [name_view(name).roman for name in names]
        self.vectorizer = TfidfVectorizer(
            analyzer='char', ngram_range=(2, 4), lowercase=False,
            sublinear_tf=True, norm='l2', dtype=np.float32)
        self.transpose = None
        if any(len(name) >= 2 for name in documents):
            # CSR transposed index is an inverted index over fixed catalog
            # n-grams. Each query performs its own sparse dot product.
            matrix = self.vectorizer.fit_transform(documents)
            self.transpose = matrix.T.tocsr()

    def search(self, name, count):
        if self.transpose is None or count <= 0:
            return []
        query = self.vectorizer.transform([name_view(name).roman])
        scores = (query @ self.transpose).tocsr()
        valid = scores.data > 0
        ids, values = scores.indices[valid], scores.data[valid]
        # Catalog is ID-sorted. Equal scores have a stable ID tie-break;
        # zero-score documents are not padded into this retrieval path.
        order = np.lexsort((ids, -values))[:count]
        return ids[order].tolist()
