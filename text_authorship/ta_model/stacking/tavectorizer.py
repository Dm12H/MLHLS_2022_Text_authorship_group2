from itertools import combinations

import scipy as sp
from sklearn.base import TransformerMixin, BaseEstimator

from stacking.multitfidf import MultiTfidf


class TAVectorizer(TransformerMixin, BaseEstimator):

    def __init__(
            self,
            cols=None,
            k=2,
            tfidf_type='classic',
            n_min=1,
            n=2,
            max_count=10_000):
        self.cols = cols
        self.k = k
        self.tfidf_type = tfidf_type
        self.n_min = n_min
        self.n = n
        self.max_count = max_count

    def fit(self, X, y=None):
        self.vectorizers_ = []

        for comb in combinations(self.cols, self.k):
            self.vectorizers_.append(MultiTfidf(
                cols=comb,
                tfidf_type=self.tfidf_type,
                n_min=self.n_min,
                n=self.n,
                max_count=self.max_count
            ).fit(X))

        self.dict_sizes_ = []

        for vec in self.vectorizers_:
            self.dict_sizes_.append(vec.dict_size_)

        return self

    def transform(self, X, y=None):
        results = []

        for vec in self.vectorizers_:
            results.append(vec.transform(X))
        return sp.sparse.hstack(results, format="csr")
