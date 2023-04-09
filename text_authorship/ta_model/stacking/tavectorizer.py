from itertools import combinations

from scipy import sparse
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from stacking.multitfidf import MultiTfidf
from typing import List, Union


class TAVectorizer(TransformerMixin, BaseEstimator):

    def __init__(
            self,
            cols: Union[str, None] = None,
            k: int = 2,
            tfidf_type: str = 'classic',
            n_min: int = 1,
            n: int = 2,
            max_count: int = 10_000):
        self.cols = cols
        self.k = k
        self.tfidf_type = tfidf_type
        self.n_min = n_min
        self.n = n
        self.max_count = max_count

    def fit(self, X: pd.DataFrame, y=None) -> "TAVectorizer":
        self.vectorizers_: List[MultiTfidf] = []

        for comb in combinations(self.cols, self.k):
            self.vectorizers_.append(MultiTfidf(
                cols=comb,
                tfidf_type=self.tfidf_type,
                n_min=self.n_min,
                n=self.n,
                max_count=self.max_count
            ).fit(X))

        self.dict_sizes_: List[int] = []

        for vec in self.vectorizers_:
            self.dict_sizes_.append(vec.dict_size_)

        return self

    def transform(self, X: pd.DataFrame, y=None) -> sparse.spmatrix:
        results: List[sparse.spmatrix] = []

        for vec in self.vectorizers_:
            results.append(vec.transform(X))
        return sparse.hstack(results, format="csr")
