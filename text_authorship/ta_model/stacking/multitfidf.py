import numpy as np
import scipy as sp
from sklearn.base import TransformerMixin, BaseEstimator

from vectorizers import get_document_vectorizer, get_author_vectorizer


class MultiTfidf(TransformerMixin, BaseEstimator):

    def __init__(
            self,
            cols=None,
            tfidf_type='classic',
            n_min=1,
            n=2,
            max_count=10_000):
        self.cols = cols
        self.tfidf_type = tfidf_type
        self.n_min = n_min
        self.n = n
        self.max_count = max_count

    def fit(self, X, y=None):
        if self.tfidf_type == 'classic':
            get_vectorizer = get_document_vectorizer
        elif self.tfidf_type == 'class_based':
            get_vectorizer = get_author_vectorizer
        else:
            raise ValueError("Unknown vectorizer")
        self.vectorizers_ = [get_vectorizer(
                X,
                n_min=self.n_min,
                n=self.n,
                max_count=self.max_count,
                column=col
                ) for col in self.cols]

        self.dict_size_ = sum(
            [len(vec.vocabulary_) for vec in self.vectorizers_])
        return self

    def transform(self, X, y=None):
        results = []
        for vec, col in zip(self.vectorizers_, self.cols):
            results.append(vec.transform(X[col]))
        return sp.sparse.hstack(results)

    def get_feature_names_out(self, input_features=None):
        features = np.array([])
        for v in self.vectorizers_:
            features = np.append(features, v.get_feature_names_out())
        return features
