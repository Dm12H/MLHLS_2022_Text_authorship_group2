import numpy as np
import pandas as pd
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from vectorizers import get_document_vectorizer, get_author_vectorizer
from .model_selection import books_cross_val

from itertools import combinations
from xgboost import XGBClassifier


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
    

class TAStack2(ClassifierMixin, BaseEstimator):

    def __init__(self,
                 vectorizer=None,
                 base_estimator=None,
                 final_estimator=None,
                 vectorized_input=False,
                 cv=None,
                 dict_sizes=None):
        self.vectorizer = vectorizer
        self.base_estimator = base_estimator
        self.final_estimator = final_estimator
        self.vectorized_input = vectorized_input
        self.cv = cv
        self.dict_sizes = dict_sizes

    def fit(self, X, y):
        if not self.vectorized_input:
            X = X.reset_index(drop=True)
            self.cv = list(books_cross_val(X))
            X = self.vectorizer.fit_transform(X)
            self.dict_sizes = self.vectorizer.dict_sizes_

        base_pipes = []

        border_idx = np.cumsum(self.dict_sizes)

        for i, idx in enumerate(border_idx):
            cols_to_keep = np.arange(
                border_idx[i - 1], idx) if i > 0 else np.arange(idx)
            column_choice = ColumnTransformer([
                ('_', 'passthrough', cols_to_keep)
            ])
            base_pipes.append(
                (
                    f'estimator_{i}',                
                    make_pipeline(column_choice, clone(self.base_estimator))
                )
            )

        self.model_ = StackingClassifier(
            base_pipes,
            final_estimator=self.final_estimator,
            cv=self.cv).fit(X, y)
        return self
    
    def predict(self, X):
        if not self.vectorized_input:
            X = self.vectorizer.transform(X)

        return self.model_.predict(X)
    

class TAStack(ClassifierMixin, BaseEstimator):
    
    def __init__(self, estimators=None, final_estimator=None):
        self.estimators = estimators
        self.final_estimator = final_estimator

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        cv = books_cross_val(X)
        self.model_ = StackingClassifier(
            self.estimators,
            final_estimator=self.final_estimator,
            cv=cv)
        self.model_.fit(X, y)
        return self
    
    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
    def predict(self, X):
        return self.model_.predict(X)
    
    def get_n_important_features(self, n=100):
        estimators = self.model_.estimators_
        features = []
        for est in estimators:
            f = est[0].get_feature_names_out()
            max_ind = np.argpartition(np.abs(est[1].coef_), -n)[:, -n:]
            features.append(f[max_ind])
        return features
    

def get_base_estimator(cols, vec_type='classic'):
    pipe = Pipeline([
        ('vectorizer', MultiTfidf(
            cols=cols,
            tfidf_type=vec_type)),
        ('model', LogisticRegression(
            class_weight='balanced',
            max_iter=500,
            C=1000))
    ])
    return pipe


def get_stacking(vec_type='classic'):
    estimators = []
    col_combinations = combinations(
        ['text_no_punkt', 'lemmas', 'tags', 'tokens'],
        2)
    for cols in col_combinations:
        estimators.append(
            (';'.join(cols),
             get_base_estimator(cols, vec_type=vec_type)))

    return TAStack(estimators, XGBClassifier())
