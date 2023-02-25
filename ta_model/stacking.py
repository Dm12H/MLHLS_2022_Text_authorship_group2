import numpy as np
import pandas as pd
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from data_preparation import get_author_vectorizer, get_document_vectorizer
from model_selection import books_cross_val



class MultiTfidf(TransformerMixin):

    def __init__(self, cols=None, tfidf_type='classic', n_min=1, n=2, max_count=10_000):
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
        return self
    
    def transform(self, X, y=None):
        results = []
        for vec, col in zip(self.vectorizers_, self.cols):
            results.append(vec.transform(X[col]))
        return sp.hstack(results)
    
    def get_feature_names_out(self, input_features=None):
        features = np.array([])
        for v in self.vectorizers_:
            features = np.append(features, v.get_feature_names_out())
        return features
    

class TAStack(ClassifierMixin, BaseEstimator):
    
    def __init__(self, estimators=None, final_estimator=None):
        self.estimators = estimators
        self.final_estimator = final_estimator
    
    def fit(self, X, y):
        X = X.reset_index(drop=True)
        cv = books_cross_val(X)
        self.model_ = StackingClassifier(self.estimators, self.final_estimator, cv=cv)
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