import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

from model_selection import books_cross_val


class TAStack(ClassifierMixin, BaseEstimator):

    def __init__(self,
                 estimators: Union[List[BaseEstimator], None] = None,
                 final_estimator: Union[BaseEstimator,
                                        LogisticRegression,
                                        None] = None):
        self.estimators = estimators
        self.final_estimator = final_estimator

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> "TAStack":
        X = X.reset_index(drop=True)
        cv = books_cross_val(X)
        self.model_ = StackingClassifier(
            self.estimators,
            final_estimator=self.final_estimator,
            cv=cv)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.model_.predict_proba(X)

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(X)

    def get_n_important_features(self, n=100):
        estimators = self.model_.estimators_
        features = []
        for est in estimators:
            f = est[0].get_feature_names_out()
            max_ind = np.argpartition(np.abs(est[1].coef_), -n)[:, -n:]
            features.append(f[max_ind])
        return features
