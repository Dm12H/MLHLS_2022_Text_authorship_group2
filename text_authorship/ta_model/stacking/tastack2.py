import numpy as np
from sklearn import clone
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline

from model_selection import books_cross_val


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
