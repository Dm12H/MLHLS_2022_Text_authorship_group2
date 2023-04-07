from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from stacking.multitfidf import MultiTfidf
from stacking.tastack import TAStack

from itertools import combinations
from xgboost import XGBClassifier


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
