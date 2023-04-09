import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from config.config_parser import parse_config

from text_authorship.ta_model import train_test_split, get_encoders
from text_authorship.ta_model.data_preparation import get_encoder
from stacking.tastack2 import TAStack2
from stacking.tavectorizer import TAVectorizer

_FEATURE_PARAMS = "feature_params"

_LOGREG_PARAMS = "logerg_params"

_VECTORIZER_PARAMS = "vectorizer_params"


def train_logreg(df: pd.DataFrame,
                 feature_params: Optional[Dict[str, Any]] = None,
                 logreg_params: Optional[Dict[str, Any]] = None
                 ) -> Tuple[LogisticRegression, float]:
    params = parse_config()

    df_train, df_test, y_train, y_test = train_test_split(df, share=0.7)
    if feature_params is None:
        feature_params = params[_FEATURE_PARAMS]
    data_enc, label_enc = get_encoders(
        df=df,
        x=df_train,
        arg_list=feature_params.keys(),
        vectorizer_params=feature_params)

    x_train = data_enc.fit_transform(df_train)
    x_test = data_enc.transform(df_test)
    y_train = label_enc.transform(y_train)
    y_test = label_enc.transform(y_test)

    if logreg_params is None:
        logreg_params = params[_LOGREG_PARAMS]

    clf = LogisticRegression(**logreg_params)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = f1_score(y_test, y_pred, average="macro")
    return clf, score


def train_stacking(df: pd.DataFrame,
                   logreg_params: Optional[Dict[str, Any]] = None
                   ) -> Tuple[TAStack2, float]:
    params = parse_config()

    x_train, x_test, y_train, y_test = train_test_split(df)
    encoder = get_encoder(x_train)
    y_train, y_test = encoder.transform(y_train), encoder.transform(y_test)
    vectorizer = TAVectorizer(**params[_VECTORIZER_PARAMS])
    if logreg_params is None:
        logreg_params = params[_LOGREG_PARAMS]
    base_estimator = LogisticRegression(**logreg_params)
    stacking = TAStack2(
        vectorizer=vectorizer,
        base_estimator=base_estimator,
        final_estimator=XGBClassifier())
    stacking.fit(x_train, y_train)
    y_pred = stacking.predict(x_test)
    score = f1_score(y_test, y_pred, average="macro")
    return stacking, score
