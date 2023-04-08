import pickle
from operator import itemgetter

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from text_authorship.ta_model.data_preparation import Featurebuilder, get_author_vectorizer

_DEFAULT_PARAMS = {
    "tokens": {
        "max_count": 15000,
        "n_min": 1,
        "n": 4,
    },
    "lemmas": {
        "max_count": 50000,
        "n_min": 1,
        "n": 1
    }
}


class LogregModel:
    def __init__(self,
                 penalty="l2",
                 random_state=10,
                 C=603,
                 class_weight="balanced",
                 data_params=None,
                 max_iter=1000):

        if data_params is None:
            self.data_params = _DEFAULT_PARAMS
        else:
            self.data_params = data_params

        self._fitted = False
        self._label_encoder = LabelEncoder()
        vecs = dict()
        for fname, params in self.data_params.items():
            n_min, n, max_count = itemgetter("n_min", "n", "max_count")(params)
            vecs["vec_" + fname] = TfidfVectorizer(ngram_range=(n_min, n), max_features=max_count, norm='l2')
        self._data_encoder = Featurebuilder(*self.data_params.keys(), **vecs)
        self.clf = LogisticRegression(penalty=penalty,
                                      random_state=random_state,
                                      C=C,
                                      class_weight=class_weight,
                                      max_iter=max_iter,
                                      verbose=True)

    def fit(self, X, y):
        vecs = dict()
        for fname, params in self.data_params.items():
            vecs["vec_" + fname] = get_author_vectorizer(X, **params, column=fname)
        self._data_encoder = Featurebuilder(*self.data_params.keys(), **vecs)
        train_data = self._data_encoder.fit_transform(X)
        labels = self._label_encoder.fit_transform(y)
        self.clf.fit(train_data, labels)
        self._fitted = True

    def save(self, path):
        if not self._fitted:
            raise ValueError("cannot save non-trained model")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            instance = pickle.load(f)
            return instance

    def predict(self, df):
        probs_df = self.predict_proba(df).iloc[0]
        best_idx = probs_df.argmax()
        return self.probs_df[best_idx]

    def predict_proba(self, df) -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("cannot run predict on non-trained model")
        data = self._data_encoder.transform(df)
        probs = self.clf.predict_proba(data)
        labels = self._label_encoder.classes_
        output = pd.DataFrame(probs, columns=labels)
        return output
