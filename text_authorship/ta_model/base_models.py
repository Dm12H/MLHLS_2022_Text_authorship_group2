from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from text_authorship.ta_model import train_test_split, get_encoders


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


def run_logreg(df):

    df_train, df_test, y_train, y_test = train_test_split(df, share=0.7)
    data_enc, label_enc = get_encoders(df, df_train, _DEFAULT_PARAMS.keys(), _DEFAULT_PARAMS)

    x_train = data_enc.fit_transform(df_train)
    x_test = data_enc.transform(df_test)
    y_train = label_enc.transform(y_train)
    y_test = label_enc.transform(y_test)

    clf = LogisticRegression(penalty="l2",
                             random_state=10,
                             C=603,
                             class_weight="balanced",
                             max_iter=1000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = f1_score(y_test, y_pred, average="macro")
    return score

