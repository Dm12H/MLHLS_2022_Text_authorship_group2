from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from text_authorship.ta_model import train_test_split, get_encoders
from text_authorship.ta_model.logreg import LogregModel
from text_authorship.ta_model.data_preparation import get_encoder
from text_authorship.ta_model.stacking import TAStack2, TAVectorizer, TASTack2Deploy
from xgboost import XGBClassifier


def train_test_logreg(df):
    df_train, df_test, y_train, y_test = train_test_split(df, share=0.7)
    clf = LogregModel()
    clf.fit(df_train, y_train)
    y_pred = clf.predict(df_test)
    score = f1_score(y_test, y_pred, average="macro")
    return clf, score


def train_logreg(df, target_col="author"):
    y_labels = df.pop(target_col)
    clf = LogregModel()
    clf.fit(df, y_labels)
    return clf


def train_test_stacking(df):
    X_train, X_test, y_train, y_test = train_test_split(df)
    encoder = get_encoder(X_train)
    y_train, y_test = encoder.transform(y_train), encoder.transform(y_test)
    vectorizer = TAVectorizer(cols=['text_no_punkt', 'lemmas', 'tags', 'tokens'])
    base_estimator = LogisticRegression(class_weight='balanced', max_iter=500, C=1000)
    stacking = TAStack2(vectorizer=vectorizer, base_estimator=base_estimator, final_estimator=XGBClassifier())
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    score = f1_score(y_test, y_pred, average="macro")
    return stacking, score


def train_stacking(df, target_col="author"):
    target = df[target_col]

    vectorizer = TAVectorizer(cols=['text_no_punkt', 'lemmas', 'tags', 'tokens'])
    base_estimator = LogisticRegression(class_weight='balanced', max_iter=500, C=1000)
    stacking = TASTack2Deploy(vectorizer=vectorizer, base_estimator=base_estimator, final_estimator=XGBClassifier())
    stacking.fit(df, target)
    return stacking