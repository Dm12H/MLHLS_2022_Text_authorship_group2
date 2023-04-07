import numpy as np
import optuna
import pandas as pd
from .data_preparation import get_author_vectorizer
from text_authorship.ta_model import FeatureBuilder
from .data_preparation import get_encoder
from sklearn.metrics import f1_score


def select_sample(dataframe, size=0.1):
    """
    сэмпл датасета, сбалансированный на вероятности классов
    :param dataframe: датафрейм для анализа
    :param size: доля общей выборки
    """
    df_size = len(dataframe)
    idx = np.random.choice(df_size, size=int(df_size * size), p=dataframe.probs)
    return dataframe.iloc[idx]


def _split_books_to_target(books, target):
    label_train_count = 0
    label_total_count = np.sum(books["counts"])
    i = 0
    for _, row in books.iloc[:-1].iterrows():
        num_segments = row.counts
        if not label_train_count:
            i += 1
            label_train_count += num_segments
            continue
        new_train_count = label_train_count + num_segments
        new_share = new_train_count / label_total_count
        old_share = label_train_count / label_total_count
        if abs(target - new_share) > abs(target - old_share):
            break
        i += 1
        label_train_count += num_segments
    return i, label_train_count, label_total_count


def _shuffle_books(df, author, random_gen, cross_val=True):
    df_slice = df[df.author == author][["book", "counts"]]
    unique_books = df_slice.drop_duplicates("book")
    if len(unique_books) < 2 and not cross_val:
        raise ValueError(f"too few books of author: {author}")
    permunation = random_gen.permutation(len(unique_books))
    shuffled_books = unique_books.iloc[permunation]
    return shuffled_books


def train_test_split(df, share=0.5, seed=10, cross_val=False):
    """
        разделяет данные на обучающую и тестовую выборки по книгам
    :param df: датафрейм для анализа
    :param share: процент данных для трейна ( тест, соответственно 1-share)
    :param seed: сид для случайных перемешиваний
    :param cross_val: флаг для использования в кросс-валидации (более жесткие требования к делению)
    :return:
    """
    rg = np.random.default_rng(seed)
    labels = df.author.unique()
    shuffled_labels = labels[rg.permutation(len(labels))]

    train_label = []
    test_label = []
    train_counts = 0
    test_counts = 0

    for label in shuffled_labels:
        shuffled_books = _shuffle_books(df=df, author=label, random_gen=rg, cross_val=cross_val)
        split_idx, segm_train, total_semg = _split_books_to_target(shuffled_books, share)

        train_label.extend(shuffled_books.iloc[:split_idx].book)
        test_label.extend(shuffled_books.iloc[split_idx:].book)
        train_counts += segm_train
        test_counts += total_semg - segm_train

    train_df = df[df["book"].isin(set(train_label))]
    test_df = df[df["book"].isin(set(test_label))]
    assert len(train_df) + len(test_df) == len(df), \
        "split is wrong"
    y_train = train_df["author"]
    y_test = test_df["author"]
    return train_df, test_df, y_train, y_test


def train_crossval_twofold(frame, clf, *args, split=0.5, vectorizer_dict=None, avg="micro"):
    x_split1, x_split2, y_split1, y_split2 = train_test_split(frame, share=split)
    pair1 = x_split1, y_split1
    pair2 = x_split2, y_split2
    scores = []
    for train_pair, test_pair in (pair1, pair2):
        df_train, y_train = train_pair
        df_test, y_test = test_pair

        if vectorizer_dict is None:
            raise ValueError("not using any vectorizer!")
        vecs = dict()
        for fname, params in vectorizer_dict.items():
            vecs["vec_" + fname] = get_author_vectorizer(df_train, **params, column=fname)

        data_encoder = FeatureBuilder(*args, **vecs)
        x_train = data_encoder.fit_transform(df_train)
        x_test = data_encoder.fit_transform(df_test)

        label_encoder = get_encoder(frame=frame)
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

        clf.fit(x_train, y_train)
        score = f1_score(clf.predict(x_test), y_test, average=avg)
        scores.append(score)
    return scores


def get_encoders(df, x, arg_list, vectorizer_params):
    if vectorizer_params is None:
        raise ValueError("not using any vectorizer!")
    vecs = dict()
    for fname, params in vectorizer_params.items():
        vecs[f"vec_{fname}"] = get_author_vectorizer(x, **params, column=fname)
    data_encoder = FeatureBuilder(*arg_list, **vecs)
    label_encoder = get_encoder(frame=df)
    return data_encoder, label_encoder


def books_cross_val(df, k=5, seed=10):
    df_remain = df
    while k > 0:
        if k == 1:
            train_idx = df.index.difference(df_remain.index)
            test_idx = df_remain.index
        else:
            share = (k - 1) / k
            df_remain, fold, _, _ = train_test_split(df_remain, share=share, seed=seed, cross_val=True)
            train_idx = df.index.difference(fold.index)
            test_idx = fold.index
        yield train_idx, test_idx
        k -= 1


def get_top_features(label_enc, data_enc, clf, n):
    names = label_enc.classes_
    coeffs = clf.coef_
    author_dict = dict()
    for i, author in enumerate(names):
        args_sorted = list(reversed(np.argsort(coeffs[i])[-n:]))
        features = [data_enc.find_idx(idx) for idx in args_sorted]
        author_dict[author] = features
    df = pd.DataFrame(author_dict)
    return df
