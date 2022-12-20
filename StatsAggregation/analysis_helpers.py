import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def load_df(path):
    """
    загружает датасет с нужными полями для работы
    """
    df = pd.read_csv(path)
    df['counts'] = df.book.map(df.book.value_counts())

    def _add_class_based_weigths(df):
        author_counts = df.author.map(df.author.value_counts())
        num_authors = len(df.author.unique())
        df["probs"] = 1 / (author_counts * num_authors)

    _add_class_based_weigths(df)
    return df


def select_sample(dataframe, size=0.1):
    """
    сэмпл датасета, сбалансированный на вероятности классов
    :param dataframe: датафрейм для анализа
    :param size: доля общей выборки
    """
    df_size = len(dataframe)
    idx = np.random.choice(df_size, size=int(df_size * size), p=dataframe.probs)
    return dataframe.iloc[idx]


def train_test_split(df, share=0.5, seed=10, cross_val=False):
    """
    разделяет данные на обучающую и тестовую выборки по книгам
    :param df: датафрейм для анализа
    :param share: процент данных для трейна ( тест, соответственно 1-share)
    :param seed: сид для случайных перемешиваний
    """
    rg = np.random.default_rng(seed)
    labels = df.author.unique()
    shuffled_labels = labels[rg.permutation(len(labels))]

    train_label = []
    test_label = []
    train_counts = 0
    test_counts = 0

    for label in shuffled_labels:
        df_slice = df[df.author == label][["book", "counts"]]
        unique_books = df_slice.drop_duplicates("book")
        if len(unique_books) < 2 and not cross_val:
            raise ValueError(f"too few books of author: {label}")
        permunation = rg.permutation(len(unique_books))
        shuffled_books = unique_books.iloc[permunation]
        label_train_count = 0
        label_total_count = np.sum(shuffled_books["counts"])
        i = 0
        for _, row in shuffled_books.iloc[:-1].iterrows():
            num_segments = row.counts
            if not label_train_count:
                i += 1
                label_train_count += num_segments
                continue
            new_train_count = label_train_count + num_segments
            new_share = new_train_count / label_total_count
            old_share = label_train_count / label_total_count
            if abs(share - new_share) > abs(share - old_share):
                break
            i += 1
            label_train_count += num_segments
        train_label.extend(shuffled_books.iloc[:i].book)
        test_label.extend(shuffled_books.iloc[i:].book)
        train_counts += label_train_count
        test_counts += label_total_count - label_train_count
    train_df = df[df["book"].isin(set(train_label))]
    test_df = df[df["book"].isin(set(test_label))]
    assert len(train_df) + len(test_df) == len(df), \
        "split is wrong"
    y_train = train_df["author"]
    y_test = test_df["author"]
    return train_df, test_df, y_train, y_test


def get_document_vectorizer(frame, n_min=1, n=2, max_count=10000, column="lemmas"):
    """
    стандартный tf-idf
    :param frame: датафрейм для анализа
    :param n_min: минимальный размер n-грамм
    :param n: максимальный размер n-грамм
    :param max_count: максимальная длина вектора фичей
    :param column: название колонки, по которой нужно векторизировать
    """
    texts_vector = frame[column]
    vectorizer = TfidfVectorizer(ngram_range=(n_min, n), max_features=max_count, norm='l2')
    vectorizer.fit(texts_vector)
    return vectorizer


def get_author_vectorizer(frame, n_min=1, n=2, max_count=10000, column="lemmas"):
    """
    class-based tf-idf
    :param frame: датафрейм для анализа
    :param frame: датафрейм для анализа
    :param n_min: минимальный размер n-грамм
    :param n: максимальный размер n-грамм
    :param max_count: максимальная длина вектора фичей
    :param column: название колонки, по которой нужно векторизировать
    """
    grouped_text = frame.groupby("author", as_index=False)[column].agg({column: ' '.join})
    vectorizer = TfidfVectorizer(ngram_range=(n_min, n), max_features=max_count, norm='l2')
    texts_vector = grouped_text[column]
    vectorizer.fit(texts_vector)
    return vectorizer


def get_encoder(frame, column="author"):
    encoder = LabelEncoder()
    encoder.fit(frame[column])
    return encoder

