from sklearn.feature_extraction.text import TfidfVectorizer


def get_document_vectorizer(
        frame,
        n_min=1,
        n=2,
        max_count=10000,
        column="lemmas"):
    """
    стандартный tf-idf
    :param frame: датафрейм для анализа
    :param n_min: минимальный размер n-грамм
    :param n: максимальный размер n-грамм
    :param max_count: максимальная длина вектора фичей
    :param column: название колонки, по которой нужно векторизировать
    """
    texts_vector = frame[column]
    vectorizer = TfidfVectorizer(
        ngram_range=(n_min, n),
        max_features=max_count, norm='l2')
    vectorizer.fit(texts_vector)
    return vectorizer


def get_author_vectorizer(
        frame,
        n_min=1,
        n=2,
        max_count=10000,
        column="lemmas"):
    """
    class-based tf-idf
    :param frame: датафрейм для анализа
    :param frame: датафрейм для анализа
    :param n_min: минимальный размер n-грамм
    :param n: максимальный размер n-грамм
    :param max_count: максимальная длина вектора фичей
    :param column: название колонки, по которой нужно векторизировать
    """
    grouped_text = frame.groupby(
        "author", as_index=False)[column].agg({column: ' '.join})
    vectorizer = TfidfVectorizer(
        ngram_range=(n_min, n),
        max_features=max_count,
        norm='l2')
    texts_vector = grouped_text[column]
    vectorizer.fit(texts_vector)
    return vectorizer
