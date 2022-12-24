import nltk
import pandas as pd
import numpy as np
import scipy as sp
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


def _count_features(df):
    tokenized_text = df.text.map(split_sentences)
    words = tokenized_text.map(make_words)
    temp_df = pd.DataFrame({"words": words, "sentences": tokenized_text})
    feature_map = {
        "word_avg_length": words.map(word_len_avg),
        "words_per_sentence": temp_df.apply(lambda x: word_per_sentence(x["sentences"], x["words"]), axis=1),
        "exclamation_density": tokenized_text.map(exclamation_density),
        "question_density": tokenized_text.map(question_density),
        "comma_density": tokenized_text.map(question_density),
        "dialogue_density": tokenized_text.map(dialogue_density),
    }
    feature_df = pd.DataFrame(feature_map)

    return pd.merge(df, feature_df, left_index=True, right_index=True)


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
    _count_features(df)
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


def split_sentences(text):
    return nltk.sent_tokenize(text)


def make_words(sentences):
    words = list(chain(*(nltk.word_tokenize(sentence) for sentence in sentences)))
    return words


def word_len_avg(words):
    total_len = sum(len(word) for word in words)
    return total_len / len(words)


def word_per_sentence(sentences, words):
    num_sentences = len(sentences)
    num_words = len(words)
    return np.log(num_words / num_sentences)


def exclamation_density(sentences):
    exclamations_cnt = len([s for s in sentences if '!' in s])
    return exclamations_cnt / len(sentences)


def question_density(sentences):
    questions_cnt = len([s for s in sentences if '?' in s])
    return questions_cnt / len(sentences)


def comma_density(sentences):
    questions_cnt = sum([s.count(",") for s in sentences])
    return questions_cnt / len(sentences)


def dialogue_density(sentences):
    long_dash = chr(8212)
    counts = len([sentence for sentence in sentences if sentence.startswith(long_dash)])
    return counts / len(sentences)


def check_seq(val):
    iterable = hasattr(val, "__iter__")
    not_string = not isinstance(val, str)
    return iterable and not_string


class Featurebuilder:
    feature_mapping = {"tokens": "vectorizer",
                       "text_no_punkt": "vectorizer",
                       "lemmas": "vectorizer",
                       "tags": "vectorizer"}

    def __init__(self, *args, **vectorizers):
        vectorizers = {k.lstrip("vec_"): v for k, v in vectorizers.items()}
        featurelist = self.pack_features(args)
        self.vectorizers = {}
        for feature in featurelist:
            processor = self.feature_mapping.get(feature, None)
            if not processor:
                self.vectorizers[feature] = None
                continue
            if processor != "vectorizer":
                raise ValueError("only vectorizing is supported for non-scalar features")
            vectorizer = vectorizers.get(feature, None)
            if vectorizer is None:
                raise ValueError(f"no vectorizer for feature: {feature}")
            self.vectorizers[feature] = vectorizer
        self.ordered_ft = list(sorted(featurelist,
                                      key=lambda x: self.feature_mapping.get(x, "")))
        self.ordered_proc = [self.feature_mapping.get(ft, None) for ft in self.ordered_ft]
        self._initialized = False
        self.feature_idx = None

    @staticmethod
    def pack_features(features):
        attrs = (ft if check_seq(ft) else (ft,) for ft in features)
        return list(chain(*attrs))

    @staticmethod
    def get_last_occurence(seq, val):
        return len(seq) - 1 - seq[::-1].index(val)

    @staticmethod
    def get_first_occurence(seq, val):
        return seq.index(val)

    def fit_transform(self, df):
        feature_positions = []
        feature_matrices = []
        for proc in set(self.ordered_proc):
            first_idx = self.get_first_occurence(self.ordered_proc, proc)
            last_idx = self.get_last_occurence(self.ordered_proc, proc)
            feature_slice = self.ordered_ft[first_idx:last_idx + 1]

            smat, fpos = self.bulk_process(df, proc, feature_slice)

            feature_matrices.append(smat)
            feature_positions.extend(fpos)

        final_matrix = sp.sparse.hstack(feature_matrices)
        counter = 0
        self.feature_idx = dict()
        for ft, length in zip(self.ordered_ft, feature_positions):
            self.feature_idx[ft] = (counter, length)
            counter += length
        self._initialized = True
        return final_matrix

    def transform(self, df):
        feature_matrices = []
        for proc in set(self.ordered_proc):
            first_idx = self.get_first_occurence(self.ordered_proc, proc)
            last_idx = self.get_last_occurence(self.ordered_proc, proc)
            feature_slice = self.ordered_ft[first_idx:last_idx + 1]
            smat, fpos = self.bulk_process(df, proc, feature_slice)
            feature_matrices.append(smat)
        final_matrix = sp.sparse.hstack(feature_matrices)
        return final_matrix

    def bulk_process(self, df, proc, featurelist):
        if proc is None:
            columns = df[featurelist].to_numpy(dtype=np.float64)
            mat = sp.sparse.csr_matrix(columns)
            indices = [1 for ft in featurelist]
            return mat, indices
        elif proc != "vectorizer":
            raise ValueError("only vectorizers supported now")
        else:
            matrices = []
            indices = []
            for i, feature in enumerate(featurelist):
                vectorizer = self.vectorizers[feature]
                mat = vectorizer.transform(df[feature])
                matrices.append(mat)
                indices.append(mat.shape[1])
            return matrices, indices

    def find_idx(self, idx):
        for key, (start, length) in self.feature_idx.items():
            if start <= idx < (start + length):
                break
        else:
            raise ValueError(f"index {idx} too big")
        vectorizer = self.vectorizers[key]
        if vectorizer is None:
            return key
        else:
            features = vectorizer.get_feature_names_out()
            return features[idx - start]


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


def train_crossval_twofold(frame, clf, *args, split=0.5, vectorizer_dict=None, avg="micro"):
    split = train_test_split(frame, share=split)
    x_split = split[:2]
    y_split = split[2:]
    scores = []
    for train_idx, test_idx in ((1, 0), (0, 1)):
        df_train = x_split[train_idx]
        df_test = x_split[test_idx]

        y_train = y_split[train_idx]
        y_test = y_split[test_idx]

        if vectorizer_dict == None:
            raise ValueError("not using any vectorizer!")
        vecs = dict()
        for fname, params in vectorizer_dict.items():
            vecs["vec_" + fname] = get_author_vectorizer(df_train, **params, column=fname)

        data_encoder = Featurebuilder(*args, **vecs)
        x_train = data_encoder.fit_transform(df_train)
        x_test = data_encoder.fit_transform(df_test)

        label_encoder = get_encoder(frame=frame)
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

        clf.fit(x_train, y_train)
        score = f1_score(clf.predict(x_test), y_test, average=avg)
        scores.append(score)
    return scores
