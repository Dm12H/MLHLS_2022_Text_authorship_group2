import pandas as pd

import sys
import unicodedata
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import numpy as np
from itertools import chain
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


punkt = ''.join([chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')])


nltk.download('stopwords')


def remove_punkt(text):
    tokens = word_tokenize(text, language='russian')
    tokens = [token.strip(punkt) for token in tokens]
    return ' '.join([token for token in tokens if token])


def lemmatize_with_tags(text, sw=set()):
    morph = MorphAnalyzer()
    lemmas = []
    tags = []
    for w in text.split():
        anls = morph.parse(w)[0]
        if anls.normal_form not in sw:
            lemmas.append(anls.normal_form)
            tags.append(f'{len(w)}_{anls.tag.POS}')
    return ' '.join(lemmas), ' '.join(tags)


class TATransformer(BaseEstimator, TransformerMixin):

    def __init__(self, use_stopwords=True, save_to_csv=True):
        self.use_stopwords = use_stopwords
        self.save_to_csv = save_to_csv
    
    def fit(self, X: pd.DataFrame, y=None):
        self.sw_ = set(X['author'].str.lower().unique())

        if self.use_stopwords:
            self.sw_ = self.sw_ | set(stopwords.words('russian'))

        return self
    
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X['text'] = X['text'].str.lower()

        X['text_no_punkt'] = X['text'].apply(remove_punkt)

        lemmas_and_tags = X['text_no_punkt'].apply(lemmatize_with_tags, sw=self.sw_)
        df_lemmas_and_tags = pd.DataFrame(lemmas_and_tags, columns=['lemmas', 'tags'])

        X = pd.concat([
            X,
            df_lemmas_and_tags
        ], axis=1)

        X['tokens'] = X['text'].apply(lambda t: ' '.join(word_tokenize(t, language='russian')))

        X.to_csv('transformed_df.csv', index=False)

        return X
    

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