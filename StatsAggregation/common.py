import hashlib
import itertools as it
import os
from abc import abstractmethod
from configparser import ConfigParser
from functools import partial, lru_cache

import ebooklib
import nltk
import numpy as np
from bs4 import BeautifulSoup
from ebooklib import epub

from .visualizers import draw_distribution, draw_3d

delete_inner = str.maketrans('', '', '\u0301\u0300\u0060\u02cb\u0027\u2019\u002d\u2010')

def get_paragraphs(chapter):
    soup = BeautifulSoup(chapter.get_body_content(), 'html.parser')
    text = [para.get_text() for para in soup.find_all('p')]
    return text


def get_books_as_text_iterator(writer, writers_dir, cutoff=-2):
    book_list = os.listdir(os.path.join(writers_dir, writer))
    full_book_path = partial(os.path.join, writers_dir, writer)
    books = map(lambda book_name: epub.read_epub(full_book_path(book_name)), book_list)
    chapters = [book.get_items_of_type(ebooklib.ITEM_DOCUMENT) for book in books][slice(None, cutoff)]
    return map(get_paragraphs, it.chain.from_iterable(chapters))


def get_dir_hash(directory):
    sha1 = hashlib.sha1()
    for root, folders, files in os.walk(directory):
        for folder in folders:
            for file in files:
                path = os.path.join(root, folder, file)
                with open(path, "rb") as f:
                    sha1.update(hashlib.sha1(f.read()).hexdigest())
    return sha1.hexdigest()


def hash_results(func, data_dir, writer, *args, out_path=None, **kwargs):
    # TODO replace Configparser with json/xml/yaml
    writers_hashfile = ConfigParser()
    hash_file = func.__name__+"_hashes.ini"
    writers_hashfile.read(os.path.join(out_path, hash_file))
    path = os.path.join(data_dir, writer)
    try:
        hash_info = writers_hashfile[writer]
        hash_existing = hash_info["hash"]
        hash_computed = get_dir_hash(path)
        if hash_existing == hash_computed:
            feature_vector = np.load(os.path.join(out_path, hash_info["path"]))
            return feature_vector
    except KeyError:
        hash_computed = get_dir_hash(path)
    result = func(data_dir, writer, *args, **kwargs)
    fname_to_save = os.path.join(out_path, f"{func.__name__}_{writer.lower()}.npy")
    np.save(file=fname_to_save, arr=result)
    writers_hashfile[writer] = {"hash": hash_computed, "path": fname_to_save}
    with open(os.path.join(out_path, hash_file), "w") as f:
        writers_hashfile.write(f)


@lru_cache(maxsize=1)
def sentence_batches(writer, writers_dir, sentences_in_batch=100, **_):
    chapters = get_books_as_text_iterator(writer, writers_dir)
    chapters_to_sentences = map(lambda chp: nltk.sent_tokenize(' '.join(chp), language='russian'), chapters)
    sentences = list(it.chain.from_iterable(chapters_to_sentences))
    samples = []
    for i in range(0, len(sentences), sentences_in_batch):
        samples.append(sentences[i:i + sentences_in_batch])
    return samples


@lru_cache(maxsize=1)
def paragraphs_limmited_by_symbols(writer, writers_dir, symbol_lim=50000, **_):
    paragraphs = list(it.chain.from_iterable(get_books_as_text_iterator(writer, writers_dir)))
    samples = []
    i = 0
    while i < len(paragraphs):
        new_sample = []
        symbol_cnt = 0
        while symbol_cnt < symbol_lim and i < len(paragraphs):
            new_sample.append(paragraphs[i])
            symbol_cnt += len(paragraphs[i])
            i += 1
        samples.append(new_sample)
    return samples


@lru_cache(maxsize=1)
def token_batches(writer, writers_dir, tokens_in_batch=2000, **_):
    chapters = get_books_as_text_iterator(writer, writers_dir)
    chapters_to_tokens = map(lambda chp: nltk.word_tokenize(' '.join(chp), language='russian'), chapters)
    tokens = list(it.chain.from_iterable(chapters_to_tokens))
    sample = []
    for i in range(0, len(tokens), tokens_in_batch):
        sample.append(tokens[i:i + tokens_in_batch])
    return sample


@lru_cache(maxsize=1)
def word_batches(writer, writers_dir, words_in_batch=2000, **_):
    chapters = get_books_as_text_iterator(writer, writers_dir)
    chapters_to_tokens = map(lambda chp: nltk.word_tokenize(' '.join(chp), language='russian'), chapters)
    tokens = it.chain.from_iterable(chapters_to_tokens)
    words = words_from_tokens(tokens)
    sample = []
    for i in range(0, len(words), words_in_batch):
        sample.append(words[i: i + words_in_batch])
    return sample


class Feature:
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def data_source(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def visualizer_single(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def visualizer_all(self):
        raise NotImplementedError

    @abstractmethod
    def _metric(self, data, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pack(data):
        raise NotImplementedError

    @classmethod
    def process(cls, data, **kwargs):
        return list(cls._metric(d, **kwargs) for d in data)

    def visualize(self, data, **kwargs):
        return self.visualizer_single(data, **kwargs)


class ScalarFeature(Feature):

    data_source = sentence_batches
    visualizer_single = draw_distribution
    visualizer_all = draw_3d

    @staticmethod
    def pack(data):
        return np.mean(data)


class VectorFeature(Feature):

    data_source = sentence_batches
    visualizer_single = draw_distribution

    @classmethod
    def visualizer_all(cls, ax, data, n_rows=10, labels=None, **kwargs):
        kwargs = kwargs
        data_packed = np.array([cls.pack(w_data) for w_data in data])
        data_mask = np.any(data_packed, axis=0)
        data_clean = data_packed[:, data_mask]
        draw_3d(ax, data_clean, n_rows=n_rows, labels=labels, mode="hist", overlap=False, **kwargs)

    @staticmethod
    def pack(data):
        return np.mean(data, axis=0)


class FeatureList:

    features = []

    @classmethod
    def register_feature(cls, feature):
        cls.features.append(feature)
        return feature

def words_from_tokens(tokens):
    return [token for token in tokens if token.translate(delete_inner).isalpha()]