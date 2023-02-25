from ..StatsAggregation.common import get_paragraphs
from data_preparation import _count_features

import os
from functools import partial, lru_cache
import ebooklib
from ebooklib import epub
import itertools as it
import pandas as pd


@lru_cache(maxsize=1)
def get_books(writer, writers_dir, cutoff=-2):
    book_list = os.listdir(os.path.join(writers_dir, writer))
    full_book_path = partial(os.path.join, writers_dir, writer)
    books = {}
    for book_name in book_list:
        book = epub.read_epub(full_book_path(book_name))
        chapters = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))[slice(None, cutoff)]
        books[book_name] = list(it.chain.from_iterable(map(get_paragraphs, chapters)))
    return books


@lru_cache(maxsize=1)
def get_data_for_df(writer, writers_dir, symbol_lim=3000):
    books = get_books(writer, writers_dir)
    data = []
    for book, paras in books.items():
        i = 0
        while i < len(paras):
            new_sample = []
            symbol_cnt = 0
            while symbol_cnt < symbol_lim and i < len(paras):
                new_sample.append(paras[i])
                symbol_cnt += len(paras[i])
                i += 1
            data.append((book, '\n'.join(new_sample)))
    return data


@lru_cache(maxsize=1)
def extract_df(writers_dir, symbol_lim=3000):    
    pre_df = []
    for writer in os.listdir(writers_dir):
        for book, text in get_data_for_df(writer, writers_dir, symbol_lim):
            pre_df.append([writer, book, text])

    df = pd.DataFrame(pre_df, columns=['author', 'book', 'text'])
    return df


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
