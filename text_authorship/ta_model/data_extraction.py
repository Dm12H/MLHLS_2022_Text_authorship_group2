from ..StatsAggregation.common import get_paragraphs
from data_preparation.features import _count_features

import os
from typing import Dict, List, Tuple
from functools import partial, lru_cache
import ebooklib
from ebooklib import epub
import itertools as it
import pandas as pd


@lru_cache(maxsize=1)
def get_books(
        writer: str,
        writers_dir: str
        ) -> Dict[str, List[str]]:
    book_list = os.listdir(os.path.join(writers_dir, writer))
    full_book_path = partial(os.path.join, writers_dir, writer)
    books = {}
    for book_name in book_list:
        book = epub.read_epub(full_book_path(book_name))
        chapters = (book.get_item_with_id(chapter_id)
                    for chapter_id, _
                    in book.spine)
        text_chapters = [ch for ch in chapters
                         if ch.get_type() == ebooklib.ITEM_DOCUMENT]
        books[book_name] = list(
            it.chain.from_iterable(
                map(get_paragraphs, text_chapters)))
    return books


@lru_cache(maxsize=1)
def get_data_for_df(
        writer: str,
        writers_dir: str,
        symbol_lim: int = 3000
        ) -> List[Tuple[str, str]]:
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
def extract_df(
        writers_dir: str,
        symbol_lim: int = 3000
        ) -> pd.DataFrame:
    pre_df = []
    for writer in os.listdir(writers_dir):
        for book, text in get_data_for_df(writer=writer,
                                          writers_dir=writers_dir,
                                          symbol_lim=symbol_lim):
            pre_df.append(
                {"author": writer,
                 "book": book,
                 "text": text})

    df = pd.DataFrame(pre_df)
    return df


def load_df(
        path: str,
        load_stats: bool = True,
        count_features: bool = True
        ) -> pd.DataFrame:
    """
    загружает датасет с нужными полями для работы
    """
    df = pd.read_csv(path)
    df['counts'] = df.book.map(df.book.value_counts())

    def _add_class_based_weigths(df: pd.DataFrame):
        author_counts = df.author.map(df.author.value_counts())
        num_authors = len(df.author.unique())
        df["probs"] = 1 / (author_counts * num_authors)

    if load_stats:
        _add_class_based_weigths(df)
    if count_features:
        df = _count_features(df)
    return df
