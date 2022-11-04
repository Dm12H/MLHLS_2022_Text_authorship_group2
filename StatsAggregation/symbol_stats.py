import argparse
import os
import unicodedata
import sys

from common import get_books_as_text_iterator

punct_deleter = dict.fromkeys(i for i in range(sys.maxunicode)
                              if unicodedata.category(chr(i)).startswith('P'))


def count_symbols_per_writer(writers_dir, writer):
    counter = 0
    for chapter in get_books_as_text_iterator(writer, writers_dir):
        for line in chapter:
            counter += len(line)
    return counter


def word_count(writers_dir, writer):
    words_cnt = 0
    for chapter in get_books_as_text_iterator(writer, writers_dir):
        for text in chapter:
            words_cnt += len(text.translate(punct_deleter).split())

    return words_cnt


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--writers_dir", help="root folder of all books", required=True)
    args = argparser.parse_args()
    symbol_counter = dict()
    word_counter = dict()
    for writer in os.listdir(args.writers_dir):
        symbol_counter[writer] = count_symbols_per_writer(args.writers_dir, writer)
        word_counter[writer] = word_count(args.writers_dir, writer)
        print(f"{writer} word count: {word_counter[writer]:,}")
        print(f"{writer} symbol count: {symbol_counter[writer]:,}")
    print(f"total word count: {sum(word_counter.values()):,}")
    print(f"total symbol count: {sum(symbol_counter.values()):,}")
