import argparse
import os
import unicodedata
import sys

from common import get_books_as_text_iterator

punct_deleter = dict.fromkeys(i for i in range(sys.maxunicode)
                              if unicodedata.category(chr(i)).startswith('P'))
space_deleter = dict.fromkeys(i for i in range(sys.maxunicode)
                              if unicodedata.category(chr(i)).startswith('Z'))


def count_stats(writers_dir="C:\\Users\\annag\\Documents\\Писатели для MLDS"):
    stats = dict()
    for writer in os.listdir(writers_dir):
        stats[writer] = dict()
        stats[writer]['words_avg_length'] = word_avg_length(writer, writers_dir)

    return stats


def word_avg_length(writer, writers_dir):
    words_cnt = 0
    total_length = 0
    for chapter in get_books_as_text_iterator(writer, writers_dir):
        text = '\n\n'.join(chapter)
        words_cnt += len(text.translate(punct_deleter).split())
        total_length += len(text.translate(punct_deleter).translate(space_deleter))

    return total_length / words_cnt


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", help="root folder of all books")
    args = argparser.parse_args()
    print(count_stats(writers_dir=args.data_dir))
