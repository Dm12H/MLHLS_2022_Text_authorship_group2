import argparse
import itertools
import os
from collections import OrderedDict

import pymystem3
import numpy as np
from pymystem3.mystem import Mystem

from common import get_books_as_text_iterator, hash_results


_pts_of_speech = {
    "A": "adj",
    "ADV": "adv",
    "ADVPRO": "adv",
    "ANUM": "adj",
    "APRO": "adj",
    "S": "noun",
    "SPRO": "noun",
    "V": "verb"
}


def _get_sp_part(word):
    try:
        info = word["analysis"][0]
    except IndexError:
        return None
    pt = info["gr"].split(",")[0]
    return _pts_of_speech.get(pt, None)


def get_ngrams_hist(text_list, n=3):
    counter = OrderedDict((ngram, 0) for ngram in itertools.product(_pts_of_speech.values(), repeat=n))
    stemmer = Mystem(mystem_bin=pymystem3.MYSTEM_BIN, entire_input=False)
    for text in text_list:
        for sentence in text:
            result = stemmer.analyze(sentence)
            speech_parts = map(_get_sp_part, result)
            clean_snt = tuple(filter(lambda x: x is not None, speech_parts))
            for i in range(len(clean_snt)-n):
                ngram = clean_snt[i:i+n]
                counter[ngram] += 1
    return counter


def ngrams_stats(writers_dir, n=3, hash_flag=False, out_path=None):
    writers = os.listdir(writers_dir)
    results_dict = dict()
    for writer in writers:
        book_chunk_chain = get_books_as_text_iterator(writer, writers_dir, cutoff=None)
        if hash_flag:
            counts = hash_results(get_ngrams_hist, writers_dir, writer, out_path=out_path, n=n)
        else:
            counts = get_ngrams_hist(book_chunk_chain, n=n)
        vals = list(counts.values())
        total_count = sum(vals)
        percentage = np.array(vals, dtype=np.float64) / total_count
        results_dict[writer] = percentage
    return results_dict


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--writers_dir", help="root folder of all books")
    argparser.add_argument("--writer", help="name of the writer")
    argparser.add_argument("--out", help="path to save results to")
    argparser.add_argument("--hash_results", help="flag to check whether to hash results", action="store_true")
    args = argparser.parse_args()
    stats = ngrams_stats(writers_dir=args.writers_dir, out_path=args.out, hash_flag=args.hash_results)
    for key, value in stats.items():
        print(key, value)
