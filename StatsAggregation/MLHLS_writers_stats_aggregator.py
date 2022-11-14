import argparse
import os
import unicodedata
import sys

import nltk
import numpy as np

from common import SwitchBoard, SOURCESNAMES

punct_deleter = dict.fromkeys(i for i in range(sys.maxunicode)
                              if unicodedata.category(chr(i)).startswith('P'))
space_deleter = dict.fromkeys(i for i in range(sys.maxunicode)
                              if unicodedata.category(chr(i)).startswith('Z'))


def count_stats(writers_dir="C:\\Users\\annag\\Documents\\Писатели для MLDS"):
    stats = dict()
    extra_params = {"sample_step": 100}
    for writer in os.listdir(writers_dir):
        stats[writer] = dict()
        for sname, sfunc in SwitchBoard.sinks.items():
            data_source = SwitchBoard.get_source(sfunc)
            data = data_source(writer, writers_dir, **extra_params)
            feature = list(map(sfunc, data))
            stats[writer][sname] = np.mean(feature)
    return stats


@SwitchBoard.request_source(SOURCESNAMES.sentences)
def word_avg_length(sentences):
    words_cnt = 0
    total_length = 0
    for sentence in sentences:
        words = [word for word in nltk.word_tokenize(sentence, language='russian') if word.isalpha()]
        words_cnt += len(words)
        total_length += len(' '.join(words))
    return total_length / words_cnt


@SwitchBoard.request_source(SOURCESNAMES.sentences)
def words_per_sentence(sentences):
    words_cnt = 0
    for sentence in sentences:
        words = [word for word in nltk.word_tokenize(sentence, language='russian') if word.isalpha()]
        words_cnt += len(words)
    return words_cnt / len(sentences)


@SwitchBoard.request_source(SOURCESNAMES.sentences, sink_name="exclamation_density")
def exclamations_per_sentence(sentences):
    exclamations_cnt = 0
    for sentence in sentences:
        if '!' in sentence:
            exclamations_cnt += 1
    return exclamations_cnt / len(sentences)


@SwitchBoard.request_source(SOURCESNAMES.sentences, sink_name="question_density")
def questions_per_sentence(sentences):
    questions_cnt = 0
    for sentence in sentences:
        if '?' in sentence:
            questions_cnt += 1
    return questions_cnt / len(sentences)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", help="root folder of all books")
    args = argparser.parse_args()
    print(count_stats(writers_dir=args.data_dir))
