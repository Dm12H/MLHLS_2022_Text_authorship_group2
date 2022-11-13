import argparse
import os
import unicodedata
import sys
import numpy as np
import nltk

from common import get_feature_sample

punct_deleter = dict.fromkeys(i for i in range(sys.maxunicode)
                              if unicodedata.category(chr(i)).startswith('P'))
space_deleter = dict.fromkeys(i for i in range(sys.maxunicode)
                              if unicodedata.category(chr(i)).startswith('Z'))


def count_stats(writers_dir="C:\\Users\\annag\\Documents\\Писатели для MLDS"):
    stats = dict()
    for writer in os.listdir(writers_dir):
        stats[writer] = dict()
        stats[writer]['words_avg_length'] = np.mean(word_avg_length(writer, writers_dir))
        stats[writer]['words_per_sentence'] = np.mean(words_per_sentence(writer, writers_dir))
        stats[writer]['exclamations_per_sentence'] = np.mean(exclamations_per_sentence(writer, writers_dir))
        stats[writer]['questions_per_sentence'] = np.mean(questions_per_sentence(writer, writers_dir))

    return stats

@get_feature_sample
def word_avg_length(sentences):
    words_cnt = 0
    total_length = 0
    for sentence in sentences:
        words = [word for word in nltk.word_tokenize(sentence, language='russian') if word.isalpha()]
        words_cnt += len(words)
        total_length += len(' '.join(words))
    return total_length / words_cnt

@get_feature_sample
def words_per_sentence(sentences):
    words_cnt = 0
    for sentence in sentences:
        words = [word for word in nltk.word_tokenize(sentence, language='russian') if word.isalpha()]
        words_cnt += len(words)
    return words_cnt / len(sentences)

@get_feature_sample
def exclamations_per_sentence(sentences):
    exclamations_cnt = 0
    for sentence in sentences:
        if '!' in sentence:
            exclamations_cnt += 1
    return exclamations_cnt / len(sentences)

@get_feature_sample
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
