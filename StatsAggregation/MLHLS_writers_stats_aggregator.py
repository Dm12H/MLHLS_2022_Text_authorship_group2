import argparse
import os
import unicodedata
import sys

import matplotlib.pyplot as plt
import nltk

from common import ScalarFeature, FeatureList

punct_deleter = dict.fromkeys(i for i in range(sys.maxunicode)
                              if unicodedata.category(chr(i)).startswith('P'))
space_deleter = dict.fromkeys(i for i in range(sys.maxunicode)
                              if unicodedata.category(chr(i)).startswith('Z'))


def count_stats(writers_dir="C:\\Users\\annag\\Documents\\Писатели для MLDS", save_pics=False, out_dir=None):
    stats = dict()
    extra_params = {"sample_step": 100}
    for writer in os.listdir(writers_dir):
        stats[writer] = dict()
        for feature in FeatureList.features:
            data_source = feature.data_source
            data = data_source(writer, writers_dir, **extra_params)
            processed_data = feature.process(data)
            stats[writer][feature.name] = processed_data
    if save_pics:
        if out_dir is None:
            raise ValueError("need out_dir to save pictures")
        for feature in FeatureList.features:
            labels = [writer for writer in stats.keys()]
            rows = [stats[writer][feature.name] for writer in labels]
            fig, ax = plt.subplots(figsize=(10, 10))
            feature.visualizer_all(ax=ax, data=rows, labels=labels)
            fig.savefig(os.path.join(out_dir, f"{feature.name}.jpeg"))
    for feature in FeatureList.features:
        for writer in stats.keys():
            data = stats[writer][feature.name]
            stats[writer][feature.name] = feature.pack(data)
    return stats


@FeatureList.register_feature
class WordAvgLength(ScalarFeature):
    name = "word_avg_length"

    @staticmethod
    def _metric(sentences, **_):
        words_cnt = 0
        total_length = 0
        for sentence in sentences:
            words = [word for word in nltk.word_tokenize(sentence, language='russian') if word.isalpha()]
            words_cnt += len(words)
            total_length += len(' '.join(words))
        return total_length / words_cnt


@FeatureList.register_feature
class WordsPerSentence(ScalarFeature):
    name = "words_per_sentence"

    @staticmethod
    def _metric(sentences, **_):
        words_cnt = 0
        for sentence in sentences:
            words = [word for word in nltk.word_tokenize(sentence, language='russian') if word.isalpha()]
            words_cnt += len(words)
        return words_cnt / len(sentences)


@FeatureList.register_feature
class ExclamationDensity(ScalarFeature):
    name = "exclamation_density"

    @staticmethod
    def _metric(sentences, **_):
        exclamations_cnt = 0
        for sentence in sentences:
            if '!' in sentence:
                exclamations_cnt += 1
        return exclamations_cnt / len(sentences)


@FeatureList.register_feature
class QuestionDensity(ScalarFeature):
    name = "question_density"

    @staticmethod
    def _metric(sentences, **_):
        questions_cnt = 0
        for sentence in sentences:
            if '?' in sentence:
                questions_cnt += 1
        return questions_cnt / len(sentences)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", help="root folder of all books")
    argparser.add_argument("--save_pics", help="flag to toggle saving images", action="store_true")
    argparser.add_argument("--out_dir", help="where to save results")
    args = argparser.parse_args()
    print(count_stats(writers_dir=args.data_dir, save_pics=args.save_pics, out_dir=args.out_dir))
