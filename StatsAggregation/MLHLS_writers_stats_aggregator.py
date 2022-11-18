import argparse
import os
import unicodedata
import sys

import matplotlib.pyplot as plt

from common import ScalarFeature, FeatureList
from common import paragraphs_limmited_by_symbols, token_batches, word_batches

punct_deleter = dict.fromkeys(i for i in range(sys.maxunicode)
                              if unicodedata.category(chr(i)).startswith('P'))
space_deleter = dict.fromkeys(i for i in range(sys.maxunicode)
                              if unicodedata.category(chr(i)).startswith('Z'))
long_dash = chr(8212)

def count_stats(writers_dir="C:\\Users\\annag\\Documents\\Писатели для MLDS", save_pics=False, out_dir=None):
    stats = dict()
    extra_params = {
        "sentences_in_batch": 100, 
        "symbol_lim": 50000, 
        "tokens_in_batch": 2000, 
        "words_in_batch": 2000
        }
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
    data_source = word_batches

    @staticmethod
    def _metric(words, **_):
        total_len = sum(len(word) for word in words)
        return total_len / len(words)


@FeatureList.register_feature
class WordsPerSentence(ScalarFeature):
    name = "words_per_sentence"

    @staticmethod
    def _metric(sentences, **_):
        words_cnt = 0
        for sentence in sentences:
            words = sentence.translate(punct_deleter).strip().split()
            words_cnt += len(words)
        return words_cnt / len(sentences)


@FeatureList.register_feature
class ExclamationDensity(ScalarFeature):
    name = "exclamation_density"

    @staticmethod
    def _metric(sentences, **_):
        exclamations_cnt = len([s for s in sentences if '!' in s])
        return exclamations_cnt / len(sentences)


@FeatureList.register_feature
class QuestionDensity(ScalarFeature):
    name = "question_density"

    @staticmethod
    def _metric(sentences, **_):
        questions_cnt = len([s for s in sentences if '?' in s])
        return questions_cnt / len(sentences)


@FeatureList.register_feature
class DialogueDensity(ScalarFeature):
    name = "dialogue_density"
    data_source = paragraphs_limmited_by_symbols

    @staticmethod
    def _metric(paras, **_):
        total_len = 0
        dialogue_len = 0
        for para in paras:
            total_len += len(para)
            if para and para[0] == long_dash:
                dialogue_len += len(para)
        return dialogue_len / total_len


@FeatureList.register_feature
class CommaDensity(ScalarFeature):
    name = "comma_density"
    data_source = token_batches

    @staticmethod
    def _metric(tokens, **_):
        return tokens.count(',') / len(tokens)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", help="root folder of all books")
    argparser.add_argument("--save_pics", help="flag to toggle saving images", action="store_true")
    argparser.add_argument("--out_dir", help="where to save results")
    args = argparser.parse_args()
    print(count_stats(writers_dir=args.data_dir, save_pics=args.save_pics, out_dir=args.out_dir))
