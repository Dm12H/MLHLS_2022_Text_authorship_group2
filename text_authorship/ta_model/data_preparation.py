from itertools import chain
from typing import List, Sequence, Any
import pandas as pd
import nltk
import numpy as np

from sklearn.preprocessing import LabelEncoder


def get_encoder(
        frame: pd.DataFrame,
        column: str = "author") -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(frame[column])
    return encoder


def split_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text)


def make_words(sentences: Sequence[str]) -> List[str]:
    words = list(
        chain(
            *(nltk.word_tokenize(sentence) for sentence in sentences)))
    return words


def word_len_avg(words: Sequence[str]) -> float:
    total_len = sum(len(word) for word in words)
    return total_len / len(words)


def word_per_sentence(sentences: Sequence, words: List[str]) -> float:
    num_sentences = len(sentences)
    num_words = len(words)
    log = np.log(num_words / num_sentences)
    return float(log)


def count_occurences(token: str, sentences: Sequence[str]) -> int:
    count = len([s for s in sentences if token in s])
    return count


def exclamation_density(sentences: Sequence[str]) -> float:
    exclamations_cnt = count_occurences(token="!", sentences=sentences)
    return exclamations_cnt / len(sentences)


def question_density(sentences: Sequence[str]) -> float:
    questions_cnt = count_occurences(token='?', sentences=sentences)
    return questions_cnt / len(sentences)


def comma_density(sentences: Sequence[str]) -> float:
    comma_cnt = count_occurences(token=',', sentences=sentences)
    return comma_cnt / len(sentences)


def dialogue_density(sentences: Sequence[str]) -> float:
    long_dash = chr(8212)
    counts = len(
        [sentence
         for sentence in sentences
         if sentence.startswith(long_dash)])
    return counts / len(sentences)


def _count_features(df: pd.DataFrame) -> pd.DataFrame:
    tokenized_text = df.text.map(split_sentences)
    words = tokenized_text.map(make_words)
    temp_df = pd.DataFrame({"words": words, "sentences": tokenized_text})
    feature_map = {
        "word_avg_length": words.map(word_len_avg),
        "words_per_sentence": temp_df.apply(
            lambda x: word_per_sentence(x["sentences"], x["words"]),
            axis=1),
        "exclamation_density": tokenized_text.map(exclamation_density),
        "question_density": tokenized_text.map(question_density),
        "comma_density": tokenized_text.map(question_density),
        "dialogue_density": tokenized_text.map(dialogue_density),
    }
    feature_df = pd.DataFrame(feature_map)

    return pd.merge(df, feature_df, left_index=True, right_index=True)


def check_seq(val: Any) -> bool:
    iterable = hasattr(val, "__iter__")
    not_string = not isinstance(val, str)
    return iterable and not_string
