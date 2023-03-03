import itertools
from collections import OrderedDict

import pymystem3
import numpy as np
from pymystem3.mystem import Mystem

from .common import FeatureList, VectorFeature


@FeatureList.register_feature
class Ngrams(VectorFeature):
    name = "ngrams"
    _pts_of_speech = {"A", "ADV", "ADVPRO", "ANUM", "APRO", "COM", "INTJ", "NUM", "PART", "S", "SPRO", "V"}

    @classmethod
    def _get_sp_part(cls, word):
        try:
            info = word["analysis"][0]
        except IndexError:
            return None
        pt = info["gr"].split(",")[0]
        return pt if pt in cls._pts_of_speech else None

    @classmethod
    def _metric(cls, sentences, ngrams=3, **kwargs):
        counter = OrderedDict((ngram, 0) for ngram in itertools.product(cls._pts_of_speech, repeat=ngrams))
        stemmer = Mystem(mystem_bin=pymystem3.MYSTEM_BIN, entire_input=False)
        for sentence in sentences:
            result = stemmer.analyze(sentence)
            speech_parts = map(cls._get_sp_part, result)
            clean_snt = tuple(filter(lambda x: x is not None, speech_parts))
            for i in range(len(clean_snt) - ngrams):
                ngram = clean_snt[i:i + ngrams]
                counter[ngram] += 1
        vals = list(counter.values())
        total_count = sum(counter.values())
        percentage = np.array(vals, dtype=np.float64) / total_count
        return percentage
