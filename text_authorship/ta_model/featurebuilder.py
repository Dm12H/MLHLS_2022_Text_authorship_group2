from itertools import chain

import numpy as np
import pandas as pd
import scipy as sp

from typing import List, Dict, Sequence, Tuple, Any, Optional
from sklearn.feature_extraction.text import CountVectorizer
from data_preparation import check_seq


class FeatureBuilder:
    feature_mapping = {"tokens": "vectorizer",
                       "text_no_punkt": "vectorizer",
                       "lemmas": "vectorizer",
                       "tags": "vectorizer"}

    def __init__(self,
                 *args: Tuple[str | Sequence[str]],
                 **vectorizers: Dict[str, CountVectorizer]):
        vectorizers = {k.lstrip("vec_"): v for k, v in vectorizers.items()}
        featurelist = self.pack_features(args)
        self.vectorizers = {}
        for feature in featurelist:
            processor = self.feature_mapping.get(feature, None)
            if not processor:
                self.vectorizers[feature] = None
                continue
            if processor != "vectorizer":
                raise ValueError(
                    "only vectorizing is supported for non-scalar features")
            vectorizer = vectorizers.get(feature, None)
            if vectorizer is None:
                raise ValueError(f"no vectorizer for feature: {feature}")
            self.vectorizers[feature] = vectorizer
        self.ordered_ft = list(
            sorted(featurelist,
                   key=lambda x: self.feature_mapping.get(x, "")))
        self.ordered_proc = [self.feature_mapping.get(ft, None)
                             for ft
                             in self.ordered_ft]
        self._initialized = False
        self.feature_idx = None

    @staticmethod
    def pack_features(
            features: Sequence[str | Sequence]
            ) -> List[str]:
        attrs = (ft if check_seq(ft) else (ft,) for ft in features)
        return list(chain(*attrs))

    @staticmethod
    def get_last_occurence(
            seq: Sequence,
            val: Any
            ) -> int:
        return len(seq) - 1 - seq[::-1].index(val)

    @staticmethod
    def get_first_occurence(
            seq: Sequence,
            val: Any
            ) -> int:
        return seq.index(val)

    def _group_transform_features(
            self,
            df: pd.DataFrame,
            processor: Optional[CountVectorizer]
            ) -> Tuple[List[np.ndarray], List[int]]:
        first_idx = self.get_first_occurence(self.ordered_proc, processor)
        last_idx = self.get_last_occurence(self.ordered_proc, processor)
        feature_slice = self.ordered_ft[first_idx:last_idx + 1]
        feature_mat, positions = self.bulk_process(df,
                                                   proc=processor,
                                                   featurelist=feature_slice)
        return feature_mat, positions

    def fit_transform(
            self,
            df: pd.DataFrame
            ) -> np.array:
        feature_positions = []
        feature_matrices = []
        for proc in set(self.ordered_proc):
            featuremat, positions = self._group_transform_features(df, proc)
            feature_matrices.extend(featuremat)
            feature_positions.extend(positions)
        final_matrix = sp.sparse.hstack(feature_matrices)
        counter = 0
        self.feature_idx = dict()
        for ft, length in zip(self.ordered_ft, feature_positions):
            self.feature_idx[ft] = (counter, length)
            counter += length
        self._initialized = True
        return final_matrix

    def transform(
            self,
            df: pd.DataFrame
            ) -> np.array:
        feature_matrices = []
        for proc in set(self.ordered_proc):
            featuremat, positions = self._group_transform_features(df, proc)
            feature_matrices.extend(featuremat)
        final_matrix = sp.sparse.hstack(feature_matrices)
        return final_matrix

    def bulk_process(
            self,
            df: pd.DataFrame,
            proc: Optional[CountVectorizer],
            featurelist: Sequence[str]
            ) -> Tuple[List[np.array], List[int]]:
        if proc is None:
            columns = df[featurelist].to_numpy(dtype=np.float64)
            mat = [sp.sparse.csr_matrix(columns)]
            indices = [1 for _ in featurelist]
            return mat, indices
        elif proc != "vectorizer":
            raise ValueError("only vectorizers supported now")
        else:
            matrices = []
            indices = []
            for i, feature in enumerate(featurelist):
                vectorizer = self.vectorizers[feature]
                mat = vectorizer.transform(df[feature])
                matrices.append(mat)
                indices.append(mat.shape[1])
            return matrices, indices

    def find_idx(
            self,
            idx: int
            ) -> int:
        for key, (start, length) in self.feature_idx.items():
            if start <= idx < (start + length):
                break
        else:
            raise ValueError(f"index {idx} too big")
        vectorizer = self.vectorizers[key]
        if vectorizer is None:
            return key
        else:
            features = vectorizer.get_feature_names_out()
            return features[idx - start]
