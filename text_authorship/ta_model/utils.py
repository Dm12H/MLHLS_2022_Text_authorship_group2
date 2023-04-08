import os
from argparse import ArgumentParser, Namespace
from typing import Sequence, Set


class DatasetArgumentParser(ArgumentParser):
    def __init__(self):
        super().__init__()
        super().add_argument("--data_dir",
                             help="root folder of all books",
                             required=True)
        super().add_argument("--output_dir",
                             help="path so save prepared dataframe",
                             required=True)

    def parse_args(self, args: Sequence[str] | None = ...) -> Namespace:
        args = super().parse_args()
        if not os.path.exists(args.data_dir):
            raise ValueError("provided data_dir does not exist")
        return args


class TrainingArgumentParser(ArgumentParser):
    def __init__(self):
        super().__init__()
        super().add_argument("--prepared_data",
                             help="source of prepared dataset",
                             default=None)
        super().add_argument("--model",
                             help="which model to train",
                             default=None)

    def parse_args(self, args: Sequence[str] | None = ...) -> Namespace:
        args = super().parse_args()
        if not os.path.exists(args.prepared_data):
            raise ValueError("must provide prepared dataset ")
        return args


def _strip_str(s: str, chars: Set[str]) -> str:
    i, j = 0, len(s)

    while i < j and s[i] in chars:
        i += 1

    while i < j and s[j - 1] in chars:
        j -= 1

    return s[i:j]
