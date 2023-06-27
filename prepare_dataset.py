#!/usr/bin/env python3
import os
import argparse
import pickle
from typing import Union

import pandas as pd

from text_authorship.ta_data_extraction.data_extraction import extract_df
from text_authorship.ta_model.data_preparation import TATransformer, load_df


def prepare_dataset(data: str,
                    output_dir: str,
                    parser: Union[str, None] = None,
                    symbol_lim: int = 3000):
    if not os.path.exists(data):
        raise ValueError("provided data_dir does not exist")
    if os.path.isdir(data):
        print("READING DATA FROM WRITERS DIRECTORY")
        df = extract_df(data, symbol_lim=symbol_lim)
    else:
        print("READING DATA FROM EXISTING DF")
        columns = ["author", "book", "text"]
        df = load_df(data, columns=columns).iloc[:50]
    transformer = TATransformer(parser=parser)
    print("FINISHED READING, PREPARING FEATURES")
    df = transformer.transform(df)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "prepared_df.csv")
    df.to_csv(output_path)
    print(f"DATASET SAVED AT: {output_path}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", help="root folder of all books", required=True)
    argparser.add_argument("--output_dir", help="path so save prepared dataframe", required=True)
    argparser.add_argument("--parser", help="select specific parser", default=None)
    argparser.add_argument("--symbol_lim", help="item text size in symbols", type=int, default=3000)
    args = argparser.parse_args()
    prepare_dataset(args.data, args.output_dir, parser=args.parser, symbol_lim=args.symbol_lim)
