#!/usr/bin/env python3
import os
import argparse
import pickle

import pandas as pd

from text_authorship.ta_data_extraction.data_extraction import extract_df
from text_authorship.ta_model.data_preparation import TATransformer, load_df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", help="root folder of all books", required=True)
    argparser.add_argument("--output_dir", help="path so save prepared dataframe", required=True)
    argparser.add_argument("--pickle", help="file to pickle transformer", default=None)
    argparser.add_argument("--parser", help="select specific parser", default=None)
    args = argparser.parse_args()
    if not os.path.exists(args.data):
        raise ValueError("provided data_dir does not exist")
    if os.path.isdir(args.data):
        print("READING DATA FROM WRITERS DIRECTORY")
        df = extract_df(args.data)
    else:
        print("READING DATA FROM EXISTING DF")
        columns = ["author", "book", "text"]
        df = load_df(args.data, columns=columns).iloc[:50]
    transformer = TATransformer(parser=args.parser).fit(df)
    print("FINISHED READING, PREPARING FEATURES")
    df = transformer.transform(df)
    output_path = os.path.join(args.output_dir, "prepared_df.csv")
    df.to_csv(output_path)
    print(f"DATASET SAVED AT: {output_path}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args = argparser.parse_args()
    if args.pickle:
        with open(args.pickle, 'wb') as f:
            pickle.dump(transformer, f, protocol=pickle.HIGHEST_PROTOCOL)
