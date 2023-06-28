#!/usr/bin/env python3
import os
import argparse
import sys
from typing import Union


from text_authorship.ta_data_extraction.data_extraction import extract_df
from text_authorship.ta_model.data_preparation import TATransformer, load_df


def prepare_dataset(data: str,
                    output_dir: str,
                    parser: Union[str, None] = None,
                    symbol_lim: int = 3000):
    if __name__ == "__main__":
        f_out = sys.stdout
    else:
        f_out = open(os.devnull, "w")
    if not os.path.exists(data):
        raise ValueError("provided data_dir does not exist")
    if os.path.isdir(data):
        print("READING DATA FROM WRITERS DIRECTORY", file=f_out)
        df = extract_df(data, symbol_lim=symbol_lim)
    else:
        print("READING DATA FROM EXISTING DF", file=f_out)
        columns = ["author", "book", "text"]
        df = load_df(data, columns=columns).iloc[:50]
    transformer = TATransformer(parser=parser)
    print("FINISHED READING, PREPARING FEATURES", file=f_out)
    df = transformer.transform(df)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "prepared_df.csv")
    df.to_csv(output_path)
    print(f"DATASET SAVED AT: {output_path}", file=f_out)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", help="root folder of all books", required=True)
    argparser.add_argument("--output_dir", help="path so save prepared dataframe", required=True)
    argparser.add_argument("--parser", help="select specific parser", default=None)
    argparser.add_argument("--symbol_lim", help="item text size in symbols", type=int, default=3000)
    args = argparser.parse_args()
    prepare_dataset(args.data, args.output_dir, parser=args.parser, symbol_lim=args.symbol_lim)
