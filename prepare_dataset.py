#!/usr/bin/env python3
import os
from text_authorship.ta_model.utils import DatasetArgumentParser
from text_authorship.ta_model.data_extraction import extract_df
from data_preparation.transformer import TATransformer

if __name__ == "__main__":
    args = DatasetArgumentParser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("READING_DATA")
    df = extract_df(args.data_dir)
    print("FINISHED READING, PREPARING FEATUERES")

    transformer = TATransformer().fit(df)
    df = transformer.transform(df)
    output_path = os.path.join(args.output_dir, "prepared_df.csv")

    df.to_csv(output_path)
    print(f"DATASET SAVED AT: {output_path}")
