#!/usr/bin/env python3
if __name__ == "__main__":
    import os
    import argparse
    import pickle
    from text_authorship.ta_data_extraction.data_extraction import extract_df
    from text_authorship.ta_model.data_preparation import TATransformer

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", help="root folder of all books", required=True)
    argparser.add_argument("--output_dir", help="path so save prepared dataframe", required=True)
    argparser.add_argument("--pickle", help="file to pickle transformer", default=None)
    args = argparser.parse_args()
    if not os.path.exists(args.data_dir):
        raise ValueError("provided data_dir does not exist")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args = argparser.parse_args()
    print("READING_DATA")
    df = extract_df(args.data_dir)
    print("FINISHED READING, PREPARING FEATUERES")
    transformer = TATransformer().fit(df)
    df = transformer.transform(df)
    output_path = os.path.join(args.output_dir, "prepared_df.csv")
    df.to_csv(output_path)

    if args.pickle:
        with open(args.pickle, 'wb') as f:
            pickle.dump(transformer, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(f"DATASET SAVED AT: {output_path}")
