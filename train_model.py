#!/usr/bin/env python3
import argparse
import pickle
from text_authorship.ta_model.data_preparation import load_df
from text_authorship.ta_model.base_models import train_logreg, train_stacking
from typing import Union


def train_model(prepared_data: Union[str, None] = None,
                model: Union[str, None] = None,
                pkl: Union[str, None] = None):
    if prepared_data:
        print("LOADING DATA")
        df = load_df(prepared_data)
        print("DATA LOADED")
    else:
        raise ValueError("must provide prepared dataset ")

    if model == "logreg":
        model_name = "logistic regression model"
        train_func = train_logreg
    elif model == "stacking":
        model_name = "stacking model"
        train_func = train_stacking
    else:
        raise ValueError("valid model options are 'logreg' and 'stacking' ")
    print(f"STARTED TRAINING {model_name}")
    model = train_func(df)
    print("TRAINING FINISHED, SAVING MODEL")
    if pkl:
        with open(pkl, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"SAVED MODEL IN {pkl}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prepared_data", help="source of prepared dataset", default=None)
    argparser.add_argument("--model", help="which model to train", default=None)
    argparser.add_argument("--pickle", help="file to pickle", default=None)
    args = argparser.parse_args()
    train_model(prepared_data=args.prepared_data, model=args.model, pkl=args.pickle)