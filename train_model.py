#!/usr/bin/env python3

from text_authorship.ta_model import load_df
from text_authorship.ta_model.base_models import train_logreg, train_stacking
from text_authorship.ta_model.utils import TrainingArgumentParser


if __name__ == "__main__":
    args = TrainingArgumentParser().parse_args()

    if args.prepared_data:
        print("LOADING DATA")
        df = load_df(args.prepared_data)
        print("DATA LOADED")
    else:
        raise ValueError("must provide prepared dataset ")

    if args.model == "logreg":
        model_name = "logistic regression model"
        train_func = train_logreg
    elif args.model == "stacking":
        model_name = "stacking model"
        train_func = train_stacking
    else:
        raise ValueError("valid model options are 'logreg' and 'stacking' ")

    print(f"TRAINING {model_name}")
    model, eval_metric = train_func(df)
    print(f"F1-macro score: {eval_metric:.03f}")
