#!/usr/bin/env python3
if __name__ == "__main__":
    import argparse
    import pickle
    from text_authorship.ta_model.data_preparation import load_df
    from text_authorship.ta_model.base_models import train_logreg, train_stacking

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prepared_data", help="source of prepared dataset", default=None)
    argparser.add_argument("--model", help="which model to train", default=None)
    argparser.add_argument("--pickle", help="file to pickle", default=None)
    args = argparser.parse_args()

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
    print(f"STARTED TRAINING {model_name}")
    model = train_func(df)
    print("TRAINING FINISHED, SAVING MODEL")
    if args.pickle:
        with open(args.pickle, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"SAVED MODEL IN {args.pickle}")