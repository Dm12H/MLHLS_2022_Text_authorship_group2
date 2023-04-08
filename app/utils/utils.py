from app.app_models.inference import select_best_pred
import pandas as pd


def get_sorted_predictions(df: pd.DataFrame):
    single_row = df.iloc[0]
    probabilities = single_row.sort_values(ascending=False)
    predictions = {author: val
                   for author, val
                   in zip(probabilities.index, probabilities)}
    return predictions
