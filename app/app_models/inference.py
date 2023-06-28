from typing import Any, List
import logging
import pandas as pd
from uuid import UUID
from ..logs import log_transform, log_evaluating


logger = logging.getLogger(__name__)


def predict_text(id: UUID, model: Any, transformer: Any, text: List[str]):
    text_df = pd.DataFrame({'text': text})
    with log_transform(logger, id):
        text_transformed: pd.DataFrame = transformer.transform(text_df)
    with log_evaluating(logger, id):
        predictions: pd.DataFrame = model.predict_proba(text_transformed)
    return predictions


def select_best_pred(probs: pd.Series):
    idx = probs.argmax()
    return probs.index[idx]
