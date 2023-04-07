from typing import Any
import logging
import pandas as pd
from uuid import UUID
from ..logs import log_transform, log_evaluating


logger = logging.getLogger(__name__)


def predict_text(id: UUID, model: Any, transformer: Any, text: str):
    text_df = pd.DataFrame({'text': [text]})
    with log_transform(logger, id):
        text_transformed: pd.DataFrame = transformer.transform(text_df)
    with log_evaluating(logger, id):
        author: str = model.predict(text_transformed)[0]
    return author
