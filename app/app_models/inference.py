from typing import Any
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def predict_text(model: Any, transformer: Any, text: str):
    text_df = pd.DataFrame({'text': [text]})
    logger.info('transforming text')
    text_transformed: pd.DataFrame = transformer.transform(text_df)
    logger.info('text transformed')
    logger.info('model evaluating')
    author: str = model.predict(text_transformed)[0]
    logger.info('model finished')
    return author
