from fastapi import Depends, Form
from typing import Any, Annotated
from ..config import get_settings
from ..logs import log_model_load, log_transformer_load
import pickle
import logging
from text_authorship.ta_model.stacking import TASTack2Deploy
from text_authorship.ta_model.logreg import LogregModel
from text_authorship.ta_model.inference_bert import InferenceBert
from text_authorship.ta_model.data_preparation import TATransformer


logger = logging.getLogger(__name__)


class ModelHolder:
    __models: dict[str, Any] = dict()
    __transformer: Any = None

    @classmethod
    def load_model(cls, name: str, path: str):
        if name in cls.__models:
            return
        with log_model_load(logger, name=name, path=path):
            if name == 'bert':
                cls.__models[name] = InferenceBert(path)
            else:
                cls.__models[name] = pickle.load(open(path, 'rb'))

    @classmethod
    def load_transformer(cls, pkl_path: str):
        if cls.__transformer:
            return
        with log_transformer_load(logger, pkl_path):
            cls.__transformer = pickle.load(open(pkl_path, 'rb'))

    @classmethod
    def load_from_settings(cls):
        settings = get_settings()
        for name, path in settings.model_paths.items():
            cls.load_model(name, path)
        cls.load_transformer(settings.transformer_path)

    @classmethod
    def get_model(cls, name: str):
        return cls.__models[name]
    
    @classmethod
    def get_transformer(cls):
        return cls.__transformer


async def get_model(model: Annotated[str, Form()]) -> Any:
    model = ModelHolder.get_model(model)
    return model


async def get_transformer() -> Any:
    transformer = ModelHolder.get_transformer()
    return transformer


ModelDep = Annotated[Any, Depends(get_model)]
TransformDep = Annotated[Any, Depends(get_transformer)]