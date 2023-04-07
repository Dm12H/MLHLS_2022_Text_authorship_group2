from fastapi import Depends, Form
from typing import Any, Annotated
from ..config import SettingsDep
import pickle
import logging
from threading import Lock
from text_authorship.ta_model.stacking import TASTack2Deploy
from text_authorship.ta_model.data_preparation import TATransformer


logger = logging.getLogger(__name__)


class ModelHolder:
    __models: dict[str, Any] = dict()
    __transformer: Any = None
    __lock: Lock = Lock()

    @classmethod
    def get_model(cls, name: str, pkl_path: str):
        with cls.__lock:
            if name not in cls.__models:
                logger.info(f'loading model {name} from {pkl_path}')
                cls.__models[name] = pickle.load(open(pkl_path, 'rb'))
                logger.info(f'model {name} loaded')
        return cls.__models[name]
    
    @classmethod
    def get_transformer(cls, pkl_path: str):
        with cls.__lock:
            if cls.__transformer is None:
                logger.info(f'loading transformer from {pkl_path}')
                cls.__transformer = pickle.load(open(pkl_path, 'rb'))
                logger.info(f'transformer loaded')
        return cls.__transformer


async def get_model(model: Annotated[str, Form()], settings: SettingsDep) -> Any:
    model = ModelHolder.get_model(model, settings.model_paths[model])
    return model


async def get_transformer(settings: SettingsDep) -> Any:
    transformer = ModelHolder.get_transformer(settings.transformer_path)
    return transformer


ModelDep = Annotated[Any, Depends(get_model)]
TransformDep = Annotated[Any, Depends(get_transformer)]