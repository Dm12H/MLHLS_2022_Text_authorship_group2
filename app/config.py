from typing import Any, Annotated
from fastapi import Depends
from pydantic import BaseSettings
import yaml
from functools import lru_cache
import logging.config


def load_yaml_settings(settings: BaseSettings) -> dict[str, Any]:
    path = getattr(settings.__config__, "yaml_settings_path", None)
    if path is None:
        return dict()
    encoding = settings.__config__.env_file_encoding
    
    with open(path, 'r', encoding=encoding) as f:
        yaml_settings = yaml.safe_load(f.read())

    return yaml_settings


class ModelConfig(BaseSettings):
    transformer_path: str
    model_paths: dict[str, str]
    log_config: dict[str, Any]

    class Config:
        yaml_settings_path = 'settings.yml'

        @classmethod
        def customise_sources(
                cls,
                init_settings,
                env_settings,
                file_secret_settings,
        ):
            return (
                init_settings,
                load_yaml_settings,
                env_settings,
                file_secret_settings,
            )


@lru_cache(maxsize=1)
def get_settings() -> ModelConfig:
    return ModelConfig()


def set_logs():
    settings = get_settings()
    logging.config.dictConfig(settings.log_config)


def get_model_names() -> list[str]:
    settings = get_settings()
    models = list(settings.model_paths.keys())
    return models


SettingsDep = Annotated[ModelConfig, Depends(get_settings)]