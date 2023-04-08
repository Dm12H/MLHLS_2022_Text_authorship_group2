import logging
import logging.config
from fastapi import Request
import time
from .config import get_settings
from contextlib import contextmanager
from contextlib import asynccontextmanager
from uuid import UUID
from typing import Union, Callable, Type, Any


LogFunc = Callable[[str, BaseException], None]

class LogManager:
    def __init__(self, logger: logging.Logger, request_id: Union[UUID, None] = None):
        self._logger = logger
        self._request_id = request_id
        self.duration = 0

    def __enter__(self):
        self.__start = time.time()
        return self
    
    def __exit__(self, 
                 exc_type: Union[Type[BaseException], None], 
                 exc_value: Union[BaseException, None], 
                 _):
        self.duration = 1000 * (time.time() - self.__start)
        if exc_type:
            log_msg = f'exception of type {exc_type} has occured'
            self.log_func(log_msg, exc_value)

    def __log_error(self, msg: str, exc_value: BaseException) -> None:
        self._logger.error(f'request {self._request_id}: {msg}', exc_info=exc_value)

    def __log_critical(self, msg: str, exc_value: BaseException) -> None:
        self._logger.critical(f'server: {msg}', exc_info=exc_value)

    @property
    def log_func(self) -> LogFunc:
        if not self._request_id:
            return self.__log_critical
        return self.__log_error
    
    @property
    def duration(self) -> float:
        return self.__duration
    
    @duration.setter
    def duration(self, val: float):
        self.__duration = val


@contextmanager
def log_server_startup(logger: logging.Logger):
    logger.info('server starting...')
    with LogManager(logger) as lm:
        yield
    logger.info(f'server started after {lm.duration:.2f}ms')


@asynccontextmanager
async def log_request(logger: logging.Logger, id: UUID, request: Request):
    logger.info(f'request {id}: request to {request.url.path} from {request.client.host}')
    with LogManager(logger, request_id=id) as lm:
        yield
    logger.info(f'request {id}: spend {lm.duration:.2f}ms on request')


@contextmanager
def log_model_load(logger: logging.Logger, name: str, path: str):
    logger.info(f'loading model {name} from {path}...')
    with LogManager(logger) as lm:
        yield
    logger.info(f'loaded model {name} after {lm.duration:.2f}ms')


@contextmanager
def log_transformer_load(logger: logging.Logger, path: str):
    logger.info(f'loading transformer from {path}...')
    with LogManager(logger) as lm:
        yield
    logger.info(f'loaded transformer after {lm.duration:.2f}ms')


@contextmanager
def log_transform(logger: logging.Logger, id: UUID):
    logger.info(f'request {id}: transforming text...')
    with LogManager(logger, request_id=id) as lm:
        yield
    logger.info(f'request {id}: text transformed after {lm.duration:.2f}ms')


@contextmanager
def log_evaluating(logger: logging.Logger, id: UUID):
    logger.info(f'request {id}: model evaluating...')
    with LogManager(logger, request_id=id) as lm:
        yield
    logger.info(f'request {id}: model finished after {lm.duration:.2f}ms')


def set_logs():
    settings = get_settings()
    logging.config.dictConfig(settings.log_config)