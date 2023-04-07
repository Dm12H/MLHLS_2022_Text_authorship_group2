import logging
import logging.config
from fastapi import Request, FastAPI
import time
from .config import get_settings
from contextlib import contextmanager
from contextlib import asynccontextmanager
from uuid import UUID


class Duration:
    def __init__(self):
        self.duration: float = 0

    def __enter__(self):
        self.__start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.duration = 1000 * (time.time() - self.__start)


@contextmanager
def log_server_startup(logger: logging.Logger):
    logger.info('server starting...')
    with Duration() as d:
        yield
    logger.info(f'server started after {d.duration:.2f}ms')


@asynccontextmanager
async def log_request(logger: logging.Logger, id: UUID, request: Request):
    logger.info(f'request {id}: request to {request.url.path} from {request.client.host}')
    with Duration() as d:
        yield
    logger.info(f'request {id}: spend {d.duration:.2f}ms on request')


@contextmanager
def log_model_load(logger: logging.Logger, name: str, path: str):
    logger.info(f'loading model {name} from {path}...')
    with Duration() as d:
        yield
    logger.info(f'loaded model {name} after {d.duration:.2f}ms')


@contextmanager
def log_transformer_load(logger: logging.Logger, path: str):
    logger.info(f'loading transformer from {path}...')
    with Duration() as d:
        yield
    logger.info(f'loaded transformer after {d.duration:.2f}ms')


@contextmanager
def log_transform(logger: logging.Logger, id: UUID):
    logger.info(f'request {id}: transforming text...')
    with Duration() as d:
        yield
    logger.info(f'request {id}: text transformed after {d.duration:.2f}ms')


@contextmanager
def log_evaluating(logger: logging.Logger, id: UUID):
    logger.info(f'request {id}: model evaluating...')
    with Duration() as d:
        yield
    logger.info(f'request {id}: model finished after {d.duration:.2f}ms')


def set_logs():
    settings = get_settings()
    logging.config.dictConfig(settings.log_config)