import logging
from fastapi import Request


def log_conection(logger_name: str, id: int, request: Request):
    logging.getLogger(logger_name).info(f'session {id}: request to {request.url.path} from {request.client.host}')


def log_duration(logger_name: str, id: int, duration: float):
    logging.getLogger(logger_name).info(f'session {id}: spend {duration:.2f}ms on request')