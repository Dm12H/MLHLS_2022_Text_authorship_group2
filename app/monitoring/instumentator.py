from prometheus_fastapi_instrumentator import Instrumentator, metrics
from .my_metrics import memory_usage_total


def create_instrumentator():
    instrumentator = Instrumentator()
    instrumentator.add(memory_usage_total())
    instrumentator.add(metrics.requests())
    instrumentator.add(metrics.latency())
    return instrumentator
