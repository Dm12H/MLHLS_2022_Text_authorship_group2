from resource import getrusage, RUSAGE_SELF
from typing import Callable
from prometheus_fastapi_instrumentator.metrics import Info
from prometheus_client import Gauge


def memory_usage_total() -> Callable[[Info], None]:
    METRIC = Gauge("memory_usage", "used memory in mb",
                   labelnames=("memory_used",),
                   unit="MB")

    def instrumentation(info: Info) -> None:
        used_mb = getrusage(RUSAGE_SELF).ru_maxrss // 1024
        print(used_mb)
        METRIC.labels("memory_used").set(used_mb)

    return instrumentation


def record_metric():
    ANSWER_COUNTER = Gauge("answers",
                           "right/wrong anwser counts",
                           labelnames=("answer_counts",))
    def inner():
        return ANSWER_COUNTER
    return inner
