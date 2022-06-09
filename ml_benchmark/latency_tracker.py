import os
from datetime import datetime, timedelta


class LatencyTracker:

    def __init__(self) -> None:
        self.recorded_latencies = []
        self.metric_pesistor = None

    def track(self, latency):
        # implement tracking routine for postgres
        self.recorded_latencies.append(latency)

    def get_recorded_latencies(self):
        return [latency.to_dict() for latency in self.recorded_latencies]


# TODO: decorator for trial latencies?

class Latency:

    def __init__(self, name) -> None:
        self.name: str = f"{name}__pid_{os.getpid()}"
        self.start_time: float = None
        self.end_time: float = None
        self.duration: float = None

    def to_dict(self):
        latency_dict = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            duration=self.duration
        )
        latency_dict = {key: self._convert_times_to_float(value) for key, value in latency_dict.items()}

        return {
            self.name: latency_dict
        }

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()
        self._calculate_duration()

    def _calculate_duration(self):
        self.duration = self.end_time - self.start_time

    def _convert_times_to_float(self, value):
        if isinstance(value, timedelta):
            return value.total_seconds()
        else:
            return value.timestamp()


def latency_decorator(func):
    # TODO add adress argument to tracker? use it as decorator in objective. Automatically tracks into right db
    def latency_func(*args, **kwargs):
        with Latency(func.__name__) as latency:
            result = func(*args, **kwargs)
        result["latency"] = latency.to_dict()
        return result
    return latency_func


if __name__ == "__main__":

    def a(b):
        return b
    with Latency(a.__name__) as latency:
        a(2)
    print(latency.duration)
    print(latency.to_dict())
