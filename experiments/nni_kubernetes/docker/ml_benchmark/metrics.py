import os
from datetime import datetime, timedelta
from uuid import uuid4
import socket
from dataclasses import dataclass, asdict


@dataclass
class Metric:
    """
    Metric Parentclass. Creates a unique identifier for every metric and gathers basic information.
    """
    process_id = os.getpid()
    hostname = socket.gethostname()
    metric_id = f"id_{uuid4()}__pid_{process_id}__hostname_{hostname}"

    def add_to_id(self, id_addition):
        self.metric_id = self.metric_id + f"__{id_addition}"

    def to_dict(self):
        return asdict(self)


class Latency(Metric):

    def __init__(self, func) -> None:
        """The latency of provided the function gives the duration of the start_time and end_time of the provided
        function.
        The id is generated from:
            * the process the function is running on, the name of the function
            * the object which the function is a class method of
            * a randomly generated number, to maintain uniqueness of the id.

        The object hash shall identify a single trail as it is assumed that a trail used 'train' and 'test'
        from the same Objective object.

        Args:
            func (_type_): _description_

        Raises:
            AttributeError: _description_
        """
        super().__init__()
        self.function_name: str = func.__name__
        try:
            self.obj_hash = hash(func.__self__)
        except AttributeError as e:
            print(e)
            raise AttributeError("Functions need to be part of a class in order to measure their latency.")
        self.add_to_id(f"function-name_{self.function_name}__objHash_{self.obj_hash}")
        self.start_time: float = None
        self.end_time: float = None
        self.duration_sec: float = None

    def to_dict(self):
        latency_dict = dict(
            metric_id=self.metric_id,
            function_name=self.function_name,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_sec=self.duration_sec
        )
        # latency_dict.update(super().to_dict())
        latency_dict = {key: self._convert_times_to_float(value) for key, value in latency_dict.items()}

        return latency_dict

    def __enter__(self):
        """Entering point of the context manager.

        Returns:
            _type_: _description_
        """
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        """
        Exit point of the context manager.
        """
        self.end_time = datetime.now()
        self._calculate_duration()

    def _calculate_duration(self):
        self.duration_sec = self.end_time - self.start_time

    def _convert_times_to_float(self, value):
        if isinstance(value, timedelta):
            return value.total_seconds()
        else:
            return str(value)
