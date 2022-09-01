import os
from abc import ABC, abstractmethod

import psycopg2
from sqlalchemy import MetaData, Table, create_engine, insert


from ml_benchmark.config import MetricsStorageConfig
from ml_benchmark.metrics import Latency


class Tracker(ABC):

    @abstractmethod
    def track(self, metric):
        pass


class LatencyTracker(Tracker):
    """
    A Tracker that establishes a connection to the Latency Table in the postgres database. The LatencyTracker
    is used to write a latency into the postgres database.

    THe Adress of the MetricsStorage has to be known and written to the ENV Variable "METRICS_STORAGE_ADRESS",
    otherwise the tracker will assumed that the tracking is performed on the same environment as the
    benchmark is running on.

    Args:
        Tracker (_type_): _description_
    """

    def __init__(self, connection_string: str = None) -> None:
        if connection_string:
            self.connection_string = connection_string
        else:
            self.connection_string = self._get_connection_string()
        if self.connection_string:
            self.engine = self._create_engine(self.connection_string)
        print(f"------------------ {self.connection_string} -----------------")

    def _create_engine(self, connection_string):
        try:
            engine = create_engine(connection_string, echo=True)
        except psycopg2.Error:
            raise ConnectionError("Could not create an Engine for the Postgres DB.")
        return engine

    def track(self, latency_obj):
        """
        Records a latency by writing it into a postgres database. First the table is retrieved, afterwards
        a connection is established and an INSERT statement is executed to save the results.

        Args:
            latency_obj (_type_): _description_
        """
        try:
            # TODO: test this shit
            metadata = MetaData(bind=self.engine)
            latency = Table("latency", metadata, autoload_with=self.engine)
            with self.engine.connect() as conn:
                stmt = insert(latency).values(latency_obj.to_dict())
                conn.execute(stmt)
            print(f"Latency Recorded! Latency: {latency_obj.to_dict()}")
        except Exception as e:
            print(f"Failed to record latency: {latency_obj.to_dict()}",e)
            

    def _get_connection_string(self):
        # XXX: list order is implicitly a priority
        connection_string_actions_registry = [
            ("env", os.environ.get("METRICS_STORAGE_HOST", None))
        ]
        for method, value in connection_string_actions_registry:
            if value:
                print(f"Tracker Connection String retrieved from: {method} using {value}")
                return self.shape_connection_string(value)
        print("No Method was succsessful. Setting Tracker URL to current Host.")
        return MetricsStorageConfig.connection_string

    def shape_connection_string(self, host):
        user = MetricsStorageConfig.user
        password = MetricsStorageConfig.password
        port = MetricsStorageConfig.port
        db = MetricsStorageConfig.db
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def latency_decorator(func):
    """A Decorator to record the latency of the decorated function. Once it is recorded the LatencyTracker
    writes the result into the postgres databse.

    Decorators overwrite a decorated function once the code is passed to the compier

    Args:
        func (_type_): _description_
    """
    def latency_func(*args, **kwargs):
        func.__self__ = args[0]
        with Latency(func) as latency:
            result = func(*args, **kwargs)
        latency_tracker = LatencyTracker()
        latency_tracker.track(latency)
        func.__self__ = None
        return result
    return latency_func


if __name__ == "__main__":

    import json

    import docker

    from ml_benchmark.metrics_storage import MetricsStorage

    storage = MetricsStorage()
    result = []

    class Test:
        metrics_storage_address = MetricsStorage.connection_string

        def __init__(self) -> None:
            pass

        @latency_decorator
        def a(self, b):
            return b

        @latency_decorator
        def c(self, b):
            return b

    try:
        storage.start_db()
        testclass = Test()
        testclass.a(2)
        testclass.c(2)
        result = storage.get_benchmark_results()
        storage.stop_db()
    except docker.errors.APIError:
        storage.stop_db()
    print(json.dumps(result))
