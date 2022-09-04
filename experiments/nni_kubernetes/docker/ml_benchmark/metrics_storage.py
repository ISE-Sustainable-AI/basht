import time
import docker
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, select

from ml_benchmark.config import MetricsStorageConfig


class MetricsStorage:

    port = MetricsStorageConfig.port
    user = MetricsStorageConfig.user
    password = MetricsStorageConfig.password
    db = MetricsStorageConfig.db
    host = MetricsStorageConfig.host
    connection_string = MetricsStorageConfig.connection_string

    def __init__(self, connection_string: str = None) -> None:
        """
        The MetricsStorage serves as the representation of the databse.
        It sets up a postgres database in a docker container and creates tables for every recorded metric.

        The default address of the Storage is constructed by its default static variables.
        The address might be different for Objective running in a different execution environment.

        Args:
            connection_string (str, optional): _description_. Defaults to None.
        """
        if connection_string:
            self.connection_string = connection_string
        self.latency = None

    def start_db(self):
        self.setup_db()
        self.engine = create_engine(self.connection_string)
        self.create_metrics_table()
        self.create_resource_table()
        return self

    def setup_db(self):
        self.client = docker.from_env()
        self.client.containers.run(
            "postgres:14.1", detach=True,
            environment=[
                f"POSTGRES_PASSWORD={self.password}", f"POSTGRES_DB={self.db}", f"POSTGRES_USER={self.user}"],
            ports={f'{self.port}/tcp': self.port},
            name="postgres",
            remove=True)
        container = self.client.containers.get("postgres")
        # checks if db is up
        while "accepting connections" not in container.exec_run("pg_isready").output.decode():
            time.sleep(2)
        print("DB-Container Running")

    def stop_db(self):
        container = self.client.containers.get("postgres")
        container.stop()

    def create_metrics_table(self):
        """
        Creates all tables in the postges database to record the Metrics form a Benchmark.
        """
        self.meta = MetaData()
        self.create_latency_table()
        self.create_resource_table()
        self.create_classification_metrics_table()
        self.meta.create_all(self.engine)

    def create_latency_table(self):
        self.latency = Table(
            "latency", self.meta,
            Column("metric_id", String, primary_key=True),
            Column("function_name", String),
            Column("start_time", String),
            Column("end_time", String),
            Column("duration_sec", Float)
        )

    def create_resource_table(self):
        pass

    def create_classification_metrics_table(self):
        pass

    def get_benchmark_results(self):
        latency = self.get_latency_results()
        resources = self.get_resource_results()
        classification = self.get_classification_results()
        return dict(latency=latency, resources=resources, classification=classification)

    def get_latency_results(self):
        result_list = []
        with self.engine.connect() as conn:
            stmt = select(self.latency)
            cursor = conn.execute(stmt)
        cursor = cursor.mappings().all()
        for row in cursor:
            result_list.append(dict(row))
        return result_list

    def get_resource_results(self):
        pass

    def get_classification_results(self):
        pass
