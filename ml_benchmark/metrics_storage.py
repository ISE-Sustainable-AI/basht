from abc import abstractmethod
import logging
import time
import docker
from docker.errors import APIError
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, select, Integer, insert, BigInteger
import psycopg2
import os 

from ml_benchmark.config import MetricsStorageConfig
from ml_benchmark.metrics import Metric

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
        logging.basicConfig()
        logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
        logging.getLogger('sqlalchemy.engine').setLevel(logging.ERROR)

        self.meta = None
        self.client = None
        self.engine = None
        if connection_string:
            self.connection_string = connection_string
        self.latency = None
        self.resources = None

    def start_db(self):
        self.setup_db()
        self.engine = create_engine(self.connection_string)
        self.create_metrics_table()
        return self

    def setup_db(self):
        self.client = docker.from_env()
        try:
            self.client.containers.run(
                "postgres:14.1", detach=True,
                environment=[
                    f"POSTGRES_PASSWORD={self.password}", f"POSTGRES_DB={self.db}", f"POSTGRES_USER={self.user}"],
                ports={f'{self.port}/tcp': self.port},
                name="postgres",
                remove=True)
        except APIError as e:
            if e.status_code == 409:
                #TODO: we maybe want to drop the database in these cases
                logging.info("Postgres is already running")
            else:
                raise e

        container = self.client.containers.get("postgres")
        # checks if db is up
        while "accepting connections" not in container.exec_run("pg_isready").output.decode():
            time.sleep(2)
            #TODO: should have a timeout condition
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
        self.meta.create_all(self.engine,checkfirst=True)
        

    def create_latency_table(self):
        self.latency = Table(
            "latency", self.meta,
            Column("metric_id", String, primary_key=True),
            Column("function_name", String),
            Column("start_time", String),
            Column("end_time", String),
            Column("duration_sec", Float)
            #TODO add fingerprint
        )

    def create_resource_table(self):
        self.resources = Table(
            "resources", self.meta,
            Column("metric_id", String, primary_key=True),
            Column("timestamp", String, primary_key=True),
            Column("cpu_usage", Float),
            Column("memory_usage", Float),
            Column("network_usage", Float),
            Column("accelerator_usage", Float),
            Column("wattage", Float),
            Column("processes", Integer),
        )

    def create_classification_metrics_table(self):
        self.classification_metrics = Table(
            "classification_metrics", self.meta,
            Column("metric_id", String, primary_key=True),
            Column("timestamp", String, primary_key=True),
            Column("value", Float),
            Column("measure", String),
            Column("hyperparameters", String),
            Column("classification_metrics", String),
            Column("process_id", Integer, nullable=True),
            Column("hostname", String),
            Column("obj_hash", BigInteger, nullable=True),
        )

    def get_benchmark_results(self):
        latency = self.get_latency_results()
        resources = self.get_resource_results()
        classification = self.get_classification_results()
        return dict(latency=latency, resources=resources, classification=classification)
    
    def _get_table_results(self,table):
        result_list = []
        with self.engine.connect() as conn:
            stmt = select(table)
            cursor = conn.execute(stmt)
        cursor = cursor.mappings().all()
        for row in cursor:
            result_list.append(dict(row))
        return result_list

    def get_latency_results(self):
        return self._get_table_results(self.latency)

    def get_resource_results(self):
        return self._get_table_results(self.resources)

    def get_classification_results(self):
        return self._get_table_results(self.classification_metrics)


class StoreStrategy(object):
    """
        Interface for swapping out different implementations of the resource store, e.g., a database, a file, etc.
    """
    @abstractmethod
    def setup(self, **kwargs):
        """
            Setup the resource store, e.g., create a database connection.
        """
        pass
    
    @abstractmethod
    def store(self, node_usage:Metric, **kwargs):
        """
            Store the node usage in the resource store.
        """
        pass

#global store engine used as a singleton to safe 
engine=None

class MetricsStorageStrategy(StoreStrategy):

    def __init__(self):
        self.engine = None

    def setup(self, **kwargs):
        if self.engine:
            return 

        #resue the global engine if it exists
        # global engine
        # if engine:
        #     self.engine = engine
            
        self.engine = self._create_engine(**kwargs)
        # engine = self.engine

    def _get_connection_string(self, **kwargs):
        # XXX: list order is implicitly a priority
        connection_string_actions_registry = [
            ("env", os.environ.get("METRICS_STORAGE_HOST", None)),
            ("args",kwargs.get("connection_string",None))
        ]
        for method, value in connection_string_actions_registry:
            if value:
                logging.debug(f"Tracker Connection String retrieved from: {method} using {value}")
                return self.shape_connection_string(value)
        logging.warn("No Method was succsessful. Setting Tracker URL to current Host.")
        return MetricsStorageConfig.connection_string
    
    def shape_connection_string(self, host):
        user = MetricsStorageConfig.user
        password = MetricsStorageConfig.password
        port = MetricsStorageConfig.port
        db = MetricsStorageConfig.db
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
        
    def _create_engine(self, **kwargs):
        connection_string = self._get_connection_string(**kwargs)
        try:
            engine = create_engine(connection_string, echo=False)
        except psycopg2.Error:
            raise ConnectionError("Could not create an Engine for the Postgres DB.")
        return engine

    def store(self, data:Metric, **kwargs):
        try:
            metadata = MetaData(bind=self.engine)
            node_usage = Table(kwargs.get("table_name","metrics"), metadata, autoload_with=self.engine)
            with self.engine.connect() as conn:
                stmt = insert(node_usage).values(data.to_dict())
                conn.execute(stmt)
        except Exception as e:
            logging.warn(f"Could not store the data in the Metrics DB {data} - {e}")

class LoggingStoreStrategy(StoreStrategy):

    def __init__(self):
        self.log = []

    def setup(self, **kwargs):
        pass

    def store(self, data,**kwargs):
        logging.info("Storing data: {}".format(data.to_dict()))
        self.log.append(data)
