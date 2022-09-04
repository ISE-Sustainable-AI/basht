import inspect
import json
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from time import sleep

import docker
import numpy as np
import torch
import logging

from ml_benchmark.latency_tracker import LatencyTracker
from ml_benchmark.metrics_storage import MetricsStorage
from ml_benchmark.resource_tracker import ResourceTracker
from ml_benchmark.metrics import Latency

class Benchmark(ABC):
    """
    This class serves as an Interface for a benchmark. All neccessary methods have to be implemented in the
    subclass that is using the interface. Make sure to use the predefined static variables. Your benchmark
    will most likely not run properly if the variables value remains to be "None".
    """

    # TODO: objective and grid are not allowed to be in the benchmark
    resources = None

    @abstractmethod
    def deploy(self) -> None:
        """
            With the completion of this step the desired architecture of the HPO Framework should be running
            on a platform, e.g,. in the case of Kubernetes it referes to the steps nassary to deploy all pods
            and services in kubernetes.
        """
        pass

    @abstractmethod
    def setup(self):
        """
        Every Operation that is needed before the actual optimization (trial) starts and that is not relevant
        for starting up workers or the necessary architecture.
        """
        pass

    @abstractmethod
    def run(self):
        """
            Executing the hyperparameter optimization on the deployed platfrom.
            use the metrics object to collect and store all measurments on the workers.
        """
        pass

    @abstractmethod
    def collect_run_results(self):
        """
        This step collects all necessary results from all performed trials. Necessary results are results that
        are used in order to retrieve the best hyperparameter setting and to collect benchmark metrics.
        """
        pass

    @abstractmethod
    def test(self):
        """
        This step tests the model instantiated with the best hyperparameter setting on the test split of the
        provided task.
        """
        pass

    @abstractmethod
    def collect_benchmark_metrics(self):
        """
            Describes the collection of all gathered metrics, which are not used by the HPO framework
            (Latencies, CPU Resources, etc.). This step runs outside of the HPO Framework.
            Ensure to optain all metrics loggs and combine into the metrics object.

            This function needs to RETURN all gathered metrics.
        """
        pass

    @abstractmethod
    def undeploy(self):
        # TODO: might be moved before collecting all metrics
        """
            The clean-up procedure to undeploy all components of the HPO Framework that were deployed in the
            Deploy step.
        """
        pass


class BenchmarkRunner():

    def __init__(
            self, benchmark_cls: Benchmark,
            resources: dict) -> None:
        """
        This class runs a Benchmark.
        It is responsibile for setting up everything that is needed upfront to run the benchmark and manages
        recording and saving of benchmark results. It aswell records the Latency of every Step of an
        object that inherits the Benchmark ABC.
        Before a Benchmark is run seeds are set to ensure identical results for every probabilistic
        interference.

        On initialization the BenchmarkRunner creates a folder to store results. Benchmark run on tasks, which
        can be varied. Data and necessary static configurations, that do not affect the Benchmark are loaded
        with the task.

        Args:
            benchmark_cls (Benchmark): _description_
            config (dict): _description_
            grid (dict): _description_
            resources (dict): _description_
            task_str (str, optional): _description_. Defaults to "mnist".
        """
        # TODO: add a benchmark validator, which checks if things are correctly defined e.g.: helper functions only: "_functionname"
        # generate a unique name from the config
        self.rundate = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        benchmark_path = os.path.abspath(os.path.dirname(inspect.getabsfile(benchmark_cls)))
        self.bench_name = f"{benchmark_cls.__name__}"
        self.bench_goal = resources.get("goal", "debug")
        self.benchmark_folder = os.path.join(benchmark_path, f"benchmark__{self.bench_name}")
        self.create_benchmark_folder(self.benchmark_folder)
        self.resources = resources

        # add input and output size to the benchmark.
        self.benchmark = benchmark_cls(resources)

        # set seeds
        self._set_all_seeds()

        # prepare tracker
        self.metrics_storage = MetricsStorage()
        self.latency_tracker = LatencyTracker(MetricsStorage.connection_string)
        if "prometheus_url" in resources:
            self.resource_tracker = ResourceTracker(resources["prometheus_url"])
        else:
            logging.warning("No Prometheus URL provided. Resource Tracker will not be used.")
            self.resource_tracker = None

    def run(self):
        """
        Runs all functions of a Benchmark and records its latencies. Saves the results afterwards
        in a predefined folder.

        Raises:
            ValueError: _description_
        """
        benchmark_results = None

        try:
            self.metrics_storage.start_db()

            # Deploy the SUT
            with Latency(self.benchmark.deploy) as latency:
                self.benchmark.deploy()
            self.latency_tracker.track(latency)

            # RUN the benchmark
            run_process = [
                self.benchmark.setup, self.benchmark.run,
                self.benchmark.collect_run_results,
                self.benchmark.test, self.benchmark.collect_benchmark_metrics]

            if self.resource_tracker is not None:
                self.resource_tracker.start()

            for benchmark_fun in run_process:
                with Latency(benchmark_fun) as latency:
                    benchmark_fun()
                self.latency_tracker.track(latency)

            # Get the results of the benchmark
            benchmark_results = self.metrics_storage.get_benchmark_results()

            # just to be save we wait a bit before killing shit.

        except (docker.errors.APIError, AttributeError, ValueError, RuntimeError) as e:
            print(e)
            raise ValueError("No Results obtained, Benchmark failed.")
        finally:
            sleep(5)
            if self.resource_tracker is not None:
                self.resource_tracker.stop()
                self.resource_tracker = None

            self.metrics_storage.stop_db()
            # Undeploy the SUT
            try:
                self.benchmark.undeploy()
            except Exception:
                pass

            try:
                self.metrics_storage.stop_db()
            except Exception:
                pass

        # TODO: move to finally block to ensure that results are always caputres if possible?
        # persist the results
        self.save_benchmark_results(benchmark_results)

    def _set_all_seeds(self):
        """
        Set all relevant seeds to exclude probabilistic interferences.
        """
        torch.manual_seed(1337)
        np.random.seed(1337)
        random.seed(1337)

    def save_benchmark_results(self, benchmark_results):
        """
        Save Benchmark Results in from of a Json. The json includes the configuration, the hyperparameter grid
        and the results of the benchmark. The grid may use custom functions of the hyperparameter framework,
        therefore saving can be troublesome.

        Args:
            benchmark_results (_type_): _description_
        """
        benchmark_config_dict = dict(
            resources=self.resources,
        )
        benchmark_result_dict = dict(
            benchmark_metrics=benchmark_results,
            benchmark_configuration=benchmark_config_dict
        )
        with open(
            os.path.join(
                self.benchmark_folder,
                f"benchmark_results__{self.rundate}__{self.bench_goal}.json"), "w"
                ) as f:
            json.dump(benchmark_result_dict, f)
        print("Results saved!")

    def create_benchmark_folder(self, folder_path):
        if os.path.isdir(folder_path):
            print(Warning("Folder already exists! No new folder will be created"))
        else:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Benchmark Folder created under: {self.benchmark_folder}")

    # def _check_json_serializabile_grid(self, to_serialize_dict):
    #     """Grid uses custom functions from optimization packages, therefore it might be anything. Make sure it
    #     is serializable.

    #     Args:
    #         to_serialize_dict (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     try:
    #         json.dumps(to_serialize_dict)
    #     except TypeError:
    #         to_serialize_dict["benchmark_configuration"]["grid"] = str(to_serialize_dict["benchmark_configuration"]["grid"])
    #     return to_serialize_dict
