import base64
import json
from datetime import datetime
from abc import ABC, abstractmethod
import inspect
import os
import torch
import numpy as np
import random

from ml_benchmark.workload.mnist.mnist_task import MnistTask
from ml_benchmark.latency_tracker import LatencyTracker, Latency


class Benchmark(ABC):
    """
    This class serves as an Interface for a benchmark. All neccessary methods have to be implemented in the
    subclass that is using the interface. Make sure to use the predefined static variables. Your benchmark
    will most likely not run properly if the variables value remains to be "None".

    Args:
        ABC (_type_): Abstract Base Class
    """

    objective = None
    grid = None
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
    task_registry = {"mnist": MnistTask}

    def __init__(
            self, benchmark_cls: Benchmark, config: dict, grid: dict,
            resources: dict, task_str: str = "mnist") -> None:
        """
            benchName: uniqueName of the bechmark, used in logging
            config: configuration object
        """

        # get task
        task = self.task_registry.get(task_str)()

        # generate a unique name from the config
        base64_bytes = base64.b64encode(json.dumps(config).encode('ascii'))
        self.config_name = str(base64_bytes, 'ascii')
        self.rundate = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        benchmark_path = os.path.abspath(os.path.dirname(inspect.getabsfile(benchmark_cls)))
        self.bench_name = f"{task.__class__.__name__}__{benchmark_cls.__name__}"
        self.benchmark_folder = os.path.join(benchmark_path, f"benchmark__{self.bench_name}")
        self.create_benchmark_folder(self.benchmark_folder)

        # create loader
        epochs = config.pop("epochs")
        train_loader, val_loader, test_loader = task.create_data_loader(**config)
        objective = task.objective_cls(
            epochs=epochs,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

        grid["input_size"] = task.input_size
        grid["output_size"] = task.output_size
        grid = grid
        resources = resources
        self.benchmark = benchmark_cls(objective, grid, resources)

        # set seeds
        self._set_all_seeds()

        # prepare tracker
        self.latency_tracker = LatencyTracker()
        # TODO: add maximum available resources??

    def run(self):
        run_process = [
            self.benchmark.deploy, self.benchmark.run, self.benchmark.collect_run_results,
            self.benchmark.test, self.benchmark.collect_benchmark_metrics,
            self.benchmark.undeploy]

        for benchmark_fun in run_process:
            with Latency(benchmark_fun.__name__) as latency:
                results = benchmark_fun()
            self.latency_tracker.track(latency)
            if benchmark_fun.__name__ == self.benchmark.collect_benchmark_metrics.__name__:
                benchmark_execution_results = results

        benchmark_process_latencies = self.latency_tracker.get_recorded_latencies()
        benchmark_results = dict(
            benchmark_process_latencies=benchmark_process_latencies,
            benchmark_execution_results=benchmark_execution_results
        )
        self.save_benchmark_results(benchmark_results)

    def _set_all_seeds(self):
        torch.manual_seed(1337)
        np.random.seed(1337)
        random.seed(1337)

    def save_benchmark_results(self, benchmark_results):
        benchmark_config_dict = dict(
            objective=self.benchmark.objective.__class__.__name__,
            grid=self.benchmark.grid,
            resources=self.benchmark.resources,
        )
        benchmark_result_dict = dict(
            benchmark_metrics=benchmark_results,
            benchmark_configuration=benchmark_config_dict
        )
        benchmark_result_dict = self._check_json_serializabile_grid(benchmark_result_dict)
        with open(
            os.path.join(
                self.benchmark_folder,
                f"benchmark_results__{self.rundate}__id_{self.config_name}.json"), "w"
                ) as f:
            json.dump(benchmark_result_dict, f)
        print("Results saved!")

    def create_benchmark_folder(self, folder_path):
        if os.path.isdir(folder_path):
            print(Warning("Folder already exists! No new folder will be created"))
        else:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Benchmark Folder created under: {self.benchmark_folder}")

    def _check_json_serializabile_grid(self, to_serialize_dict):
        """Grid uses custom functions from optimization packages, therefore it might be anything. Make sure it
        is serializable.

        Args:
            to_serialize_dict (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            json.dumps(to_serialize_dict)
        except TypeError:
            to_serialize_dict["benchmark_configuration"]["grid"] = str(to_serialize_dict["benchmark_configuration"]["grid"])
        return to_serialize_dict