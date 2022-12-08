from experiments.raytune_kubernetes.raytune_benchmark import RaytuneBenchmark
from experiments.optuna_kubernetes.optuna_kubernetes_benchmark import OptunaKubernetesBenchmark
from basht.utils.yaml import YamlTemplateFiller, YMLHandler
import os
from basht.benchmark_runner import BenchmarkRunner
from urllib.request import urlopen
from basht.config import Path
import itertools


class Experiment:

    def __init__(
        self, benchmark_cls, k8s_context="admin@smile", k8s_namespace="k8s-study",
        k8s_master_ip="130.149.158.143",
            dockertag="tawalaya/ccgrid-study:latest",
            metrics_ip=None, prometheus_url="http://130.149.158.143:30041", name=None, reps=1) -> None:

        self.benchmark_cls = benchmark_cls
        self.name = name
        self.reps = reps
        self.template_path = os.path.join(os.path.dirname(__file__), "resource_template.yml")
        self.resource_definition = dict(
                    kubecontext=k8s_context,
                    namespace=k8s_namespace,
                    kubemasterIP=k8s_master_ip,
                    image_tag=dockertag,
                    metrics_ip=metrics_ip if metrics_ip else urlopen("https://checkip.amazonaws.com").read().decode("utf-8").strip(),
                    prometheus_url=prometheus_url,
                    workerCpu=4,
                    workerMemory=4,
                    workerCount=4,
                    pruning=None
                )

    def horizontal_exp(self):
        start = 2
        end = 6

        for worker_num in range(start, end+1):
            self.resource_definition.update(
                dict(
                    workerCount=worker_num,
                    workerCpu=4,
                    workerMemory=4
                )
            )
            self.resource_definition.update({"goal": f"worker_{worker_num}"})
            self.start_benchmark("horizontal")

    def vertical_exp(self):

        memory_list = [4, 6, 8, 10, 12]
        cpu_list = [4, 6, 8, 10, 12]

        for cpu, memory in zip(cpu_list, memory_list):
            self.resource_definition.update(dict(
                workerCpu=cpu,
                workerMemory=memory,
                workerCount=2
            ))
            self.resource_definition.update({"goal": f"cpuem_{cpu}_{memory}"})
            self.start_benchmark("vertical")

    def pruning_exp(self):
        # For pruning we use 75 Epochs
        pruners = ["median", "hyperband"]
        search_spaces = ["small", "medium", "big", "vbig", "large"]
        search_spaces_folder_path = os.path.join(Path.root_path, "experiments/ccgrid_benchmark/search_spaces")

        for search_space, pruning in itertools.product(search_spaces, pruners):
            search_space_values = YMLHandler.load_yaml(
                os.path.join(search_spaces_folder_path, f"{search_space}.yml"))
            run_values = dict(
                pruning=pruning,
                hyperparameter=search_space_values,
                goal=f"pruning_{pruning}_space_{search_space}"
            )
            self.resource_definition.update(run_values)
            self.start_benchmark("pruning")

    def start_benchmark(self, exp_type):
        filled_template = YamlTemplateFiller.load_and_fill_yaml_template(
            self.template_path, self.resource_definition, as_dict=True)
        filled_template_path = os.path.join(self.benchmark_cls._path, "resource_definition.yml")
        YMLHandler.as_yaml(filled_template_path, filled_template)
        for i in range(self.reps):
            runner = BenchmarkRunner(
                benchmark_cls=self.benchmark_cls, resources=filled_template,
                name=f"__{self.name}" + f"__{exp_type}" + f"__{i}")
            runner.run()


if __name__ == "__main__":
    for benchmark_cls in [RaytuneBenchmark]:
        if benchmark_cls is OptunaKubernetesBenchmark:
            experiment = Experiment(
                benchmark_cls=benchmark_cls, name="ccgrid_run2", reps=2, dockertag="tawalaya/optuna-trial:latest")
        else:
            experiment = Experiment(
                benchmark_cls=benchmark_cls, name="ccgrid_run2", reps=2)
        experiment.pruning_exp()

# Raytune Horizontal is missing
