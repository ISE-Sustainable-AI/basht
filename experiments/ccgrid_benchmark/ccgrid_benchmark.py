from experiments.raytune_kubernetes.raytune_benchmark import RaytuneBenchmark
from experiments.optuna_kubernetes.optuna_kubernetes_benchmark import OptunaKubernetesBenchmark
from basht.utils.yaml import YamlTemplateFiller, YMLHandler
import os
from basht.benchmark_runner import BenchmarkRunner
from urllib.request import urlopen


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
                    workerCpu=2,
                    workerMemory=1,
                    workerCount=1,
                )

    def horizontal_exp(self):
        start = 2
        end = 3

        for worker_num in range(start, end+1):
            self.resource_definition.update(
                dict(
                    workerCount=worker_num,
                    workerCpu=4,
                    workerMemory=4
                )
            )
            self.start_benchmark("horizontal")

    def vertical_exp(self):
        memory_list = [4, 6]  # [4, 6, 8, 10, 12]
        cpu_list = [4, 6]  # # [4, 6, 8, 10, 12]

        for cpu, memory in zip(cpu_list, memory_list):
            self.resource_definition.update(dict(
                workerCpu=cpu,
                workerMemory=memory,
                workerCount=2
            ))
            self.start_benchmark("vertical")

    def pruning_exp(self):
        pruning = None
        search_spaces = None
        pass

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
    for benchmark_cls in [RaytuneBenchmark, OptunaKubernetesBenchmark]:
        if benchmark_cls is OptunaKubernetesBenchmark:
            experiment = Experiment(
                benchmark_cls=benchmark_cls, name="ccgrid_run2", reps=2, dockertag="tawalaya/optuna-trial:latest")
        else:
            experiment = Experiment(
                benchmark_cls=benchmark_cls, name="ccgrid_run2", reps=2)
        experiment.horizontal_exp()
