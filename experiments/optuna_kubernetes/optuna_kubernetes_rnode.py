import logging
from os import path
from time import sleep
from experiments.optuna_kubernetes.optuna_kubernetes_benchmark import OptunaKubernetesBenchmark
from ml_benchmark.benchmark_runner import BenchmarkRunner
from urllib.request import urlopen
from ml_benchmark.utils.yml_parser import YMLParser

if __name__ == "__main__":
    metricsIP = urlopen("https://checkip.amazonaws.com").read().decode("utf-8").strip()

    # read in base configuration
    resources = YMLParser.parse(path.join(path.dirname(__file__),"resource_definition.yml"))
    # TODO: XXX remove this hardcoded values
    to_automate = {
        "metricsIP": metricsIP,
        "dockerImageTag": "tawalaya/optuna-trial:latest",
        "dockerImageBuilder": "docker",
        #force random namespaces to reduce conflicts
        # "kubernetesNamespace": "optuna-study",
        "kubernetesContext": "admin@smile",
        "kubernetesMasterIP": "130.149.158.143",
        "prometheus_url": "http://130.149.158.143:30041",
        "deleteAfterRun":True,
    	"epochs": 50,
    }
    resources.update(to_automate)

    repetions = 2
    for i in range(1,repetions+1):
        for n in range(1,7):
                sleep(3)
                logging.info(f"Starting Run {i} with {n} nodes with n_trails 100")
                try:
                    resources["trials"] = 100
                    resources["workerCount"] = n
                    resources["goal"] = f"rnode{n}-100-{i}"
                    runner = BenchmarkRunner(benchmark_cls=OptunaKubernetesBenchmark, resources=resources)
                    runner.run()
                    sleep(7)
                    runner = None
                except Exception as e:
                    logging.warning(f'Failed Run {i} with {n} nodes and n_trails 100 - {e}')
