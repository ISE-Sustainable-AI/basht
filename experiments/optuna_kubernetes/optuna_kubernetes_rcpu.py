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
    for trials in [6,12,18]:
        for cpu in range(2,8):
            for i in range(1,repetions+1):
                    sleep(3)
                    logging.info(f"Starting Run {i} with 3x{cpu} vCPUs with n_trails {trials}")
                    try:
                        resources["trials"] = trials
                        resources["workerCpu"] = (cpu/2.0)
                        resources["goal"] = f"rcpu{cpu}-{trials}-{i}"
                        runner = BenchmarkRunner(benchmark_cls=OptunaKubernetesBenchmark, resources=resources)
                        runner.run()
                        sleep(7)
                        runner = None
                    except Exception as e:
                        logging.warning(f'Failed Run {i} with 3x{cpu} vCPUs with n_trails {trials} - {e}')
