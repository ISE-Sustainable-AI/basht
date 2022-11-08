import imp
import logging
from basht.metrics_storage import MetricsStorage
from basht.resource_tracker import ResourceTracker
from basht.results_tracker import ResultTracker
from basht.workload.objective import Objective
import torch
from time import sleep


def test_metrics(prometeus_url, resource_definition):
    # setup
    workload_def = resource_definition["workload"]
    dl_framework = workload_def["dl_framework"]
    model_cls = workload_def["model_cls"]
    epochs = workload_def["epochs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = workload_def["task"]
    objective = Objective(
            dl_framework=dl_framework, model_cls=model_cls, epochs=epochs, device=device,
            task=task)
    metrics_storage = MetricsStorage()
    resourceTracker = ResourceTracker(prometheus_url=prometeus_url)
    try:
        metrics_storage.start_db()
        sleep(2)
        resourceTracker.start()
        objective.train()
        score = objective.validate()
        objective.test()

        sleep(15)

        result = metrics_storage.get_benchmark_results()
        logging.info(result)

        assert len(result["latency"]) > 0
        assert len(result["classification"]) > 0
        assert len(result["resources"]) > 0
    except Exception as e:
        assert False, e
    finally:
        resourceTracker.stop()
        metrics_storage.stop_db()
