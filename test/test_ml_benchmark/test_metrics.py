import imp
import logging
from ml_benchmark.metrics_storage import MetricsStorage
from ml_benchmark.resource_tracker import ResourceTracker
from ml_benchmark.results_tracker import ResultTracker

from ml_benchmark.workload.mnist.mnist_task import MnistTask
from time import sleep

def test_metrics(prometeus_url):
    task = MnistTask({"epochs": 1})
    objective = task.create_objective()
    metrics_storage = MetricsStorage()
    resourceTracker = ResourceTracker(prometheus_url=prometeus_url)
    try:
        metrics_storage.start_db()
        sleep(2)
        resourceTracker.start()
        objective.set_hyperparameters({"learning_rate":1e-3})
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
