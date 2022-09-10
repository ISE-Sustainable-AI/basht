from ml_benchmark.metrics_storage import MetricsStorage
import docker
import json
import os


def test_latency_decorator(objective):
    objective = objective()
    metrics_storage = MetricsStorage()

    try:
        metrics_storage.start_db()
        objective.train()
        objective.validate()
        objective.test()
        result = metrics_storage.get_benchmark_results()
        metrics_storage.stop_db()
    except docker.errors.APIError:
        metrics_storage.stop_db()

    assert isinstance(json.dumps(result), str)


def test_latency_decorator_using_env(objective):
    objective = objective()
    metrics_storage = MetricsStorage()

    try:
        metrics_storage.start_db()
        os.environ["METRICS_STORAGE_HOST"] = MetricsStorage.host
        objective.train()
        objective.validate()
        objective.test()
        result = metrics_storage.get_benchmark_results()
        metrics_storage.stop_db()
    except docker.errors.APIError:
        metrics_storage.stop_db()

    assert isinstance(json.dumps(result), str)
