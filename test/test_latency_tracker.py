from basht.metrics_storage import MetricsStorage
import docker
import json
import os


def test_latency_decorator(prepared_objective):
    objective = prepared_objective
    metrics_storage = MetricsStorage()

    try:
        metrics_storage.start_db()
        objective.load()
        objective.train()
        objective.validate()
        objective.test()
        result = metrics_storage.get_benchmark_results()
        metrics_storage.stop_db()
    except docker.errors.APIError:
        metrics_storage.stop_db()

    assert isinstance(json.dumps(result), str)


def test_latency_decorator_using_env(prepared_objective):
    objective = prepared_objective
    metrics_storage = MetricsStorage()

    try:
        metrics_storage.start_db()
        os.environ["METRICS_STORAGE_HOST"] = MetricsStorage.host
        objective.load()
        objective.train()
        objective.validate()
        objective.test()
        result = metrics_storage.get_benchmark_results()
        metrics_storage.stop_db()
    except docker.errors.APIError:
        metrics_storage.stop_db()

    assert isinstance(json.dumps(result), str)
