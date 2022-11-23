import pytest
import requests
import os

from basht.decorators import latency_decorator, validation_latency_decorator
from basht.workload.objective import Objective
from basht.resources import Resources
import torch


@pytest.fixture
def objective():
    class TestObjective:

        def __init__(self) -> None:
            pass

        @latency_decorator
        def train(self):
            pass

        def get_hyperparameters(self) -> dict:
            return {"test": True}

        def set_hyperparameters(self, hyperparameters: dict):
            pass

        @validation_latency_decorator
        def validate(self):
            return {"macro avg": {"f1-score": 0.5}}

        @latency_decorator
        def test(self):
            return {"score": 0.5}
    return TestObjective


@pytest.fixture
def prometeus_url():
    url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            pytest.skip("Prometheus is availible")
    except Exception:
        pytest.skip("Could not connect to Prometheus.")
    return url


@pytest.fixture
def resource_definition():
    res = Resources()
    res.trials = 2
    res.workload.epochs = 2

    return res


@pytest.fixture
def prepared_objective():
    res = Resouces()
    res.trials = 2
    res.workload.epochs = 2
    res.workload.device = "cuda" if torch.cuda.is_available() else "cpu"
    task = res.workload.task.to_dict()

    # test
    objective = Objective(
        dl_framework=res.workload.dl_framework, model_cls=res.workload.model_cls, epochs=res.workload.epochs, device=res.workload.device,
        task=task)
    return objective
