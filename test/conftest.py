import pytest
import requests
import os

from basht.decorators import latency_decorator, validation_latency_decorator
from basht.config import Path
from basht.utils.yaml import YMLHandler
from basht.workload.objective import Objective
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
    test_file_path = os.path.join(Path.root_path, "test/test.yaml")
    def_dict = YMLHandler.load_yaml(test_file_path)
    return def_dict


@pytest.fixture
def prepared_objective():
    test_file_path = os.path.join(Path.root_path, "test/test.yaml")
    resource_definition = YMLHandler.load_yaml(test_file_path)
    workload_def = resource_definition["workload"]
    dl_framework = workload_def["dl_framework"]
    model_cls = workload_def["model_cls"]
    epochs = workload_def["epochs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = workload_def["task"]

    # test
    objective = Objective(
        dl_framework=dl_framework, model_cls=model_cls, epochs=epochs, device=device,
        task=task)
    return objective


@pytest.fixture
def exception_function():

    def cause_exception(cause):
        if cause:
            raise AttributeError("This is an Example")
        else:
            return True

    return cause_exception
