import pytest
import requests
import os

from ml_benchmark.decorators import latency_decorator, validation_latency_decorator
from ml_benchmark.workload.objective import Objective


@pytest.fixture
def objective():
    class TestObjective(Objective):

        def __init__(self) -> None:
            pass
        @latency_decorator
        def train(self):
            pass

        def get_hyperparameters(self) -> dict:
            return {"test":True}

        def set_hyperparameters(self, hyperparameters: dict):
            pass

        @validation_latency_decorator
        def validate(self):
            return {"macro avg":{"f1-score":0.5}}

        @latency_decorator
        def test(self):
            return {"score": 0.5}
    return TestObjective

@pytest.fixture
def prometeus_url():
    url =  os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            pytest.skip("Prometheus is availible")
    except Exception:
        pytest.skip("Could not connect to Prometheus.")
    return url
