import pytest
import requests
import os

from basht.decorators import latency_decorator, validation_latency_decorator
from basht.config import Path
from basht.utils.yaml import YMLHandler

# @pytest.fixture
# def objective():
#     class TestObjective(Objective):

#         def __init__(self) -> None:
#             pass
#         @latency_decorator
#         def train(self):
#             pass

#         def get_hyperparameters(self) -> dict:
#             return {"test":True}

#         def set_hyperparameters(self, hyperparameters: dict):
#             pass

#         @validation_latency_decorator
#         def validate(self):
#             return {"macro avg":{"f1-score":0.5}}

#         @latency_decorator
#         def test(self):
#             return {"score": 0.5}
#     return TestObjective

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


@pytest.fixture
def resource_definition():
    test_file_path = os.path.join(Path.root_path, "test/test.yaml")
    def_dict = YMLHandler.load_yaml(test_file_path)
    return def_dict
