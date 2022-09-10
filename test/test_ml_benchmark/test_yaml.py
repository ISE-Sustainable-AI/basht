from os import path
from ml_benchmark.utils.yaml import YMLHandler


def test_yaml():
    resources = YMLHandler.load_yaml(path.join(path.dirname(__file__),"test.yaml"))
    assert resources["deleteAfterRun"]

    print(resources["hyperparameter"])

    YMLHandler.as_yaml(path.join(path.dirname(__file__),"hyperparameter_space.yml"), resources["hyperparameter"])
    params = YMLHandler.load_yaml(path.join(path.dirname(__file__),"hyperparameter_space.yml"))
    assert params == resources["hyperparameter"]

