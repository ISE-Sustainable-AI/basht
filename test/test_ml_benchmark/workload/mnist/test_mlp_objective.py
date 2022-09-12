from ml_benchmark.workload.mnist.mnist_task import MnistTask
import pytest
import torch


@pytest.mark.parametrize(
    "device",
    [None, pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="no cuda device detected"))]
)
def test_mlp_objective_device(device):
    # setup
    task = MnistTask({"epochs": 10})
    objective = task.create_objective()
    objective.set_hyperparameters({"learning_rate": 1e-2, "weight_decay": 1e-6})
    objective.set_device(device)

    # perform
    objective.train()
    classification_report = objective.validate()

    # validate
    assert classification_report
