import torch
from basht.workload.objective import Objective


class TestObjective:

    def test_objective(self, resource_definition):
        # setup
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
        objective.train()

        # assert
        assert objective
        assert isinstance(objective.workload_definition, dict)
        assert objective._functional_objective
