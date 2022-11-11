import torch
from basht.workload.objective import Objective
from basht.resources import Resouces

class TestObjective:

    def test_objective(self, resource_definition):
        

        # test
        objective = Objective(
            dl_framework=resource_definition.workload.dl_framework, model_cls=resource_definition.workload.model_cls, epochs=resource_definition.workload.epochs, device=resource_definition.workload.device,
            task=resource_definition.workload.task.to_dict())
        objective.train()

        # assert
        assert objective
        assert objective._functional_objective
