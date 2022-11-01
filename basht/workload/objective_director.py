from basht.workload.models.mlp import MLP
from basht.workload.tasks.mnist_task import MnistTask
from basht.workload.torch_objective import Objective
from basht.config import WorkloadDataEnum, WorkloadFrameworkEnum, WorkloadModelEnum
from basht.workload.torch_objective_builder import TorchObjectiveBuilder
from basht.config import MLPHyperparameter


class ObjectiveDirector:

    building_directory = {
        "mlp": MLP,
        "mnist": MnistTask
    }

    builder_directory = {
        "torch": TorchObjectiveBuilder
    }

    def __init__(self, workload_definition: dict) -> None:
        self.builder = workload_definition.pop("dl_framework")
        self.builder = self.builder(**workload_definition)

    def build_objective(self) -> None:
        self._builder.build_task()
        print("Finished Task Building")
        self._builder.build_model()
        print("Finished Model Building")
        self._builder.create_objective()
        print("Created Objective from: ")

    def get_objective(self) -> Objective:
        return self._builder.objective
