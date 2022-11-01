from basht.workload.models.mlp import MLP
from basht.workload.torch_objective import Objective
from basht.workload.torch_objective_builder import TorchObjectiveBuilder


class ObjectiveBuildingManual:

    def __init__(self) -> None:
        pass


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
        # TODO: director shouöld provide a building manual by mapping config inputs to classes

    def build_objective(self) -> None:
        self._builder.build_task()
        print("Finished Task Building")
        self._builder.build_model()
        print("Finished Model Building")
        self._builder.create_objective()
        print("Created Objective from: ")

    def get_objective(self) -> Objective:
        return self._builder.objective
