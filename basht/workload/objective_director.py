from basht.workload.models import *
from basht.workload.task_components import *
from basht.workload.torch_objective import Objective
from basht.workload.torch_objective_builder import TorchObjectiveBuilder


class ObjectiveBuildingManual:

    def __init__(self) -> None:
        pass


class ObjectiveDirector:

    building_directory = {
        "mlp": mlp.MLP
    }

    builder_directory = {
        "torch": TorchObjectiveBuilder
    }

    def __init__(self, workload_definition: dict) -> None:
        self.builder = workload_definition.pop("dl_framework")
        self.builder = self.builder(**workload_definition)
        # TODO: director shouöld provide a building manual by mapping config inputs to classes

    def build_objective(self) -> None:
        self.builder.reset()
        self.builder.add_task_loader(loader)
        self.builder.add_task_preprocessor(preprocessor) # TODO: requires loop
        self.builder.add_splitter(splitter)  # TODO: requires Loop
        self.builder.add_batcher(batcher)
        print("Finished Task Building")
        self.builder.add_model_cls(model_cls)
        self.builder.add_task_to_objective()
        return self.builder.get_objective()

    def get_objective(self) -> Objective:
        return self._builder.objective
