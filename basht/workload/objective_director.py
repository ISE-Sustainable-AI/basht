from basht.workload.models import *
from basht.workload.task_components import *
from basht.workload.objective_builder import TorchObjectiveBuilder
from basht.workload.functional_objectives import FunctionalObjective


class ObjectiveBuildingManual:

    def __init__(self) -> None:
        pass


class ObjectiveDirector:

    builder_directory = {
        "torch": TorchObjectiveBuilder
    }

    def __init__(self, workload_definition: dict) -> None:
        self.builder = self.builder_directory.get(workload_definition.pop("dl_framework"))
        self.builder = self.builder(**workload_definition)
        # TODO: director shouöld provide a building manual by mapping config inputs to classes

    def build_objective(self, objective_building_manual: ObjectiveBuildingManual) -> FunctionalObjective:
        self.builder.reset()
        self.builder.add_task_loader(objective_building_manual.loader)
        for preprocessor in objective_building_manual.preprocessor:
            self.builder.add_task_preprocessor(preprocessor)
        for splitter in objective_building_manual.splitter:
            self.builder.add_splitter(splitter)
        self.builder.add_batcher(objective_building_manual.batcher)
        print("Finished Task Building")
        self.builder.add_model_cls(objective_building_manual.model_cls)
        self.builder.add_task_to_objective()
        print("Finished Objective Building")
        return self.builder.get_objective()
