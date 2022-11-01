from basht.workload.objective_builder import TorchObjectiveBuilder, TFObjectiveBuilder
from basht.workload.functional_objectives import FunctionalObjective


class ObjectiveDirector:

    builder_directory = {
        "torch": TorchObjectiveBuilder,
        "tensorflow": TFObjectiveBuilder
    }

    def __init__(self, workload_definition: dict) -> None:
        self.builder = self.builder_directory.get(workload_definition.get("dl_framework"))()

    def build_objective(self, workload_definition: dict) -> FunctionalObjective:
        self.builder.reset()
        self.builder.create_building_manual(workload_definition)
        self.builder.add_task_loader()
        self.builder.add_task_preprocessors()
        self.builder.add_splitters()
        self.builder.add_batcher()
        print("Finished Task Building")
        self.builder.add_model_cls()
        self.builder.add_epochs()
        self.builder.add_task_to_objective()
        print("Finished Objective Building")
        return self.builder.get_objective()
