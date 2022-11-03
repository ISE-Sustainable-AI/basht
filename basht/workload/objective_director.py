from basht.workload.objective_builder import ObjectiveBuilder
from basht.workload.functional_objectives import FunctionalObjective


class ObjectiveDirector:

    def __init__(self) -> None:
        self.builder = ObjectiveBuilder()

    def build_objective(self, workload_definition: dict) -> FunctionalObjective:
        task_definition = workload_definition.get("task")
        if task_definition:
            self.builder.add_task_loader(task_definition.get("loader"))
            self.builder.add_task_preprocessors(task_definition.get("preprocessors"))
            self.builder.add_splitters(task_definition.get("splitters"))
            self.builder.add_batcher(task_definition.get("batcher"))
            print("Finished Task Building")
        self.builder.add_task_to_objective()
        print("Finished Objective Building")
        return self.builder.get_objective()
