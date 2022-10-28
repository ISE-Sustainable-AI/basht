from ml_benchmark.workload.torch_objective import TorchObjective
from ml_benchmark.workload.builder_interface import Builder


class TorchObjectiveBuilder(Builder):

    # use: https://refactoring.guru/design-patterns/builder/python/example

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.objective = TorchObjective()

    def build_task(self, task):
        pass

    def build_model(self, model):
        pass
