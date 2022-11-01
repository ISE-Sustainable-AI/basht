from abc import ABC, abstractmethod
from typing import List

from basht.workload.task import TorchTask
from basht.workload.models.model_interface import ObjModel
from basht.workload.functional_objectives import TorchObjective
from basht.workload.task_components import Splitter, Loader, Batcher, Preprocessor


class TorchBuildingManual:

    def __init__(self) -> None:
        pass


class Builder(ABC):

    task = None
    model_cls = None
    objective = None

    # use: https://refactoring.guru/design-patterns/builder/python/example

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self):
        pass


class TorchObjectiveBuilder(Builder):

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.objective = TorchObjective()
        self.task = TorchTask()
        self.model_cls = None
        self.building_manual = None

    def add_task_loader(self, loader: Loader):
        self.task.add_loader(loader)

    def add_task_preprocessors(self):
        for preprocessor in preprocessors:
            self.task.add_preprocessor(preprocessor)

    def add_task_splitters(self):
        for splitter in splitters:
            self.task.add_splitter(splitter)

    def add_task_batcher(self):
        self.task.add_batcher(batcher)

    def add_model_cls(self):
        self.objective._add_model_cls(model_cls)

    def add_epochs(self):
        self.objective._add_epochs(epochs)

    def add_task_to_objective(self):
        self.objective._add_task(self.task)

    def create_building_manual(self, workload_definition: dict):
        self.building_manual = TorchBuildingManual(workload_definition)
