from abc import ABC, abstractmethod


from basht.workload.task import TorchTask
from basht.workload.models.model_interface import ObjModel
from basht.workload.functional_objectives import TorchObjective
from basht.workload.task_components import Splitter, Loader, Batcher, Preprocessor


class Builder(ABC):

    task = None
    model = None
    objective = None

    # use: https://refactoring.guru/design-patterns/builder/python/example

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def build_task(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def add_manual(self, manual):
        pass


class TorchObjectiveBuilder(Builder):

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.objective = TorchObjective()
        self.task = TorchTask()
        self.model_cls = None

    def add_task_loader(self, loader: Loader):
        self.task.add_loader(loader)

    def add_task_preprocessor(self, preprocessor: Preprocessor):
        self.task.add_preprocessor(preprocessor)

    def add_task_splitter(self, splitter: Splitter):
        self.task.add_splitter(splitter)

    def add_task_batcher(self, batcher: Batcher):
        self.task.add_batcher(batcher)

    def add_model_cls(self, model_cls: ObjModel):
        self.objective._add_model_cls(model_cls)

    def add_task_to_objective(self):
        self.objective._add_task(self.task)
