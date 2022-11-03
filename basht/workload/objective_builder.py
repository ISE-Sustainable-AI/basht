from abc import ABC, abstractmethod

from basht.workload.task import TorchTask
from basht.workload.functional_objectives import TorchObjective
from basht.workload.task_components import Splitter, Loader, Batcher, Preprocessor, TorchImageFlattner, \
    TorchStandardBatcher, StandardTorchSplitter
from basht.workload.models import MLP


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


class BuilderMapper:

    objective = {
        "torch": TorchObjective
    }
    task = {
        "torch": TorchTask
    }
    torch_objects = [
        TorchImageFlattner, TorchStandardBatcher, StandardTorchSplitter, MLP
    ]
    tf_objects = [
    ]
    torch_components = {cls.name: cls for cls in torch_objects}
    tf_components = {cls.name: cls for cls in tf_objects}

    components = {
        "torch": torch_components,
        "tensorflow": tf_components}

    def __init__(self, dl_framework) -> None:
        self.components = self.components.get(dl_framework)
        self.task = self.task.get(dl_framework)
        self.objective = self.objective.get(dl_framework)


class ObjectiveBuilder(Builder):

    def __init__(self, dl_framework: str) -> None:
        self.mapper = BuilderMapper(dl_framework)
        self.objective = self.mapper.objective()
        self.task = self.mapper.task()

    def add_task_loader(self, loader: str):
        self.task.add_loader(self.mapper.components.get(loader))

    def add_task_preprocessors(self, preprocessors: list):
        for preprocessor in self.mapper.components.get(preprocessors):
            self.task.add_preprocessor(preprocessor)

    def add_task_splitters(self, splitter: dict):
        splitter_cls = self.mapper.components.get(splitter.get("type"))
        splitter_config = splitter.get("config")
        self.task.add_splitter(splitter_cls(splitter_config))

    def add_task_batcher(self, batcher: dict):
        batcher_cls = self.mapper.components.get(batcher.get("type"))
        batcher_config = batcher.get("config")
        self.task.add_batcher(batcher_cls(batcher_config))

    def add_task_to_objective(self):
        self.objective._add_task(self.task)
