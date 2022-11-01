from basht.workload.torch_objective import TorchObjective
from basht.workload.builder_interface import Builder
from basht.workload.task import TorchTask
from basht.workload.models.model_interface import ObjModel
from basht.workload.task_components import Splitter, Loader, Batcher, Preprocessor


class TorchObjectiveBuilder(Builder):

    # use: https://refactoring.guru/design-patterns/builder/python/example

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

    def get_objective(self):
        # TODO: wraps everything up in a objective object, which the user uses as an interface
