from basht.workload.torch_objective import TorchObjective
from basht.workload.builder_interface import Builder
from basht.workload.task import TorchTask
from basht.workload.models.model_interface import ObjModel


class TorchObjectiveBuilder(Builder):

    # use: https://refactoring.guru/design-patterns/builder/python/example

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.objective = TorchObjective()
        self.task = TorchTask()
        self.model_cls = None

    def add_task_loader(self, loader):
        self.task.add_loader(loader)

    def add_task_preprocessor(self, preprocessor):
        self.task.add_preprocessor(preprocessor)

    def add_task_splitter(self, splitter):
        self.task.add_splitter(splitter)

    def add_task_batcher(self, batcher):
        self.task.add_batcher(batcher)

    def add_model_cls(self, model_cls: ObjModel):
        self.model_cls = model_cls

    # def create_objective(self):
    #     train_loader, val_loader, test_loader = self.create_data_loader(self.mnist_config)
    #     return self.objective_cls(
    #         self.mnist_config.epochs, train_loader, val_loader, test_loader, self.input_size,
    #         self.output_size)
