from regex import splititer
from basht.workload.torch_objective import TorchObjective
from basht.workload.builder_interface import Builder


class TorchObjectiveBuilder(Builder):

    # use: https://refactoring.guru/design-patterns/builder/python/example

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.objective = TorchObjective()
        self.task = Task()

    def add_task(self):
        pass

    def add_hyperparameters(self):
        pass

    def add_model_cls(self):
        pass

    def build_task(self, task):
        self.task.loader = loader
        self.task.preprocessor = preprocessor
        self.task.splitter = splititer
        self.task.data_batcher = batcher
        return task

    # def create_objective(self):
    #     train_loader, val_loader, test_loader = self.create_data_loader(self.mnist_config)
    #     return self.objective_cls(
    #         self.mnist_config.epochs, train_loader, val_loader, test_loader, self.input_size,
    #         self.output_size)
